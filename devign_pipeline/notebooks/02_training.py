# %% [markdown]
# # Devign Model Training Pipeline - Hybrid BiGRU + V2 Features
# 
# Train vulnerability detection models on preprocessed Devign dataset.
# 
# **Environment**: Kaggle with 2x NVIDIA T4 GPU (32GB total VRAM), 13GB RAM
# 
# **Model**: Hybrid BiGRU (Tokens) + Dense MLP (V2 Features) with Additive Attention
# 
# **Features**:
# - Multi-GPU training with DataParallel
# - Mixed precision training (AMP)
# - Gradient accumulation
# - OneCycleLR / ReduceLROnPlateau scheduling
# - Early stopping & checkpointing
# - Label smoothing option
# - Comprehensive metrics (Accuracy, Precision, Recall, F1, AUC-ROC)
# - **Dynamic Threshold Optimization** (maximizes Validation F1)
# - **Hybrid Architecture** combining code tokens and static vulnerability features

# %% [markdown]
# ## 1. Setup & Configuration

# %%
import os
import sys
import time
import json
import random
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.amp import autocast, GradScaler
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, confusion_matrix, classification_report
)
from tqdm.auto import tqdm

# Environment setup
if os.path.exists('/kaggle'):
    WORKING_DIR = '/kaggle/working'
    DATA_DIR = '/kaggle/input/devign-final/processed'
    sys.path.insert(0, '/tmp/devign_pipeline')
else:
    WORKING_DIR = '/media/hdi/Hdii/Work/C Vul Devign'
    DATA_DIR = '/media/hdi/Hdii/Work/C Vul Devign/Dataset/devign slice'
    sys.path.insert(0, '/media/hdi/Hdii/Work/C Vul Devign/devign_pipeline')

MODEL_DIR = os.path.join(WORKING_DIR, 'models')
LOG_DIR = os.path.join(WORKING_DIR, 'logs')

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# Device setup
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
N_GPUS = torch.cuda.device_count()

if torch.cuda.is_available():
    import torch.backends.cudnn as cudnn
    cudnn.benchmark = True
    gpu_info = ', '.join([f"{torch.cuda.get_device_properties(i).name}" for i in range(N_GPUS)])
    print(f"Device: {DEVICE} ({N_GPUS}x {gpu_info})")

# %% [markdown]
# ## 2. Training Configuration

# %%
def load_data_config(data_dir: str) -> Dict:
    """Load config from preprocessed data"""
    config_path = os.path.join(data_dir, 'config.json')
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return json.load(f)
    return {}

data_config = load_data_config(DATA_DIR)

class TrainConfig:
    """Training hyperparameters - optimized for Devign dataset"""
    # Model (REDUCED CAPACITY to combat overfitting - vocab=266, 21K samples)
    # Oracle analysis showed Train F1=0.79 vs Best Val F1=0.71 at epoch 17
    vocab_size: int = data_config.get('vocab_size', 266)
    embed_dim: int = 64           # DOWN from 128 - smaller vocab needs less embedding capacity
    hidden_dim: int = 128         # DOWN from 256 - BiGRU output = 256 (bidirectional)
    num_layers: int = 1           # DOWN from 2 - single layer less prone to overfitting
    rnn_dropout: float = 0.3      # Dropout between GRU layers (not used for num_layers=1)
    embedding_dropout: float = 0.15  # NEW - dropout after embedding layer
    classifier_dropout: float = 0.4  # DOWN from 0.5 - rebalanced with embedding dropout
    bidirectional: bool = True
    
    # Hybrid Model Features (V2)
    use_vuln_features: bool = True
    vuln_feature_dim: int = len(data_config.get('vuln_feature_names', [])) or 26
    vuln_feature_hidden_dim: int = 64
    vuln_feature_dropout: float = 0.2
    
    # Training (OPTIMIZED for dual T4 - 32GB VRAM total)
    batch_size: int = 128         # UP from 64 - tận dụng 32GB VRAM
    accumulation_steps: int = 1   # DOWN from 2 - batch đã đủ lớn
    learning_rate: float = 5e-4   # DOWN from 1e-3 - more stable convergence
    max_lr: float = 1.5e-3        # DOWN from 2e-3 - for OneCycleLR (if used)
    weight_decay: float = 1e-2    # UP from 5e-3 - stronger L2 regularization
    max_epochs: int = 35          # DOWN from 50 - oracle showed best at epoch 17
    grad_clip: float = 1.0
    
    # Regularization
    label_smoothing: float = 0.05  # Helps with noisy labels
    
    # Early stopping (MORE AGGRESSIVE)
    patience: int = 5             # DOWN from 10 - stop sooner when plateauing
    min_delta: float = 1e-4
    
    # Data
    max_seq_length: int = 512
    num_workers: int = 2          # DOWN from 4 - Kaggle chỉ 2 vCPU
    
    # Checkpointing
    save_every: int = 5
    
    # Mixed precision
    use_amp: bool = True
    
    # Scheduler: 'plateau' more stable than 'onecycle' for small datasets
    scheduler_type: str = 'plateau'  # CHANGED from 'onecycle'
    
    # Threshold Optimization
    use_optimal_threshold: bool = True
    threshold_min: float = 0.1
    threshold_max: float = 0.9
    threshold_step: float = 0.01
    
    def to_dict(self) -> Dict:
        return {k: v for k, v in self.__class__.__dict__.items() 
                if not k.startswith('_') and not callable(v)}

class LargeTrainConfig(TrainConfig):
    """
    Larger model configuration for better performance.
    - hidden_dim: 128 → 192 (BiGRU output = 384)
    - num_layers: 1 → 2 (stacked BiGRU)
    - Stronger regularization
    - Packed sequences enabled for 30-40% speedup
    """
    # Model capacity (INCREASED)
    embed_dim: int = 64
    hidden_dim: int = 192         # UP from 128 - BiGRU output = 384
    num_layers: int = 2           # UP from 1 - stacked BiGRU

    # Regularization (STRONGER)
    rnn_dropout: float = 0.4      # UP from 0.3 (active with 2 layers!)
    embedding_dropout: float = 0.2   # UP from 0.15
    classifier_dropout: float = 0.5  # UP from 0.4
    weight_decay: float = 2e-2    # UP from 1e-2 - stronger L2
    label_smoothing: float = 0.08 # UP from 0.05

    # Vuln features MLP
    vuln_feature_hidden_dim: int = 96  # UP from 64
    vuln_feature_dropout: float = 0.25 # UP from 0.2

    # Training (adjusted for larger model)
    batch_size: int = 96          # DOWN from 128 - larger model needs more memory
    accumulation_steps: int = 1
    learning_rate: float = 4e-4   # DOWN from 5e-4 - more conservative
    max_lr: float = 1.2e-3        # DOWN from 1.5e-3
    max_epochs: int = 40          # UP from 35 - allow more time to converge

    # Optimization features (NEW)
    use_packed_sequences: bool = True  # Enable packed sequences for 30-40% speedup

    # Early stopping
    patience: int = 5
    min_delta: float = 1e-4

    # Scheduler
    scheduler_type: str = 'plateau'

    def to_dict(self) -> Dict:
        return {k: v for k, v in self.__class__.__dict__.items() 
                if not k.startswith('_') and not callable(v)}


class RegularizedConfig(LargeTrainConfig):
    """
    Regularized config to reduce overfitting based on Oracle analysis.
    
    Key changes from LargeTrainConfig:
    - Reduced max_epochs (25) - Oracle showed best at epoch 14-16
    - Lower weight_decay (1e-4) - avoid over-regularization
    - Increased dropout slightly for better generalization
    - Shorter patience for early stopping (4)
    - Lower learning rate (3e-4) for more stable convergence
    """
    # Training - reduced epochs, stop earlier
    max_epochs: int = 25
    patience: int = 4
    learning_rate: float = 3e-4
    max_lr: float = 1.0e-3
    
    # Weight decay - reduced from 2e-2 to 1e-4 (per Oracle recommendation)
    weight_decay: float = 1e-4
    
    # Dropout - slightly increased for better generalization
    rnn_dropout: float = 0.4           # Keep same
    embedding_dropout: float = 0.25    # UP from 0.2
    classifier_dropout: float = 0.5    # Keep same
    vuln_feature_dropout: float = 0.30 # UP from 0.25
    
    # Label smoothing - reduced to avoid over-smoothing
    label_smoothing: float = 0.05
    
    # Scheduler settings
    scheduler_type: str = 'plateau'
    scheduler_patience: int = 2        # NEW: for ReduceLROnPlateau
    scheduler_factor: float = 0.5      # NEW: LR reduction factor
    scheduler_min_lr: float = 1e-6     # NEW: minimum LR
    
    # Data
    batch_size: int = 96
    max_seq_length: int = 512
    accumulation_steps: int = 1
    
    # Packed sequences
    use_packed_sequences: bool = True

    def to_dict(self) -> Dict:
        return {k: v for k, v in self.__class__.__dict__.items() 
                if not k.startswith('_') and not callable(v)}


class ImprovedConfig(TrainConfig):
    """
    Improved config based on Oracle analysis of training results.
    
    Key changes:
    - Reduced hidden_dim (192 → 128) - less overfitting with smaller capacity
    - Reduced vuln_feature_hidden_dim (96 → 48) - simpler MLP for 26 features
    - Added LayerNorm before classifier (enabled via use_layer_norm flag)
    - Token augmentation (dropout + masking) for robustness
    - SWA (Stochastic Weight Averaging) for better generalization
    - Tighter early stopping (patience=3, min_delta=5e-4)
    """
    # Model capacity (REDUCED for better generalization)
    embed_dim: int = 64
    hidden_dim: int = 128             # DOWN from 192 - BiGRU output = 256
    num_layers: int = 2               # Keep 2 layers for expressiveness
    
    # Regularization
    rnn_dropout: float = 0.35         # Slightly reduced for smaller model
    embedding_dropout: float = 0.2
    classifier_dropout: float = 0.5
    weight_decay: float = 1e-4
    label_smoothing: float = 0.05
    
    # Vuln features MLP (REDUCED)
    vuln_feature_hidden_dim: int = 48  # DOWN from 96
    vuln_feature_dropout: float = 0.25
    
    # NEW: LayerNorm before classifier
    use_layer_norm: bool = True
    
    # NEW: Token augmentation (only during training)
    use_token_augmentation: bool = True
    token_dropout_prob: float = 0.1    # Randomly drop 10% of tokens
    token_mask_prob: float = 0.05      # Randomly mask 5% of tokens
    mask_token_id: int = 1             # UNK token as mask
    
    # NEW: SWA (Stochastic Weight Averaging)
    use_swa: bool = True
    swa_start_epoch: int = 10          # Start SWA after epoch 10
    swa_lr: float = 1e-4               # Lower LR for SWA phase
    
    # Training
    batch_size: int = 96
    learning_rate: float = 3e-4
    max_lr: float = 1.0e-3
    max_epochs: int = 25
    
    # Early stopping (TIGHTER)
    patience: int = 3                  # DOWN from 4
    min_delta: float = 5e-4            # UP from 1e-4
    
    # Scheduler
    scheduler_type: str = 'plateau'
    scheduler_patience: int = 2
    scheduler_factor: float = 0.5
    scheduler_min_lr: float = 1e-6
    
    # Data
    max_seq_length: int = 512
    accumulation_steps: int = 1
    num_workers: int = 2
    
    # Packed sequences
    use_packed_sequences: bool = True

    def to_dict(self) -> Dict:
        return {k: v for k, v in self.__class__.__dict__.items() 
                if not k.startswith('_') and not callable(v)}


class EnhancedConfig(ImprovedConfig):
    """
    Enhanced configuration with:
    - hidden_dim: 128 → 160 (more capacity, same regularization)
    - Threshold search narrowed to [0.35, 0.45] with finer step
    - Multi-head attention pooling (replaces additive attention)
    - Ensemble training support (3-5 models with different seeds)
    """
    # Capacity (INCREASED slightly)
    hidden_dim: int = 160              # UP from 128 - BiGRU output = 320
    
    # Threshold optimization: narrow, finer grid
    threshold_min: float = 0.35
    threshold_max: float = 0.45
    threshold_step: float = 0.005      # Finer than 0.01
    
    # Multi-head attention pooling
    use_multihead_attention: bool = True
    num_attention_heads: int = 4
    attention_dropout: float = 0.1
    
    # Ensemble training
    ensemble_size: int = 5
    ensemble_base_seed: int = 1337
    
    # SWA: start earlier based on Oracle advice
    swa_start_epoch: int = 6           # DOWN from 10 - start SWA earlier
    
    def to_dict(self) -> Dict:
        return {k: v for k, v in self.__class__.__dict__.items() 
                if not k.startswith('_') and not callable(v)}


class OptimizedConfig(EnhancedConfig):
    """
    Optimized configuration targeting stable F1 > 0.72.

    Key changes vs EnhancedConfig:
    - Threshold search widened to [0.1, 0.9] with finer step 0.005 (for ensemble).
    - SWA start epoch moved closer to observed best (17–18) → 14.
    - Classifier dropout reduced to 0.4 (model not overfitting).
    - Dynamic label smoothing: 0.05 until epoch 15, then 0.0.
    - 5-seed ensemble + 1 SWA model (6-model ensemble at eval).
    - Optional fine-tuning tail at low LR (1e-5) after plateau.
    """

    # Threshold optimization for ensemble / validation (WIDENED)
    threshold_min: float = 0.1
    threshold_max: float = 0.9
    threshold_step: float = 0.005

    # Classifier dropout (down from 0.5 - model not overfitting)
    classifier_dropout: float = 0.4

    # SWA timing (moved closer to best epoch 17-18)
    swa_start_epoch: int = 14

    # Dynamic label smoothing schedule
    label_smoothing: float = 0.05
    label_smoothing_warmup_epochs: int = 15  # use smoothing through epoch 15, then 0.0

    # Ensemble: keep 5 base seeds; SWA becomes 6th model at eval
    ensemble_size: int = 5

    # Fine-tuning tail after plateau
    use_finetune_tail: bool = True
    finetune_epochs: int = 3          # number of low-LR epochs
    finetune_lr: float = 1e-5         # low LR for tail

    def to_dict(self) -> Dict:
        """Export a full config dict including inherited attributes."""
        cfg: Dict[str, Any] = {}
        for cls in reversed(self.__class__.mro()):
            if cls is object:
                continue
            for k, v in cls.__dict__.items():
                if not k.startswith('_') and not callable(v):
                    cfg[k] = v
        return cfg


# Use OptimizedConfig for this training run
config = OptimizedConfig()

# %% [markdown]
# ## 3. Dataset & DataLoader

# %%
class DevignDataset(Dataset):
    """Load preprocessed single .npz file (tokens + vuln features)"""
    
    def __init__(self, npz_path: str, max_seq_length: int = 512, load_vuln_features: bool = True):
        self.max_seq_length = max_seq_length
        self.load_vuln_features = load_vuln_features
        
        data = np.load(npz_path)
        self.input_ids = data['input_ids']
        self.labels = data['labels']
        
        # Handle attention_mask if present, otherwise generate
        if 'attention_mask' in data:
            self.attention_mask = data['attention_mask']
        else:
            self.attention_mask = None
        
        # Load vulnerability features if requested
        self.vuln_features = None
        if self.load_vuln_features:
            path_obj = Path(npz_path)
            vuln_path = path_obj.with_name(f"{path_obj.stem}_vuln.npz")
            
            if vuln_path.exists():
                vuln_data = np.load(vuln_path)
                if 'features' in vuln_data:
                    self.vuln_features = vuln_data['features']
                elif 'vuln_features' in vuln_data:
                    self.vuln_features = vuln_data['vuln_features']
                    
                if self.vuln_features is not None:
                    if len(self.vuln_features) != len(self.labels):
                        self.vuln_features = None
    
    def __len__(self) -> int:
        return len(self.labels)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Raw sequence for this sample
        raw_ids = self.input_ids[idx]
        
        # Truncate to max_seq_length
        input_ids = raw_ids[:self.max_seq_length]
        
        # Pad if needed
        padding_length = self.max_seq_length - len(input_ids)
        if padding_length > 0:
            input_ids = np.pad(input_ids, (0, padding_length), constant_values=0)
        
        # Attention mask (1 for real tokens, 0 for padding)
        if self.attention_mask is not None:
            attention_mask = self.attention_mask[idx][:self.max_seq_length]
            if len(attention_mask) < self.max_seq_length:
                attention_mask = np.pad(
                    attention_mask,
                    (0, self.max_seq_length - len(attention_mask)),
                    constant_values=0
                )
            # Compute length as number of real tokens after truncation
            orig_len = int(np.sum(attention_mask[:self.max_seq_length]))
        else:
            attention_mask = (input_ids != 0).astype(np.float32)
            orig_len = int(np.sum(attention_mask))
        
        # Clamp to [1, max_seq_length] to satisfy pack_padded_sequence
        orig_len = max(1, min(orig_len, self.max_seq_length))
            
        item = {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.float),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long),
            'lengths': torch.tensor(orig_len, dtype=torch.long)
        }
        
        # Add vulnerability features if available
        if self.vuln_features is not None:
            item['vuln_features'] = torch.tensor(self.vuln_features[idx], dtype=torch.float)
        else:
            # Fallback zero vector if missing but expected (to prevent crashing)
            # Assuming 26 dims based on config
            item['vuln_features'] = torch.zeros(26, dtype=torch.float)
            
        return item


def create_dataloaders(
    data_dir: str, 
    batch_size: int, 
    max_seq_length: int,
    num_workers: int = 4
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train/val/test dataloaders"""
    
    train_path = Path(data_dir) / 'train.npz'
    val_path = Path(data_dir) / 'val.npz'
    test_path = Path(data_dir) / 'test.npz'
    
    if train_path.exists():
        train_dataset = DevignDataset(str(train_path), max_seq_length)
        val_dataset = DevignDataset(str(val_path), max_seq_length)
        test_dataset = DevignDataset(str(test_path), max_seq_length)
    else:
        raise ValueError(f"Could not find train.npz in {data_dir}")
    
    print(f"Data: {len(train_dataset)} train, {len(val_dataset)} val, {len(test_dataset)} test")
    
    # Tối ưu DataLoader cho dual T4
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, 
        num_workers=num_workers, pin_memory=True, drop_last=True,
        persistent_workers=num_workers > 0, prefetch_factor=2 if num_workers > 0 else None
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
        persistent_workers=num_workers > 0, prefetch_factor=2 if num_workers > 0 else None
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
        persistent_workers=num_workers > 0, prefetch_factor=2 if num_workers > 0 else None
    )
    
    return train_loader, val_loader, test_loader

# %%
train_loader, val_loader, test_loader = create_dataloaders(
    DATA_DIR, 
    config.batch_size, 
    config.max_seq_length,
    config.num_workers
)

# Class distribution
train_labels = train_loader.dataset.labels
print(f"Class: neg={np.sum(train_labels==0)}, pos={np.sum(train_labels==1)}")

# %% [markdown]
# ## 4. Hybrid BiGRU Model with V2 Features

# %%
class MultiHeadSelfAttentionPooling(nn.Module):
    """
    Multi-head self-attention pooling over BiGRU outputs.
    Uses a single learned query to attend over the sequence.
    
    Inputs:
      - rnn_outputs: [B, T, D]
      - attention_mask: [B, T] (1 = keep, 0 = pad)
    Output:
      - context: [B, D]
    """
    def __init__(self, input_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.input_dim = input_dim
        self.num_heads = num_heads
        
        self.query = nn.Parameter(torch.randn(1, 1, input_dim))
        self.mha = nn.MultiheadAttention(
            embed_dim=input_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, rnn_outputs: torch.Tensor, attention_mask: torch.Tensor):
        bsz = rnn_outputs.size(0)
        
        query = self.query.expand(bsz, -1, -1)  # [B, 1, D]
        key_padding_mask = ~attention_mask.bool()  # True = ignore
        
        attn_output, _ = self.mha(
            query, rnn_outputs, rnn_outputs,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        
        context = attn_output.squeeze(1)  # [B, D]
        context = self.dropout(context)
        return context


class ImprovedHybridBiGRUVulnDetector(nn.Module):
    """
    Improved Hybrid BiGRU with:
    1. Packed sequences support (30-40% speedup)
    2. Dropout on attention context
    3. Token augmentation (dropout + masking) during training
    4. LayerNorm before classifier for better generalization
    5. Reduced capacity (hidden_dim=128) to combat overfitting
    """
    
    def __init__(self, config: TrainConfig):
        super().__init__()
        self.config = config
        
        # Token augmentation params
        self.use_token_augmentation = getattr(config, 'use_token_augmentation', False)
        self.token_dropout_prob = getattr(config, 'token_dropout_prob', 0.1)
        self.token_mask_prob = getattr(config, 'token_mask_prob', 0.05)
        self.mask_token_id = getattr(config, 'mask_token_id', 1)  # UNK token
        
        # Embedding
        self.embedding = nn.Embedding(
            config.vocab_size, 
            config.embed_dim, 
            padding_idx=0
        )
        embed_drop_rate = getattr(config, 'embedding_dropout', 0.15)
        self.embed_dropout = nn.Dropout(embed_drop_rate)
        
        # BiGRU
        self.gru = nn.GRU(
            config.embed_dim,
            config.hidden_dim,
            num_layers=config.num_layers,
            bidirectional=config.bidirectional,
            batch_first=True,
            dropout=config.rnn_dropout if config.num_layers > 1 else 0.0
        )
        
        self.rnn_out_dim = config.hidden_dim * (2 if config.bidirectional else 1)
        
        # Attention mechanism (configurable: additive or multi-head)
        self.use_multihead_attention = getattr(config, 'use_multihead_attention', False)
        if self.use_multihead_attention:
            self.attention = MultiHeadSelfAttentionPooling(
                input_dim=self.rnn_out_dim,
                num_heads=getattr(config, 'num_attention_heads', 4),
                dropout=getattr(config, 'attention_dropout', 0.1),
            )
        else:
            self.attention = nn.Sequential(
                nn.Linear(self.rnn_out_dim, self.rnn_out_dim // 2),
                nn.Tanh(),
                nn.Linear(self.rnn_out_dim // 2, 1, bias=False)
            )
        # Dropout on attention context (regularization)
        self.context_dropout = nn.Dropout(0.2)
        
        # Vuln features branch
        if config.use_vuln_features:
            self.vuln_bn_in = nn.BatchNorm1d(config.vuln_feature_dim)
            vuln_hidden = getattr(config, 'vuln_feature_hidden_dim', 64)
            self.vuln_mlp = nn.Sequential(
                nn.Linear(config.vuln_feature_dim, vuln_hidden),
                nn.BatchNorm1d(vuln_hidden),
                nn.GELU(),
                nn.Dropout(config.vuln_feature_dropout)
            )
            self.combined_dim = self.rnn_out_dim + vuln_hidden
        else:
            self.combined_dim = self.rnn_out_dim
        
        # LayerNorm before classifier (for better generalization)
        self.use_layer_norm = getattr(config, 'use_layer_norm', False)
        if self.use_layer_norm:
            self.pre_classifier_ln = nn.LayerNorm(self.combined_dim)
            
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(self.combined_dim, self.combined_dim // 2),
            nn.BatchNorm1d(self.combined_dim // 2),
            nn.GELU(),
            nn.Dropout(config.classifier_dropout),
            nn.Linear(self.combined_dim // 2, 2)
        )
    
    def apply_token_augmentation(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Apply token-level augmentation during training:
        1. Token dropout: randomly replace tokens with PAD (0)
        2. Token masking: randomly replace tokens with MASK/UNK token
        
        Only applied to non-special tokens (skip PAD, BOS, EOS).
        """
        if not self.training or not self.use_token_augmentation:
            return input_ids
        
        augmented = input_ids.clone()
        B, L = input_ids.shape
        
        # Create mask for valid tokens (not padding, not special tokens 0,2,3)
        valid_mask = (input_ids > 3) & (attention_mask > 0)
        
        # Token dropout: replace with PAD (0)
        if self.token_dropout_prob > 0:
            dropout_mask = torch.rand(B, L, device=input_ids.device) < self.token_dropout_prob
            dropout_mask = dropout_mask & valid_mask
            augmented = augmented.masked_fill(dropout_mask, 0)
        
        # Token masking: replace with MASK token (UNK=1)
        if self.token_mask_prob > 0:
            mask_mask = torch.rand(B, L, device=input_ids.device) < self.token_mask_prob
            mask_mask = mask_mask & valid_mask & (augmented > 0)  # Don't mask already dropped tokens
            augmented = augmented.masked_fill(mask_mask, self.mask_token_id)
        
        return augmented
        
    def forward(self, input_ids, attention_mask, vuln_features=None, lengths=None):
        """
        Args:
            input_ids: [B, L]
            attention_mask: [B, L]
            vuln_features: [B, F] or None
            lengths: [B] actual (non-pad) sequence lengths for packed sequences
        """
        B, L = input_ids.shape
        
        # Apply token augmentation during training
        augmented_ids = self.apply_token_augmentation(input_ids, attention_mask)
        
        # Embedding
        embedded = self.embedding(augmented_ids)  # [B, L, E]
        embedded = self.embed_dropout(embedded)
        
        # GRU with optional packing
        use_packing = getattr(self.config, 'use_packed_sequences', False)
        
        if use_packing and lengths is not None:
            # Pack sequences (skip padding for 30-40% speedup)
            lengths_cpu = lengths.cpu()
            packed = nn.utils.rnn.pack_padded_sequence(
                embedded, lengths_cpu, batch_first=True, enforce_sorted=False
            )
            packed_out, _ = self.gru(packed)
            rnn_out, _ = nn.utils.rnn.pad_packed_sequence(
                packed_out, batch_first=True, total_length=L
            )  # [B, L, D]
        else:
            # Standard GRU
            rnn_out, _ = self.gru(embedded)  # [B, L, D]
        
        # Attention (handles both additive and multi-head)
        if self.use_multihead_attention:
            context_vector = self.attention(rnn_out, attention_mask)  # [B, D]
        else:
            att_scores = self.attention(rnn_out)  # [B, L, 1]
            mask = attention_mask.unsqueeze(-1)  # [B, L, 1]
            att_scores = att_scores.masked_fill(mask == 0, -1e4)
            att_weights = F.softmax(att_scores, dim=1)
            context_vector = torch.sum(rnn_out * att_weights, dim=1)  # [B, D]
            context_vector = self.context_dropout(context_vector)
        
        # Vuln features
        if self.config.use_vuln_features and vuln_features is not None:
            # vuln_features: [B, F]
            feat_out = self.vuln_bn_in(vuln_features)
            feat_out = self.vuln_mlp(feat_out)
            
            # Concatenate
            combined = torch.cat([context_vector, feat_out], dim=1)
        else:
            combined = context_vector
        
        # LayerNorm before classifier (if enabled)
        if self.use_layer_norm:
            combined = self.pre_classifier_ln(combined)
            
        # Classification
        logits = self.classifier(combined)
        return logits

# Initialize model
model = ImprovedHybridBiGRUVulnDetector(config)
model.to(DEVICE)

# DataParallel for multi-GPU
if N_GPUS > 1:
    model = nn.DataParallel(model)

params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Model: {params:,} params, hidden={config.hidden_dim}, layers={config.num_layers}")

# %% [markdown]
# ## 5. Training Utilities with Threshold Optimization

# %%
class EarlyStopping:
    """
    Early stopping for maximizing metrics (e.g., F1, AUC-ROC).
    Stops training when metric doesn't improve for `patience` epochs.
    """
    def __init__(self, patience=7, min_delta=0, verbose=True):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_metric):
        score = val_metric
        
        if self.best_score is None:
            self.best_score = score
            if self.verbose:
                print(f"  EarlyStopping: initialized with score={score:.4f}")
        elif score <= self.best_score + self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f"  EarlyStopping: no improvement ({self.counter}/{self.patience})")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            if self.verbose:
                print(f"  EarlyStopping: improved {self.best_score:.4f} → {score:.4f}")
            self.best_score = score
            self.counter = 0
            
class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1, weight=None):
        super().__init__()
        self.smoothing = smoothing
        if weight is not None:
            self.register_buffer("weight", weight.float())
        else:
            self.weight = None

    def forward(self, preds, target):
        n_classes = preds.size(1)
        log_preds = F.log_softmax(preds, dim=1)
        loss = -log_preds.sum(dim=1)
        
        weight = self.weight
        if weight is not None:
            weight = weight.to(device=preds.device, dtype=preds.dtype)
            
        return F.cross_entropy(preds, target, weight=weight, label_smoothing=self.smoothing)

def find_optimal_threshold(y_true, y_probs, min_t=0.1, max_t=0.9, step=0.005):
    """Find probability threshold that maximizes F1 score.
    
    Args:
        y_true: Ground truth labels
        y_probs: Predicted probabilities
        min_t: Minimum threshold to search (default: 0.1)
        max_t: Maximum threshold to search (default: 0.9)
        step: Step size for threshold search (default: 0.005 for finer search)
    """
    thresholds = np.arange(min_t, max_t + step, step)
    best_t = 0.5
    best_f1 = 0.0
    
    for t in thresholds:
        y_pred = (y_probs >= t).astype(int)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_t = t
            
    return best_t, best_f1


@torch.no_grad()
def update_bn_dict_loader(loader, model, device=None):
    """
    Custom update_bn for DataLoaders that yield dict batches.
    Recomputes BatchNorm running statistics for SWA models.
    """
    from torch.nn.modules.batchnorm import _BatchNorm
    
    has_bn = any(isinstance(m, _BatchNorm) for m in model.modules())
    if not has_bn:
        return
    
    was_training = model.training
    model.train()
    
    momenta = {}
    for module in model.modules():
        if isinstance(module, _BatchNorm):
            module.running_mean = torch.zeros_like(module.running_mean)
            module.running_var = torch.ones_like(module.running_var)
            momenta[module] = module.momentum
            module.momentum = None
    
    n = 0
    for batch in loader:
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        vuln_features = batch.get("vuln_features", None)
        lengths = batch.get("lengths", None)
        
        if device is not None:
            input_ids = input_ids.to(device, non_blocking=True)
            attention_mask = attention_mask.to(device, non_blocking=True)
            if vuln_features is not None:
                vuln_features = vuln_features.to(device, non_blocking=True)
            if lengths is not None:
                lengths = lengths.to(device, non_blocking=True)
        
        b = input_ids.size(0)
        momentum = b / float(n + b)
        for module in momenta.keys():
            module.momentum = momentum
        
        model(input_ids, attention_mask, vuln_features, lengths)
        n += b
    
    for module, mom in momenta.items():
        module.momentum = mom
    
    model.train(was_training)


def train_epoch(model, loader, optimizer, criterion, scaler, grad_clip, accum_steps, use_amp):
    model.train()
    total_loss = 0
    all_preds, all_labels = [], []
    
    optimizer.zero_grad()
    
    for i, batch in enumerate(tqdm(loader, desc="Training", leave=False)):
        input_ids = batch['input_ids'].to(DEVICE, non_blocking=True)
        attention_mask = batch['attention_mask'].to(DEVICE, non_blocking=True)
        labels = batch['labels'].to(DEVICE, non_blocking=True)
        vuln_features = batch['vuln_features'].to(DEVICE, non_blocking=True) if 'vuln_features' in batch else None
        lengths = batch['lengths'].to(DEVICE, non_blocking=True)
        
        with autocast(device_type='cuda', enabled=use_amp):
            logits = model(input_ids, attention_mask, vuln_features, lengths)
            loss = criterion(logits, labels)
            loss = loss / accum_steps
            
        scaler.scale(loss).backward()
        
        if (i + 1) % accum_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            
        total_loss += loss.item() * accum_steps
        
        preds = logits.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())
        
    avg_loss = total_loss / len(loader)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    try:
        auc_roc = roc_auc_score(all_labels, all_preds)
    except:
        auc_roc = 0.5
        
    return {
        'loss': avg_loss,
        'f1': f1,
        'auc_roc': auc_roc
    }

@torch.no_grad()
def evaluate(model, loader, criterion, use_amp, threshold=None, find_threshold=False):
    model.eval()
    total_loss = 0
    all_labels = []
    all_probs = []
    
    for batch in tqdm(loader, desc="Evaluating", leave=False):
        input_ids = batch['input_ids'].to(DEVICE, non_blocking=True)
        attention_mask = batch['attention_mask'].to(DEVICE, non_blocking=True)
        labels = batch['labels'].to(DEVICE, non_blocking=True)
        vuln_features = batch['vuln_features'].to(DEVICE, non_blocking=True) if 'vuln_features' in batch else None
        lengths = batch['lengths'].to(DEVICE, non_blocking=True)
        
        with autocast(device_type='cuda', enabled=use_amp):
            logits = model(input_ids, attention_mask, vuln_features, lengths)
            loss = criterion(logits, labels)
            
        total_loss += loss.item()
        
        probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs)
        
    avg_loss = total_loss / len(loader)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # Determine threshold
    best_t = 0.5
    if find_threshold:
        best_t, best_f1 = find_optimal_threshold(all_labels, all_probs)
        used_t = best_t
    elif threshold is not None:
        used_t = threshold
    else:
        used_t = 0.5
        
    # Apply threshold
    preds = (all_probs >= used_t).astype(int)
    
    # Metrics
    accuracy = accuracy_score(all_labels, preds)
    precision = precision_score(all_labels, preds, zero_division=0)
    recall = recall_score(all_labels, preds, zero_division=0)
    f1 = f1_score(all_labels, preds, zero_division=0)
    try:
        auc_roc = roc_auc_score(all_labels, all_probs)
    except:
        auc_roc = 0.5
        
    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc_roc': auc_roc,
        'best_threshold': best_t if find_threshold else used_t,
        'labels': all_labels,    # For ensemble thresholding
        'probs': all_probs,      # For ensemble thresholding
    }


# %% [markdown]
# ## 5.5 Ensemble Training Utilities

# %%
def set_global_seed(seed: int):
    """Set all random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model(config):
    """Build and initialize model"""
    model = ImprovedHybridBiGRUVulnDetector(config)
    model.to(DEVICE)
    if N_GPUS > 1:
        model = nn.DataParallel(model)
    return model


@torch.no_grad()
def predict_ensemble(models, loader, threshold, use_amp: bool):
    """
    Run ensemble inference by averaging probabilities from multiple models.
    
    Returns metrics dict with accuracy, precision, recall, f1, auc_roc.
    """
    for m in models:
        m.eval()
    
    all_labels = []
    all_probs_ensemble = []
    
    for batch in tqdm(loader, desc="Ensemble Inference", leave=False):
        input_ids = batch['input_ids'].to(DEVICE, non_blocking=True)
        attention_mask = batch['attention_mask'].to(DEVICE, non_blocking=True)
        vuln_features = batch.get('vuln_features', None)
        if vuln_features is not None:
            vuln_features = vuln_features.to(DEVICE, non_blocking=True)
        lengths = batch['lengths'].to(DEVICE, non_blocking=True)
        
        probs_sum = 0.0
        for model in models:
            with autocast(device_type='cuda', enabled=use_amp):
                logits = model(input_ids, attention_mask, vuln_features, lengths)
                probs = torch.softmax(logits, dim=1)[:, 1]
            probs_sum += probs
        
        probs_avg = (probs_sum / len(models)).cpu().numpy()
        all_probs_ensemble.extend(probs_avg)
        all_labels.extend(batch['labels'].numpy())
    
    all_labels = np.array(all_labels)
    all_probs_ensemble = np.array(all_probs_ensemble)
    preds = (all_probs_ensemble >= threshold).astype(int)
    
    accuracy = accuracy_score(all_labels, preds)
    precision = precision_score(all_labels, preds, zero_division=0)
    recall = recall_score(all_labels, preds, zero_division=0)
    f1 = f1_score(all_labels, preds, zero_division=0)
    try:
        auc_roc = roc_auc_score(all_labels, all_probs_ensemble)
    except:
        auc_roc = 0.5
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc_roc': auc_roc,
        'labels': all_labels,
        'probs': all_probs_ensemble,
    }

# %% [markdown]
# ## 6. Main Training Loop

# %%
def train(model, train_loader, val_loader, config):
    """
    Main training loop with dynamic label smoothing, fine-tuning tail, and SWA.
    
    Improvements in OptimizedConfig:
    - Dynamic label smoothing: 0.05 until epoch 15, then 0.0 for sharper boundaries
    - Fine-tuning tail: low-LR (1e-5) epochs after plateau for fine-grained optimization
    - SWA starts at epoch 14 (closer to best epoch 17-18)
    """
    # Calculate class weights
    all_labels = []
    for batch in train_loader:
        all_labels.extend(batch['labels'].numpy())
    
    neg_count = np.sum(np.array(all_labels) == 0)
    pos_count = np.sum(np.array(all_labels) == 1)
    
    # Slight boost to minority class if imbalanced
    ratio = float(neg_count) / float(pos_count)
    weight = torch.tensor([1.0, ratio], dtype=torch.float32, device=DEVICE)
    
    # Dynamic label smoothing schedule
    base_smoothing = float(getattr(config, 'label_smoothing', 0.0))
    smoothing_warmup_epochs = int(getattr(config, 'label_smoothing_warmup_epochs', 0))
    
    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    
    # Scheduler
    if config.scheduler_type == 'onecycle':
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=config.max_lr,
            steps_per_epoch=len(train_loader) // config.accumulation_steps,
            epochs=config.max_epochs,
            pct_start=0.1,
            anneal_strategy='cos'
        )
    else:
        # ReduceLROnPlateau - monitors val_auc_roc for stability
        scheduler_patience = getattr(config, 'scheduler_patience', 2)
        scheduler_factor = getattr(config, 'scheduler_factor', 0.5)
        scheduler_min_lr = getattr(config, 'scheduler_min_lr', 1e-6)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='max', 
            factor=scheduler_factor, 
            patience=scheduler_patience,
            threshold=1e-3,
            min_lr=scheduler_min_lr
        )
    
    scaler = GradScaler(enabled=config.use_amp)
    early_stopping = EarlyStopping(config.patience, config.min_delta)
    
    # SWA setup (Stochastic Weight Averaging)
    use_swa = getattr(config, 'use_swa', False)
    swa_model = None
    swa_scheduler = None
    swa_start_epoch = int(getattr(config, 'swa_start_epoch', 10))
    
    if use_swa:
        # Create SWA model wrapper
        base_model = model.module if isinstance(model, nn.DataParallel) else model
        swa_model = AveragedModel(base_model)
        swa_lr = float(getattr(config, 'swa_lr', 1e-4))
        swa_scheduler = SWALR(optimizer, swa_lr=swa_lr, anneal_epochs=2)
        print(f"SWA enabled: starts at epoch {swa_start_epoch}, lr={swa_lr}")
    
    # Fine-tuning tail config
    use_finetune_tail = bool(getattr(config, 'use_finetune_tail', False))
    finetune_lr = float(getattr(config, 'finetune_lr', 1e-5))
    finetune_epochs = int(getattr(config, 'finetune_epochs', 3))
    in_finetune = False
    finetune_epochs_done = 0
    
    history = {'train': [], 'val': []}
    best_f1 = 0.0
    best_epoch = 0
    best_threshold_overall = 0.5
    
    print("\n" + "="*50)
    print(f"Training: batch={config.batch_size}, epochs={config.max_epochs}, AMP={config.use_amp}")
    if smoothing_warmup_epochs > 0:
        print(f"Dynamic label smoothing: {base_smoothing} until epoch {smoothing_warmup_epochs}, then 0.0")
    if use_finetune_tail:
        print(f"Fine-tuning tail: {finetune_epochs} epochs at LR={finetune_lr}")
    print("="*50)
    
    for epoch in range(1, config.max_epochs + 1):
        epoch_start = time.time()
        
        # Dynamic label smoothing: base_smoothing until warmup_epochs, then 0.0
        if base_smoothing > 0 and (smoothing_warmup_epochs == 0 or epoch <= smoothing_warmup_epochs):
            current_smoothing = base_smoothing
        else:
            current_smoothing = 0.0
        
        # Create criterion with current smoothing
        if current_smoothing > 0:
            criterion = LabelSmoothingCrossEntropy(
                smoothing=current_smoothing, 
                weight=weight
            )
        else:
            criterion = nn.CrossEntropyLoss(weight=weight)
        
        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, criterion, scaler,
            config.grad_clip, config.accumulation_steps, config.use_amp
        )
        
        # Validate (with dynamic threshold search)
        val_metrics = evaluate(
            model, val_loader, criterion, config.use_amp,
            find_threshold=config.use_optimal_threshold
        )
        
        epoch_time = time.time() - epoch_start
        
        # Log
        history['train'].append(train_metrics)
        history['val'].append(val_metrics)
        
        current_lr = optimizer.param_groups[0]['lr']
        val_t = val_metrics['best_threshold']
        
        print(
            f"Ep {epoch:2d}/{config.max_epochs} ({epoch_time:.0f}s) | "
            f"Train: L={train_metrics['loss']:.3f} F1={train_metrics['f1']:.3f} | "
            f"Val: L={val_metrics['loss']:.3f} F1={val_metrics['f1']:.3f} "
            f"AUC={val_metrics['auc_roc']:.3f} T={val_t:.2f} | "
            f"LR={current_lr:.2e} | smooth={current_smoothing:.3f}"
        )
        
        # SWA update (after swa_start_epoch)
        if use_swa and epoch >= swa_start_epoch:
            base_model = model.module if isinstance(model, nn.DataParallel) else model
            swa_model.update_parameters(base_model)
            swa_scheduler.step()
            print(f"  [SWA] Updated averaged model (epoch {epoch})")
        elif config.scheduler_type == 'plateau' and not in_finetune:
            # Only use regular scheduler before SWA kicks in and not in finetune
            scheduler.step(val_metrics['auc_roc'])
        
        # Save best model (based on F1)
        if val_metrics['f1'] > best_f1:
            best_f1 = val_metrics['f1']
            best_epoch = epoch
            best_threshold_overall = val_t
            
            save_dict = {
                'epoch': epoch,
                'model_state_dict': model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_metrics': val_metrics,
                'config': config.to_dict(),
                'best_threshold': best_threshold_overall
            }
            torch.save(save_dict, os.path.join(MODEL_DIR, 'best_model.pt'))
            print(f"  ★ Best F1={best_f1:.4f}")
        
        # Regular checkpoint
        if epoch % config.save_every == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict(),
                'best_threshold': val_t
            }, os.path.join(MODEL_DIR, f'checkpoint_epoch_{epoch}.pt'))
        
        # Early stopping + optional fine-tuning tail
        if early_stopping(val_metrics['f1']):
            if use_finetune_tail and not in_finetune:
                # Enter fine-tuning tail at low LR instead of stopping immediately
                in_finetune = True
                finetune_epochs_done = 0
                early_stopping.counter = 0
                early_stopping.early_stop = False
                for pg in optimizer.param_groups:
                    pg['lr'] = finetune_lr
                print(f"  → Entering fine-tuning tail: LR={finetune_lr:.2e} for up to {finetune_epochs} epochs")
            else:
                print(f"Early stopping at epoch {epoch}")
                break
        
        if in_finetune:
            finetune_epochs_done += 1
            if finetune_epochs_done >= finetune_epochs:
                print(f"Completed fine-tuning tail for {finetune_epochs} epochs. Stopping training.")
                break
    
    # SWA: Update BatchNorm statistics and save SWA model
    swa_threshold = best_threshold_overall
    if use_swa and swa_model is not None and epoch >= swa_start_epoch:
        print("\n[SWA] Updating BatchNorm statistics...")
        # Update BN stats with training data (custom function for dict batches)
        swa_model.to(DEVICE)
        update_bn_dict_loader(train_loader, swa_model, device=DEVICE)
        
        # Evaluate SWA model
        print("[SWA] Evaluating SWA model...")
        swa_metrics = evaluate(
            swa_model, val_loader, criterion, config.use_amp,
            find_threshold=config.use_optimal_threshold
        )
        swa_threshold = swa_metrics['best_threshold']
        print(f"[SWA] Val: F1={swa_metrics['f1']:.4f} AUC={swa_metrics['auc_roc']:.4f} T={swa_threshold:.2f}")
        
        # Save SWA model
        swa_save_dict = {
            'epoch': epoch,
            'model_state_dict': swa_model.state_dict(),
            'val_metrics': swa_metrics,
            'config': config.to_dict(),
            'best_threshold': swa_threshold
        }
        torch.save(swa_save_dict, os.path.join(MODEL_DIR, 'swa_model.pt'))
        print(f"[SWA] Saved SWA model to {MODEL_DIR}/swa_model.pt")
        
        # Compare SWA vs best model
        if swa_metrics['f1'] > best_f1:
            print(f"[SWA] SWA model is BETTER: F1={swa_metrics['f1']:.4f} > {best_f1:.4f}")
            best_f1 = swa_metrics['f1']
            best_threshold_overall = swa_threshold
        else:
            print(f"[SWA] Best model still better: F1={best_f1:.4f} > {swa_metrics['f1']:.4f}")
    
    print(f"\n{'='*50}")
    print(f"Done! Best F1={best_f1:.4f} at epoch {best_epoch} (T={best_threshold_overall:.2f})")
    print("="*50)
    
    # Save history (convert numpy types to native Python for JSON serialization)
    def convert_to_serializable(obj):
        if isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(v) for v in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        return obj
    
    with open(os.path.join(LOG_DIR, 'training_history.json'), 'w') as f:
        json.dump(convert_to_serializable(history), f, indent=2)
    
    return history, best_threshold_overall, swa_model if use_swa else None

# %% [markdown]
# ## 7. Run Training

# %%
# Check if ensemble mode is enabled
USE_ENSEMBLE = getattr(config, 'ensemble_size', 1) > 1

if USE_ENSEMBLE:
    # ============ ENSEMBLE TRAINING ============
    print(f"\n{'='*50}")
    print(f"ENSEMBLE TRAINING: {config.ensemble_size} models (plus SWA)")
    print(f"{'='*50}\n")
    
    ensemble_models = []
    last_swa_model = None
    
    for i in range(config.ensemble_size):
        seed = config.ensemble_base_seed + i
        print(f"\n{'='*50}")
        print(f"Training model {i+1}/{config.ensemble_size} (seed={seed})")
        print(f"{'='*50}\n")
        
        set_global_seed(seed)
        model = build_model(config)
        
        history, best_threshold, swa_model = train(model, train_loader, val_loader, config)
        
        # Load best checkpoint for this model
        checkpoint = torch.load(os.path.join(MODEL_DIR, 'best_model.pt'), weights_only=False)
        if isinstance(model, nn.DataParallel):
            model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint['model_state_dict'])
        
        ensemble_models.append(model)
        
        # Keep track of the last SWA model for 6-model ensemble
        if swa_model is not None:
            last_swa_model = swa_model
        
        # Save this base model
        torch.save({
            'model_state_dict': model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict(),
            'seed': seed,
        }, os.path.join(MODEL_DIR, f'ensemble_model_{i}.pt'))
    
    # Add SWA model as the 6th ensemble member (if available)
    if last_swa_model is not None:
        last_swa_model.to(DEVICE)
        ensemble_models.append(last_swa_model)
        print(f"\n[ENSEMBLE] Added SWA model to ensemble → total {len(ensemble_models)} models")
    
    # Ensemble threshold optimization on validation set using all ensemble members
    val_ens_metrics = predict_ensemble(
        ensemble_models, val_loader, threshold=0.5, use_amp=config.use_amp
    )
    all_val_labels = val_ens_metrics['labels']
    all_val_probs = val_ens_metrics['probs']
    
    best_threshold, best_f1 = find_optimal_threshold(
        all_val_labels, all_val_probs,
        min_t=config.threshold_min,
        max_t=config.threshold_max,
        step=config.threshold_step,
    )
    print(f"\n[ENSEMBLE] Optimal threshold: {best_threshold:.3f} (Val F1={best_f1:.4f})")
    print(f"[ENSEMBLE] Threshold search range: [{config.threshold_min}, {config.threshold_max}], step={config.threshold_step}")
    
    # Final ensemble model reference for evaluation
    model = ensemble_models[0]  # Keep first for single-model eval comparison
    swa_model = last_swa_model
    
else:
    # ============ SINGLE MODEL TRAINING ============
    result = train(model, train_loader, val_loader, config)
    if len(result) == 3:
        history, best_threshold, swa_model = result
    else:
        history, best_threshold = result
        swa_model = None
    ensemble_models = None

# %% [markdown]
# ## 8. Final Evaluation on Test Set

# %%
if USE_ENSEMBLE and ensemble_models is not None:
    # ============ ENSEMBLE EVALUATION ============
    print(f"\n{'='*50}")
    print("ENSEMBLE TEST RESULTS")
    print(f"{'='*50}")
    
    test_metrics = predict_ensemble(
        ensemble_models, test_loader, best_threshold, config.use_amp
    )
    print(f"Acc={test_metrics['accuracy']:.4f} P={test_metrics['precision']:.4f} R={test_metrics['recall']:.4f} F1={test_metrics['f1']:.4f} AUC={test_metrics['auc_roc']:.4f} T={best_threshold:.3f}")
    
    labels = test_metrics['labels']
    probs = test_metrics['probs']
    preds = (probs >= best_threshold).astype(int)
    
else:
    # ============ SINGLE MODEL EVALUATION ============
    # Load best model
    checkpoint = torch.load(os.path.join(MODEL_DIR, 'best_model.pt'), weights_only=False)
    
    if isinstance(model, nn.DataParallel):
        model.module.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint['model_state_dict'])
    
    # Evaluate on test set
    criterion = nn.CrossEntropyLoss()
    test_metrics = evaluate(
        model, test_loader, criterion, config.use_amp, 
        threshold=best_threshold
    )
    
    print(f"\n{'='*50}")
    print("TEST RESULTS")
    print(f"{'='*50}")
    print(f"Acc={test_metrics['accuracy']:.4f} P={test_metrics['precision']:.4f} R={test_metrics['recall']:.4f} F1={test_metrics['f1']:.4f} AUC={test_metrics['auc_roc']:.4f} T={best_threshold:.2f}")
    
    labels = test_metrics['labels']
    probs = test_metrics['probs']
    preds = (probs >= best_threshold).astype(int)

# %%
# Detailed classification report (using labels/probs/preds from evaluation above)
print("\nClassification Report:")
print(classification_report(labels, preds, target_names=['Non-Vulnerable', 'Vulnerable']))

print("\nConfusion Matrix:")
cm = confusion_matrix(labels, preds)
print(f"  TN: {cm[0,0]:5d}  FP: {cm[0,1]:5d}")
print(f"  FN: {cm[1,0]:5d}  TP: {cm[1,1]:5d}")

# %% [markdown]
# ## 9. Training Curves

# %%
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

epochs = range(1, len(history['train']) + 1)

# Loss
axes[0].plot(epochs, [m['loss'] for m in history['train']], 'b-', label='Train')
axes[0].plot(epochs, [m['loss'] for m in history['val']], 'r-', label='Val')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].set_title('Loss')
axes[0].legend()
axes[0].grid(True)

# F1
axes[1].plot(epochs, [m['f1'] for m in history['train']], 'b-', label='Train')
axes[1].plot(epochs, [m['f1'] for m in history['val']], 'r-', label='Val')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('F1 Score')
axes[1].set_title('F1 Score')
axes[1].legend()
axes[1].grid(True)

# AUC-ROC
axes[2].plot(epochs, [m['auc_roc'] for m in history['train']], 'b-', label='Train')
axes[2].plot(epochs, [m['auc_roc'] for m in history['val']], 'r-', label='Val')
axes[2].set_xlabel('Epoch')
axes[2].set_ylabel('AUC-ROC')
axes[2].set_title('AUC-ROC')
axes[2].legend()
axes[2].grid(True)

plt.tight_layout()
plt.savefig(os.path.join(LOG_DIR, 'training_curves.png'), dpi=150)
plt.show()

# %% [markdown]
# ## 10. Export Model for Inference

# %%
# Save final model with config
final_export = {
    'model_state_dict': model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict(),
    'config': config.to_dict(),
    'test_metrics': test_metrics,
    'best_threshold': best_threshold,
    'timestamp': datetime.now().isoformat(),
    'model_type': 'HybridBiGRU'
}
torch.save(final_export, os.path.join(MODEL_DIR, 'bigru_vuln_detector_final.pt'))
print(f"Model saved to {MODEL_DIR}/bigru_vuln_detector_final.pt")
None  # Suppress cell output
