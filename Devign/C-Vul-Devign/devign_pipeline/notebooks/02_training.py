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


class RefinedConfig(EnhancedConfig):
    """
    Refined configuration based on Oracle analysis of OptimizedConfig results.
    
    Training showed: Best Val F1=0.7147, AUC=0.8178 at epoch 21.
    Model is slightly UNDERFITTING (train AUC ~0.74, val AUC ~0.82).
    
    Key changes (Oracle recommendations):
    1. Reduce dropout: 0.4 → 0.3 (model underfitting, not overfitting)
    2. Weaken label smoothing: 0.05 → 0.03, stop at epoch 10 (sharper boundaries)
    3. Delay SWA: epoch 14 → 17 (closer to convergence)
    4. Fix threshold on best checkpoint only (not per-epoch)
    5. Slightly more capacity in vuln MLP
    
    Target: Stable F1 ≥ 0.72-0.73 with AUC ≥ 0.82
    """
    
    # Dropout REDUCED (model is underfitting)
    classifier_dropout: float = 0.3       # DOWN from 0.4
    rnn_dropout: float = 0.3              # DOWN from 0.35
    embedding_dropout: float = 0.15       # DOWN from 0.2
    vuln_feature_dropout: float = 0.2     # DOWN from 0.25
    
    # Label smoothing WEAKENED (sharper decision boundary for F1)
    label_smoothing: float = 0.03         # DOWN from 0.05
    label_smoothing_warmup_epochs: int = 10  # DOWN from 15
    
    # SWA DELAYED (start closer to convergence)
    swa_start_epoch: int = 17             # UP from 14
    swa_lr: float = 5e-5                  # Lower for stability
    
    # Vuln features MLP slightly larger
    vuln_feature_hidden_dim: int = 64     # UP from 48
    
    # Training epochs extended slightly
    max_epochs: int = 28                  # UP from 25
    
    # Early stopping: more patient since we're less regularized
    patience: int = 5                     # UP from 3
    min_delta: float = 3e-4               # DOWN from 5e-4
    
    # Threshold: use wider range but will fix on best checkpoint
    threshold_min: float = 0.2
    threshold_max: float = 0.8
    threshold_step: float = 0.005
    
    # Keep ensemble
    ensemble_size: int = 5
    
    # Fine-tuning tail
    use_finetune_tail: bool = True
    finetune_epochs: int = 3
    finetune_lr: float = 1e-5
    
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


class FinalConfig(RefinedConfig):
    """
    Final configuration for Devign Hybrid BiGRU + Vuln Features model.
    Based on Oracle analysis of RefinedConfig results (F1≈0.72, AUC≈0.82).

    Key changes from RefinedConfig:
    1. Increased regularization: rnn_dropout 0.3→0.35, classifier_dropout 0.3→0.35
    2. Larger vuln MLP: vuln_feature_hidden_dim 64→80
    3. Reduced label smoothing: 0.03→0.02 for better calibration
    4. Lower weight_decay: 5e-4 (balanced regularization)
    5. Extended max_epochs: 28→32
    6. SWA start adjusted: 17→18
    7. Higher attention dropout: 0.1→0.15

    Target: Stable F1 ≥ 0.72-0.73 with better calibration (threshold closer to 0.5)
    """

    # --- Model capacity (keep as RefinedConfig, tune vuln MLP) ---
    hidden_dim: int = 160           # keep - works well for F1 ~0.72+
    num_layers: int = 2             # stacked BiGRU
    bidirectional: bool = True

    # Multi-head attention
    use_multihead_attention: bool = True
    num_attention_heads: int = 4
    attention_dropout: float = 0.15  # UP from 0.1 - more regularization

    # --- Dropout & regularization ---
    embedding_dropout: float = 0.15  # keep

    # GRU: increased dropout between layers for stability
    rnn_dropout: float = 0.35        # UP from 0.30

    # Classifier: increased dropout for FC layers
    classifier_dropout: float = 0.35  # UP from 0.30

    # Vuln features branch - larger MLP
    vuln_feature_hidden_dim: int = 80    # UP from 64
    vuln_feature_dropout: float = 0.25   # UP from 0.20

    # --- Training hyperparameters ---
    batch_size: int = 128
    accumulation_steps: int = 1

    # Learning rate / weight decay
    learning_rate: float = 3e-4
    max_lr: float = 1.0e-3
    weight_decay: float = 5e-4       # balanced between 1e-4 and 1e-3

    max_epochs: int = 32             # UP from 28

    # Gradient clipping
    grad_clip: float = 1.0

    # --- Label smoothing ---
    # Reduced for better calibration (threshold closer to 0.5)
    label_smoothing: float = 0.02    # DOWN from 0.03
    label_smoothing_warmup_epochs: int = 10

    # --- Early stopping ---
    patience: int = 4
    min_delta: float = 5e-4

    # --- SWA (Stochastic Weight Averaging) ---
    use_swa: bool = True
    swa_start_epoch: int = 18        # UP from 17 - start in plateau
    swa_lr: float = 5e-5

    # --- Scheduler ---
    scheduler_type: str = "plateau"
    scheduler_patience: int = 2
    scheduler_factor: float = 0.5
    scheduler_min_lr: float = 1e-6

    # --- Data / loader ---
    max_seq_length: int = 512
    num_workers: int = 2

    # Packed sequences
    use_packed_sequences: bool = True

    # Threshold optimization
    use_optimal_threshold: bool = True
    threshold_min: float = 0.1
    threshold_max: float = 0.9
    threshold_step: float = 0.01

    # Token augmentation
    use_token_augmentation: bool = True
    token_dropout_prob: float = 0.1
    token_mask_prob: float = 0.05
    mask_token_id: int = 1

    # LayerNorm
    use_layer_norm: bool = True

    # Ensemble
    ensemble_size: int = 5
    ensemble_base_seed: int = 1337

    # Fine-tuning tail
    use_finetune_tail: bool = True
    finetune_epochs: int = 3
    finetune_lr: float = 1e-5

    def to_dict(self) -> Dict:
        """Export config to plain dict."""
        cfg: Dict[str, Any] = {}
        for cls in reversed(self.__class__.mro()):
            if cls is object:
                continue
            for k, v in cls.__dict__.items():
                if not k.startswith('_') and not callable(v):
                    cfg[k] = v
        return cfg


class QuickWinConfig(FinalConfig):
    """
    Quick Win configuration based on Oracle analysis.
    Target: F1 ≥ 75%, AUC-ROC ≥ 85%
    
    Changes from FinalConfig:
    1. Precision-focused class weighting (boost negative class weight)
    2. Disable label smoothing for sharper decision boundary
    3. Reduced token augmentation (less aggressive)
    4. LayerNorm enabled (already in FinalConfig)
    5. Diverse ensemble with per-model dropout variation
    """
    
    # --- Quick Win 1.2: Disable label smoothing ---
    label_smoothing: float = 0.0              # OFF - sharper boundary
    label_smoothing_warmup_epochs: int = 0
    
    # --- Quick Win 1.3: Reduced token augmentation ---
    token_dropout_prob: float = 0.05          # DOWN from 0.1
    token_mask_prob: float = 0.03             # DOWN from 0.05
    
    # --- Quick Win 1.4: LayerNorm (already True, confirm) ---
    use_layer_norm: bool = True
    
    # --- Quick Win 1.1: Precision-focused class weighting ---
    # New flag to enable precision-focused weighting in training loop
    use_precision_focused_weight: bool = True
    neg_weight_boost: float = 1.15            # Boost negative class weight by 15%
    pos_weight_reduce: float = 0.85           # Reduce positive class weight by 15%
    
    # --- Quick Win 1.5: Diverse ensemble ---
    # Dropout variations per ensemble member (indexed by seed offset)
    use_diverse_ensemble: bool = True
    ensemble_dropout_variations: tuple = (0.0, -0.05, +0.05, -0.03, +0.03)
    
    # Keep other settings from FinalConfig
    max_epochs: int = 30                      # Slightly reduced
    patience: int = 5                         # More patient for new config
    
    def to_dict(self) -> Dict:
        """Export config to plain dict."""
        cfg: Dict[str, Any] = {}
        for cls in reversed(self.__class__.mro()):
            if cls is object:
                continue
            for k, v in cls.__dict__.items():
                if not k.startswith('_') and not callable(v):
                    cfg[k] = v
        return cfg


class AdvancedConfig(FinalConfig):
    """
    Advanced configuration targeting F1 ≥ 75%, AUC-ROC ≥ 85%.
    
    Based on analysis of QuickWinConfig results:
    - Val F1 ~72%, AUC ~82% (not meeting targets)
    - Validation metrics > training (dropout effect - normal)
    - Overfitting starts around epoch 15-20
    
    Key changes:
    1. STRONGER class weighting: boost neg 30%, reduce pos 30% (more aggressive)
    2. Focal Loss style: use_focal_weight for hard example mining
    3. EARLIER SWA: start at epoch 12 (before overfitting)
    4. HIGHER model capacity: hidden_dim 160 → 192
    5. Cosine annealing with warm restarts
    6. LIGHTER regularization: less dropout since model underfits on val
    7. LARGER ensemble: 7 models with more diversity
    """
    
    # --- Model Capacity INCREASED ---
    hidden_dim: int = 192                     # UP from 160 - more capacity
    num_attention_heads: int = 6              # UP from 4 - richer attention
    vuln_feature_hidden_dim: int = 96         # UP from 80
    
    # --- Dropout REDUCED (model underfitting on validation) ---
    classifier_dropout: float = 0.25          # DOWN from 0.35
    rnn_dropout: float = 0.25                 # DOWN from 0.35
    embedding_dropout: float = 0.1            # DOWN from 0.15
    attention_dropout: float = 0.1            # DOWN from 0.15
    vuln_feature_dropout: float = 0.15        # DOWN from 0.25
    
    # --- STRONGER class weighting for precision ---
    use_precision_focused_weight: bool = True
    neg_weight_boost: float = 1.30            # UP from 1.15 - more aggressive
    pos_weight_reduce: float = 0.70           # DOWN from 0.85
    
    # --- Focal Loss style weighting ---
    use_focal_weight: bool = True             # NEW: focus on hard examples
    focal_gamma: float = 2.0                  # Focal loss gamma
    
    # --- Label smoothing OFF ---
    label_smoothing: float = 0.0
    label_smoothing_warmup_epochs: int = 0
    
    # --- Token augmentation MINIMAL ---
    use_token_augmentation: bool = True
    token_dropout_prob: float = 0.03          # DOWN from 0.05
    token_mask_prob: float = 0.02             # DOWN from 0.03
    
    # --- Learning rate with Cosine Annealing ---
    scheduler_type: str = 'cosine'            # CHANGED from 'plateau'
    learning_rate: float = 5e-4               # UP from 3e-4
    max_lr: float = 2e-3                      # UP from 1e-3
    warmup_epochs: int = 3                    # NEW: warmup period
    
    # --- SWA EARLIER (before overfitting) ---
    use_swa: bool = True
    swa_start_epoch: int = 12                 # DOWN from 18
    swa_lr: float = 1e-4                      # UP from 5e-5
    
    # --- Training duration ---
    max_epochs: int = 25                      # DOWN from 30 (early stopping anyway)
    patience: int = 6                         # UP from 5
    min_delta: float = 3e-4                   # More sensitive
    
    # --- LARGER & MORE DIVERSE ensemble ---
    ensemble_size: int = 7                    # UP from 5
    use_diverse_ensemble: bool = True
    ensemble_dropout_variations: tuple = (0.0, -0.08, +0.08, -0.05, +0.05, -0.03, +0.03)
    
    # --- Threshold optimization FINER ---
    threshold_min: float = 0.30
    threshold_max: float = 0.60
    threshold_step: float = 0.005
    
    # --- Fine-tuning tail ---
    use_finetune_tail: bool = True
    finetune_epochs: int = 5                  # UP from 3
    finetune_lr: float = 5e-6                 # DOWN from 1e-5
    
    # --- Gradient clipping ---
    grad_clip: float = 0.5                    # DOWN from 1.0 - more stable
    
    # --- Batch size ---
    batch_size: int = 96                      # DOWN from 128 - more regularization effect
    accumulation_steps: int = 2               # UP from 1 - effective batch = 192
    
    def to_dict(self) -> Dict:
        """Export config to plain dict."""
        cfg: Dict[str, Any] = {}
        for cls in reversed(self.__class__.mro()):
            if cls is object:
                continue
            for k, v in cls.__dict__.items():
                if not k.startswith('_') and not callable(v):
                    cfg[k] = v
        return cfg


class AdvancedConfigV2(FinalConfig):
    """
    AdvancedConfigV2 - Oracle-optimized configuration for F1 ≥ 75%, AUC-ROC ≥ 85%.
    
    Based on Oracle analysis of AdvancedConfig results:
    - Val F1 ~72%, AUC ~82%, Precision ~49%, Recall ~90%
    - Overfitting from epoch 10-15 (too aggressive capacity + low dropout)
    - Focal Loss + strong class weights = noisy optimization
    - Threshold ~0.35 too low → too many false positives
    
    Key changes from AdvancedConfig:
    1. DISABLE Focal Loss - use weighted BCE instead
    2. SOFTER class weights: neg +10%, pos -10% (instead of 30%)
    3. INCREASE dropout back to prevent overfitting
    4. REDUCE model capacity: hidden_dim 192→160, heads 6→4
    5. ADD mild label smoothing (0.03) for better calibration
    6. EARLIER SWA: epoch 8 (before overfitting at 10)
    7. Use F0.5 or precision-constrained threshold selection
    """
    
    # --- Model Capacity REDUCED for better generalization ---
    hidden_dim: int = 160                     # DOWN from 192
    num_attention_heads: int = 4              # DOWN from 6
    vuln_feature_hidden_dim: int = 80         # DOWN from 96
    
    # --- Dropout INCREASED to prevent overfitting ---
    classifier_dropout: float = 0.35          # UP from 0.25
    rnn_dropout: float = 0.35                 # UP from 0.25
    embedding_dropout: float = 0.15           # UP from 0.1
    attention_dropout: float = 0.15           # UP from 0.1
    vuln_feature_dropout: float = 0.25        # UP from 0.15
    
    # --- SOFTER class weighting ---
    use_precision_focused_weight: bool = True
    neg_weight_boost: float = 1.10            # DOWN from 1.30 - less aggressive
    pos_weight_reduce: float = 0.90           # UP from 0.70
    
    # --- Focal Loss DISABLED ---
    use_focal_weight: bool = False            # DISABLED - causes noisy optimization
    focal_gamma: float = 1.0                  # Reduced if re-enabled later
    
    # --- Label smoothing for better calibration ---
    label_smoothing: float = 0.03             # UP from 0.0
    label_smoothing_warmup_epochs: int = 0
    
    # --- Token augmentation ---
    use_token_augmentation: bool = True
    token_dropout_prob: float = 0.05          # UP from 0.03
    token_mask_prob: float = 0.03             # UP from 0.02
    
    # --- Learning rate REDUCED for stability ---
    scheduler_type: str = 'plateau'           # CHANGED back from 'cosine' for stability
    learning_rate: float = 3e-4               # DOWN from 5e-4
    max_lr: float = 1e-3                      # DOWN from 2e-3
    scheduler_patience: int = 2
    scheduler_factor: float = 0.5
    scheduler_min_lr: float = 1e-6
    
    # --- SWA MUCH EARLIER (before overfitting at epoch 10) ---
    use_swa: bool = True
    swa_start_epoch: int = 8                  # DOWN from 12
    swa_lr: float = 1e-4
    
    # --- Training duration ---
    max_epochs: int = 25
    patience: int = 5                         # DOWN from 6
    min_delta: float = 3e-4
    
    # --- Ensemble ---
    ensemble_size: int = 5                    # DOWN from 7 - diminishing returns
    use_diverse_ensemble: bool = True
    ensemble_dropout_variations: tuple = (0.0, -0.05, +0.05, -0.03, +0.03)
    
    # --- Threshold optimization: wider range, use F0.5 objective ---
    threshold_min: float = 0.35               # UP from 0.30
    threshold_max: float = 0.70               # UP from 0.60
    threshold_step: float = 0.005
    threshold_objective: str = 'f0.5'         # NEW: 'f1', 'f0.5', or 'precision_constrained'
    min_precision_constraint: float = 0.60    # NEW: for precision_constrained mode
    
    # --- Fine-tuning tail ---
    use_finetune_tail: bool = True
    finetune_epochs: int = 3
    finetune_lr: float = 1e-5
    
    # --- Gradient clipping ---
    grad_clip: float = 0.5
    
    # --- Batch size ---
    batch_size: int = 96
    accumulation_steps: int = 2               # effective batch = 192
    
    def to_dict(self) -> Dict:
        """Export config to plain dict."""
        cfg: Dict[str, Any] = {}
        for cls in reversed(self.__class__.mro()):
            if cls is object:
                continue
            for k, v in cls.__dict__.items():
                if not k.startswith('_') and not callable(v):
                    cfg[k] = v
        return cfg


class AdvancedConfigV3(FinalConfig):
    """
    AdvancedConfigV3 - Oracle-optimized after V2 regression analysis.
    Target: F1 ≥ 75%, AUC-ROC ≥ 85%
    
    V2 Results (REGRESSION):
    - F1 ~66% (DOWN from 72% baseline)
    - AUC ~80% (DOWN from 82% baseline)
    - Precision ~75-81% (UP from 62%)
    - Recall ~50-59% (DOWN from 86%) ← COLLAPSED
    
    Root Cause: Precision-focus changes (F0.5, precision_constrained, softer pos weight)
    were too aggressive → recall collapsed → F1 dropped despite higher precision.
    
    Oracle Recommendations for V3:
    1. RESTORE model capacity: hidden_dim=192, heads=6 (undo V2 reduction)
    2. REDUCE dropout: 0.25-0.30 (model was underpowered)
    3. BRING BACK Focal Loss (milder gamma=1.5) for hard positives
    4. STRONGER positive class weight (undo softening)
    5. RECALL-CONSTRAINED threshold: maximize F1 subject to recall ≥ 0.80
    6. F1 objective instead of F0.5 (undo precision bias)
    """
    
    # --- Model Capacity RESTORED (Oracle: undo V2 reduction) ---
    hidden_dim: int = 192                     # UP from 160 - more capacity
    num_attention_heads: int = 6              # UP from 4 - richer attention
    vuln_feature_hidden_dim: int = 96         # UP from 80
    
    # --- Dropout REDUCED (Oracle: model underpowered in V2) ---
    classifier_dropout: float = 0.28          # DOWN from 0.35
    rnn_dropout: float = 0.28                 # DOWN from 0.35
    embedding_dropout: float = 0.12           # DOWN from 0.15
    attention_dropout: float = 0.12           # DOWN from 0.15
    vuln_feature_dropout: float = 0.20        # DOWN from 0.25
    
    # --- Class weighting RESTORED (Oracle: undo V2 softening) ---
    # V2 was neg_boost=1.10, pos_reduce=0.90 → too soft, killed recall
    # Restore baseline-like weighting to recover recall
    use_precision_focused_weight: bool = True
    neg_weight_boost: float = 1.0             # NEUTRAL (was 1.10)
    pos_weight_reduce: float = 1.0            # NEUTRAL (was 0.90)
    
    # --- Focal Loss RESTORED with milder gamma ---
    # Oracle: bring back focal loss for hard positive mining
    use_focal_weight: bool = True             # ENABLED
    focal_gamma: float = 1.5                  # Milder than V1's 2.0
    
    # --- Label smoothing REDUCED for sharper boundary ---
    label_smoothing: float = 0.02             # DOWN from 0.03
    label_smoothing_warmup_epochs: int = 8    # Early epochs only
    
    # --- Token augmentation (moderate) ---
    use_token_augmentation: bool = True
    token_dropout_prob: float = 0.08          # UP from 0.05 - more augmentation
    token_mask_prob: float = 0.04             # UP from 0.03
    
    # --- Learning rate / scheduler ---
    scheduler_type: str = 'cosine'            # Cosine annealing for smooth decay
    learning_rate: float = 4e-4               # UP from 3e-4
    max_lr: float = 1.5e-3                    # UP from 1e-3
    warmup_epochs: int = 2                    # Quick warmup
    
    # --- SWA timing (Oracle: start after stable convergence) ---
    use_swa: bool = True
    swa_start_epoch: int = 12                 # UP from 8 - not too early
    swa_lr: float = 8e-5                      # Moderate
    
    # --- Training duration ---
    max_epochs: int = 28                      # UP from 25
    patience: int = 6                         # More patient
    min_delta: float = 2e-4                   # More sensitive
    
    # --- Ensemble ---
    ensemble_size: int = 5
    use_diverse_ensemble: bool = True
    ensemble_dropout_variations: tuple = (0.0, -0.05, +0.05, -0.03, +0.03)
    
    # --- Threshold optimization: RECALL-CONSTRAINED (Oracle key fix) ---
    # Instead of F0.5 or precision_constrained, use recall_constrained
    # Maximize F1 subject to recall ≥ 0.80 (baseline was 86%)
    threshold_min: float = 0.25               # DOWN from 0.35 - allow lower thresholds
    threshold_max: float = 0.60               # DOWN from 0.70
    threshold_step: float = 0.005
    threshold_objective: str = 'recall_constrained'  # NEW: maximize F1 with recall floor
    min_recall_constraint: float = 0.80       # NEW: recall must be ≥ 80%
    
    # --- Fine-tuning tail ---
    use_finetune_tail: bool = True
    finetune_epochs: int = 4                  # UP from 3
    finetune_lr: float = 8e-6                 # Slightly higher
    
    # --- Gradient clipping ---
    grad_clip: float = 0.8                    # UP from 0.5 - more lenient
    
    # --- Batch size ---
    batch_size: int = 96
    accumulation_steps: int = 2               # effective batch = 192
    
    def to_dict(self) -> Dict:
        """Export config to plain dict."""
        cfg: Dict[str, Any] = {}
        for cls in reversed(self.__class__.mro()):
            if cls is object:
                continue
            for k, v in cls.__dict__.items():
                if not k.startswith('_') and not callable(v):
                    cfg[k] = v
        return cfg


class AdvancedConfigV4(FinalConfig):
    """
    AdvancedConfigV4 - Oracle-optimized after V3 confusion matrix bias analysis.
    Target: F1 ≥ 75%, AUC-ROC ≥ 85%
    
    V3 Results:
    - F1 ~71% (slightly below baseline 72%)
    - AUC ~80% (below baseline 82%)
    - Precision ~63%, Recall ~81-92%
    - Problem: Confusion matrix biased toward predicting "vulnerable" (too many FP)
    
    Root Cause Analysis:
    1. Recall-constrained threshold forced low threshold (~0.41-0.47) → many FP
    2. Focal Loss without alpha balancing → negatives under-penalized
    3. Low dropout (0.28) → model overconfident → miscalibration
    4. Neutral class weights → no pressure to reduce FP
    
    Oracle Recommendations for V4:
    1. Change threshold selection: pick HIGHEST threshold that meets recall≥0.80 (not best F1)
    2. Add alpha balancing to Focal Loss: alpha_neg > alpha_pos to penalize FP
    3. Increase dropout to 0.32-0.35 to reduce overconfidence
    4. Slightly boost negative class weight to discourage FP
    5. Keep model capacity (192, 6 heads) - separation is the issue, not capacity
    """
    
    # --- Model Capacity (keep from V3 - capacity is not the issue) ---
    hidden_dim: int = 192
    num_attention_heads: int = 6
    vuln_feature_hidden_dim: int = 96
    
    # --- Dropout INCREASED to reduce overconfidence (Oracle V4 fix) ---
    classifier_dropout: float = 0.35          # UP from 0.28
    rnn_dropout: float = 0.32                 # UP from 0.28
    embedding_dropout: float = 0.15           # UP from 0.12
    attention_dropout: float = 0.15           # UP from 0.12
    vuln_feature_dropout: float = 0.25        # UP from 0.20
    
    # --- Class weighting: BOOST NEGATIVE to reduce FP (Oracle V4 key fix) ---
    use_precision_focused_weight: bool = True
    neg_weight_boost: float = 1.20            # UP from 1.0 - penalize FP more
    pos_weight_reduce: float = 0.95           # Slight reduction to balance
    
    # --- Focal Loss with ALPHA BALANCING (Oracle V4 key fix) ---
    # alpha_neg > alpha_pos to penalize false positives more
    use_focal_weight: bool = True
    focal_gamma: float = 1.5                  # Keep milder gamma
    use_focal_alpha: bool = True              # NEW: enable alpha balancing
    focal_alpha_pos: float = 0.4              # NEW: lower weight for positives
    focal_alpha_neg: float = 0.6              # NEW: higher weight for negatives (penalize FP)
    
    # --- Label smoothing (slightly higher for better calibration) ---
    label_smoothing: float = 0.03             # UP from 0.02
    label_smoothing_warmup_epochs: int = 10   # UP from 8
    
    # --- Token augmentation (keep moderate) ---
    use_token_augmentation: bool = True
    token_dropout_prob: float = 0.08
    token_mask_prob: float = 0.04
    
    # --- Learning rate / scheduler ---
    scheduler_type: str = 'cosine'
    learning_rate: float = 3e-4               # DOWN from 4e-4 - more stable
    max_lr: float = 1.2e-3                    # DOWN from 1.5e-3
    warmup_epochs: int = 3                    # UP from 2 - longer warmup
    
    # --- SWA timing ---
    use_swa: bool = True
    swa_start_epoch: int = 14                 # UP from 12 - later start
    swa_lr: float = 5e-5                      # DOWN from 8e-5 - more conservative
    
    # --- Training duration ---
    max_epochs: int = 30                      # UP from 28
    patience: int = 6
    min_delta: float = 2e-4
    
    # --- Ensemble ---
    ensemble_size: int = 5
    use_diverse_ensemble: bool = True
    ensemble_dropout_variations: tuple = (0.0, -0.05, +0.05, -0.03, +0.03)
    
    # --- Threshold optimization: HIGHEST threshold meeting recall constraint ---
    # Instead of "maximize F1 subject to recall>=0.80", pick "highest threshold with recall>=0.80"
    # This directly reduces FP while respecting recall floor
    threshold_min: float = 0.30               # UP from 0.25
    threshold_max: float = 0.65               # UP from 0.60
    threshold_step: float = 0.005
    threshold_objective: str = 'recall_constrained_strict'  # NEW: highest threshold mode
    min_recall_constraint: float = 0.80
    
    # --- Fine-tuning tail ---
    use_finetune_tail: bool = True
    finetune_epochs: int = 4
    finetune_lr: float = 5e-6                 # DOWN from 8e-6
    
    # --- Gradient clipping ---
    grad_clip: float = 1.0                    # UP from 0.8 - standard
    
    # --- Batch size ---
    batch_size: int = 96
    accumulation_steps: int = 2               # effective batch = 192
    
    def to_dict(self) -> Dict:
        """Export config to plain dict."""
        cfg: Dict[str, Any] = {}
        for cls in reversed(self.__class__.mro()):
            if cls is object:
                continue
            for k, v in cls.__dict__.items():
                if not k.startswith('_') and not callable(v):
                    cfg[k] = v
        return cfg


class AdvancedConfigV5(FinalConfig):
    """
    AdvancedConfigV5 - SIMPLIFIED "Branch A" approach based on Oracle analysis.
    Target: F1 ≥ 75%, AUC-ROC ≥ 85%
    
    V4 Results (REGRESSION from baseline):
    - Best F1 ~69.8% (DOWN from baseline 72%)
    - AUC ~79.2% (DOWN from baseline 82%)
    - Precision ~62%, Recall ~80%
    
    Root Cause (Oracle Analysis):
    1. TOO MANY CONFLICTING imbalance methods fighting each other:
       - Focal Loss + alpha balancing + neg_boost + label smoothing + high dropout
       - Combined, they HURT ranking quality (AUC dropped 82% → 79%)
    2. Label smoothing compresses logits → hurts precision/AUC
    3. High dropout (0.32-0.35) → underfitting (Train F1 < Val F1)
    4. Threshold tricks can't recover lost AUC separability
    
    V5 Strategy - "Branch A" SIMPLIFICATION:
    1. REMOVE Focal Loss entirely - use plain BCE with pos_weight
    2. REMOVE label smoothing (set to 0)
    3. REMOVE alpha balancing, neg_boost, pos_reduce
    4. REDUCE dropout to 0.20-0.28 (stop underfitting)
    5. Use simple pos_weight-based BCE (compute from class imbalance)
    6. Use F1-maximizing threshold (not recall_constrained)
    7. Keep model capacity (192, 6 heads) - capacity was fine
    """
    
    # --- Model Capacity (KEEP - capacity is not the issue) ---
    hidden_dim: int = 192
    num_attention_heads: int = 6
    vuln_feature_hidden_dim: int = 96
    
    # --- Dropout REDUCED to stop underfitting (Oracle V5 key fix) ---
    # V4 had Train F1 ~64% < Val F1 ~70% = underfitting
    classifier_dropout: float = 0.25          # DOWN from 0.35
    rnn_dropout: float = 0.22                 # DOWN from 0.32
    embedding_dropout: float = 0.10           # DOWN from 0.15
    attention_dropout: float = 0.10           # DOWN from 0.15
    vuln_feature_dropout: float = 0.18        # DOWN from 0.25
    
    # --- Class weighting SIMPLIFIED (Oracle V5 key fix) ---
    # Remove all the competing methods, use only pos_weight in BCE
    use_precision_focused_weight: bool = False  # DISABLED - no manual boost
    neg_weight_boost: float = 1.0              # NEUTRAL
    pos_weight_reduce: float = 1.0             # NEUTRAL
    
    # --- Focal Loss DISABLED (Oracle V5 key fix) ---
    # Focal + alpha + weights = too many methods fighting
    use_focal_weight: bool = False             # DISABLED
    use_focal_alpha: bool = False              # DISABLED
    focal_gamma: float = 0.0                   # DISABLED
    focal_alpha_pos: float = 0.5               # NEUTRAL
    focal_alpha_neg: float = 0.5               # NEUTRAL
    
    # --- Label smoothing DISABLED (Oracle V5 key fix) ---
    # Label smoothing compresses logits → hurts precision and AUC
    label_smoothing: float = 0.0               # DISABLED (was 0.03)
    label_smoothing_warmup_epochs: int = 0     # DISABLED
    
    # --- Token augmentation MINIMAL ---
    # Keep minimal augmentation for regularization without hurting signal
    use_token_augmentation: bool = True
    token_dropout_prob: float = 0.05           # DOWN from 0.08
    token_mask_prob: float = 0.02              # DOWN from 0.04
    
    # --- Learning rate / scheduler (stable plateau) ---
    scheduler_type: str = 'plateau'            # CHANGED from 'cosine' for stability
    learning_rate: float = 4e-4                # UP from 3e-4 - less regularization needs lower LR
    max_lr: float = 1.5e-3
    scheduler_patience: int = 3                # UP from 2 - more patient
    scheduler_factor: float = 0.5
    scheduler_min_lr: float = 1e-6
    
    # --- SWA timing (start after stable convergence) ---
    use_swa: bool = True
    swa_start_epoch: int = 15                  # Moderate - not too early
    swa_lr: float = 8e-5                       # Moderate
    
    # --- Training duration ---
    max_epochs: int = 30
    patience: int = 7                          # UP from 6 - more patient with simpler model
    min_delta: float = 2e-4
    
    # --- Ensemble DISABLED for faster iteration ---
    # Re-enable after V5 baseline is established
    ensemble_size: int = 1                     # SINGLE MODEL (was 5)
    use_diverse_ensemble: bool = False         # DISABLED
    ensemble_dropout_variations: tuple = (0.0,)
    
    # --- Threshold optimization: F1-MAXIMIZING (Oracle V5 key fix) ---
    # Use simple F1 maximization, not recall_constrained
    threshold_min: float = 0.30
    threshold_max: float = 0.70
    threshold_step: float = 0.01
    threshold_objective: str = 'f1'            # CHANGED from 'recall_constrained_strict'
    min_recall_constraint: float = 0.70        # Lower floor if using constrained mode
    
    # --- Fine-tuning tail ---
    use_finetune_tail: bool = True
    finetune_epochs: int = 3
    finetune_lr: float = 1e-5
    
    # --- Gradient clipping (standard) ---
    grad_clip: float = 1.0
    
    # --- Batch size (larger for stability) ---
    batch_size: int = 128                      # UP from 96
    accumulation_steps: int = 1                # No accumulation needed
    
    def to_dict(self) -> Dict:
        """Export config to plain dict."""
        cfg: Dict[str, Any] = {}
        for cls in reversed(self.__class__.mro()):
            if cls is object:
                continue
            for k, v in cls.__dict__.items():
                if not k.startswith('_') and not callable(v):
                    cfg[k] = v
        return cfg


class AggressiveRegConfig(TrainConfig):
    """
    AggressiveRegConfig - Optimized based on first training run analysis.
    
    First Run Results:
    - Best Val F1: 68.2% at epoch 5
    - Val AUC: 78.2%
    - Train-Val F1 gap: 5.6% (overfitting from epoch 5)
    
    Key Fixes Applied:
    1. EarlyStopping now returns bool (bug fixed)
    2. Scheduler steps on val_f1 instead of val_auc (bug fixed)
    3. train_epoch AUC now uses probs instead of preds (bug fixed)
    
    Config Strategy:
    - Aggressive early stopping (patience=3)
    - Stronger regularization (higher dropout, weight_decay)
    - Plateau scheduler on F1
    - SWA starts early (epoch 4) near observed best epoch
    - Simpler architecture (fewer conflicting regularizers)
    """
    
    # --- Model (moderate capacity) ---
    vocab_size: int = data_config.get('vocab_size', 266)
    embed_dim: int = 64
    hidden_dim: int = 160                     # Moderate - not too large
    num_layers: int = 2
    bidirectional: bool = True
    
    # Hybrid Model Features
    use_vuln_features: bool = True
    vuln_feature_dim: int = len(data_config.get('vuln_feature_names', [])) or 26
    vuln_feature_hidden_dim: int = 64         # Moderate MLP
    
    # Multi-head attention
    use_multihead_attention: bool = True
    num_attention_heads: int = 4
    
    # --- Dropout INCREASED for regularization ---
    embedding_dropout: float = 0.20           # UP from 0.15
    rnn_dropout: float = 0.45                 # Strong (for 2 layers)
    attention_dropout: float = 0.20           # UP
    classifier_dropout: float = 0.50          # Strong
    vuln_feature_dropout: float = 0.30        # UP
    
    # --- Training (aggressive early stopping) ---
    batch_size: int = 96                      # Slightly smaller for regularization
    accumulation_steps: int = 1
    learning_rate: float = 3e-4               # Conservative
    max_lr: float = 1e-3
    weight_decay: float = 2e-3                # Strong L2 regularization
    max_epochs: int = 20                      # Reduced (best was epoch 5)
    grad_clip: float = 0.5                    # Tighter clipping
    
    # --- Early stopping AGGRESSIVE ---
    patience: int = 3                         # Quick stop
    min_delta: float = 3e-4
    
    # --- Scheduler (ReduceLROnPlateau on F1) ---
    scheduler_type: str = 'plateau'
    scheduler_patience: int = 1               # Reduce LR quickly when F1 stalls
    scheduler_factor: float = 0.5
    scheduler_min_lr: float = 1e-6
    
    # --- Regularization (SIMPLIFIED - no conflicting methods) ---
    label_smoothing: float = 0.02             # Light smoothing
    label_smoothing_warmup_epochs: int = 3    # Only first few epochs
    
    # --- NO Focal Loss (keep simple) ---
    use_focal_weight: bool = False
    use_focal_alpha: bool = False
    
    # --- NO precision-focused weighting (keep simple) ---
    use_precision_focused_weight: bool = False
    neg_weight_boost: float = 1.0
    pos_weight_reduce: float = 1.0
    
    # --- Token augmentation (moderate) ---
    use_token_augmentation: bool = True
    token_dropout_prob: float = 0.07
    token_mask_prob: float = 0.04
    mask_token_id: int = 1
    
    # --- SWA (start EARLY near observed best epoch 5) ---
    use_swa: bool = True
    swa_start_epoch: int = 4                  # Start near observed peak
    swa_lr: float = 7e-5
    
    # --- Threshold optimization ---
    use_optimal_threshold: bool = True
    threshold_min: float = 0.20
    threshold_max: float = 0.80
    threshold_step: float = 0.005
    threshold_objective: str = 'f1'
    min_precision_constraint: float = 0.60
    min_recall_constraint: float = 0.80
    
    # --- Single model (faster iteration) ---
    ensemble_size: int = 1
    use_diverse_ensemble: bool = False
    ensemble_dropout_variations: tuple = (0.0,)
    
    # --- Fine-tuning tail ---
    use_finetune_tail: bool = True
    finetune_epochs: int = 3
    finetune_lr: float = 1e-5
    
    # --- Data ---
    max_seq_length: int = 512
    num_workers: int = 2
    save_every: int = 1                       # Save every epoch for analysis
    use_amp: bool = True
    use_packed_sequences: bool = True
    use_layer_norm: bool = True
    
    def to_dict(self) -> Dict:
        """Export config to plain dict."""
        cfg: Dict[str, Any] = {}
        for cls in reversed(self.__class__.mro()):
            if cls is object:
                continue
            for k, v in cls.__dict__.items():
                if not k.startswith('_') and not callable(v):
                    cfg[k] = v
        return cfg


class BalancedConfig(TrainConfig):
    """
    BalancedConfig - Oracle-optimized to reduce Positive prediction bias.
    
    Problem from AggressiveRegConfig training:
    - Val Precision 57-68% (LOW - too many false positives)
    - Val Recall 70-82% (HIGH - over-predicting positive/vulnerable)
    - Best threshold 0.45 (too low)
    
    Oracle Recommendations Applied:
    1. Weighted BCE: neg_weight=1.3 to penalize false positives
    2. Reduced dropout (0.45-0.50 → 0.30-0.35) to reduce score compression
    3. Later SWA start (epoch 6) to let base model settle
    4. Higher inference threshold (0.55) to reduce false positives
    5. Slightly reduced weight_decay (2e-3 → 1e-3)
    
    Expected Outcome:
    - Precision: ↑ +5-15 points
    - Recall: ↓ moderate drop (acceptable)
    - AUC: ~0.78-0.80 (similar)
    """
    
    # --- Model (same as AggressiveRegConfig) ---
    vocab_size: int = data_config.get('vocab_size', 266)
    embed_dim: int = 64
    hidden_dim: int = 160
    num_layers: int = 2
    bidirectional: bool = True
    
    # Hybrid Model Features
    use_vuln_features: bool = True
    vuln_feature_dim: int = len(data_config.get('vuln_feature_names', [])) or 26
    vuln_feature_hidden_dim: int = 64
    
    # Multi-head attention
    use_multihead_attention: bool = True
    num_attention_heads: int = 4
    
    # --- Dropout REDUCED (Oracle: reduce score compression) ---
    embedding_dropout: float = 0.20           # Keep same
    rnn_dropout: float = 0.30                 # DOWN from 0.45
    attention_dropout: float = 0.15           # DOWN from 0.20
    classifier_dropout: float = 0.35          # DOWN from 0.50
    vuln_feature_dropout: float = 0.25        # DOWN from 0.30
    
    # --- Training ---
    batch_size: int = 96
    accumulation_steps: int = 1
    learning_rate: float = 3e-4
    max_lr: float = 1e-3
    weight_decay: float = 1e-3                # DOWN from 2e-3
    max_epochs: int = 20
    grad_clip: float = 0.5
    
    # --- Early stopping ---
    patience: int = 4                         # UP from 3 (more stable)
    min_delta: float = 3e-4
    
    # --- Scheduler ---
    scheduler_type: str = 'plateau'
    scheduler_patience: int = 2               # UP from 1 (more stable)
    scheduler_factor: float = 0.5
    scheduler_min_lr: float = 1e-6
    
    # --- Label smoothing (light) ---
    label_smoothing: float = 0.02
    label_smoothing_warmup_epochs: int = 3
    
    # --- WEIGHTED BCE (Oracle: penalize false positives) ---
    use_focal_weight: bool = False            # No focal loss
    use_focal_alpha: bool = False
    use_precision_focused_weight: bool = True  # ENABLED
    neg_weight_boost: float = 1.30            # Penalize FP more
    pos_weight_reduce: float = 1.00           # Keep pos weight normal
    
    # --- Token augmentation (moderate) ---
    use_token_augmentation: bool = True
    token_dropout_prob: float = 0.07
    token_mask_prob: float = 0.04
    mask_token_id: int = 1
    
    # --- SWA (start LATER to let base model settle) ---
    use_swa: bool = True
    swa_start_epoch: int = 6                  # UP from 4
    swa_lr: float = 7e-5
    
    # --- Threshold optimization (HIGHER range to reduce FP) ---
    use_optimal_threshold: bool = True
    threshold_min: float = 0.40               # UP from 0.20
    threshold_max: float = 0.75               # Narrower range
    threshold_step: float = 0.005
    threshold_objective: str = 'f1'
    min_precision_constraint: float = 0.65    # UP from 0.60
    min_recall_constraint: float = 0.70       # DOWN from 0.80 (allow lower recall)
    
    # --- Single model ---
    ensemble_size: int = 1
    use_diverse_ensemble: bool = False
    ensemble_dropout_variations: tuple = (0.0,)
    
    # --- Fine-tuning tail ---
    use_finetune_tail: bool = True
    finetune_epochs: int = 3
    finetune_lr: float = 1e-5
    
    # --- Data ---
    max_seq_length: int = 512
    num_workers: int = 2
    save_every: int = 1
    use_amp: bool = True
    use_packed_sequences: bool = True
    use_layer_norm: bool = True
    
    def to_dict(self) -> Dict:
        """Export config to plain dict."""
        cfg: Dict[str, Any] = {}
        for cls in reversed(self.__class__.mro()):
            if cls is object:
                continue
            for k, v in cls.__dict__.items():
                if not k.startswith('_') and not callable(v):
                    cfg[k] = v
        return cfg


class BalancedConfigV2(TrainConfig):
    """
    BalancedConfigV2 - Oracle-optimized after BalancedConfig analysis.
    
    Problem from BalancedConfig training:
    - Threshold stuck at 0.40 (lower bound) → model needs lower cutoff
    - Positive bias STILL EXISTS: Best Prec=62% vs Recall=82%
    - Epoch 6 volatility due to SWA start
    - neg_weight_boost=1.30 was not strong enough
    
    Oracle Recommendations Applied:
    1. Stronger neg_weight_boost (1.30 → 1.60) + pos_weight_reduce (1.0 → 0.90)
    2. Wider threshold range [0.20, 0.85] to find true optimal
    3. Use F0.5 threshold objective (favors precision over recall)
    4. Later SWA start (epoch 8) to avoid epoch 6 volatility
    5. Reduced dropout for more stable training
    6. Longer patience (5) and scheduler_patience (3)
    
    Expected Outcome:
    - Precision: ↑ significant increase
    - Recall: ↓ moderate drop (acceptable trade-off)
    - F1: maintain ~70%
    - AUC: maintain ~80%
    """
    
    # --- Model (same architecture) ---
    vocab_size: int = data_config.get('vocab_size', 266)
    embed_dim: int = 64
    hidden_dim: int = 160
    num_layers: int = 2
    bidirectional: bool = True
    
    # Hybrid Model Features
    use_vuln_features: bool = True
    vuln_feature_dim: int = len(data_config.get('vuln_feature_names', [])) or 26
    vuln_feature_hidden_dim: int = 64
    
    # Multi-head attention
    use_multihead_attention: bool = True
    num_attention_heads: int = 4
    
    # --- Dropout REDUCED (Oracle: more stable training) ---
    embedding_dropout: float = 0.18           # DOWN from 0.20
    rnn_dropout: float = 0.25                 # DOWN from 0.30
    attention_dropout: float = 0.12           # DOWN from 0.15
    classifier_dropout: float = 0.30          # DOWN from 0.35
    vuln_feature_dropout: float = 0.20        # DOWN from 0.25
    
    # --- Training ---
    batch_size: int = 96
    accumulation_steps: int = 1
    learning_rate: float = 3e-4
    max_lr: float = 1e-3
    weight_decay: float = 1e-3
    max_epochs: int = 25                      # UP from 20 (more time for convergence)
    grad_clip: float = 0.5
    
    # --- Early stopping (more patient) ---
    patience: int = 5                         # UP from 4
    min_delta: float = 3e-4
    
    # --- Scheduler (more patient) ---
    scheduler_type: str = 'plateau'
    scheduler_patience: int = 3               # UP from 2
    scheduler_factor: float = 0.5
    scheduler_min_lr: float = 1e-6
    
    # --- Label smoothing (light) ---
    label_smoothing: float = 0.02
    label_smoothing_warmup_epochs: int = 3
    
    # --- STRONGER WEIGHTED BCE (Oracle: penalize FP more aggressively) ---
    use_focal_weight: bool = False
    use_focal_alpha: bool = False
    use_precision_focused_weight: bool = True
    neg_weight_boost: float = 1.60            # UP from 1.30 - stronger FP penalty
    pos_weight_reduce: float = 0.90           # DOWN from 1.00 - slightly reduce pos weight
    
    # --- Token augmentation (moderate) ---
    use_token_augmentation: bool = True
    token_dropout_prob: float = 0.07
    token_mask_prob: float = 0.04
    mask_token_id: int = 1
    
    # --- SWA (start LATER to avoid epoch 6 volatility) ---
    use_swa: bool = True
    swa_start_epoch: int = 8                  # UP from 6
    swa_lr: float = 7e-5
    
    # --- Threshold optimization (WIDER range + F0.5 objective) ---
    use_optimal_threshold: bool = True
    threshold_min: float = 0.20               # DOWN from 0.40 - find true optimal
    threshold_max: float = 0.85               # UP from 0.75 - wider range
    threshold_step: float = 0.01              # Finer step
    threshold_objective: str = 'f0.5'         # CHANGED from 'f1' - favor precision
    min_precision_constraint: float = 0.70    # UP from 0.65
    min_recall_constraint: float = 0.60       # DOWN from 0.70 (allow lower recall)
    
    # --- Single model ---
    ensemble_size: int = 1
    use_diverse_ensemble: bool = False
    ensemble_dropout_variations: tuple = (0.0,)
    
    # --- Fine-tuning tail ---
    use_finetune_tail: bool = True
    finetune_epochs: int = 3
    finetune_lr: float = 1e-5
    
    # --- Data ---
    max_seq_length: int = 512
    num_workers: int = 2
    save_every: int = 1
    use_amp: bool = True
    use_packed_sequences: bool = True
    use_layer_norm: bool = True
    
    def to_dict(self) -> Dict:
        """Export config to plain dict."""
        cfg: Dict[str, Any] = {}
        for cls in reversed(self.__class__.mro()):
            if cls is object:
                continue
            for k, v in cls.__dict__.items():
                if not k.startswith('_') and not callable(v):
                    cfg[k] = v
        return cfg


class BalancedConfigV3(TrainConfig):
    """
    BalancedConfigV3 - Recover F1 >= 70% after V2 recall collapse.
    
    Problem from BalancedConfigV2:
    - F1 DROPPED: 70.54% → 64.37% (regression!)
    - Recall COLLAPSED: 82% → 56% (missing half of vulnerabilities)
    - Precision OVERCORRECTED: 62% → 76%
    - Cause: neg_weight_boost=1.60 + F0.5 objective = too aggressive
    
    Oracle Recommendations Applied:
    1. MILD class weighting: neg_weight_boost=1.20 (down from 1.60)
    2. KEEP pos_weight_reduce=1.00 (don't reduce positive importance)
    3. REVERT to F1 objective with recall floor >= 0.65
    4. REDUCE dropout slightly (help model learn positives better)
    5. Threshold sweep with recall constraint
    
    Target Metrics:
    - F1 >= 70%
    - Precision >= 68%
    - Recall >= 65%
    - AUC >= 80%
    """
    
    # --- Model (same architecture) ---
    vocab_size: int = data_config.get('vocab_size', 266)
    embed_dim: int = 64
    hidden_dim: int = 160
    num_layers: int = 2
    bidirectional: bool = True
    
    # Hybrid Model Features
    use_vuln_features: bool = True
    vuln_feature_dim: int = len(data_config.get('vuln_feature_names', [])) or 26
    vuln_feature_hidden_dim: int = 64
    
    # Multi-head attention
    use_multihead_attention: bool = True
    num_attention_heads: int = 4
    
    # --- Dropout REDUCED (Oracle: help model learn positives better) ---
    embedding_dropout: float = 0.15           # DOWN from 0.18
    rnn_dropout: float = 0.20                 # DOWN from 0.25
    attention_dropout: float = 0.10           # DOWN from 0.12
    classifier_dropout: float = 0.25          # DOWN from 0.30
    vuln_feature_dropout: float = 0.18        # DOWN from 0.20
    
    # --- Training ---
    batch_size: int = 96
    accumulation_steps: int = 1
    learning_rate: float = 3e-4
    max_lr: float = 1e-3
    weight_decay: float = 1e-3
    max_epochs: int = 25
    grad_clip: float = 0.5
    
    # --- Early stopping ---
    patience: int = 5
    min_delta: float = 3e-4
    
    # --- Scheduler ---
    scheduler_type: str = 'plateau'
    scheduler_patience: int = 3
    scheduler_factor: float = 0.5
    scheduler_min_lr: float = 1e-6
    
    # --- Label smoothing (light) ---
    label_smoothing: float = 0.02
    label_smoothing_warmup_epochs: int = 3
    
    # --- MILD WEIGHTED BCE (Oracle: reduce from 1.60 to 1.20) ---
    use_focal_weight: bool = False
    use_focal_alpha: bool = False
    use_precision_focused_weight: bool = True
    neg_weight_boost: float = 1.20            # DOWN from 1.60 - mild FP penalty
    pos_weight_reduce: float = 1.00           # KEEP at 1.00 - don't reduce positives
    
    # --- Token augmentation ---
    use_token_augmentation: bool = True
    token_dropout_prob: float = 0.05          # DOWN from 0.07 - less augmentation
    token_mask_prob: float = 0.03             # DOWN from 0.04
    mask_token_id: int = 1
    
    # --- SWA ---
    use_swa: bool = True
    swa_start_epoch: int = 8
    swa_lr: float = 7e-5
    
    # --- Threshold optimization (F1 with RECALL FLOOR) ---
    use_optimal_threshold: bool = True
    threshold_min: float = 0.20
    threshold_max: float = 0.80
    threshold_step: float = 0.01
    threshold_objective: str = 'f1'           # REVERT from 'f0.5' to 'f1'
    min_precision_constraint: float = 0.65
    min_recall_constraint: float = 0.65       # UP from 0.60 - ensure recall >= 65%
    
    # --- Single model ---
    ensemble_size: int = 1
    use_diverse_ensemble: bool = False
    ensemble_dropout_variations: tuple = (0.0,)
    
    # --- Fine-tuning tail ---
    use_finetune_tail: bool = True
    finetune_epochs: int = 3
    finetune_lr: float = 1e-5
    
    # --- Data ---
    max_seq_length: int = 512
    num_workers: int = 2
    save_every: int = 1
    use_amp: bool = True
    use_packed_sequences: bool = True
    use_layer_norm: bool = True
    
    def to_dict(self) -> Dict:
        """Export config to plain dict."""
        cfg: Dict[str, Any] = {}
        for cls in reversed(self.__class__.mro()):
            if cls is object:
                continue
            for k, v in cls.__dict__.items():
                if not k.startswith('_') and not callable(v):
                    cfg[k] = v
        return cfg


class OptimalConfig(TrainConfig):
    """
    OptimalConfig - Final Oracle-optimized configuration.
    
    BalancedConfigV3 Results:
    - F1: 71.7% ✅, AUC: 81.95% ✅
    - Precision: 63.3% (target 68% not met)
    - Recall: 82.6% (high - positive bias)
    - Threshold: 0.35 (too low)
    
    Oracle Analysis:
    - Ranking is good (AUC 82%) - model separates classes well
    - Problem is OPERATING POINT - threshold selection favors recall
    - Need recall-constrained threshold to push precision up
    
    Key Changes from V3:
    1. threshold_objective: 'recall_constrained' with min_recall=0.78
    2. neg_weight_boost: 1.25 (mild increase from 1.20)
    3. label_smoothing: 0.01 (reduce probability squashing)
    4. Wider threshold range for better operating point
    
    Target Metrics:
    - F1 >= 71%
    - Precision >= 68%
    - Recall >= 78%
    - AUC >= 82%
    """
    
    # --- Model (same architecture) ---
    vocab_size: int = data_config.get('vocab_size', 266)
    embed_dim: int = 64
    hidden_dim: int = 160
    num_layers: int = 2
    bidirectional: bool = True
    
    # Hybrid Model Features
    use_vuln_features: bool = True
    vuln_feature_dim: int = len(data_config.get('vuln_feature_names', [])) or 26
    vuln_feature_hidden_dim: int = 64
    
    # Multi-head attention
    use_multihead_attention: bool = True
    num_attention_heads: int = 4
    
    # --- Dropout (same as V3 - working well) ---
    embedding_dropout: float = 0.15
    rnn_dropout: float = 0.20
    attention_dropout: float = 0.10
    classifier_dropout: float = 0.25
    vuln_feature_dropout: float = 0.18
    
    # --- Training ---
    batch_size: int = 96
    accumulation_steps: int = 1
    learning_rate: float = 3e-4
    max_lr: float = 1e-3
    weight_decay: float = 1e-3
    max_epochs: int = 25
    grad_clip: float = 0.5
    
    # --- Early stopping ---
    patience: int = 5
    min_delta: float = 3e-4
    
    # --- Scheduler ---
    scheduler_type: str = 'plateau'
    scheduler_patience: int = 3
    scheduler_factor: float = 0.5
    scheduler_min_lr: float = 1e-6
    
    # --- Label smoothing REDUCED (Oracle: reduce probability squashing) ---
    label_smoothing: float = 0.01             # DOWN from 0.02
    label_smoothing_warmup_epochs: int = 2    # DOWN from 3
    
    # --- MILD WEIGHTED BCE (Oracle: slight increase) ---
    use_focal_weight: bool = False
    use_focal_alpha: bool = False
    use_precision_focused_weight: bool = True
    neg_weight_boost: float = 1.25            # UP from 1.20 - slightly more FP penalty
    pos_weight_reduce: float = 1.00           # Keep at 1.00
    
    # --- Token augmentation ---
    use_token_augmentation: bool = True
    token_dropout_prob: float = 0.05
    token_mask_prob: float = 0.03
    mask_token_id: int = 1
    
    # --- SWA ---
    use_swa: bool = True
    swa_start_epoch: int = 10                 # UP from 8 - let model stabilize more
    swa_lr: float = 5e-5                      # DOWN from 7e-5
    
    # --- Threshold optimization (RECALL-CONSTRAINED for better precision) ---
    use_optimal_threshold: bool = True
    threshold_min: float = 0.25               # UP from 0.20
    threshold_max: float = 0.80
    threshold_step: float = 0.005             # Finer step
    threshold_objective: str = 'recall_constrained'  # CHANGED from 'f1'
    min_precision_constraint: float = 0.65
    min_recall_constraint: float = 0.78       # UP from 0.65 - ensure recall >= 78%
    
    # --- Single model ---
    ensemble_size: int = 1
    use_diverse_ensemble: bool = False
    ensemble_dropout_variations: tuple = (0.0,)
    
    # --- Fine-tuning tail ---
    use_finetune_tail: bool = True
    finetune_epochs: int = 3
    finetune_lr: float = 1e-5
    
    # --- Data ---
    max_seq_length: int = 512
    num_workers: int = 2
    save_every: int = 1
    use_amp: bool = True
    use_packed_sequences: bool = True
    use_layer_norm: bool = True
    
    def to_dict(self) -> Dict:
        """Export config to plain dict."""
        cfg: Dict[str, Any] = {}
        for cls in reversed(self.__class__.mro()):
            if cls is object:
                continue
            for k, v in cls.__dict__.items():
                if not k.startswith('_') and not callable(v):
                    cfg[k] = v
        return cfg


class OptimalConfigV2(TrainConfig):
    """
    OptimalConfigV2 - Oracle-optimized after analyzing OptimalConfig results.
    
    OptimalConfig Results:
    - F1: 71.7% ✅ (target met)
    - AUC: 80% ❌ (target 82%)
    - Precision: 63% ❌ (target 68%)
    - Recall: 83% ✅ (exceeds target 78%)
    - Threshold: stuck at low values
    
    Issues Identified:
    1. Overfitting after epoch 5-6 (val loss plateaus, train keeps dropping)
    2. Precision too low - threshold selecting too low
    3. Train AUC > Val AUC gap widening
    
    Key Changes from OptimalConfig:
    1. neg_weight_boost: 1.25 → 1.35 (more FP penalty)
    2. pos_weight_reduce: 1.00 → 0.95 (slight positive down-weight)
    3. threshold_objective: 'f0.5_recall_constrained' (precision-focused)
    4. threshold_min: 0.25 → 0.35 (avoid too-low thresholds)
    5. swa_start_epoch: 10 → 7 (lock generalization before overfit)
    6. classifier_dropout: 0.25 → 0.33 (reduce overfitting)
    7. scheduler_patience: 3 → 1 (react faster to plateau)
    8. Increased dropout across layers for better generalization
    
    Target Metrics:
    - F1 >= 71%
    - Precision >= 68%
    - Recall >= 78%
    - AUC >= 82%
    """
    
    # --- Model (same architecture) ---
    vocab_size: int = data_config.get('vocab_size', 266)
    embed_dim: int = 64
    hidden_dim: int = 160
    num_layers: int = 2
    bidirectional: bool = True
    
    # Hybrid Model Features
    use_vuln_features: bool = True
    vuln_feature_dim: int = len(data_config.get('vuln_feature_names', [])) or 26
    vuln_feature_hidden_dim: int = 64
    
    # Multi-head attention
    use_multihead_attention: bool = True
    num_attention_heads: int = 4
    
    # --- Dropout INCREASED (Oracle: reduce overfitting) ---
    embedding_dropout: float = 0.15             # UP from 0.15 (keep)
    rnn_dropout: float = 0.25                   # UP from 0.20
    attention_dropout: float = 0.15             # UP from 0.10
    classifier_dropout: float = 0.33            # UP from 0.25 - key change!
    vuln_feature_dropout: float = 0.25          # UP from 0.18
    
    # --- Training ---
    batch_size: int = 96
    accumulation_steps: int = 1
    learning_rate: float = 3e-4
    max_lr: float = 1e-3
    weight_decay: float = 1e-2                  # UP from 1e-3 - stronger L2
    max_epochs: int = 25
    grad_clip: float = 0.5
    
    # --- Early stopping ---
    patience: int = 5
    min_delta: float = 3e-4
    
    # --- Scheduler FASTER reaction (Oracle: react to plateau sooner) ---
    scheduler_type: str = 'plateau'
    scheduler_patience: int = 1                 # DOWN from 3 - react faster!
    scheduler_factor: float = 0.5
    scheduler_min_lr: float = 1e-6
    
    # --- Label smoothing (keep low) ---
    label_smoothing: float = 0.01
    label_smoothing_warmup_epochs: int = 2
    
    # --- STRONGER WEIGHTED BCE (Oracle: precision focus) ---
    use_focal_weight: bool = False
    use_focal_alpha: bool = False
    use_precision_focused_weight: bool = True
    neg_weight_boost: float = 1.35              # UP from 1.25 - more FP penalty
    pos_weight_reduce: float = 0.95             # DOWN from 1.00 - slight positive down-weight
    
    # --- Token augmentation ---
    use_token_augmentation: bool = True
    token_dropout_prob: float = 0.05
    token_mask_prob: float = 0.03
    mask_token_id: int = 1
    
    # --- SWA EARLIER (Oracle: lock generalization before epoch 5-6 overfit) ---
    use_swa: bool = True
    swa_start_epoch: int = 7                    # DOWN from 10 - earlier!
    swa_lr: float = 1e-4                        # UP from 5e-5
    
    # --- Threshold optimization PRECISION-FOCUSED (Oracle: avoid low thresholds) ---
    use_optimal_threshold: bool = True
    threshold_min: float = 0.35                 # UP from 0.25 - avoid too-low
    threshold_max: float = 0.75                 # DOWN from 0.80
    threshold_step: float = 0.005
    threshold_objective: str = 'f0.5_recall_constrained'  # CHANGED from 'recall_constrained'
    min_precision_constraint: float = 0.68      # UP from 0.65 - force precision
    min_recall_constraint: float = 0.78
    
    # --- Single model ---
    ensemble_size: int = 1
    use_diverse_ensemble: bool = False
    ensemble_dropout_variations: tuple = (0.0,)
    
    # --- Fine-tuning tail ---
    use_finetune_tail: bool = True
    finetune_epochs: int = 3
    finetune_lr: float = 1e-5
    
    # --- Data ---
    max_seq_length: int = 512
    num_workers: int = 2
    save_every: int = 1
    use_amp: bool = True
    use_packed_sequences: bool = True
    use_layer_norm: bool = True
    
    def to_dict(self) -> Dict:
        """Export config to plain dict."""
        cfg: Dict[str, Any] = {}
        for cls in reversed(self.__class__.mro()):
            if cls is object:
                continue
            for k, v in cls.__dict__.items():
                if not k.startswith('_') and not callable(v):
                    cfg[k] = v
        return cfg


class OptimalConfigV3(TrainConfig):
    """
    OptimalConfigV3 - Oracle recovery after V2 regression.
    
    OptimalConfigV2 Results (REGRESSION):
    - F1: 65% ❌ (down from 71.7%)
    - AUC: 73-80% ❌ (down from 81-82%)
    - Precision: 61% ❌
    - Recall: 69%
    - Early stopped at epoch 10
    
    Root Cause Analysis (Oracle):
    1. neg_weight_boost=1.35 + pos_weight_reduce=0.95 TOO AGGRESSIVE
       → Suppressed positives → recall drop → F1 collapse
    2. scheduler_patience=1 → LR decay too early → under-training → AUC drop
    3. weight_decay=1e-2 TOO STRONG → hurt separability
    4. classifier_dropout=0.33 + above → overregularized → underfit
    5. swa_start_epoch=7 with early stop at 10 → averaged too early
    
    V3 Strategy: RECOVER AUC first, then use THRESHOLDING for precision
    
    Key Changes from V2:
    1. neg_weight_boost: 1.35 → 1.25 (undo aggressive penalty)
    2. pos_weight_reduce: 0.95 → 1.00 (restore positive weight)
    3. scheduler_patience: 1 → 3 (allow more training)
    4. weight_decay: 1e-2 → 1e-3 (less regularization)
    5. classifier_dropout: 0.33 → 0.25 (better separation)
    6. swa_start_epoch: 7 → 12 (SWA after convergence)
    7. threshold_min: 0.35 → 0.25 (allow recall recovery)
    8. threshold_objective: precision_constrained (target precision directly)
    
    Target Metrics:
    - F1 >= 71%
    - Precision >= 68%
    - Recall >= 78%
    - AUC >= 82%
    """
    
    # --- Model (same architecture) ---
    vocab_size: int = data_config.get('vocab_size', 266)
    embed_dim: int = 64
    hidden_dim: int = 160
    num_layers: int = 2
    bidirectional: bool = True
    
    # Hybrid Model Features
    use_vuln_features: bool = True
    vuln_feature_dim: int = len(data_config.get('vuln_feature_names', [])) or 26
    vuln_feature_hidden_dim: int = 64
    
    # Multi-head attention
    use_multihead_attention: bool = True
    num_attention_heads: int = 4
    
    # --- Dropout REDUCED (Oracle: V2 was overregularized) ---
    embedding_dropout: float = 0.15
    rnn_dropout: float = 0.20                   # DOWN from 0.25
    attention_dropout: float = 0.10             # DOWN from 0.15
    classifier_dropout: float = 0.25            # DOWN from 0.33 - key fix!
    vuln_feature_dropout: float = 0.18          # DOWN from 0.25
    
    # --- Training ---
    batch_size: int = 96
    accumulation_steps: int = 1
    learning_rate: float = 3e-4
    max_lr: float = 1e-3
    weight_decay: float = 1e-3                  # DOWN from 1e-2 - key fix!
    max_epochs: int = 30                        # UP from 25 - allow more training
    grad_clip: float = 0.5
    
    # --- Early stopping ---
    patience: int = 6                           # UP from 5 - more patient
    min_delta: float = 3e-4
    
    # --- Scheduler SLOWER reaction (Oracle: V2 decayed too fast) ---
    scheduler_type: str = 'plateau'
    scheduler_patience: int = 3                 # UP from 1 - key fix!
    scheduler_factor: float = 0.5
    scheduler_min_lr: float = 1e-6
    
    # --- Label smoothing (keep low) ---
    label_smoothing: float = 0.01
    label_smoothing_warmup_epochs: int = 2
    
    # --- BALANCED WEIGHTED BCE (Oracle: undo V2 aggressive weighting) ---
    use_focal_weight: bool = False
    use_focal_alpha: bool = False
    use_precision_focused_weight: bool = True
    neg_weight_boost: float = 1.25              # DOWN from 1.35 - key fix!
    pos_weight_reduce: float = 1.00             # UP from 0.95 - key fix!
    
    # --- Token augmentation ---
    use_token_augmentation: bool = True
    token_dropout_prob: float = 0.05
    token_mask_prob: float = 0.03
    mask_token_id: int = 1
    
    # --- SWA LATER (Oracle: V2 averaged too early) ---
    use_swa: bool = True
    swa_start_epoch: int = 12                   # UP from 7 - key fix!
    swa_lr: float = 5e-5                        # DOWN from 1e-4
    
    # --- Threshold: PRECISION-CONSTRAINED (Oracle: target precision directly) ---
    use_optimal_threshold: bool = True
    threshold_min: float = 0.25                 # DOWN from 0.35 - allow recall recovery
    threshold_max: float = 0.75
    threshold_step: float = 0.005
    threshold_objective: str = 'precision_constrained'  # CHANGED - target precision directly
    min_precision_constraint: float = 0.68      # Target precision
    min_recall_constraint: float = 0.78         # Keep as backup
    
    # --- Single model ---
    ensemble_size: int = 1
    use_diverse_ensemble: bool = False
    ensemble_dropout_variations: tuple = (0.0,)
    
    # --- Fine-tuning tail ---
    use_finetune_tail: bool = True
    finetune_epochs: int = 3
    finetune_lr: float = 1e-5
    
    # --- Data ---
    max_seq_length: int = 512
    num_workers: int = 2
    save_every: int = 1
    use_amp: bool = True
    use_packed_sequences: bool = True
    use_layer_norm: bool = True
    
    def to_dict(self) -> Dict:
        """Export config to plain dict."""
        cfg: Dict[str, Any] = {}
        for cls in reversed(self.__class__.mro()):
            if cls is object:
                continue
            for k, v in cls.__dict__.items():
                if not k.startswith('_') and not callable(v):
                    cfg[k] = v
        return cfg


# Use OptimalConfigV3 for this training run (Oracle recovery)
config = OptimalConfigV3()

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
        
        return self.early_stop
            
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


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance by down-weighting easy examples.
    
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    
    Args:
        gamma: Focusing parameter (default: 2.0). Higher values focus more on hard examples.
        alpha: Class weights tensor [neg_weight, pos_weight] or None for equal weights.
        reduction: 'mean', 'sum', or 'none'
    """
    def __init__(self, gamma: float = 2.0, alpha: torch.Tensor = None, reduction: str = 'mean'):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        if alpha is not None:
            self.register_buffer('alpha', alpha.float())
        else:
            self.alpha = None
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Get probabilities
        probs = torch.softmax(logits, dim=1)
        
        # Get probability of true class
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        
        # Get p_t (probability of correct class)
        p_t = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        
        # Focal weight: (1 - p_t)^gamma
        focal_weight = (1 - p_t) ** self.gamma
        
        # Apply focal weight
        focal_loss = focal_weight * ce_loss
        
        # Apply class weights (alpha)
        if self.alpha is not None:
            alpha_t = self.alpha.gather(0, targets)
            focal_loss = alpha_t * focal_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss

def find_optimal_threshold(y_true, y_probs, min_t=0.1, max_t=0.9, step=0.005,
                           objective='f1', min_precision=0.60, min_recall=0.80):
    """Find probability threshold that maximizes the specified objective.
    
    Args:
        y_true: Ground truth labels
        y_probs: Predicted probabilities
        min_t: Minimum threshold to search (default: 0.1)
        max_t: Maximum threshold to search (default: 0.9)
        step: Step size for threshold search (default: 0.005 for finer search)
        objective: Optimization objective:
            - 'f1': maximize F1 score
            - 'f0.5': maximize F0.5 score (precision-weighted)
            - 'precision_constrained': maximize F1 subject to precision >= min_precision
            - 'recall_constrained': maximize F1 subject to recall >= min_recall
            - 'recall_constrained_strict': pick HIGHEST threshold that meets recall >= min_recall
              (Oracle V4: directly reduces FP while respecting recall floor)
            - 'f0.5_recall_constrained': maximize F0.5 subject to recall >= min_recall
              (Oracle V2: precision-weighted with recall floor)
        min_precision: Minimum precision constraint for 'precision_constrained' mode
        min_recall: Minimum recall constraint for 'recall_constrained' modes (default: 0.80)
    
    Returns:
        best_t: Optimal threshold
        best_score: Best score achieved (F1 or F0.5 depending on objective)
    """
    from sklearn.metrics import fbeta_score
    
    thresholds = np.arange(min_t, max_t + step, step)
    best_t = 0.5
    best_score = 0.0
    
    # For recall_constrained_strict: find highest threshold meeting recall constraint
    if objective == 'recall_constrained_strict':
        # Search from HIGH to LOW threshold, pick first one meeting recall >= min_recall
        for t in reversed(thresholds):
            y_pred = (y_probs >= t).astype(int)
            rec = recall_score(y_true, y_pred, zero_division=0)
            if rec >= min_recall:
                f1 = f1_score(y_true, y_pred, zero_division=0)
                return t, f1
        # Fallback: no threshold meets recall constraint, use max F1
        return find_optimal_threshold(y_true, y_probs, min_t, max_t, step, 'f1')
    
    for t in thresholds:
        y_pred = (y_probs >= t).astype(int)
        
        if objective == 'f0.5':
            # F0.5 score: weights precision higher than recall
            score = fbeta_score(y_true, y_pred, beta=0.5, zero_division=0)
        elif objective == 'precision_constrained':
            # Maximize F1 subject to precision >= min_precision
            prec = precision_score(y_true, y_pred, zero_division=0)
            if prec >= min_precision:
                score = f1_score(y_true, y_pred, zero_division=0)
            else:
                score = 0.0  # Skip thresholds that don't meet precision constraint
        elif objective == 'recall_constrained':
            # Maximize F1 subject to recall >= min_recall
            # Oracle recommendation: maintain recall >= 80% while improving precision
            rec = recall_score(y_true, y_pred, zero_division=0)
            if rec >= min_recall:
                score = f1_score(y_true, y_pred, zero_division=0)
            else:
                score = 0.0  # Skip thresholds that don't meet recall constraint
        elif objective == 'f0.5_recall_constrained':
            # Maximize F0.5 (precision-weighted) subject to recall >= min_recall
            # Oracle V2: balance precision focus with recall floor
            rec = recall_score(y_true, y_pred, zero_division=0)
            if rec >= min_recall:
                score = fbeta_score(y_true, y_pred, beta=0.5, zero_division=0)
            else:
                score = 0.0  # Skip thresholds that don't meet recall constraint
        else:
            # Default: maximize F1
            score = f1_score(y_true, y_pred, zero_division=0)
        
        if score > best_score:
            best_score = score
            best_t = t
    
    # If precision_constrained found no valid threshold, fall back to max F1
    if objective == 'precision_constrained' and best_score == 0.0:
        return find_optimal_threshold(y_true, y_probs, min_t, max_t, step, 'f1')
    
    # If recall_constrained found no valid threshold, fall back to max F1
    if objective == 'recall_constrained' and best_score == 0.0:
        return find_optimal_threshold(y_true, y_probs, min_t, max_t, step, 'f1')
    
    # If f0.5_recall_constrained found no valid threshold, fall back to max F0.5
    if objective == 'f0.5_recall_constrained' and best_score == 0.0:
        return find_optimal_threshold(y_true, y_probs, min_t, max_t, step, 'f0.5')
            
    return best_t, best_score


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


@torch.no_grad()
def evaluate_split(
    model,
    dataloader,
    device,
    threshold_grid=(0.1, 0.9, 0.01),
    use_amp=True,
    verbose=False,
):
    """
    Evaluate model on a data split with proper eval mode.
    
    This function provides accurate metrics by:
    1. Running model in eval() mode (dropout/augmentation off)
    2. Computing metrics on entire split (not batch-wise average)
    3. Grid search for optimal F1 threshold
    
    Use this to compare Train vs Val metrics fairly.
    """
    model.eval()
    all_probs = []
    all_labels = []

    for batch in tqdm(dataloader, desc="Eval Split", leave=False):
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        labels = batch["labels"].cpu().numpy()
        vuln_feats = batch.get("vuln_features")
        lengths = batch.get("lengths")

        if vuln_feats is not None:
            vuln_feats = vuln_feats.to(device, non_blocking=True)
        if lengths is not None:
            lengths = lengths.to(device, non_blocking=True)

        with autocast(device_type='cuda', enabled=use_amp):
            logits = model(input_ids, attention_mask, vuln_feats, lengths)
        probs = torch.softmax(logits, dim=-1)[:, 1].detach().cpu().numpy()

        all_probs.append(probs)
        all_labels.append(labels)

    all_probs = np.concatenate(all_probs)
    all_labels = np.concatenate(all_labels)

    # AUC (threshold-free)
    try:
        auc_roc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc_roc = float("nan")

    # Threshold optimization for F1
    t_min, t_max, t_step = threshold_grid
    thresholds = np.arange(t_min, t_max + 1e-8, t_step)

    best_f1 = -1.0
    best_thr = 0.5
    best_prec = 0.0
    best_rec = 0.0

    for thr in thresholds:
        preds = (all_probs >= thr).astype(int)
        f1 = f1_score(all_labels, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thr = thr
            best_prec = precision_score(all_labels, preds, zero_division=0)
            best_rec = recall_score(all_labels, preds, zero_division=0)

    if verbose:
        print(
            f"[Eval Split] AUC={auc_roc:.4f} | "
            f"F1={best_f1:.4f} (thr={best_thr:.2f}) | "
            f"Prec={best_prec:.4f} | Rec={best_rec:.4f}"
        )

    return {
        "auc_roc": auc_roc,
        "best_f1": best_f1,
        "best_threshold": best_thr,
        "precision": best_prec,
        "recall": best_rec,
        "labels": all_labels,
        "probs": all_probs,
    }


def train_epoch(model, loader, optimizer, criterion, scaler, grad_clip, accum_steps, use_amp):
    model.train()
    total_loss = 0
    all_preds, all_labels, all_probs = [], [], []
    
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
        
        probs = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
        preds = logits.argmax(dim=1).cpu().numpy()
        all_probs.extend(probs)
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())
        
    avg_loss = total_loss / len(loader)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
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
        'auc_roc': auc_roc
    }

@torch.no_grad()
def evaluate(model, loader, criterion, use_amp, threshold=None, find_threshold=False,
             threshold_objective='f1', min_precision_constraint=0.60, min_recall_constraint=0.80,
             threshold_min=0.1, threshold_max=0.9, threshold_step=0.005):
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
        best_t, best_score = find_optimal_threshold(
            all_labels, all_probs,
            min_t=threshold_min,
            max_t=threshold_max,
            step=threshold_step,
            objective=threshold_objective,
            min_precision=min_precision_constraint,
            min_recall=min_recall_constraint
        )
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
    
    Quick Win improvements:
    - Precision-focused class weighting: boost negative weight to reduce FP
    """
    # Calculate class weights
    all_labels = []
    for batch in train_loader:
        all_labels.extend(batch['labels'].numpy())
    
    neg_count = np.sum(np.array(all_labels) == 0)
    pos_count = np.sum(np.array(all_labels) == 1)
    
    # Quick Win 1.1: Precision-focused class weighting
    use_precision_focused = getattr(config, 'use_precision_focused_weight', False)
    if use_precision_focused:
        # Boost negative class weight to reduce false positives (improve precision)
        neg_boost = getattr(config, 'neg_weight_boost', 1.15)
        pos_reduce = getattr(config, 'pos_weight_reduce', 0.85)
        
        # Compute base ratio and apply precision focus
        base_ratio = float(neg_count) / float(pos_count)
        w_neg = neg_boost  # Increase penalty for FP (predicting 1 when actual is 0)
        w_pos = base_ratio * pos_reduce  # Slightly reduce weight on positives
        
        weight = torch.tensor([w_neg, w_pos], dtype=torch.float32, device=DEVICE)
        print(f"Precision-focused weights: neg={w_neg:.3f}, pos={w_pos:.3f} (boost={neg_boost}, reduce={pos_reduce})")
    else:
        # Original: Slight boost to minority class if imbalanced
        ratio = float(neg_count) / float(pos_count)
        weight = torch.tensor([1.0, ratio], dtype=torch.float32, device=DEVICE)
        print(f"Standard class weights: neg=1.0, pos={ratio:.3f}")
    
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
    elif config.scheduler_type == 'cosine':
        # Cosine Annealing with Warm Restarts
        warmup_epochs = getattr(config, 'warmup_epochs', 3)
        # Linear warmup scheduler
        warmup_scheduler = optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=warmup_epochs
        )
        # Cosine annealing after warmup
        cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.max_epochs - warmup_epochs,
            eta_min=getattr(config, 'scheduler_min_lr', 1e-6)
        )
        # Sequential: warmup then cosine
        scheduler = optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_epochs]
        )
        print(f"Cosine scheduler: warmup {warmup_epochs} epochs, then cosine annealing")
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
        
        # Create criterion based on config
        use_focal = getattr(config, 'use_focal_weight', False)
        if use_focal:
            # Focal Loss for hard example mining
            focal_gamma = getattr(config, 'focal_gamma', 2.0)
            
            # Check if using separate focal alpha (Oracle V4 fix)
            use_focal_alpha = getattr(config, 'use_focal_alpha', False)
            if use_focal_alpha:
                # Use explicit focal alpha for class balancing (penalize FP more)
                focal_alpha_pos = getattr(config, 'focal_alpha_pos', 0.5)
                focal_alpha_neg = getattr(config, 'focal_alpha_neg', 0.5)
                focal_weight = torch.tensor([focal_alpha_neg, focal_alpha_pos], 
                                           dtype=torch.float32, device=DEVICE)
                criterion = FocalLoss(gamma=focal_gamma, alpha=focal_weight)
            else:
                # Use class weight as focal alpha
                criterion = FocalLoss(gamma=focal_gamma, alpha=weight)
        elif current_smoothing > 0:
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
        threshold_objective = getattr(config, 'threshold_objective', 'f1')
        min_precision_constraint = getattr(config, 'min_precision_constraint', 0.60)
        min_recall_constraint = getattr(config, 'min_recall_constraint', 0.80)
        val_metrics = evaluate(
            model, val_loader, criterion, config.use_amp,
            find_threshold=config.use_optimal_threshold,
            threshold_objective=threshold_objective,
            min_precision_constraint=min_precision_constraint,
            min_recall_constraint=min_recall_constraint,
            threshold_min=config.threshold_min,
            threshold_max=config.threshold_max,
            threshold_step=config.threshold_step
        )
        
        epoch_time = time.time() - epoch_start
        
        # Log
        history['train'].append(train_metrics)
        history['val'].append(val_metrics)
        
        current_lr = optimizer.param_groups[0]['lr']
        val_t = val_metrics['best_threshold']
        
        loss_type = "focal" if use_focal else f"smooth={current_smoothing:.3f}"
        print(
            f"Ep {epoch:2d}/{config.max_epochs} ({epoch_time:.0f}s) | "
            f"Train: L={train_metrics['loss']:.3f} F1={train_metrics['f1']:.3f} | "
            f"Val: L={val_metrics['loss']:.3f} F1={val_metrics['f1']:.3f} "
            f"AUC={val_metrics['auc_roc']:.3f} T={val_t:.2f} | "
            f"LR={current_lr:.2e} | {loss_type}"
        )
        
        # Scheduler step (depends on type)
        if use_swa and epoch >= swa_start_epoch:
            # SWA update (after swa_start_epoch)
            base_model = model.module if isinstance(model, nn.DataParallel) else model
            swa_model.update_parameters(base_model)
            swa_scheduler.step()
            print(f"  [SWA] Updated averaged model (epoch {epoch})")
        elif config.scheduler_type == 'plateau' and not in_finetune:
            # ReduceLROnPlateau: step with Val F1 (not AUC) for better F1 optimization
            scheduler.step(val_metrics['f1'])
        elif config.scheduler_type in ['cosine', 'onecycle'] and not in_finetune:
            # Cosine/OneCycle: step every epoch
            scheduler.step()
        
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
            find_threshold=config.use_optimal_threshold,
            threshold_objective=getattr(config, 'threshold_objective', 'f1'),
            min_precision_constraint=getattr(config, 'min_precision_constraint', 0.60),
            min_recall_constraint=getattr(config, 'min_recall_constraint', 0.80),
            threshold_min=config.threshold_min,
            threshold_max=config.threshold_max,
            threshold_step=config.threshold_step
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
    
    # Quick Win 1.5: Diverse ensemble with dropout variations
    use_diverse = getattr(config, 'use_diverse_ensemble', False)
    dropout_variations = getattr(config, 'ensemble_dropout_variations', (0.0,) * 5)
    if use_diverse:
        print(f"Diverse ensemble enabled: dropout variations = {dropout_variations}")
    
    ensemble_models = []
    last_swa_model = None
    
    for i in range(config.ensemble_size):
        seed = config.ensemble_base_seed + i
        print(f"\n{'='*50}")
        print(f"Training model {i+1}/{config.ensemble_size} (seed={seed})")
        
        # Quick Win 1.5: Apply dropout variation for this ensemble member
        if use_diverse and i < len(dropout_variations):
            dropout_delta = dropout_variations[i]
            # Create a modified config with adjusted dropout
            class DiverseConfig:
                pass
            diverse_config = DiverseConfig()
            for k, v in config.to_dict().items():
                setattr(diverse_config, k, v)
            
            # Adjust dropout values
            diverse_config.classifier_dropout = max(0.1, min(0.6, config.classifier_dropout + dropout_delta))
            diverse_config.rnn_dropout = max(0.1, min(0.5, config.rnn_dropout + dropout_delta))
            diverse_config.to_dict = config.to_dict  # Keep method reference
            
            print(f"Dropout variation: delta={dropout_delta:+.2f} → classifier={diverse_config.classifier_dropout:.2f}, rnn={diverse_config.rnn_dropout:.2f}")
            model_config = diverse_config
        else:
            model_config = config
        
        print(f"{'='*50}\n")
        
        set_global_seed(seed)
        model = build_model(model_config)
        
        history, best_threshold, swa_model = train(model, train_loader, val_loader, model_config)
        
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
    
    threshold_objective = getattr(config, 'threshold_objective', 'f1')
    min_precision_constraint = getattr(config, 'min_precision_constraint', 0.60)
    min_recall_constraint = getattr(config, 'min_recall_constraint', 0.80)
    best_threshold, best_score = find_optimal_threshold(
        all_val_labels, all_val_probs,
        min_t=config.threshold_min,
        max_t=config.threshold_max,
        step=config.threshold_step,
        objective=threshold_objective,
        min_precision=min_precision_constraint,
        min_recall=min_recall_constraint,
    )
    print(f"\n[ENSEMBLE] Optimal threshold: {best_threshold:.3f} (Val score={best_score:.4f}, objective={threshold_objective})")
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
