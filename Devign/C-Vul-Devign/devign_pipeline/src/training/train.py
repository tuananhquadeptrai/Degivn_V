"""Training script for SliceAttBiGRU vulnerability detection model.

Usage:
    python3 -m src.training.train --data_dir "Output data" --epochs 50
    
Features:
- Hierarchical BiGRU with slice attention
- Vulnerability feature fusion
- Class-weighted loss for imbalanced data
- Early stopping on validation F1
- Mixed precision training (optional)
- Checkpointing and resume support
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    raise ImportError("PyTorch is required. Install with: pip install torch")

from sklearn.metrics import (
    f1_score, precision_score, recall_score, 
    roc_auc_score, confusion_matrix, classification_report
)

from src.training.visualize import (
    plot_training_history, plot_confusion_matrix,
    plot_roc_curve, plot_precision_recall_curve
)


def find_optimal_threshold(
    labels: np.ndarray, 
    probs: np.ndarray, 
    metric: str = 'f1'
) -> Tuple[float, float]:
    """
    Find optimal decision threshold for classification.
    
    Args:
        labels: True labels
        probs: Predicted probabilities
        metric: 'f1', 'precision', 'recall', or 'balanced' (P=R)
    
    Returns:
        Tuple of (optimal_threshold, best_metric_value)
    """
    thresholds = np.arange(0.1, 0.9, 0.01)
    best_threshold = 0.5
    best_value = 0.0
    
    results = []
    
    for thresh in thresholds:
        pred_labels = (probs >= thresh).astype(int)
        
        if metric == 'f1':
            value = f1_score(labels, pred_labels, zero_division=0)
        elif metric == 'precision':
            value = precision_score(labels, pred_labels, zero_division=0)
        elif metric == 'recall':
            value = recall_score(labels, pred_labels, zero_division=0)
        elif metric == 'balanced':
            p = precision_score(labels, pred_labels, zero_division=0)
            r = recall_score(labels, pred_labels, zero_division=0)
            value = -abs(p - r)  # Minimize difference (maximize negative)
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
        results.append((thresh, value))
        
        if value > best_value:
            best_value = value
            best_threshold = thresh
    
    return best_threshold, best_value

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.models.slice_attention_bigru import SliceAttBiGRU, create_model


@dataclass
class TrainConfig:
    """Training configuration."""
    # Data
    data_dir: str = "Output data"
    output_dir: str = "checkpoints"
    
    # Model
    vocab_size: int = 238
    emb_dim: int = 128
    hidden_dim: int = 128
    feat_dim: int = 26  # Number of vuln features
    num_layers: int = 1
    dropout: float = 0.3
    embed_dropout: float = 0.3
    gru_dropout: float = 0.3
    classifier_dropout: float = 0.5
    feat_dropout: float = 0.5
    
    # Training
    epochs: int = 50
    batch_size: int = 32
    learning_rate: float = 5e-4
    weight_decay: float = 1e-4
    embed_weight_decay: float = 1e-3  # Extra weight decay for embedding
    warmup_epochs: int = 5
    label_smoothing: float = 0.1
    
    # Slices
    max_slices: int = 4
    slice_max_len: int = 256
    
    # Early stopping
    patience: int = 7
    min_delta: float = 0.001
    
    # Hardware
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers: int = 4
    mixed_precision: bool = True
    
    # Misc
    seed: int = 42
    log_interval: int = 50
    save_best_only: bool = True
    
    def save(self, path: str):
        with open(path, 'w') as f:
            json.dump(asdict(self), f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> 'TrainConfig':
        with open(path) as f:
            data = json.load(f)
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


class DevignSliceDataset(Dataset):
    """Dataset for slice-based vulnerability detection."""
    
    def __init__(
        self, 
        data_path: str,
        vuln_path: str,
        max_slices: int = 4,
        transform_features: bool = True
    ):
        """
        Args:
            data_path: Path to train.npz / val.npz / test.npz
            vuln_path: Path to train_vuln.npz / val_vuln.npz / test_vuln.npz
            max_slices: Number of slices to use (default 4)
            transform_features: Apply log transform to features
        """
        # Load main data
        data = np.load(data_path)
        self.slice_input_ids = data['slice_input_ids'][:, :max_slices, :]  # [N, 4, 256]
        self.slice_attention_mask = data['slice_attention_mask'][:, :max_slices, :]
        self.slice_count = np.minimum(data['slice_count'], max_slices)  # Cap at max_slices
        self.labels = data['labels']
        
        # Create slice_mask from slice_count
        self.slice_mask = np.zeros((len(self.labels), max_slices), dtype=np.float32)
        for i, count in enumerate(self.slice_count):
            self.slice_mask[i, :int(count)] = 1.0
        
        # Load vuln features
        vuln_data = np.load(vuln_path, allow_pickle=True)
        self.features = vuln_data['features'].astype(np.float32)  # [N, 26]
        
        if transform_features:
            self.features = self._transform_features(self.features)
        
        self.max_slices = max_slices
        self.n_samples = len(self.labels)
        
    def _transform_features(self, features: np.ndarray) -> np.ndarray:
        """Apply log1p transform and normalize features."""
        # Replace any NaN/Inf with 0
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Log transform for count/ratio features (handles large values like 2994)
        transformed = np.log1p(np.abs(features)) * np.sign(features)
        
        # Clip extreme values
        transformed = np.clip(transformed, -10, 10)
        
        # Standardize (z-score) instead of min-max for better gradient behavior
        mean_vals = transformed.mean(axis=0, keepdims=True)
        std_vals = transformed.std(axis=0, keepdims=True)
        std_vals[std_vals == 0] = 1  # Avoid division by zero
        
        normalized = (transformed - mean_vals) / std_vals
        
        # Clip final values to prevent extreme outliers
        normalized = np.clip(normalized, -5, 5)
        
        return normalized.astype(np.float32)
    
    def __len__(self) -> int:
        return self.n_samples
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            'input_ids': torch.from_numpy(self.slice_input_ids[idx].astype(np.int64)),
            'attention_mask': torch.from_numpy(self.slice_attention_mask[idx].astype(np.float32)),
            'slice_mask': torch.from_numpy(self.slice_mask[idx]),
            'vuln_features': torch.from_numpy(self.features[idx]),
            'label': torch.tensor(self.labels[idx], dtype=torch.float32),
        }
    
    def get_class_weights(self) -> torch.Tensor:
        """Compute class weights for imbalanced data."""
        n_samples = len(self.labels)
        n_pos = self.labels.sum()
        n_neg = n_samples - n_pos
        
        # Inverse frequency weighting
        weight_pos = n_samples / (2 * n_pos) if n_pos > 0 else 1.0
        weight_neg = n_samples / (2 * n_neg) if n_neg > 0 else 1.0
        
        return torch.tensor([weight_neg, weight_pos], dtype=torch.float32)
    
    def get_pos_weight(self) -> torch.Tensor:
        """Get positive class weight for BCEWithLogitsLoss."""
        n_pos = self.labels.sum()
        n_neg = len(self.labels) - n_pos
        return torch.tensor(n_neg / n_pos if n_pos > 0 else 1.0, dtype=torch.float32)


class EarlyStopping:
    """Early stopping handler."""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.001, mode: str = 'max'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
            return True  # First epoch, save model
        
        if self.mode == 'max':
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta
        
        if improved:
            self.best_score = score
            self.counter = 0
            return True  # Improved, save model
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
            return False  # Not improved


class Trainer:
    """Main training class."""
    
    def __init__(self, config: TrainConfig):
        self.config = config
        self.device = torch.device(config.device)
        
        # Set seed
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.seed)
        
        # Create output directory
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.scaler = None
        self.train_loader = None
        self.val_loader = None
        self.early_stopping = None
        
        # Metrics history
        self.history = {
            'train_loss': [], 'val_loss': [],
            'train_f1': [], 'val_f1': [],
            'train_auc': [], 'val_auc': [],
            'val_precision': [], 'val_recall': [],
        }
        
    def setup(self):
        """Setup model, data, optimizer."""
        print(f"Setting up training on {self.device}...")
        
        # Load datasets
        data_dir = Path(self.config.data_dir)
        
        self.train_dataset = DevignSliceDataset(
            data_path=str(data_dir / "train.npz"),
            vuln_path=str(data_dir / "train_vuln.npz"),
            max_slices=self.config.max_slices,
        )
        
        self.val_dataset = DevignSliceDataset(
            data_path=str(data_dir / "val.npz"),
            vuln_path=str(data_dir / "val_vuln.npz"),
            max_slices=self.config.max_slices,
        )
        
        print(f"Train samples: {len(self.train_dataset)}")
        print(f"Val samples: {len(self.val_dataset)}")
        
        # DataLoaders
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=True,
            drop_last=True,
        )
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.config.batch_size * 2,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=True,
        )
        
        # Model
        model_config = {
            'vocab_size': self.config.vocab_size,
            'emb_dim': self.config.emb_dim,
            'hidden_dim': self.config.hidden_dim,
            'feat_dim': self.config.feat_dim,
            'num_layers': self.config.num_layers,
            'dropout': self.config.dropout,
            'embed_dropout': self.config.embed_dropout,
            'gru_dropout': self.config.gru_dropout,
            'classifier_dropout': self.config.classifier_dropout,
            'feat_dropout': self.config.feat_dropout,
        }
        self.model = create_model(model_config).to(self.device)
        
        n_params = sum(p.numel() for p in self.model.parameters())
        print(f"Model parameters: {n_params:,}")
        
        # Loss with class weighting
        pos_weight = self.train_dataset.get_pos_weight().to(self.device)
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        print(f"Positive class weight: {pos_weight.item():.3f}")
        
        # Optimizer with separate weight decay for embedding
        embed_params = list(self.model.embedding.parameters())
        other_params = [p for n, p in self.model.named_parameters() if 'embedding' not in n]
        
        self.optimizer = AdamW([
            {'params': embed_params, 'weight_decay': self.config.embed_weight_decay},
            {'params': other_params, 'weight_decay': self.config.weight_decay},
        ], lr=self.config.learning_rate)
        
        # Scheduler
        steps_per_epoch = len(self.train_loader)
        total_steps = steps_per_epoch * self.config.epochs
        warmup_steps = steps_per_epoch * self.config.warmup_epochs
        
        self.scheduler = OneCycleLR(
            self.optimizer,
            max_lr=self.config.learning_rate,
            total_steps=total_steps,
            pct_start=warmup_steps / total_steps,
            anneal_strategy='cos',
        )
        
        # Mixed precision
        if self.config.mixed_precision and self.device.type == 'cuda':
            self.scaler = torch.amp.GradScaler('cuda')
        
        # Early stopping
        self.early_stopping = EarlyStopping(
            patience=self.config.patience,
            min_delta=self.config.min_delta,
            mode='max'
        )
        
        # Save config
        self.config.save(str(self.output_dir / "config.json"))
        
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        
        total_loss = 0.0
        all_preds = []
        all_labels = []
        
        for batch_idx, batch in enumerate(self.train_loader):
            # Move to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            slice_mask = batch['slice_mask'].to(self.device)
            vuln_features = batch['vuln_features'].to(self.device)
            labels = batch['label'].to(self.device)
            
            self.optimizer.zero_grad()
            
            # Apply label smoothing (only if > 0)
            if self.config.label_smoothing > 0:
                labels_smooth = labels * (1 - self.config.label_smoothing) + 0.5 * self.config.label_smoothing
            else:
                labels_smooth = labels
            
            # Forward pass
            if self.scaler is not None:
                with torch.amp.autocast('cuda'):
                    logits = self.model(input_ids, attention_mask, slice_mask, vuln_features)
                    loss = self.criterion(logits.squeeze(-1), labels_smooth)
                
                # Check for NaN loss
                if torch.isnan(loss):
                    print(f"  WARNING: NaN loss at batch {batch_idx}, skipping...")
                    continue
                
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)  # Tighter clipping
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                logits = self.model(input_ids, attention_mask, slice_mask, vuln_features)
                loss = self.criterion(logits.squeeze(-1), labels_smooth)
                
                # Check for NaN loss
                if torch.isnan(loss):
                    print(f"  WARNING: NaN loss at batch {batch_idx}, skipping...")
                    continue
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)  # Tighter clipping
                self.optimizer.step()
            
            self.scheduler.step()
            
            # Accumulate metrics
            total_loss += loss.item()
            preds = torch.sigmoid(logits.squeeze(-1)).detach().cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
            
            # Log progress
            if (batch_idx + 1) % self.config.log_interval == 0:
                lr = self.scheduler.get_last_lr()[0]
                print(f"  Batch {batch_idx+1}/{len(self.train_loader)} | "
                      f"Loss: {loss.item():.4f} | LR: {lr:.2e}")
        
        # Compute metrics
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        pred_labels = (all_preds >= 0.5).astype(int)
        
        metrics = {
            'loss': total_loss / len(self.train_loader),
            'f1': f1_score(all_labels, pred_labels),
            'auc': roc_auc_score(all_labels, all_preds) if len(np.unique(all_labels)) > 1 else 0.0,
        }
        
        return metrics
    
    @torch.no_grad()
    def validate(self, return_probs: bool = False) -> Dict[str, Any]:
        """Validate the model.
        
        Args:
            return_probs: If True, also return raw probabilities and labels
        """
        self.model.eval()
        
        total_loss = 0.0
        all_preds = []
        all_labels = []
        
        for batch in self.val_loader:
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            slice_mask = batch['slice_mask'].to(self.device)
            vuln_features = batch['vuln_features'].to(self.device)
            labels = batch['label'].to(self.device)
            
            if self.scaler is not None:
                with torch.amp.autocast('cuda'):
                    logits = self.model(input_ids, attention_mask, slice_mask, vuln_features)
                    loss = self.criterion(logits.squeeze(-1), labels)
            else:
                logits = self.model(input_ids, attention_mask, slice_mask, vuln_features)
                loss = self.criterion(logits.squeeze(-1), labels)
            
            total_loss += loss.item()
            preds = torch.sigmoid(logits.squeeze(-1)).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
        
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        pred_labels = (all_preds >= 0.5).astype(int)
        
        metrics = {
            'loss': total_loss / len(self.val_loader),
            'f1': f1_score(all_labels, pred_labels),
            'auc': roc_auc_score(all_labels, all_preds) if len(np.unique(all_labels)) > 1 else 0.0,
            'precision': precision_score(all_labels, pred_labels),
            'recall': recall_score(all_labels, pred_labels),
        }
        
        if return_probs:
            metrics['probs'] = all_preds
            metrics['labels'] = all_labels
        
        return metrics
    
    def save_checkpoint(self, epoch: int, metrics: Dict[str, float], 
                        is_best: bool = False, optimal_threshold: float = 0.5):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': metrics,
            'config': asdict(self.config),
            'optimal_threshold': optimal_threshold,
        }
        
        if self.scaler is not None:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        # Save latest
        torch.save(checkpoint, self.output_dir / "checkpoint_latest.pt")
        
        # Save best
        if is_best:
            torch.save(checkpoint, self.output_dir / "checkpoint_best.pt")
            print(f"  Saved best model (F1: {metrics['f1']:.4f})")
    
    def load_checkpoint(self, path: str) -> int:
        """Load checkpoint and return start epoch."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        if self.scaler is not None and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
        return checkpoint['epoch'] + 1
    
    def train(self, resume_path: Optional[str] = None, tune_threshold: bool = False):
        """Main training loop."""
        self.setup()
        
        start_epoch = 0
        if resume_path and Path(resume_path).exists():
            start_epoch = self.load_checkpoint(resume_path)
        
        print(f"\nStarting training from epoch {start_epoch}...")
        print("=" * 60)
        
        optimal_threshold = 0.5
        
        for epoch in range(start_epoch, self.config.epochs):
            print(f"\nEpoch {epoch+1}/{self.config.epochs}")
            print("-" * 40)
            
            # Train
            train_metrics = self.train_epoch(epoch)
            print(f"Train | Loss: {train_metrics['loss']:.4f} | "
                  f"F1: {train_metrics['f1']:.4f} | AUC: {train_metrics['auc']:.4f}")
            
            # Validate
            val_metrics = self.validate()
            print(f"Val   | Loss: {val_metrics['loss']:.4f} | "
                  f"F1: {val_metrics['f1']:.4f} | AUC: {val_metrics['auc']:.4f} | "
                  f"P: {val_metrics['precision']:.4f} | R: {val_metrics['recall']:.4f}")
            
            # Update history
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['train_f1'].append(train_metrics['f1'])
            self.history['train_auc'].append(train_metrics['auc'])
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['val_f1'].append(val_metrics['f1'])
            self.history['val_auc'].append(val_metrics['auc'])
            self.history['val_precision'].append(val_metrics['precision'])
            self.history['val_recall'].append(val_metrics['recall'])
            
            # Early stopping check
            is_best = self.early_stopping(val_metrics['f1'])
            
            if is_best or not self.config.save_best_only:
                self.save_checkpoint(epoch, val_metrics, is_best, optimal_threshold)
            
            if self.early_stopping.early_stop:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break
        
        # Threshold tuning after training
        if tune_threshold:
            print("\n" + "=" * 60)
            print("Tuning decision threshold on validation set...")
            val_metrics = self.validate(return_probs=True)
            
            optimal_threshold, best_f1 = find_optimal_threshold(
                val_metrics['labels'], 
                val_metrics['probs'], 
                metric='f1'
            )
            
            # Print threshold sweep results
            print(f"\nThreshold sweep results:")
            print(f"  Optimal threshold: {optimal_threshold:.2f}")
            print(f"  Best F1 at threshold: {best_f1:.4f}")
            print(f"  Default F1 (0.5): {val_metrics['f1']:.4f}")
            print(f"  Improvement: {(best_f1 - val_metrics['f1'])*100:.2f}%")
            
            # Re-save best checkpoint with optimal threshold
            best_ckpt_path = self.output_dir / "checkpoint_best.pt"
            if best_ckpt_path.exists():
                checkpoint = torch.load(best_ckpt_path, map_location=self.device, weights_only=False)
                checkpoint['optimal_threshold'] = optimal_threshold
                torch.save(checkpoint, best_ckpt_path)
                print(f"  Updated best checkpoint with optimal threshold: {optimal_threshold:.2f}")
            
            self.history['optimal_threshold'] = optimal_threshold
            self.history['threshold_tuned_f1'] = best_f1
        
        # Save training history
        with open(self.output_dir / "history.json", 'w') as f:
            json.dump(self.history, f, indent=2)
        
        # Generate training plots
        print("\n" + "=" * 60)
        print("Generating evaluation plots...")
        plots_dir = self.output_dir / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            plot_training_history(self.history, str(plots_dir))
            print(f"Training plots saved to: {plots_dir}")
        except Exception as e:
            print(f"Warning: Could not generate plots: {e}")
        
        print("\n" + "=" * 60)
        print("Training completed!")
        print(f"Best validation F1: {self.early_stopping.best_score:.4f}")
        if tune_threshold:
            print(f"Optimal threshold: {optimal_threshold:.2f}")
        
        return self.history


def evaluate_model(
    model_path: str,
    data_dir: str,
    split: str = 'test',
    device: str = 'cuda',
    threshold: Optional[float] = None
) -> Dict[str, Any]:
    """Evaluate a trained model on test set.
    
    Args:
        model_path: Path to checkpoint
        data_dir: Directory containing data files
        split: Data split to evaluate ('val' or 'test')
        device: Device to use
        threshold: Decision threshold (None = use optimal from checkpoint, or 0.5)
    """
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    config = TrainConfig(**checkpoint['config'])
    
    # Get threshold from checkpoint if not specified
    if threshold is None:
        threshold = checkpoint.get('optimal_threshold', 0.5)
    
    # Create model
    model_config = {
        'vocab_size': config.vocab_size,
        'emb_dim': config.emb_dim,
        'hidden_dim': config.hidden_dim,
        'feat_dim': config.feat_dim,
        'num_layers': config.num_layers,
        'dropout': config.dropout,
        'embed_dropout': getattr(config, 'embed_dropout', 0.3),
        'gru_dropout': getattr(config, 'gru_dropout', 0.3),
        'classifier_dropout': getattr(config, 'classifier_dropout', 0.5),
        'feat_dropout': getattr(config, 'feat_dropout', 0.5),
    }
    model = create_model(model_config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # Load test data
    data_dir = Path(data_dir)
    dataset = DevignSliceDataset(
        data_path=str(data_dir / f"{split}.npz"),
        vuln_path=str(data_dir / f"{split}_vuln.npz"),
        max_slices=config.max_slices,
    )
    
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size * 2,
        shuffle=False,
        num_workers=config.num_workers,
    )
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            slice_mask = batch['slice_mask'].to(device)
            vuln_features = batch['vuln_features'].to(device)
            labels = batch['label']
            
            logits = model(input_ids, attention_mask, slice_mask, vuln_features)
            preds = torch.sigmoid(logits.squeeze(-1)).cpu().numpy()
            
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    pred_labels = (all_preds >= threshold).astype(int)
    
    cm = confusion_matrix(all_labels, pred_labels)
    results = {
        'f1': f1_score(all_labels, pred_labels),
        'auc': roc_auc_score(all_labels, all_preds),
        'precision': precision_score(all_labels, pred_labels),
        'recall': recall_score(all_labels, pred_labels),
        'confusion_matrix': cm.tolist(),
        'classification_report': classification_report(all_labels, pred_labels),
        'threshold': threshold,
    }
    
    print(f"\n{split.upper()} Results (threshold={threshold:.2f}):")
    print(f"  F1: {results['f1']:.4f}")
    print(f"  AUC: {results['auc']:.4f}")
    print(f"  Precision: {results['precision']:.4f}")
    print(f"  Recall: {results['recall']:.4f}")
    print(f"\nConfusion Matrix:")
    print(f"  {results['confusion_matrix']}")
    print(f"\n{results['classification_report']}")
    
    # Generate evaluation plots
    checkpoint_dir = Path(model_path).parent
    plots_dir = checkpoint_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        plot_confusion_matrix(cm, str(plots_dir))
        plot_roc_curve(all_labels, all_preds, str(plots_dir))
        plot_precision_recall_curve(all_labels, all_preds, str(plots_dir))
        print(f"\nEvaluation plots saved to: {plots_dir}")
    except Exception as e:
        print(f"Warning: Could not generate evaluation plots: {e}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Train SliceAttBiGRU model")
    parser.add_argument('--data_dir', type=str, default="Output data",
                        help="Directory containing train/val/test npz files")
    parser.add_argument('--output_dir', type=str, default="checkpoints",
                        help="Directory to save checkpoints")
    parser.add_argument('--epochs', type=int, default=50,
                        help="Number of training epochs")
    parser.add_argument('--batch_size', type=int, default=32,
                        help="Batch size")
    parser.add_argument('--lr', type=float, default=1e-3,
                        help="Learning rate")
    parser.add_argument('--max_slices', type=int, default=4,
                        help="Number of slices to use")
    parser.add_argument('--resume', type=str, default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument('--eval_only', action='store_true',
                        help="Only evaluate, don't train")
    parser.add_argument('--eval_split', type=str, default='test',
                        help="Split to evaluate (val/test)")
    parser.add_argument('--tune_threshold', action='store_true',
                        help="Tune decision threshold on validation set after training")
    
    args = parser.parse_args()
    
    if args.eval_only:
        # Evaluation mode
        model_path = args.resume or "checkpoints/checkpoint_best.pt"
        evaluate_model(model_path, args.data_dir, args.eval_split)
    else:
        # Training mode
        config = TrainConfig(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            max_slices=args.max_slices,
        )
        
        trainer = Trainer(config)
        trainer.train(resume_path=args.resume, tune_threshold=args.tune_threshold)
        
        # Evaluate on test set
        print("\n" + "=" * 60)
        print("Evaluating on test set...")
        evaluate_model(
            str(Path(args.output_dir) / "checkpoint_best.pt"),
            args.data_dir,
            'test'
        )


if __name__ == "__main__":
    main()
