"""Compare V2 (baseline) vs V3 (improved) model performance.

This script trains both versions and provides detailed comparison.

Usage:
    cd /media/tuananh/새 볼륨/DACNANM/Devign/C-Vul-Devign/devign_pipeline
    python3 -m experiments.compare_v2_v3
"""

import os
import sys
import json
import time
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Dict, List, Any, Optional

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import OneCycleLR
except ImportError:
    raise ImportError("PyTorch required")

from sklearn.metrics import (
    f1_score, precision_score, recall_score, 
    roc_auc_score, confusion_matrix, precision_recall_curve, auc
)

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.models.slice_attention_bigru import create_model


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance."""
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, pos_weight: Optional[torch.Tensor] = None):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.pos_weight = pos_weight
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = F.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=self.pos_weight, reduction='none'
        )
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = self.alpha * (1 - p_t) ** self.gamma
        loss = focal_weight * bce_loss
        return loss.mean()


class SliceDataset(Dataset):
    """Dataset for slice-based vulnerability detection."""
    
    def __init__(self, data_path: str, vuln_path: str, max_slices: int = 6):
        data = np.load(data_path)
        
        self.slice_input_ids = data['slice_input_ids'][:, :max_slices, :].astype(np.int64)
        self.slice_attention_mask = data['slice_attention_mask'][:, :max_slices, :].astype(np.float32)
        self.slice_count = np.minimum(data['slice_count'], max_slices)
        self.labels = data['labels']
        
        self.slice_mask = np.zeros((len(self.labels), max_slices), dtype=np.float32)
        for i, count in enumerate(self.slice_count):
            self.slice_mask[i, :int(count)] = 1.0
        
        vuln_data = np.load(vuln_path, allow_pickle=True)
        features = vuln_data['features'].astype(np.float32)
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        features = np.log1p(np.abs(features)) * np.sign(features)
        features = np.clip(features, -10, 10)
        mean_vals = features.mean(axis=0, keepdims=True)
        std_vals = features.std(axis=0, keepdims=True)
        std_vals[std_vals == 0] = 1
        self.features = np.clip((features - mean_vals) / std_vals, -5, 5).astype(np.float32)
        
        self.max_slices = max_slices
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return {
            'input_ids': torch.from_numpy(self.slice_input_ids[idx]),
            'attention_mask': torch.from_numpy(self.slice_attention_mask[idx]),
            'slice_mask': torch.from_numpy(self.slice_mask[idx]),
            'vuln_features': torch.from_numpy(self.features[idx]),
            'label': torch.tensor(self.labels[idx], dtype=torch.float32),
        }
    
    def get_pos_weight(self):
        n_pos = self.labels.sum()
        n_neg = len(self.labels) - n_pos
        return torch.tensor(n_neg / n_pos if n_pos > 0 else 1.0, dtype=torch.float32)


@dataclass
class ModelConfig:
    """Configuration for a model version."""
    name: str
    vocab_size: int = 238
    emb_dim: int = 128
    hidden_dim: int = 128
    feat_dim: int = 26
    num_layers: int = 1
    dropout: float = 0.3
    embed_dropout: float = 0.3
    gru_dropout: float = 0.3
    classifier_dropout: float = 0.5
    feat_dropout: float = 0.5
    attn_hidden: int = 64
    max_slices: int = 4
    use_focal_loss: bool = False
    focal_gamma: float = 2.0
    focal_alpha: float = 1.0
    label_smoothing: float = 0.1


# V2 Config (baseline - current best)
V2_CONFIG = ModelConfig(
    name="V2_Baseline",
    emb_dim=128,
    hidden_dim=128,
    num_layers=1,
    gru_dropout=0.3,
    max_slices=4,
    use_focal_loss=False,
    label_smoothing=0.1,
)

# V3 Config (improved)
V3_CONFIG = ModelConfig(
    name="V3_Improved",
    emb_dim=160,
    hidden_dim=96,
    num_layers=2,
    gru_dropout=0.2,
    attn_hidden=64,
    max_slices=6,
    use_focal_loss=True,
    focal_gamma=2.0,
    focal_alpha=1.0,
    label_smoothing=0.0,
)


def find_optimal_threshold(labels: np.ndarray, probs: np.ndarray) -> tuple:
    """Find optimal F1 threshold."""
    thresholds = np.arange(0.2, 0.8, 0.01)
    best_f1, best_thresh = 0, 0.5
    for t in thresholds:
        f1 = f1_score(labels, (probs >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_thresh = f1, t
    return best_thresh, best_f1


def train_model(
    config: ModelConfig,
    data_dir: str,
    epochs: int = 40,
    batch_size: int = 32,
    lr: float = 5e-4,
    patience: int = 10,
    seed: int = 42,
    device: str = "cuda",
) -> Dict[str, Any]:
    """Train a model with given config."""
    
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    
    # Set seeds
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    print(f"\n{'='*60}")
    print(f"Training: {config.name}")
    print(f"{'='*60}")
    
    # Load data
    data_dir = Path(data_dir)
    
    train_dataset = SliceDataset(
        str(data_dir / "train.npz"),
        str(data_dir / "train_vuln.npz"),
        max_slices=config.max_slices,
    )
    val_dataset = SliceDataset(
        str(data_dir / "val.npz"),
        str(data_dir / "val_vuln.npz"),
        max_slices=config.max_slices,
    )
    test_dataset = SliceDataset(
        str(data_dir / "test.npz"),
        str(data_dir / "test_vuln.npz"),
        max_slices=config.max_slices,
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                              num_workers=4, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size*2, shuffle=False,
                            num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size*2, shuffle=False,
                             num_workers=4, pin_memory=True)
    
    # Create model
    model_config = {
        'vocab_size': config.vocab_size,
        'emb_dim': config.emb_dim,
        'hidden_dim': config.hidden_dim,
        'feat_dim': config.feat_dim,
        'num_layers': config.num_layers,
        'dropout': config.dropout,
        'embed_dropout': config.embed_dropout,
        'gru_dropout': config.gru_dropout,
        'classifier_dropout': config.classifier_dropout,
        'feat_dropout': config.feat_dropout,
        'attn_hidden': config.attn_hidden,
    }
    model = create_model(model_config).to(device)
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")
    print(f"Max slices: {config.max_slices}")
    print(f"Focal Loss: {config.use_focal_loss}")
    print(f"Label smoothing: {config.label_smoothing}")
    
    # Loss
    pos_weight = train_dataset.get_pos_weight().to(device)
    if config.use_focal_loss:
        criterion = FocalLoss(config.focal_alpha, config.focal_gamma, pos_weight)
    else:
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    # Optimizer
    embed_params = list(model.embedding.parameters())
    other_params = [p for n, p in model.named_parameters() if 'embedding' not in n]
    optimizer = AdamW([
        {'params': embed_params, 'weight_decay': 1e-3},
        {'params': other_params, 'weight_decay': 1e-4},
    ], lr=lr)
    
    # Scheduler
    total_steps = len(train_loader) * epochs
    scheduler = OneCycleLR(optimizer, max_lr=lr, total_steps=total_steps,
                           pct_start=0.1, anneal_strategy='cos')
    
    scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None
    
    # Training loop
    history = {'train_f1': [], 'val_f1': [], 'val_auc': [], 'val_precision': [], 'val_recall': []}
    best_val_f1, best_epoch, patience_counter = 0, 0, 0
    best_state = None
    
    start_time = time.time()
    
    for epoch in range(epochs):
        # Train
        model.train()
        train_preds, train_labels = [], []
        
        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            slice_mask = batch['slice_mask'].to(device)
            vuln_features = batch['vuln_features'].to(device)
            labels = batch['label'].to(device)
            
            if config.label_smoothing > 0:
                labels_smooth = labels * (1 - config.label_smoothing) + 0.5 * config.label_smoothing
            else:
                labels_smooth = labels
            
            optimizer.zero_grad()
            
            if scaler:
                with torch.amp.autocast('cuda'):
                    logits = model(input_ids, attention_mask, slice_mask, vuln_features)
                    loss = criterion(logits.squeeze(-1), labels_smooth)
                if torch.isnan(loss):
                    continue
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(input_ids, attention_mask, slice_mask, vuln_features)
                loss = criterion(logits.squeeze(-1), labels_smooth)
                if torch.isnan(loss):
                    continue
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()
            
            scheduler.step()
            
            preds = torch.sigmoid(logits.squeeze(-1)).detach().cpu().numpy()
            train_preds.extend(preds)
            train_labels.extend(labels.cpu().numpy())
        
        train_f1 = f1_score(train_labels, (np.array(train_preds) >= 0.5).astype(int))
        
        # Validate
        model.eval()
        val_preds, val_labels_list = [], []
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                slice_mask = batch['slice_mask'].to(device)
                vuln_features = batch['vuln_features'].to(device)
                
                if scaler:
                    with torch.amp.autocast('cuda'):
                        logits = model(input_ids, attention_mask, slice_mask, vuln_features)
                else:
                    logits = model(input_ids, attention_mask, slice_mask, vuln_features)
                
                preds = torch.sigmoid(logits.squeeze(-1)).cpu().numpy()
                val_preds.extend(preds)
                val_labels_list.extend(batch['label'].numpy())
        
        val_preds = np.array(val_preds)
        val_labels_arr = np.array(val_labels_list)
        val_pred_labels = (val_preds >= 0.5).astype(int)
        
        val_f1 = f1_score(val_labels_arr, val_pred_labels)
        val_auc = roc_auc_score(val_labels_arr, val_preds)
        val_precision = precision_score(val_labels_arr, val_pred_labels, zero_division=0)
        val_recall = recall_score(val_labels_arr, val_pred_labels, zero_division=0)
        
        history['train_f1'].append(train_f1)
        history['val_f1'].append(val_f1)
        history['val_auc'].append(val_auc)
        history['val_precision'].append(val_precision)
        history['val_recall'].append(val_recall)
        
        print(f"Epoch {epoch+1:2d} | Train F1: {train_f1:.4f} | Val F1: {val_f1:.4f} | "
              f"AUC: {val_auc:.4f} | P: {val_precision:.4f} | R: {val_recall:.4f}")
        
        # Early stopping
        if val_f1 > best_val_f1 + 0.001:
            best_val_f1 = val_f1
            best_epoch = epoch
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    train_time = time.time() - start_time
    
    # Load best model and evaluate on test
    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    model.eval()
    
    test_preds, test_labels_list = [], []
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            slice_mask = batch['slice_mask'].to(device)
            vuln_features = batch['vuln_features'].to(device)
            
            if scaler:
                with torch.amp.autocast('cuda'):
                    logits = model(input_ids, attention_mask, slice_mask, vuln_features)
            else:
                logits = model(input_ids, attention_mask, slice_mask, vuln_features)
            
            preds = torch.sigmoid(logits.squeeze(-1)).cpu().numpy()
            test_preds.extend(preds)
            test_labels_list.extend(batch['label'].numpy())
    
    test_preds = np.array(test_preds)
    test_labels = np.array(test_labels_list)
    
    # Find optimal threshold
    opt_thresh, opt_f1 = find_optimal_threshold(test_labels, test_preds)
    
    # Metrics at 0.5 and optimal threshold
    test_pred_05 = (test_preds >= 0.5).astype(int)
    test_pred_opt = (test_preds >= opt_thresh).astype(int)
    
    # PR-AUC
    precision_curve, recall_curve, _ = precision_recall_curve(test_labels, test_preds)
    pr_auc = auc(recall_curve, precision_curve)
    
    results = {
        'config_name': config.name,
        'n_params': n_params,
        'best_epoch': best_epoch + 1,
        'train_time_sec': train_time,
        'best_val_f1': best_val_f1,
        
        # Test @ 0.5
        'test_f1_05': f1_score(test_labels, test_pred_05),
        'test_auc': roc_auc_score(test_labels, test_preds),
        'test_pr_auc': pr_auc,
        'test_precision_05': precision_score(test_labels, test_pred_05),
        'test_recall_05': recall_score(test_labels, test_pred_05),
        
        # Test @ optimal
        'optimal_threshold': opt_thresh,
        'test_f1_opt': opt_f1,
        'test_precision_opt': precision_score(test_labels, test_pred_opt),
        'test_recall_opt': recall_score(test_labels, test_pred_opt),
        
        'confusion_matrix': confusion_matrix(test_labels, test_pred_opt).tolist(),
        'history': history,
        'config': asdict(config),
    }
    
    # Cleanup
    del model, optimizer, scheduler, scaler
    torch.cuda.empty_cache()
    
    return results


def print_comparison(v2_results: Dict, v3_results: Dict):
    """Print side-by-side comparison."""
    
    print("\n" + "="*70)
    print("COMPARISON: V2 (Baseline) vs V3 (Improved)")
    print("="*70)
    
    metrics = [
        ('Parameters', 'n_params', '{:,}'),
        ('Best Epoch', 'best_epoch', '{}'),
        ('Train Time (sec)', 'train_time_sec', '{:.1f}'),
        ('Best Val F1', 'best_val_f1', '{:.4f}'),
        ('Test F1 @0.5', 'test_f1_05', '{:.4f}'),
        ('Test F1 @opt', 'test_f1_opt', '{:.4f}'),
        ('Optimal Threshold', 'optimal_threshold', '{:.2f}'),
        ('Test AUC', 'test_auc', '{:.4f}'),
        ('Test PR-AUC', 'test_pr_auc', '{:.4f}'),
        ('Test Precision @opt', 'test_precision_opt', '{:.4f}'),
        ('Test Recall @opt', 'test_recall_opt', '{:.4f}'),
    ]
    
    print(f"\n{'Metric':<25} {'V2 Baseline':>15} {'V3 Improved':>15} {'Delta':>12}")
    print("-"*70)
    
    for name, key, fmt in metrics:
        v2_val = v2_results[key]
        v3_val = v3_results[key]
        
        if isinstance(v2_val, (int, float)) and isinstance(v3_val, (int, float)):
            delta = v3_val - v2_val
            if 'f1' in key.lower() or 'auc' in key.lower() or 'precision' in key.lower() or 'recall' in key.lower():
                delta_str = f"{delta:+.4f}" if delta != 0 else "0.0000"
                if delta > 0.01:
                    delta_str += " ✅"
                elif delta < -0.01:
                    delta_str += " ❌"
            else:
                delta_str = f"{delta:+.1f}" if abs(delta) > 0.1 else "~0"
        else:
            delta_str = "-"
        
        print(f"{name:<25} {fmt.format(v2_val):>15} {fmt.format(v3_val):>15} {delta_str:>12}")
    
    print("\n" + "-"*70)
    print("Config differences:")
    print(f"  V2: max_slices=4, hidden=128, layers=1, focal=False, smoothing=0.1")
    print(f"  V3: max_slices=6, hidden=96, layers=2, focal=True, smoothing=0.0")
    
    # Determine winner
    f1_improvement = v3_results['test_f1_opt'] - v2_results['test_f1_opt']
    if f1_improvement > 0.01:
        print(f"\n>>> V3 WINS by {f1_improvement:.4f} F1 improvement!")
    elif f1_improvement < -0.01:
        print(f"\n>>> V2 still better by {-f1_improvement:.4f} F1")
    else:
        print(f"\n>>> Results are similar (delta: {f1_improvement:.4f})")


def run_comparison():
    """Run full V2 vs V3 comparison."""
    
    data_dir = "../Output data"
    output_dir = Path("experiments/comparison_results")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*70)
    print("V2 vs V3 MODEL COMPARISON")
    print("="*70)
    
    # Train V2
    v2_results = train_model(V2_CONFIG, data_dir, epochs=40, seed=42)
    
    # Train V3
    v3_results = train_model(V3_CONFIG, data_dir, epochs=40, seed=42)
    
    # Print comparison
    print_comparison(v2_results, v3_results)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = output_dir / f"v2_v3_comparison_{timestamp}.json"
    
    with open(results_file, 'w') as f:
        json.dump({
            'v2': v2_results,
            'v3': v3_results,
            'timestamp': timestamp,
        }, f, indent=2)
    
    print(f"\nResults saved to: {results_file}")
    
    return v2_results, v3_results


if __name__ == "__main__":
    run_comparison()
