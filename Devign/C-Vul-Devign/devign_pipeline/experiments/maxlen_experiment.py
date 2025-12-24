"""Experiment: Compare different max_len settings (128, 150, 256).

This script runs a controlled experiment to find the optimal slice max_len
by testing 3 settings with 3 seeds each.

Usage:
    cd /media/tuananh/새 볼륨/DACNANM/Devign/C-Vul-Devign/devign_pipeline
    python3 -m experiments.maxlen_experiment
"""

import os
import sys
import json
import time
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Any

import numpy as np

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import OneCycleLR
except ImportError:
    raise ImportError("PyTorch required")

from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.models.slice_attention_bigru import create_model


@dataclass
class ExperimentConfig:
    """Experiment configuration."""
    # Data
    data_dir: str = "Output data"
    output_dir: str = "experiments/maxlen_results"
    
    # Model (fixed)
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
    
    # Training (fixed)
    epochs: int = 30  # Reduced for faster experiments
    batch_size: int = 32
    learning_rate: float = 5e-4
    weight_decay: float = 1e-4
    embed_weight_decay: float = 1e-3
    warmup_epochs: int = 3
    label_smoothing: float = 0.1
    patience: int = 7
    
    # Slices
    max_slices: int = 4
    
    # Hardware
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers: int = 4


class TruncatedSliceDataset(Dataset):
    """Dataset with configurable max_len truncation."""
    
    def __init__(
        self,
        data_path: str,
        vuln_path: str,
        max_slices: int = 4,
        max_len: int = 256,  # Can be 128, 150, 256
    ):
        data = np.load(data_path)
        
        # Load and truncate to max_len
        slice_input_ids = data['slice_input_ids'][:, :max_slices, :max_len]
        slice_attention_mask = data['slice_attention_mask'][:, :max_slices, :max_len]
        
        self.slice_input_ids = slice_input_ids.astype(np.int64)
        self.slice_attention_mask = slice_attention_mask.astype(np.float32)
        self.slice_count = np.minimum(data['slice_count'], max_slices)
        self.labels = data['labels']
        
        # Create slice_mask
        self.slice_mask = np.zeros((len(self.labels), max_slices), dtype=np.float32)
        for i, count in enumerate(self.slice_count):
            self.slice_mask[i, :int(count)] = 1.0
        
        # Load and transform vuln features
        vuln_data = np.load(vuln_path, allow_pickle=True)
        features = vuln_data['features'].astype(np.float32)
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        features = np.log1p(np.abs(features)) * np.sign(features)
        features = np.clip(features, -10, 10)
        mean_vals = features.mean(axis=0, keepdims=True)
        std_vals = features.std(axis=0, keepdims=True)
        std_vals[std_vals == 0] = 1
        self.features = np.clip((features - mean_vals) / std_vals, -5, 5).astype(np.float32)
        
        self.max_len = max_len
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


def train_single_run(
    config: ExperimentConfig,
    max_len: int,
    seed: int,
) -> Dict[str, Any]:
    """Train a single model with given max_len and seed."""
    
    device = torch.device(config.device)
    
    # Set seeds
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    # Load data
    data_dir = Path(config.data_dir)
    
    train_dataset = TruncatedSliceDataset(
        str(data_dir / "train.npz"),
        str(data_dir / "train_vuln.npz"),
        max_slices=config.max_slices,
        max_len=max_len,
    )
    
    val_dataset = TruncatedSliceDataset(
        str(data_dir / "val.npz"),
        str(data_dir / "val_vuln.npz"),
        max_slices=config.max_slices,
        max_len=max_len,
    )
    
    test_dataset = TruncatedSliceDataset(
        str(data_dir / "test.npz"),
        str(data_dir / "test_vuln.npz"),
        max_slices=config.max_slices,
        max_len=max_len,
    )
    
    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size,
        shuffle=True, num_workers=config.num_workers,
        pin_memory=True, drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=config.batch_size * 2,
        shuffle=False, num_workers=config.num_workers, pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, batch_size=config.batch_size * 2,
        shuffle=False, num_workers=config.num_workers, pin_memory=True
    )
    
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
    }
    model = create_model(model_config).to(device)
    
    # Loss
    pos_weight = train_dataset.get_pos_weight().to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    # Optimizer
    embed_params = list(model.embedding.parameters())
    other_params = [p for n, p in model.named_parameters() if 'embedding' not in n]
    
    optimizer = AdamW([
        {'params': embed_params, 'weight_decay': config.embed_weight_decay},
        {'params': other_params, 'weight_decay': config.weight_decay},
    ], lr=config.learning_rate)
    
    # Scheduler
    total_steps = len(train_loader) * config.epochs
    warmup_steps = len(train_loader) * config.warmup_epochs
    
    scheduler = OneCycleLR(
        optimizer,
        max_lr=config.learning_rate,
        total_steps=total_steps,
        pct_start=warmup_steps / total_steps,
        anneal_strategy='cos',
    )
    
    scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None
    
    # Training loop
    best_val_f1 = 0.0
    best_epoch = 0
    patience_counter = 0
    
    history = {'train_f1': [], 'val_f1': [], 'val_auc': []}
    
    start_time = time.time()
    
    for epoch in range(config.epochs):
        # Train
        model.train()
        train_preds, train_labels = [], []
        
        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            slice_mask = batch['slice_mask'].to(device)
            vuln_features = batch['vuln_features'].to(device)
            labels = batch['label'].to(device)
            
            # Label smoothing
            labels_smooth = labels * (1 - config.label_smoothing) + 0.5 * config.label_smoothing
            
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
                labels = batch['label']
                
                if scaler:
                    with torch.amp.autocast('cuda'):
                        logits = model(input_ids, attention_mask, slice_mask, vuln_features)
                else:
                    logits = model(input_ids, attention_mask, slice_mask, vuln_features)
                
                preds = torch.sigmoid(logits.squeeze(-1)).cpu().numpy()
                val_preds.extend(preds)
                val_labels_list.extend(labels.numpy())
        
        val_preds = np.array(val_preds)
        val_labels_arr = np.array(val_labels_list)
        val_pred_labels = (val_preds >= 0.5).astype(int)
        
        val_f1 = f1_score(val_labels_arr, val_pred_labels)
        val_auc = roc_auc_score(val_labels_arr, val_preds)
        
        history['train_f1'].append(train_f1)
        history['val_f1'].append(val_f1)
        history['val_auc'].append(val_auc)
        
        # Early stopping
        if val_f1 > best_val_f1 + 0.001:
            best_val_f1 = val_f1
            best_epoch = epoch
            patience_counter = 0
            # Save best model state
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= config.patience:
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
            labels = batch['label']
            
            if scaler:
                with torch.amp.autocast('cuda'):
                    logits = model(input_ids, attention_mask, slice_mask, vuln_features)
            else:
                logits = model(input_ids, attention_mask, slice_mask, vuln_features)
            
            preds = torch.sigmoid(logits.squeeze(-1)).cpu().numpy()
            test_preds.extend(preds)
            test_labels_list.extend(labels.numpy())
    
    test_preds = np.array(test_preds)
    test_labels_arr = np.array(test_labels_list)
    test_pred_labels = (test_preds >= 0.5).astype(int)
    
    results = {
        'max_len': max_len,
        'seed': seed,
        'best_epoch': best_epoch + 1,
        'train_time_sec': train_time,
        'best_val_f1': best_val_f1,
        'test_f1': f1_score(test_labels_arr, test_pred_labels),
        'test_auc': roc_auc_score(test_labels_arr, test_preds),
        'test_precision': precision_score(test_labels_arr, test_pred_labels),
        'test_recall': recall_score(test_labels_arr, test_pred_labels),
    }
    
    # Cleanup
    del model, optimizer, scheduler, scaler
    torch.cuda.empty_cache()
    
    return results


def run_experiment():
    """Run full experiment comparing max_len settings."""
    
    config = ExperimentConfig()
    
    # Create output directory
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Experiment settings
    max_lens = [128, 150, 256]
    seeds = [42, 123, 456]
    
    print("=" * 70)
    print("MAX_LEN EXPERIMENT")
    print(f"Settings: max_lens={max_lens}, seeds={seeds}")
    print(f"Total runs: {len(max_lens) * len(seeds)}")
    print("=" * 70)
    
    all_results = []
    
    for max_len in max_lens:
        print(f"\n{'='*70}")
        print(f"MAX_LEN = {max_len}")
        print("=" * 70)
        
        for seed in seeds:
            print(f"\n--- Seed {seed} ---")
            
            results = train_single_run(config, max_len, seed)
            all_results.append(results)
            
            print(f"  Best epoch: {results['best_epoch']}")
            print(f"  Val F1: {results['best_val_f1']:.4f}")
            print(f"  Test F1: {results['test_f1']:.4f}, AUC: {results['test_auc']:.4f}")
            print(f"  Test P: {results['test_precision']:.4f}, R: {results['test_recall']:.4f}")
            print(f"  Time: {results['train_time_sec']:.1f}s")
    
    # Aggregate results
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    summary = {}
    
    for max_len in max_lens:
        runs = [r for r in all_results if r['max_len'] == max_len]
        
        test_f1s = [r['test_f1'] for r in runs]
        test_aucs = [r['test_auc'] for r in runs]
        times = [r['train_time_sec'] for r in runs]
        
        summary[max_len] = {
            'test_f1_mean': np.mean(test_f1s),
            'test_f1_std': np.std(test_f1s),
            'test_auc_mean': np.mean(test_aucs),
            'test_auc_std': np.std(test_aucs),
            'time_mean': np.mean(times),
        }
        
        print(f"\nmax_len={max_len}:")
        print(f"  Test F1: {summary[max_len]['test_f1_mean']:.4f} ± {summary[max_len]['test_f1_std']:.4f}")
        print(f"  Test AUC: {summary[max_len]['test_auc_mean']:.4f} ± {summary[max_len]['test_auc_std']:.4f}")
        print(f"  Avg Time: {summary[max_len]['time_mean']:.1f}s")
    
    # Find best
    best_max_len = max(max_lens, key=lambda x: summary[x]['test_f1_mean'])
    print(f"\n>>> BEST: max_len={best_max_len} (F1={summary[best_max_len]['test_f1_mean']:.4f})")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = output_dir / f"results_{timestamp}.json"
    
    with open(results_file, 'w') as f:
        json.dump({
            'config': asdict(config),
            'max_lens': max_lens,
            'seeds': seeds,
            'all_results': all_results,
            'summary': {str(k): v for k, v in summary.items()},
            'best_max_len': best_max_len,
        }, f, indent=2)
    
    print(f"\nResults saved to: {results_file}")
    
    return all_results, summary


if __name__ == "__main__":
    run_experiment()
