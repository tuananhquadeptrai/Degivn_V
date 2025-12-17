# %% [markdown]
# # Devign Model Training Pipeline
# 
# Train vulnerability detection models on preprocessed Devign dataset.
# 
# **Environment**: Kaggle with 2x NVIDIA T4 GPU (32GB total VRAM), 13GB RAM
# 
# **Models**:
# - LSTM Baseline (this notebook)
# - Transformer, CodeBERT, GNN (separate notebooks)
# 
# **Features**:
# - Multi-GPU training with DataParallel
# - Early stopping & checkpointing
# - Comprehensive metrics (Accuracy, Precision, Recall, F1, AUC-ROC)

# %% [markdown]
# ## 1. Setup & Configuration

# %%
import os
import sys
import time
import json
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, confusion_matrix, classification_report
)
from tqdm.auto import tqdm

# Environment setup
if os.path.exists('/kaggle'):
    WORKING_DIR = '/kaggle/working'
    sys.path.insert(0, '/kaggle/working/devign_pipeline')
else:
    WORKING_DIR = '/media/hdi/Hdii/Work/C Vul Devign'
    sys.path.insert(0, '/media/hdi/Hdii/Work/C Vul Devign/devign_pipeline')

DATA_DIR = os.path.join(WORKING_DIR, 'processed')
MODEL_DIR = os.path.join(WORKING_DIR, 'models')
LOG_DIR = os.path.join(WORKING_DIR, 'logs')

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# Device setup
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
N_GPUS = torch.cuda.device_count()

print(f"Device: {DEVICE}")
print(f"GPU count: {N_GPUS}")
if torch.cuda.is_available():
    for i in range(N_GPUS):
        gpu = torch.cuda.get_device_properties(i)
        print(f"  GPU {i}: {gpu.name} ({gpu.total_memory / 1024**3:.1f} GB)")

# %% [markdown]
# ## 2. Training Configuration

# %%
class TrainConfig:
    """Training hyperparameters"""
    # Model
    vocab_size: int = 50000
    embed_dim: int = 256
    hidden_dim: int = 256
    num_layers: int = 2
    dropout: float = 0.3
    bidirectional: bool = True
    
    # Training
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    max_epochs: int = 50
    grad_clip: float = 1.0
    
    # Early stopping
    patience: int = 7
    min_delta: float = 1e-4
    
    # Data
    max_seq_length: int = 512
    num_workers: int = 4
    
    # Checkpointing
    save_every: int = 1
    
    def to_dict(self) -> Dict:
        return {k: v for k, v in self.__class__.__dict__.items() 
                if not k.startswith('_') and not callable(v)}

config = TrainConfig()
print("Training config:")
for k, v in config.to_dict().items():
    print(f"  {k}: {v}")

# %% [markdown]
# ## 3. Dataset & DataLoader

# %%
class DevignDataset(Dataset):
    """Load preprocessed .npz chunks"""
    
    def __init__(self, chunk_paths: List[str], max_seq_length: int = 512):
        self.max_seq_length = max_seq_length
        self.samples = []
        
        for path in tqdm(chunk_paths, desc="Loading chunks"):
            data = np.load(path)
            input_ids = data['input_ids']
            labels = data['labels']
            
            for i in range(len(labels)):
                self.samples.append({
                    'input_ids': input_ids[i],
                    'label': labels[i]
                })
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        input_ids = sample['input_ids'][:self.max_seq_length]
        
        # Pad if needed
        padding_length = self.max_seq_length - len(input_ids)
        if padding_length > 0:
            input_ids = np.pad(input_ids, (0, padding_length), constant_values=0)
        
        # Attention mask (1 for real tokens, 0 for padding)
        attention_mask = (input_ids != 0).astype(np.float32)
        
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.float),
            'labels': torch.tensor(sample['label'], dtype=torch.long)
        }


def create_dataloaders(
    data_dir: str, 
    batch_size: int, 
    max_seq_length: int,
    num_workers: int = 4
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train/val/test dataloaders"""
    
    train_paths = sorted(Path(data_dir).glob('train/*.npz'))
    val_paths = sorted(Path(data_dir).glob('val/*.npz'))
    test_paths = sorted(Path(data_dir).glob('test/*.npz'))
    
    print(f"Found: {len(train_paths)} train, {len(val_paths)} val, {len(test_paths)} test chunks")
    
    train_dataset = DevignDataset([str(p) for p in train_paths], max_seq_length)
    val_dataset = DevignDataset([str(p) for p in val_paths], max_seq_length)
    test_dataset = DevignDataset([str(p) for p in test_paths], max_seq_length)
    
    print(f"Samples: {len(train_dataset)} train, {len(val_dataset)} val, {len(test_dataset)} test")
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, 
        num_workers=num_workers, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    
    return train_loader, val_loader, test_loader

# %%
train_loader, val_loader, test_loader = create_dataloaders(
    DATA_DIR, 
    config.batch_size, 
    config.max_seq_length,
    config.num_workers
)

# Test batch
batch = next(iter(train_loader))
print(f"\nBatch shapes:")
print(f"  input_ids: {batch['input_ids'].shape}")
print(f"  attention_mask: {batch['attention_mask'].shape}")
print(f"  labels: {batch['labels'].shape}")

# %% [markdown]
# ## 4. LSTM Baseline Model

# %%
class LSTMVulnDetector(nn.Module):
    """Bidirectional LSTM for vulnerability detection"""
    
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 256,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.3,
        bidirectional: bool = True,
        num_classes: int = 2,
        padding_idx: int = 0
    ):
        super().__init__()
        
        self.embedding = nn.Embedding(
            vocab_size, embed_dim, padding_idx=padding_idx
        )
        
        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        
        self.attention = nn.Sequential(
            nn.Linear(lstm_output_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(lstm_output_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name and param.dim() > 1:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
    
    def forward(
        self, 
        input_ids: torch.Tensor, 
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        # Embed
        embeds = self.embedding(input_ids)  # (B, L, E)
        
        # LSTM
        lstm_out, _ = self.lstm(embeds)  # (B, L, H*2)
        
        # Attention pooling
        attn_weights = self.attention(lstm_out)  # (B, L, 1)
        attn_weights = attn_weights.masked_fill(
            attention_mask.unsqueeze(-1) == 0, float('-inf')
        )
        attn_weights = torch.softmax(attn_weights, dim=1)
        
        context = torch.sum(attn_weights * lstm_out, dim=1)  # (B, H*2)
        
        # Classify
        logits = self.classifier(context)  # (B, 2)
        
        return logits


# %%
# Initialize model
model = LSTMVulnDetector(
    vocab_size=config.vocab_size,
    embed_dim=config.embed_dim,
    hidden_dim=config.hidden_dim,
    num_layers=config.num_layers,
    dropout=config.dropout,
    bidirectional=config.bidirectional
)

# Multi-GPU with DataParallel
if N_GPUS > 1:
    print(f"Using DataParallel on {N_GPUS} GPUs")
    model = nn.DataParallel(model)

model = model.to(DEVICE)

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\nModel parameters: {total_params:,} total, {trainable_params:,} trainable")

# %% [markdown]
# ## 5. Training Utilities

# %%
class EarlyStopping:
    """Early stopping to prevent overfitting"""
    
    def __init__(self, patience: int = 7, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.should_stop = False
    
    def __call__(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        else:
            self.best_score = score
            self.counter = 0
        
        return self.should_stop


class MetricsTracker:
    """Track and compute training metrics"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.losses = []
        self.predictions = []
        self.labels = []
        self.probabilities = []
    
    def update(
        self, 
        loss: float, 
        preds: np.ndarray, 
        labels: np.ndarray,
        probs: np.ndarray
    ):
        self.losses.append(loss)
        self.predictions.extend(preds.tolist())
        self.labels.extend(labels.tolist())
        self.probabilities.extend(probs.tolist())
    
    def compute(self) -> Dict[str, float]:
        preds = np.array(self.predictions)
        labels = np.array(self.labels)
        probs = np.array(self.probabilities)
        
        metrics = {
            'loss': np.mean(self.losses),
            'accuracy': accuracy_score(labels, preds),
            'precision': precision_score(labels, preds, zero_division=0),
            'recall': recall_score(labels, preds, zero_division=0),
            'f1': f1_score(labels, preds, zero_division=0),
        }
        
        # AUC-ROC (only if both classes present)
        if len(np.unique(labels)) > 1:
            metrics['auc_roc'] = roc_auc_score(labels, probs)
        else:
            metrics['auc_roc'] = 0.0
        
        return metrics


def save_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    epoch: int,
    metrics: Dict,
    path: str
):
    """Save model checkpoint"""
    # Handle DataParallel
    model_state = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
    
    torch.save({
        'epoch': epoch,
        'model_state_dict': model_state,
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics,
        'config': config.to_dict()
    }, path)


def load_checkpoint(path: str, model: nn.Module, optimizer: optim.Optimizer = None):
    """Load model checkpoint"""
    checkpoint = torch.load(path, map_location=DEVICE)
    
    # Handle DataParallel
    if isinstance(model, nn.DataParallel):
        model.module.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return checkpoint['epoch'], checkpoint['metrics']

# %% [markdown]
# ## 6. Training Loop

# %%
def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    grad_clip: float
) -> Dict[str, float]:
    """Train one epoch"""
    model.train()
    tracker = MetricsTracker()
    
    pbar = tqdm(loader, desc="Training", leave=False)
    for batch in pbar:
        input_ids = batch['input_ids'].to(DEVICE)
        attention_mask = batch['attention_mask'].to(DEVICE)
        labels = batch['labels'].to(DEVICE)
        
        optimizer.zero_grad()
        
        logits = model(input_ids, attention_mask)
        loss = criterion(logits, labels)
        
        loss.backward()
        
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        
        optimizer.step()
        
        # Track metrics
        probs = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
        preds = logits.argmax(dim=1).detach().cpu().numpy()
        
        tracker.update(
            loss.item(),
            preds,
            labels.cpu().numpy(),
            probs
        )
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    return tracker.compute()


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module
) -> Dict[str, float]:
    """Evaluate model"""
    model.eval()
    tracker = MetricsTracker()
    
    for batch in tqdm(loader, desc="Evaluating", leave=False):
        input_ids = batch['input_ids'].to(DEVICE)
        attention_mask = batch['attention_mask'].to(DEVICE)
        labels = batch['labels'].to(DEVICE)
        
        logits = model(input_ids, attention_mask)
        loss = criterion(logits, labels)
        
        probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        preds = logits.argmax(dim=1).cpu().numpy()
        
        tracker.update(
            loss.item(),
            preds,
            labels.cpu().numpy(),
            probs
        )
    
    return tracker.compute()


# %%
def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: TrainConfig
) -> Dict:
    """Full training loop"""
    
    # Class weights for imbalanced data
    labels = [s['label'] for s in train_loader.dataset.samples]
    class_counts = np.bincount(labels)
    class_weights = torch.tensor(
        len(labels) / (len(class_counts) * class_counts),
        dtype=torch.float
    ).to(DEVICE)
    
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3, verbose=True
    )
    
    early_stopping = EarlyStopping(config.patience, config.min_delta)
    
    history = {'train': [], 'val': []}
    best_f1 = 0.0
    best_epoch = 0
    
    print("\n" + "="*60)
    print("Starting training...")
    print("="*60)
    
    for epoch in range(1, config.max_epochs + 1):
        epoch_start = time.time()
        
        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, criterion, config.grad_clip
        )
        
        # Validate
        val_metrics = evaluate(model, val_loader, criterion)
        
        epoch_time = time.time() - epoch_start
        
        # Log
        history['train'].append(train_metrics)
        history['val'].append(val_metrics)
        
        print(f"\nEpoch {epoch}/{config.max_epochs} ({epoch_time:.1f}s)")
        print(f"  Train - Loss: {train_metrics['loss']:.4f}, F1: {train_metrics['f1']:.4f}, AUC: {train_metrics['auc_roc']:.4f}")
        print(f"  Val   - Loss: {val_metrics['loss']:.4f}, F1: {val_metrics['f1']:.4f}, AUC: {val_metrics['auc_roc']:.4f}")
        
        # Learning rate scheduling
        scheduler.step(val_metrics['f1'])
        
        # Save best model
        if val_metrics['f1'] > best_f1:
            best_f1 = val_metrics['f1']
            best_epoch = epoch
            save_checkpoint(
                model, optimizer, epoch, val_metrics,
                os.path.join(MODEL_DIR, 'best_model.pt')
            )
            print(f"  ✓ New best model saved (F1: {best_f1:.4f})")
        
        # Regular checkpoint
        if epoch % config.save_every == 0:
            save_checkpoint(
                model, optimizer, epoch, val_metrics,
                os.path.join(MODEL_DIR, f'checkpoint_epoch_{epoch}.pt')
            )
        
        # Early stopping
        if early_stopping(val_metrics['f1']):
            print(f"\nEarly stopping at epoch {epoch}")
            break
    
    print("\n" + "="*60)
    print(f"Training complete! Best F1: {best_f1:.4f} at epoch {best_epoch}")
    print("="*60)
    
    # Save history
    with open(os.path.join(LOG_DIR, 'training_history.json'), 'w') as f:
        json.dump(history, f, indent=2)
    
    return history

# %% [markdown]
# ## 7. Run Training

# %%
history = train(model, train_loader, val_loader, config)

# %% [markdown]
# ## 8. Final Evaluation on Test Set

# %%
# Load best model
print("Loading best model for evaluation...")
load_checkpoint(os.path.join(MODEL_DIR, 'best_model.pt'), model)

# Evaluate on test set
criterion = nn.CrossEntropyLoss()
test_metrics = evaluate(model, test_loader, criterion)

print("\n" + "="*60)
print("TEST SET RESULTS")
print("="*60)
print(f"Accuracy:  {test_metrics['accuracy']:.4f}")
print(f"Precision: {test_metrics['precision']:.4f}")
print(f"Recall:    {test_metrics['recall']:.4f}")
print(f"F1 Score:  {test_metrics['f1']:.4f}")
print(f"AUC-ROC:   {test_metrics['auc_roc']:.4f}")

# %%
# Detailed classification report
@torch.no_grad()
def get_predictions(model, loader):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    
    for batch in loader:
        input_ids = batch['input_ids'].to(DEVICE)
        attention_mask = batch['attention_mask'].to(DEVICE)
        
        logits = model(input_ids, attention_mask)
        probs = torch.softmax(logits, dim=1)[:, 1]
        preds = logits.argmax(dim=1)
        
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(batch['labels'].numpy())
        all_probs.extend(probs.cpu().numpy())
    
    return np.array(all_preds), np.array(all_labels), np.array(all_probs)

preds, labels, probs = get_predictions(model, test_loader)

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
    'timestamp': datetime.now().isoformat()
}
torch.save(final_export, os.path.join(MODEL_DIR, 'lstm_vuln_detector_final.pt'))
print(f"\nFinal model exported to {MODEL_DIR}/lstm_vuln_detector_final.pt")

# %% [markdown]
# ## Next Steps
# 
# 1. **Hyperparameter tuning**: Use Optuna or Ray Tune
# 2. **Better models**: Try Transformer, CodeBERT, GraphCodeBERT
# 3. **Graph-based**: Use GNN on CFG/DFG (see `03_graph_training.ipynb`)
# 4. **Ensemble**: Combine sequence + graph models
