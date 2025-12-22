# Devign V2 Training - Hierarchical BiGRU with Oracle Improvements
# Improvements: Symmetric gating, pos_weight tuning, multi-seed ensemble, calibration
import os, sys, json, random, numpy as np
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.amp import autocast, GradScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.isotonic import IsotonicRegression
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm

# Config paths
if os.path.exists('/kaggle'):
    DATA_DIR = '/kaggle/input/devign-v2-processed/processed'
    MODEL_DIR = '/kaggle/working/models'
else:
    DATA_DIR = '/media/tuananh/새 볼륨/DACNANM/Devign/C-Vul-Devign/Output data/results/processed'
    MODEL_DIR = './models'

os.makedirs(MODEL_DIR, exist_ok=True)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {DEVICE}, GPUs: {torch.cuda.device_count()}")

# ===== CONFIGURATION =====
NUM_SEEDS = 3  # Multi-seed ensemble (set to 1 for single model)
POS_WEIGHT_SCALE = 1.12  # Oracle recommendation: 1.08-1.16
GATE_INIT = 0.3  # Initial gate strength (bounded)

# Load config
with open(f'{DATA_DIR}/config.json') as f:
    data_config = json.load(f)
print(f"Config: vocab={data_config['vocab_size']}, version={data_config['version']}")

# Dataset
class DevignV2Dataset(Dataset):
    def __init__(self, npz_path, max_len=512):
        data = np.load(npz_path)
        self.input_ids = data['input_ids'][:, :max_len]
        self.attention_mask = data['attention_mask'][:, :max_len]
        self.labels = data['labels']
        self.slice_input_ids = data.get('slice_input_ids')
        self.slice_attention_mask = data.get('slice_attention_mask')
        self.slice_count = data.get('slice_count')
        
        vuln_path = Path(npz_path).with_name(f"{Path(npz_path).stem}_vuln.npz")
        self.vuln_features = self.slice_vuln_features = self.slice_rel_features = None
        if vuln_path.exists():
            v = np.load(vuln_path, allow_pickle=True)
            self.vuln_features = v.get('features')
            self.slice_vuln_features = v.get('slice_vuln_features')
            self.slice_rel_features = v.get('slice_rel_features')
        print(f"  Loaded {len(self.labels)} samples")
    
    def __len__(self): return len(self.labels)
    
    def __getitem__(self, i):
        item = {
            'input_ids': torch.tensor(self.input_ids[i], dtype=torch.long),
            'attention_mask': torch.tensor(self.attention_mask[i], dtype=torch.float),
            'labels': torch.tensor(self.labels[i], dtype=torch.long),
        }
        if self.slice_input_ids is not None:
            item['slice_input_ids'] = torch.tensor(self.slice_input_ids[i], dtype=torch.long)
            item['slice_attention_mask'] = torch.tensor(self.slice_attention_mask[i], dtype=torch.float)
            item['slice_count'] = torch.tensor(self.slice_count[i], dtype=torch.long)
        if self.vuln_features is not None:
            item['vuln_features'] = torch.tensor(self.vuln_features[i], dtype=torch.float)
        if self.slice_vuln_features is not None:
            item['slice_vuln_features'] = torch.tensor(self.slice_vuln_features[i], dtype=torch.float)
        if self.slice_rel_features is not None:
            item['slice_rel_features'] = torch.tensor(self.slice_rel_features[i], dtype=torch.float)
        return item

print("Loading data...")
train_ds = DevignV2Dataset(f'{DATA_DIR}/train.npz')
val_ds = DevignV2Dataset(f'{DATA_DIR}/val.npz')
test_ds = DevignV2Dataset(f'{DATA_DIR}/test.npz')

n_neg, n_pos = np.sum(train_ds.labels==0), np.sum(train_ds.labels==1)
print(f"Classes: neg={n_neg}, pos={n_pos}")

# Model - Enhanced with all Oracle recommendations
class HierarchicalBiGRU(nn.Module):
    def __init__(self, vocab_size=238, embed_dim=96, hidden_dim=192, slice_hidden=160,
                 vuln_dim=26, slice_feat_dim=52, gate_init=0.3):
        super().__init__()
        self.slice_hidden = slice_hidden
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.embed_drop = nn.Dropout(0.15)
        
        # Global encoder
        self.global_gru = nn.GRU(embed_dim, hidden_dim, num_layers=2, bidirectional=True, batch_first=True, dropout=0.25)
        self.global_attn = nn.Sequential(nn.Linear(hidden_dim*2, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, 1))
        
        # Slice encoder
        self.slice_gru = nn.GRU(embed_dim, slice_hidden, bidirectional=True, batch_first=True)
        self.slice_attn = nn.Sequential(nn.Linear(slice_hidden*2, slice_hidden), nn.Tanh(), nn.Linear(slice_hidden, 1))
        
        # Slice feature fusion (concat+MLP)
        self.slice_feat_mlp = nn.Sequential(nn.Linear(slice_feat_dim, 128), nn.LayerNorm(128), nn.GELU(), nn.Dropout(0.2))
        self.slice_fusion = nn.Sequential(
            nn.Linear(slice_hidden*2 + 128, slice_hidden*2),
            nn.GELU(),
            nn.Dropout(0.2)
        )
        self.slice_level_attn = nn.Sequential(nn.Linear(slice_hidden*2, slice_hidden), nn.Tanh(), nn.Linear(slice_hidden, 1))
        
        # Vuln features MLP (dynamic dim)
        self.vuln_dim = vuln_dim
        self.vuln_mlp = nn.Sequential(nn.BatchNorm1d(vuln_dim), nn.Linear(vuln_dim, 64), nn.GELU(), nn.Dropout(0.2))
        
        # Symmetric defense-aware gating (Oracle improvement: allows both increase/decrease)
        self.defense_gate = nn.Sequential(
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        # Bounded gate strength: sigmoid keeps it in (0, 1)
        self.gate_strength_raw = nn.Parameter(torch.tensor(gate_init))
        
        # Classifier: global(384) + slice(320) + vuln(64) = 768
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim*2 + slice_hidden*2 + 64), 
            nn.Linear(hidden_dim*2 + slice_hidden*2 + 64, 256), 
            nn.GELU(), nn.Dropout(0.3), 
            nn.Linear(256, 2)
        )
    
    @property
    def gate_strength(self):
        return torch.sigmoid(self.gate_strength_raw)  # Bounded to (0, 1)
    
    def encode_global(self, ids, mask):
        emb = self.embed_drop(self.embedding(ids))
        out, _ = self.global_gru(emb)
        scores = self.global_attn(out).masked_fill(mask.unsqueeze(-1)==0, -65000.0)
        return (out * F.softmax(scores, dim=1)).sum(dim=1)
    
    def encode_slices(self, slice_ids, slice_mask, slice_count, slice_vuln=None, slice_rel=None):
        B, S, L = slice_ids.shape
        emb = self.embed_drop(self.embedding(slice_ids.view(B*S, L)))
        out, _ = self.slice_gru(emb)
        scores = self.slice_attn(out).masked_fill(slice_mask.view(B*S,L).unsqueeze(-1)==0, -65000.0)
        slice_repr = (out * F.softmax(scores, dim=1)).sum(dim=1).view(B, S, -1)
        
        if slice_vuln is not None and slice_rel is not None:
            feat = self.slice_feat_mlp(torch.cat([slice_vuln, slice_rel], dim=-1))
            slice_repr = self.slice_fusion(torch.cat([slice_repr, feat], dim=-1))
        
        s_mask = torch.arange(S, device=slice_count.device).expand(B,S) < slice_count.unsqueeze(1)
        s_scores = self.slice_level_attn(slice_repr).masked_fill(~s_mask.unsqueeze(-1), -65000.0)
        return (slice_repr * F.softmax(s_scores, dim=1)).sum(dim=1)
    
    def forward(self, input_ids, attention_mask, slice_input_ids=None, slice_attention_mask=None, 
                slice_count=None, vuln_features=None, slice_vuln_features=None, slice_rel_features=None, **kw):
        g = self.encode_global(input_ids, attention_mask)
        s = self.encode_slices(slice_input_ids, slice_attention_mask, slice_count, slice_vuln_features, slice_rel_features) if slice_input_ids is not None else torch.zeros(g.size(0), self.slice_hidden*2, device=g.device)
        v = self.vuln_mlp(vuln_features) if vuln_features is not None else torch.zeros(g.size(0), 64, device=g.device)
        
        h = torch.cat([g, s, v], dim=1)
        logits = self.classifier(h)
        
        # Symmetric gating: (gate - 0.5) allows both positive and negative adjustment
        if vuln_features is not None:
            gate = self.defense_gate(v).squeeze(-1)  # [B], in (0, 1)
            delta = self.gate_strength * (gate - 0.5)  # Symmetric around 0
            logits = logits.clone()
            logits[:, 0] = logits[:, 0] + delta  # Push toward negative when gate > 0.5
            logits[:, 1] = logits[:, 1] - delta  # Push toward positive when gate < 0.5
        
        return logits

# ===== TRAINING FUNCTIONS =====
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def train_epoch(model, loader, criterion, optimizer, scaler):
    model.train()
    total_loss, all_preds, all_labels, all_probs = 0, [], [], []
    for batch in tqdm(loader, desc="Train"):
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        optimizer.zero_grad()
        with autocast(device_type='cuda'):
            logits = model(**batch)
            loss = criterion(logits, batch['labels'])
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer); torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        scaler.step(optimizer); scaler.update()
        total_loss += loss.item()
        all_probs.extend(F.softmax(logits.detach(), dim=1)[:,1].cpu().numpy())
        all_preds.extend(logits.argmax(1).cpu().numpy())
        all_labels.extend(batch['labels'].cpu().numpy())
    return {'loss': total_loss/len(loader), 'f1': f1_score(all_labels, all_preds), 'auc': roc_auc_score(all_labels, all_probs)}

@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    total_loss, all_preds, all_labels, all_probs = 0, [], [], []
    for batch in tqdm(loader, desc="Eval"):
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        with autocast(device_type='cuda'):
            logits = model(**batch)
            loss = criterion(logits, batch['labels'])
        total_loss += loss.item()
        all_probs.extend(F.softmax(logits, dim=1)[:,1].cpu().numpy())
        all_preds.extend(logits.argmax(1).cpu().numpy())
        all_labels.extend(batch['labels'].cpu().numpy())
    all_labels, all_probs = np.array(all_labels), np.array(all_probs)
    best_f1, best_t = 0, 0.5
    for t in np.arange(0.25, 0.75, 0.01):
        f1 = f1_score(all_labels, (all_probs>=t).astype(int))
        if f1 > best_f1: best_f1, best_t = f1, t
    opt_preds = (all_probs >= best_t).astype(int)
    return {'loss': total_loss/len(loader), 'f1': f1_score(all_labels, all_preds), 'auc': roc_auc_score(all_labels, all_probs),
            'opt_f1': best_f1, 'opt_prec': precision_score(all_labels, opt_preds), 'opt_rec': recall_score(all_labels, opt_preds), 
            'opt_t': best_t, 'labels': all_labels, 'probs': all_probs}

@torch.no_grad()
def get_predictions(model, loader):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    for batch in tqdm(loader, desc="Predict"):
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        with autocast(device_type='cuda'):
            logits = model(**batch)
        all_probs.extend(F.softmax(logits, dim=1)[:,1].cpu().numpy())
        all_preds.extend(logits.argmax(1).cpu().numpy())
        all_labels.extend(batch['labels'].cpu().numpy())
    return np.array(all_labels), np.array(all_preds), np.array(all_probs)

def train_single_model(seed, train_ds, val_ds, n_neg, n_pos, data_config):
    """Train a single model with given seed."""
    print(f"\n{'#'*60}\nTraining model with seed {seed}\n{'#'*60}")
    set_seed(seed)
    
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=2, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=64, num_workers=2, pin_memory=True)
    
    model = HierarchicalBiGRU(vocab_size=data_config['vocab_size'], gate_init=GATE_INIT).to(DEVICE)
    if torch.cuda.device_count() > 1: 
        model = nn.DataParallel(model)
    print(f"Model: {sum(p.numel() for p in model.parameters()):,} params")
    
    pos_weight = (n_neg / n_pos) * POS_WEIGHT_SCALE
    criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0, pos_weight], dtype=torch.float32, device=DEVICE), label_smoothing=0.0)
    optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=2, factor=0.5)
    scaler = GradScaler()
    
    best_f1, patience = 0, 0
    model_path = f'{MODEL_DIR}/best_v2_seed{seed}.pt'
    
    for epoch in range(1, 31):
        print(f"\n{'='*50}\nEpoch {epoch}/30 (seed={seed})")
        train_m = train_epoch(model, train_loader, criterion, optimizer, scaler)
        val_m = evaluate(model, val_loader, criterion)
        scheduler.step(val_m['opt_f1'])
        print(f"Train: loss={train_m['loss']:.4f}, F1={train_m['f1']:.4f}, AUC={train_m['auc']:.4f}")
        print(f"Val: F1={val_m['f1']:.4f}, AUC={val_m['auc']:.4f}, OptF1={val_m['opt_f1']:.4f}, Prec={val_m['opt_prec']:.4f}, Rec={val_m['opt_rec']:.4f}")
        if val_m['opt_f1'] > best_f1:
            best_f1 = val_m['opt_f1']; patience = 0
            torch.save(model.state_dict(), model_path)
            print(f"★ Best F1: {best_f1:.4f}")
        else:
            patience += 1
            if patience >= 6: print("Early stop!"); break
    
    # Load best model
    model.load_state_dict(torch.load(model_path))
    return model, best_f1

# ===== MAIN TRAINING LOOP =====
print(f"\n{'='*60}")
print(f"Training {NUM_SEEDS} model(s) with pos_weight_scale={POS_WEIGHT_SCALE}")
print(f"{'='*60}")

models = []
val_probs_list = []
test_probs_list = []

val_loader = DataLoader(val_ds, batch_size=64, num_workers=2, pin_memory=True)
test_loader = DataLoader(test_ds, batch_size=64, num_workers=2, pin_memory=True)

for seed_idx in range(NUM_SEEDS):
    seed = 42 + seed_idx * 1000
    model, best_f1 = train_single_model(seed, train_ds, val_ds, n_neg, n_pos, data_config)
    models.append(model)
    
    # Get predictions for ensemble
    _, _, val_probs = get_predictions(model, val_loader)
    _, _, test_probs = get_predictions(model, test_loader)
    val_probs_list.append(val_probs)
    test_probs_list.append(test_probs)

# ===== ENSEMBLE PREDICTIONS =====
print(f"\n{'='*60}")
print(f"Ensemble Evaluation ({NUM_SEEDS} models)")
print(f"{'='*60}")

val_labels = val_ds.labels
test_labels = test_ds.labels

# Average probabilities
val_probs_ens = np.mean(val_probs_list, axis=0)
test_probs_ens = np.mean(test_probs_list, axis=0)

# Calibration using Isotonic Regression on validation set
print("Applying isotonic calibration...")
calibrator = IsotonicRegression(y_min=0, y_max=1, out_of_bounds='clip')
calibrator.fit(val_probs_ens, val_labels)
val_probs_cal = calibrator.predict(val_probs_ens)
test_probs_cal = calibrator.predict(test_probs_ens)

# Find optimal threshold on calibrated validation predictions
best_f1, best_t = 0, 0.5
for t in np.arange(0.25, 0.75, 0.01):
    f1 = f1_score(val_labels, (val_probs_cal >= t).astype(int))
    if f1 > best_f1: best_f1, best_t = f1, t
print(f"Validation (ensemble+calibrated): OptF1={best_f1:.4f} at t={best_t:.2f}")

# Evaluate on test set
test_preds_cal = (test_probs_cal >= best_t).astype(int)
test_preds_05 = (test_probs_cal >= 0.5).astype(int)

test_f1_05 = f1_score(test_labels, test_preds_05)
test_f1_opt = f1_score(test_labels, test_preds_cal)
test_prec = precision_score(test_labels, test_preds_cal)
test_rec = recall_score(test_labels, test_preds_cal)
test_auc = roc_auc_score(test_labels, test_probs_cal)

print(f"\n{'='*60}")
print(f"FINAL TEST RESULTS (Ensemble + Calibration)")
print(f"{'='*60}")
print(f"Test F1 (t=0.5): {test_f1_05:.4f}")
print(f"Test OptF1 (t={best_t:.2f}): {test_f1_opt:.4f}")
print(f"Test Precision: {test_prec:.4f}")
print(f"Test Recall: {test_rec:.4f}")
print(f"Test AUC: {test_auc:.4f}")

# ===== CONFUSION MATRIX =====
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Default threshold (0.5)
cm1 = confusion_matrix(test_labels, test_preds_05)
sns.heatmap(cm1, annot=True, fmt='d', cmap='Blues', ax=axes[0],
            xticklabels=['Non-Vuln', 'Vuln'], yticklabels=['Non-Vuln', 'Vuln'])
axes[0].set_title(f'Confusion Matrix (t=0.5)\nF1={test_f1_05:.4f}')
axes[0].set_xlabel('Predicted'); axes[0].set_ylabel('Actual')

# Optimal threshold
cm2 = confusion_matrix(test_labels, test_preds_cal)
sns.heatmap(cm2, annot=True, fmt='d', cmap='Greens', ax=axes[1],
            xticklabels=['Non-Vuln', 'Vuln'], yticklabels=['Non-Vuln', 'Vuln'])
axes[1].set_title(f'Confusion Matrix (t={best_t:.2f})\nOptF1={test_f1_opt:.4f}')
axes[1].set_xlabel('Predicted'); axes[1].set_ylabel('Actual')

plt.tight_layout()
plt.savefig(f'{MODEL_DIR}/confusion_matrix.png', dpi=150)
plt.show()
print(f"Saved confusion matrix to {MODEL_DIR}/confusion_matrix.png")

# Save ensemble config
ensemble_config = {
    'num_seeds': NUM_SEEDS,
    'pos_weight_scale': POS_WEIGHT_SCALE,
    'gate_init': GATE_INIT,
    'optimal_threshold': float(best_t),
    'test_f1_05': float(test_f1_05),
    'test_opt_f1': float(test_f1_opt),
    'test_precision': float(test_prec),
    'test_recall': float(test_rec),
    'test_auc': float(test_auc)
}
with open(f'{MODEL_DIR}/ensemble_config.json', 'w') as f:
    json.dump(ensemble_config, f, indent=2)
print(f"Saved ensemble config to {MODEL_DIR}/ensemble_config.json")
