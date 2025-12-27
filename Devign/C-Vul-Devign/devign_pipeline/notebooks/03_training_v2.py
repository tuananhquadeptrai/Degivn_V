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
    DATA_DIR = '/kaggle/input/outputdata/Output data'
    OUTPUT_DIR = '/kaggle/working/output'
else:
    DATA_DIR = '/media/tuananh/새 볼륨/DACNANM/Devign/C-Vul-Devign/Output data'
    OUTPUT_DIR = './output'

MODEL_DIR = f'{OUTPUT_DIR}/models'
PLOTS_DIR = f'{OUTPUT_DIR}/plots'

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {DEVICE}, GPUs: {torch.cuda.device_count()}")

# ===== CONFIGURATION =====
NUM_SEEDS = 3  # Multi-seed ensemble for better AUC
POS_WEIGHT_SCALE = 1.12  # Oracle recommendation: 1.08-1.16
GATE_INIT = 0.4  # Initial gate strength (bounded) - increased for feature gating
FOCAL_GAMMA = 1.5  # Reduced from 2.0 to improve recall
FOCAL_ALPHA_SCALE = 1.4  # Increased to prioritize positive class (reduce FN)
MAX_EPOCHS = 24  # Reduced from 30 to avoid overfitting

# Load config
with open(f'{DATA_DIR}/config.json') as f:
    data_config = json.load(f)
print(f"Config: vocab={data_config['vocab_size']}, version={data_config['version']}")

# ===== DATA AUGMENTATION FOR VULNERABLE CLASS =====
class VulnerableAugmenter:
    """Augmentation techniques for vulnerable code samples."""
    
    # C keywords that should not be modified
    C_KEYWORDS = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                  21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40}
    
    def __init__(self, dropout_rate=0.07, shuffle_prob=0.1, seed=42):
        self.dropout_rate = dropout_rate  # 5-10% token dropout
        self.shuffle_prob = shuffle_prob  # Probability of swapping adjacent tokens
        self.rng = np.random.RandomState(seed)
    
    def _is_keyword(self, token_id):
        """Check if token is a C keyword (should not be modified)."""
        return int(token_id) in self.C_KEYWORDS or token_id == 0  # 0 is padding
    
    def token_dropout(self, input_ids, attention_mask):
        """Randomly drop 5-10% tokens (except keywords and padding)."""
        ids = input_ids.copy()
        mask = attention_mask.copy()
        
        for i in range(len(ids)):
            if mask[i] == 1 and not self._is_keyword(ids[i]):
                if self.rng.random() < self.dropout_rate:
                    ids[i] = 0  # Set to padding
                    mask[i] = 0
        return ids, mask
    
    def token_shuffle(self, input_ids, attention_mask):
        """Swap adjacent non-keyword tokens with given probability."""
        ids = input_ids.copy()
        mask = attention_mask.copy()
        
        i = 0
        while i < len(ids) - 1:
            if (mask[i] == 1 and mask[i+1] == 1 and 
                not self._is_keyword(ids[i]) and not self._is_keyword(ids[i+1])):
                if self.rng.random() < self.shuffle_prob:
                    ids[i], ids[i+1] = ids[i+1], ids[i]
                    i += 2  # Skip next token to avoid double-swapping
                    continue
            i += 1
        return ids, mask
    
    def synonym_replacement(self, input_ids, attention_mask):
        """Re-index VAR_X and FUNC_X tokens (e.g., VAR_0 -> VAR_5)."""
        ids = input_ids.copy()
        
        # Assume VAR tokens are in range 100-119 (VAR_0 to VAR_19)
        # Assume FUNC tokens are in range 120-139 (FUNC_0 to FUNC_19)
        VAR_START, VAR_END = 100, 119
        FUNC_START, FUNC_END = 120, 139
        
        for i in range(len(ids)):
            if attention_mask[i] == 0:
                continue
            
            token = ids[i]
            if VAR_START <= token <= VAR_END:
                offset = self.rng.randint(1, 10)  # Shift by 1-9
                new_idx = VAR_START + ((token - VAR_START + offset) % 20)
                ids[i] = new_idx
            elif FUNC_START <= token <= FUNC_END:
                offset = self.rng.randint(1, 10)
                new_idx = FUNC_START + ((token - FUNC_START + offset) % 20)
                ids[i] = new_idx
        
        return ids, attention_mask
    
    def augment(self, input_ids, attention_mask):
        """Apply random combination of augmentations."""
        # Choose which augmentations to apply
        aug_choice = self.rng.randint(0, 3)
        
        if aug_choice == 0:
            return self.token_dropout(input_ids, attention_mask)
        elif aug_choice == 1:
            return self.token_shuffle(input_ids, attention_mask)
        else:
            return self.synonym_replacement(input_ids, attention_mask)


class AugmentedDevignDataset(Dataset):
    """Wrapper dataset that augments vulnerable samples 2x."""
    
    def __init__(self, base_dataset, num_augmentations=2, seed=42):
        self.base_dataset = base_dataset
        self.num_augmentations = num_augmentations
        self.augmenter = VulnerableAugmenter(seed=seed)
        
        # Build index mapping
        self.index_map = []  # (base_idx, aug_idx) where aug_idx=0 means original
        
        for i in range(len(base_dataset)):
            label = base_dataset.labels[i]
            self.index_map.append((i, 0))  # Original sample
            if label == 1:  # Vulnerable sample
                for aug_idx in range(1, num_augmentations + 1):
                    self.index_map.append((i, aug_idx))
        
        # Count samples
        n_orig = len(base_dataset)
        n_vuln = np.sum(base_dataset.labels == 1)
        n_aug = n_vuln * num_augmentations
        print(f"  AugmentedDataset: {n_orig} original + {n_aug} augmented = {len(self.index_map)} total")
        print(f"  New class ratio: {n_orig - n_vuln} clean vs {n_vuln + n_aug} vuln")
    
    def __len__(self):
        return len(self.index_map)
    
    def __getitem__(self, idx):
        base_idx, aug_idx = self.index_map[idx]
        item = self.base_dataset[base_idx]
        
        if aug_idx > 0:  # Apply augmentation
            # Set seed based on idx for reproducibility
            self.augmenter.rng = np.random.RandomState(42 + idx * 100 + aug_idx)
            
            # Augment main input
            input_ids = item['input_ids'].numpy()
            attention_mask = item['attention_mask'].numpy()
            aug_ids, aug_mask = self.augmenter.augment(input_ids, attention_mask)
            item['input_ids'] = torch.tensor(aug_ids, dtype=torch.long)
            item['attention_mask'] = torch.tensor(aug_mask, dtype=torch.float)
            
            # Augment slices if present
            if 'slice_input_ids' in item:
                slice_ids = item['slice_input_ids'].numpy()
                slice_mask = item['slice_attention_mask'].numpy()
                S, L = slice_ids.shape
                for s in range(S):
                    aug_s_ids, aug_s_mask = self.augmenter.augment(slice_ids[s], slice_mask[s])
                    slice_ids[s] = aug_s_ids
                    slice_mask[s] = aug_s_mask
                item['slice_input_ids'] = torch.tensor(slice_ids, dtype=torch.long)
                item['slice_attention_mask'] = torch.tensor(slice_mask, dtype=torch.float)
        
        return item


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
train_ds_base = DevignV2Dataset(f'{DATA_DIR}/train.npz')
val_ds = DevignV2Dataset(f'{DATA_DIR}/val.npz')
test_ds = DevignV2Dataset(f'{DATA_DIR}/test.npz')

n_neg, n_pos = np.sum(train_ds_base.labels==0), np.sum(train_ds_base.labels==1)
print(f"Original classes: neg={n_neg}, pos={n_pos}")

# Apply augmentation to training set (2x augmentation for vulnerable samples)
print("Applying data augmentation for vulnerable class...")
train_ds = AugmentedDevignDataset(train_ds_base, num_augmentations=2, seed=42)

# Model - Enhanced with all Oracle recommendations
class HierarchicalBiGRU(nn.Module):
    def __init__(self, vocab_size=238, embed_dim=96, hidden_dim=192, slice_hidden=160,
                 vuln_dim=26, slice_feat_dim=52, gate_init=0.3):
        super().__init__()
        self.slice_hidden = slice_hidden
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.embed_drop = nn.Dropout(0.3)  # Increased from 0.15
        
        # Global encoder
        self.global_gru = nn.GRU(embed_dim, hidden_dim, num_layers=2, bidirectional=True, batch_first=True, dropout=0.4)  # Increased from 0.25
        self.global_attn = nn.Sequential(nn.Linear(hidden_dim*2, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, 1))
        
        # Slice encoder
        self.slice_gru = nn.GRU(embed_dim, slice_hidden, bidirectional=True, batch_first=True)
        self.slice_attn = nn.Sequential(nn.Linear(slice_hidden*2, slice_hidden), nn.Tanh(), nn.Linear(slice_hidden, 1))
        
        # Slice-sequence BiGRU for inter-slice dependencies (Oracle improvement)
        self.slice_seq_gru = nn.GRU(slice_hidden*2, slice_hidden, bidirectional=True, batch_first=True)
        self.slice_seq_attn = nn.Sequential(nn.Linear(slice_hidden*2, slice_hidden), nn.Tanh(), nn.Linear(slice_hidden, 1))
        
        # Slice feature fusion (concat+MLP)
        self.slice_feat_mlp = nn.Sequential(nn.Linear(slice_feat_dim, 128), nn.LayerNorm(128), nn.GELU(), nn.Dropout(0.4))  # Increased from 0.2
        self.slice_fusion = nn.Sequential(
            nn.Linear(slice_hidden*2 + 128, slice_hidden*2),
            nn.GELU(),
            nn.Dropout(0.4)  # Increased from 0.2
        )
        self.slice_level_attn = nn.Sequential(nn.Linear(slice_hidden*2, slice_hidden), nn.Tanh(), nn.Linear(slice_hidden, 1))
        
        # Vuln features MLP (dynamic dim)
        self.vuln_dim = vuln_dim
        self.vuln_mlp = nn.Sequential(nn.BatchNorm1d(vuln_dim), nn.Linear(vuln_dim, 64), nn.GELU(), nn.Dropout(0.4))  # Increased from 0.2
        
        # Feature-level gating over vuln representation (Oracle improvement: replaces logit-shift)
        self.feature_gate = nn.Sequential(
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Linear(32, 64),
            nn.Sigmoid()
        )
        # Bounded gate strength: sigmoid keeps it in (0, 1)
        self.gate_strength_raw = nn.Parameter(torch.tensor(gate_init))
        
        # Classifier: global(384) + slice(320) + vuln(64) = 768
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim*2 + slice_hidden*2 + 64), 
            nn.Linear(hidden_dim*2 + slice_hidden*2 + 64, 256), 
            nn.GELU(), nn.Dropout(0.5),  # Increased from 0.3 
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
        
        # Existing slice-level attention (bag-of-slices summary)
        s_scores = self.slice_level_attn(slice_repr).masked_fill(~s_mask.unsqueeze(-1), -65000.0)
        slice_attn_repr = (slice_repr * F.softmax(s_scores, dim=1)).sum(dim=1)  # [B, 2*slice_hidden]
        
        # New BiGRU over the slice sequence (Oracle improvement: inter-slice dependencies)
        slice_repr_masked = slice_repr * s_mask.unsqueeze(-1).float()
        seq_out, _ = self.slice_seq_gru(slice_repr_masked)  # [B, S, 2*slice_hidden]
        seq_scores = self.slice_seq_attn(seq_out).masked_fill(~s_mask.unsqueeze(-1), -65000.0)
        slice_seq_repr = (seq_out * F.softmax(seq_scores, dim=1)).sum(dim=1)  # [B, 2*slice_hidden]
        
        # Combine orderless + sequential views
        return 0.5 * (slice_attn_repr + slice_seq_repr)
    
    def forward(self, input_ids, attention_mask, slice_input_ids=None, slice_attention_mask=None, 
                slice_count=None, vuln_features=None, slice_vuln_features=None, slice_rel_features=None, **kw):
        g = self.encode_global(input_ids, attention_mask)
        s = self.encode_slices(slice_input_ids, slice_attention_mask, slice_count, slice_vuln_features, slice_rel_features) if slice_input_ids is not None else torch.zeros(g.size(0), self.slice_hidden*2, device=g.device)
        
        # Feature-level gating (Oracle improvement: modulates vuln features directly instead of logit-shift)
        if vuln_features is not None:
            v = self.vuln_mlp(vuln_features)  # [B, 64]
            gate = self.feature_gate(v)  # [B, 64] in (0, 1)
            v = v * (1.0 + self.gate_strength * (gate - 0.5))  # Symmetric modulation
        else:
            v = torch.zeros(g.size(0), 64, device=g.device)
        
        h = torch.cat([g, s, v], dim=1)
        logits = self.classifier(h)
        
        return logits

# ===== TRAINING FUNCTIONS =====
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# Focal Loss (Oracle improvement: emphasizes hard examples)
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        alpha_t = torch.where(targets == 1, self.alpha, 1.0 - self.alpha)
        loss = alpha_t * (1.0 - pt) ** self.gamma * ce_loss
        return loss.mean()

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
    # Focal Loss (Oracle improvement: focuses on hard examples, reduces FN)
    pos_ratio = n_pos / (n_pos + n_neg)
    alpha_pos = (1.0 - pos_ratio) * FOCAL_ALPHA_SCALE  # More weight on minority class
    criterion = FocalLoss(alpha=alpha_pos, gamma=FOCAL_GAMMA).to(DEVICE)
    print(f"Using Focal Loss: alpha={alpha_pos:.3f}, gamma={FOCAL_GAMMA}")
    optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=2, factor=0.5)
    scaler = GradScaler()
    
    best_val_loss = float('inf')
    best_f1, patience_counter = 0, 0
    EARLY_STOP_PATIENCE = 4  # Dừng sớm hơn để tránh overfitting
    model_path = f'{MODEL_DIR}/best_v2_seed{seed}.pt'
    
    # Training history
    history = {
        'train_loss': [], 'train_f1': [], 'train_auc': [],
        'val_loss': [], 'val_f1': [], 'val_auc': [],
        'val_opt_f1': [], 'val_prec': [], 'val_rec': []
    }
    
    for epoch in range(1, MAX_EPOCHS + 1):
        print(f"\n{'='*50}\nEpoch {epoch}/{MAX_EPOCHS} (seed={seed})")
        train_m = train_epoch(model, train_loader, criterion, optimizer, scaler)
        val_m = evaluate(model, val_loader, criterion)
        scheduler.step(val_m['opt_f1'])
        print(f"Train: loss={train_m['loss']:.4f}, F1={train_m['f1']:.4f}, AUC={train_m['auc']:.4f}")
        print(f"Val: loss={val_m['loss']:.4f}, F1={val_m['f1']:.4f}, AUC={val_m['auc']:.4f}, OptF1={val_m['opt_f1']:.4f}, Prec={val_m['opt_prec']:.4f}, Rec={val_m['opt_rec']:.4f}")
        
        # Save history
        history['train_loss'].append(train_m['loss'])
        history['train_f1'].append(train_m['f1'])
        history['train_auc'].append(train_m['auc'])
        history['val_loss'].append(val_m['loss'])
        history['val_f1'].append(val_m['f1'])
        history['val_auc'].append(val_m['auc'])
        history['val_opt_f1'].append(val_m['opt_f1'])
        history['val_prec'].append(val_m['opt_prec'])
        history['val_rec'].append(val_m['opt_rec'])
        
        # Early stopping: val_loss as primary criterion, val_opt_f1 as tie-breaker
        curr_val_loss = val_m['loss']
        curr_opt_f1 = val_m['opt_f1']
        
        improved = False
        if curr_val_loss < best_val_loss - 1e-3:
            improved = True
        elif curr_val_loss <= best_val_loss + 1e-3 and curr_opt_f1 > best_f1:
            improved = True
        
        if improved:
            best_val_loss = curr_val_loss
            best_f1 = curr_opt_f1
            patience_counter = 0
            torch.save(model.state_dict(), model_path)
            print(f"★ Best loss: {best_val_loss:.4f}, F1: {best_f1:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOP_PATIENCE:
                print(f"Early stop at epoch {epoch}!")
                break
    
    # Load best model
    model.load_state_dict(torch.load(model_path, weights_only=False))
    return model, best_f1, history

# ===== MAIN TRAINING LOOP =====
print(f"\n{'='*60}")
print(f"Training {NUM_SEEDS} model(s) with pos_weight_scale={POS_WEIGHT_SCALE}")
print(f"{'='*60}")

models = []
val_probs_list = []
test_probs_list = []
all_histories = []

val_loader = DataLoader(val_ds, batch_size=64, num_workers=2, pin_memory=True)
test_loader = DataLoader(test_ds, batch_size=64, num_workers=2, pin_memory=True)

for seed_idx in range(NUM_SEEDS):
    seed = 42 + seed_idx * 1000
    model, best_f1, history = train_single_model(seed, train_ds, val_ds, n_neg, n_pos, data_config)
    models.append(model)
    all_histories.append(history)
    
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

# Find optimal threshold using Fβ (β=1.5) with precision constraint
THRESHOLD_BETA = 1.5  # Prioritize recall over precision
MIN_PRECISION = 0.78  # Minimum acceptable precision

best_fbeta, best_t = 0, 0.5
for t in np.arange(0.20, 0.80, 0.01):
    preds = (val_probs_cal >= t).astype(int)
    prec = precision_score(val_labels, preds, zero_division=0)
    rec = recall_score(val_labels, preds, zero_division=0)
    if prec < MIN_PRECISION:
        continue  # Skip thresholds that violate precision constraint
    fbeta = (1 + THRESHOLD_BETA**2) * prec * rec / (THRESHOLD_BETA**2 * prec + rec + 1e-8)
    if fbeta > best_fbeta:
        best_fbeta, best_t = fbeta, t

# Fallback to standard F1 if no threshold meets precision constraint
if best_fbeta == 0:
    print("Warning: No threshold meets precision constraint, using F1 optimization")
    for t in np.arange(0.25, 0.75, 0.01):
        f1 = f1_score(val_labels, (val_probs_cal >= t).astype(int))
        if f1 > best_fbeta: best_fbeta, best_t = f1, t

print(f"Validation (ensemble+calibrated): F{THRESHOLD_BETA}={best_fbeta:.4f} at t={best_t:.2f} (min_prec={MIN_PRECISION})")

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
test_fbeta_opt = (1 + THRESHOLD_BETA**2) * test_prec * test_rec / (THRESHOLD_BETA**2 * test_prec + test_rec + 1e-8)
print(f"Test F1 (t=0.5): {test_f1_05:.4f}")
print(f"Test F{THRESHOLD_BETA} (t={best_t:.2f}): {test_fbeta_opt:.4f}")
print(f"Test F1 (t={best_t:.2f}): {test_f1_opt:.4f}")
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
axes[1].set_title(f'Confusion Matrix (t={best_t:.2f})\nF{THRESHOLD_BETA}={test_fbeta_opt:.4f}')
axes[1].set_xlabel('Predicted'); axes[1].set_ylabel('Actual')

plt.tight_layout()
plt.savefig(f'{PLOTS_DIR}/confusion_matrix.png', dpi=150)
plt.show()
print(f"Saved confusion matrix to {PLOTS_DIR}/confusion_matrix.png")

# Save calibrator for inference
import joblib
calibrator_path = f'{MODEL_DIR}/calibrator.joblib'
joblib.dump(calibrator, calibrator_path)
print(f"Saved isotonic calibrator to {calibrator_path}")

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
with open(f'{OUTPUT_DIR}/ensemble_config.json', 'w') as f:
    json.dump(ensemble_config, f, indent=2)
print(f"Saved ensemble config to {OUTPUT_DIR}/ensemble_config.json")

# ===== ADDITIONAL PLOTS =====
from sklearn.metrics import roc_curve, precision_recall_curve, auc

# ROC Curve
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

fpr, tpr, _ = roc_curve(test_labels, test_probs_cal)
roc_auc = auc(fpr, tpr)
axes[0].plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
axes[0].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', alpha=0.5)
axes[0].set_xlim([0.0, 1.0])
axes[0].set_ylim([0.0, 1.05])
axes[0].set_xlabel('False Positive Rate', fontsize=12)
axes[0].set_ylabel('True Positive Rate', fontsize=12)
axes[0].set_title('ROC Curve', fontsize=14, fontweight='bold')
axes[0].legend(loc="lower right")
axes[0].grid(True, alpha=0.3)

# Precision-Recall Curve
precision, recall, _ = precision_recall_curve(test_labels, test_probs_cal)
pr_auc = auc(recall, precision)
axes[1].plot(recall, precision, color='green', lw=2, label=f'PR curve (AUC = {pr_auc:.4f})')
axes[1].axhline(y=test_prec, color='red', linestyle='--', alpha=0.5, label=f'Precision @ t={best_t:.2f} = {test_prec:.4f}')
axes[1].set_xlim([0.0, 1.0])
axes[1].set_ylim([0.0, 1.05])
axes[1].set_xlabel('Recall', fontsize=12)
axes[1].set_ylabel('Precision', fontsize=12)
axes[1].set_title('Precision-Recall Curve', fontsize=14, fontweight='bold')
axes[1].legend(loc="lower left")
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{PLOTS_DIR}/roc_pr_curves.png', dpi=150)
plt.show()
print(f"Saved ROC & PR curves to {PLOTS_DIR}/roc_pr_curves.png")

# Metrics Summary
fig, ax = plt.subplots(figsize=(10, 6))
metrics_names = ['F1 (t=0.5)', 'F1 (optimal)', 'Precision', 'Recall', 'AUC']
metrics_values = [test_f1_05, test_f1_opt, test_prec, test_rec, test_auc]
colors = ['#3498db', '#2ecc71', '#9b59b6', '#e74c3c', '#f39c12']
bars = ax.bar(metrics_names, metrics_values, color=colors, edgecolor='black', linewidth=1.2)
for bar, val in zip(bars, metrics_values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f'{val:.4f}', 
            ha='center', va='bottom', fontsize=12, fontweight='bold')
ax.set_ylim(0, 1.1)
ax.set_ylabel('Score', fontsize=12)
ax.set_title(f'Test Metrics Summary (Ensemble + Calibration, t={best_t:.2f})', fontsize=14, fontweight='bold')
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(f'{PLOTS_DIR}/metrics_summary.png', dpi=150)
plt.show()
print(f"Saved metrics summary to {PLOTS_DIR}/metrics_summary.png")

# ===== TRAINING HISTORY PLOTS =====
print("\nGenerating training history plots...")

# Plot for each seed (or average if multiple seeds)
if len(all_histories) == 1:
    history = all_histories[0]
else:
    # Average histories across seeds
    history = {}
    for key in all_histories[0].keys():
        min_len = min(len(h[key]) for h in all_histories)
        history[key] = [np.mean([h[key][i] for h in all_histories]) for i in range(min_len)]

epochs = range(1, len(history['train_loss']) + 1)

# 1. Loss curve
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

axes[0, 0].plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
axes[0, 0].plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Loss')
axes[0, 0].set_title('Training & Validation Loss', fontweight='bold')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# 2. F1 curve
axes[0, 1].plot(epochs, history['train_f1'], 'b-', label='Train F1', linewidth=2)
axes[0, 1].plot(epochs, history['val_f1'], 'r-', label='Val F1', linewidth=2)
axes[0, 1].plot(epochs, history['val_opt_f1'], 'g--', label='Val OptF1', linewidth=2)
best_epoch = np.argmax(history['val_opt_f1']) + 1
best_f1_val = max(history['val_opt_f1'])
axes[0, 1].axvline(x=best_epoch, color='green', linestyle=':', alpha=0.7, label=f'Best (epoch {best_epoch})')
axes[0, 1].scatter([best_epoch], [best_f1_val], color='green', s=100, zorder=5)
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('F1 Score')
axes[0, 1].set_title('F1 Score over Epochs', fontweight='bold')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# 3. AUC curve
axes[1, 0].plot(epochs, history['train_auc'], 'b-', label='Train AUC', linewidth=2)
axes[1, 0].plot(epochs, history['val_auc'], 'r-', label='Val AUC', linewidth=2)
axes[1, 0].set_xlabel('Epoch')
axes[1, 0].set_ylabel('AUC')
axes[1, 0].set_title('AUC over Epochs', fontweight='bold')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# 4. Precision/Recall curve
axes[1, 1].plot(epochs, history['val_prec'], 'g-', label='Precision', linewidth=2)
axes[1, 1].plot(epochs, history['val_rec'], 'm-', label='Recall', linewidth=2)
axes[1, 1].plot(epochs, history['val_opt_f1'], 'b--', label='Val OptF1', linewidth=2, alpha=0.7)
axes[1, 1].set_xlabel('Epoch')
axes[1, 1].set_ylabel('Score')
axes[1, 1].set_title('Precision, Recall & F1 (Validation)', fontweight='bold')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.suptitle('Training History', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(f'{PLOTS_DIR}/training_history.png', dpi=150)
plt.show()
print(f"Saved training history to {PLOTS_DIR}/training_history.png")

# Save history to JSON
with open(f'{OUTPUT_DIR}/training_history.json', 'w') as f:
    json.dump(history, f, indent=2)
print(f"Saved training history data to {OUTPUT_DIR}/training_history.json")

print(f"\n{'='*60}")
print(f"All outputs saved to: {OUTPUT_DIR}")
print(f"  - Models: {MODEL_DIR}/")
print(f"  - Plots: {PLOTS_DIR}/")
print(f"{'='*60}")
