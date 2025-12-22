# Phân Tích và Cải Tiến Hệ Thống Phát Hiện Lỗ Hổng C Code
## BiGRU Vulnerability Detection - Devign Dataset

**Ngày phân tích:** 22/12/2024  
**Hiệu suất hiện tại:** F1 ~72%, AUC-ROC ~82%  
**Mục tiêu:** F1 ≥ 75%, AUC-ROC ≥ 85%

---

## 📊 I. NHẬN XÉT (Observations)

### 1. Kiến Trúc Hybrid BiGRU + V2 Features

#### ✅ Điểm mạnh
| Component | Mô tả | Đánh giá |
|-----------|-------|----------|
| **Dual-branch architecture** | BiGRU cho tokens + MLP cho V2 features | Thiết kế hợp lý, kết hợp ngữ nghĩa và tri thức tĩnh |
| **Multi-head attention pooling** | 4-6 heads thay vì simple pooling | Giúp tập trung vào đoạn code quan trọng |
| **V2 Features "Missing Defenses"** | Đếm thiếu phòng thủ thay vì chỉ đếm nguy hiểm | Phù hợp với cách audit code thực tế |
| **SWA + Ensemble** | 5-7 models với dropout variations | Tăng robustness và generalization |

#### ⚠️ Điểm yếu
| Vấn đề | Chi tiết | Tác động |
|--------|----------|----------|
| **Không tận dụng đồ thị** | AST/CFG/DFG chỉ được nén thành vector thống kê | Bỏ phí compute cho graph mà không dùng GNN/path encoding |
| **Vocab quá compact** | ~266 tokens với normalize_vars=True | Hạn chế khả năng phân biệt patterns tinh vi |
| **V2 features global** | Không gắn với vị trí trong sequence | BiGRU không biết token nào liên quan đến "missing defense" |
| **Regex-based detection** | Nhiều logic V2 dựa vào regex đơn giản | False positive/negative cao, noise giới hạn trần mô hình |

### 2. Pipeline Preprocessing (10 bước)

```
load → vuln_features → ast → cfg → dfg → slice → tokenize → normalize → vocab → vectorize
```

#### ✅ Điểm mạnh
- **Checkpointing & Chunking**: Resume được khi bị interrupt
- **GC sau mỗi chunk**: Quản lý bộ nhớ tốt trên Kaggle
- **joblib parallelization**: Tăng tốc xử lý
- **Đầy đủ graph stats**: `cfg_block_count`, `dfg_node_count`, etc.

#### ⚠️ Điểm yếu
| Vấn đề | Giải thích |
|--------|------------|
| **Tokenization tuyến tính** | AST/CFG/DFG không được dùng để xây sequence structure cao hơn |
| **Slicing cứng nhắc** | Một slice duy nhất, bỏ qua multi-view (forward + backward) |
| **window_size cố định** | Không adaptive theo độ dài hàm |
| **Slicing không biết V2** | Không tận dụng V2 features để chọn đoạn code quan trọng |

### 3. Code Slicing Strategy

#### Hiện tại
- **Backward slice**: Default, dựa trên `vul_lines` 
- **Forward slice**: Có sẵn nhưng ít dùng
- **Window fallback**: ±15 dòng khi parse fail

#### Vấn đề
```
Hàm dài (LOC > 200) → Slice quá to → Nhiều noise
Hàm ngắn (LOC < 50) → Window gần như full hàm → Không có ích
Không có vul_lines → Fallback full code → Attention khó học
```

### 4. V2 Features - Missing Defenses

#### Danh sách features hiện có
```python
# Dangerous calls
dangerous_call_without_check_count/ratio

# Pointer operations  
pointer_deref_without_null_check_count/ratio

# Array access
array_access_without_bounds_check_count/ratio

# Memory management
malloc_without_free_count/ratio
free_without_null_check_count/ratio

# Return values
unchecked_return_value_count/ratio

# Defense metrics
defense_ratio
null_check_density
```

#### Phân tích
- **Ưu điểm**: Capture được pattern quan trọng, gần với cách human audit
- **Nhược điểm**: 
  - Không path-sensitive, không fully control/data-flow-sensitive
  - Tất cả features đều global trên toàn hàm/slice
  - Không có feature selection hay scaling chuyên biệt

### 5. Training Configuration Analysis

#### Progression của các configs
```
TrainConfig → LargeTrainConfig → RegularizedConfig → ImprovedConfig 
    → EnhancedConfig → OptimizedConfig → RefinedConfig → FinalConfig 
    → QuickWinConfig → AdvancedConfig → AdvancedConfigV2 → AdvancedConfigV3
```

#### Kết quả thực nghiệm
| Config | F1 | AUC | Precision | Recall | Ghi chú |
|--------|-----|-----|-----------|--------|---------|
| Baseline | ~72% | ~82% | ~49% | ~90% | High recall, low precision |
| AdvancedV2 | ~66% | ~80% | ~75-81% | ~50-59% | Recall collapsed |
| AdvancedV3 | ~72% | ~82% | - | - | Restored but plateaued |

#### Kết luận
> **Bottleneck không còn ở optimizer/hyperparam** mà ở **representation** (model & features, cách encode AST/CFG/DFG/slices)

---

## 🚀 II. CẢI TIẾN (Improvements)

### Mức 1: Thay đổi ít - Tác động nhanh ⚡

#### 1.1 Multi-Slice / Multi-Instance Learning

**Ý tưởng**: Thay vì 1 slice/hàm, tạo nhiều slices với perspectives khác nhau

```python
# Pseudo-code
class MultiSliceDataset(Dataset):
    def __getitem__(self, idx):
        sample = self.data[idx]
        
        # Tạo nhiều slices
        slices = []
        for vul_line in sample['vul_lines']:
            slices.append(backward_slice(sample['code'], vul_line))
            slices.append(forward_slice(sample['code'], vul_line))
        
        # Fallback window nếu không có vul_lines
        if not slices:
            slices.append(window_slice(sample['code']))
        
        return {
            'slice_tokens': [tokenize(s) for s in slices],
            'slice_count': len(slices),
            'v2_features': sample['v2_features'],
            'label': sample['label']
        }

# Model với slice-level attention
class MultiSliceModel(nn.Module):
    def forward(self, batch):
        # Encode mỗi slice
        slice_embeddings = []
        for slice_tokens in batch['slice_tokens']:
            h = self.bigru_encoder(slice_tokens)
            slice_embed = self.token_attention_pool(h)
            slice_embeddings.append(slice_embed)
        
        # Attention over slices (multi-instance)
        stacked = torch.stack(slice_embeddings)
        final_embed = self.slice_attention_pool(stacked)
        
        return self.classifier(final_embed)
```

**Lợi ích**:
- Tách riêng contexts: backward capture nguyên nhân, forward capture hậu quả
- Model có thể học bỏ qua slice noisy thông qua attention

#### 1.2 Distance-to-Criterion Token Feature

**Ý tưởng**: Thêm positional feature cho biết token gần vul_line bao nhiêu

```python
def compute_distance_feature(tokens, token_lines, criterion_lines):
    """
    Tính khoảng cách từ mỗi token đến vul_line gần nhất
    """
    distances = []
    for line in token_lines:
        min_dist = min(abs(line - c) for c in criterion_lines)
        distances.append(min_dist)
    
    # Normalize và embed
    max_dist = 20  # clamp
    normalized = [min(d / max_dist, 1.0) for d in distances]
    return normalized

# Trong model
class EnhancedBiGRU(nn.Module):
    def __init__(self, ...):
        self.dist_embedding = nn.Linear(1, 16)  # Hoặc embedding table
        
    def forward(self, tokens, distance_features):
        token_embed = self.token_embedding(tokens)
        dist_embed = self.dist_embedding(distance_features.unsqueeze(-1))
        
        # Concatenate
        combined = torch.cat([token_embed, dist_embed], dim=-1)
        return self.bigru(combined)
```

**Lợi ích**: Giúp attention ưu tiên tokens gần vùng nghi vấn

#### 1.3 Adaptive Slicing Parameters

```python
def get_adaptive_slice_config(code):
    loc = len(code.split('\n'))
    
    if loc > 200:
        return SliceConfig(window_size=10, max_depth=3)
    elif loc > 100:
        return SliceConfig(window_size=15, max_depth=4)
    elif loc < 50:
        return SliceConfig(window_size=loc, max_depth=5)  # Full function
    else:
        return SliceConfig(window_size=15, max_depth=5)  # Default
```

---

### Mức 2: Cải thiện V2 Features 📊

#### 2.1 Feature Scaling & Selection

```python
from sklearn.preprocessing import StandardScaler
import numpy as np

# Các ratio features thường rất skewed (nhiều 0, một ít 1)
RATIO_FEATURES = [
    'malloc_without_free_ratio',
    'free_without_null_check_ratio', 
    'array_access_without_bounds_check_ratio',
    'dangerous_call_without_check_ratio',
    'pointer_deref_without_null_check_ratio',
]

def transform_v2_features(features):
    transformed = {}
    for key, value in features.items():
        if key in RATIO_FEATURES:
            # Log transform for skewed distributions
            transformed[key] = np.log1p(value * 10)  # Scale up before log
        elif '_count' in key:
            # Log transform for counts
            transformed[key] = np.log1p(value)
        else:
            transformed[key] = value
    return transformed

# Feature importance analysis
from sklearn.ensemble import RandomForestClassifier

def analyze_feature_importance(X_v2, y):
    """Chạy RF để đo feature importance"""
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_v2, y)
    
    importance = dict(zip(X_v2.columns, rf.feature_importances_))
    return sorted(importance.items(), key=lambda x: -x[1])
```

#### 2.2 Localized V2 Features (per-slice)

```python
def extract_v2_features_per_slice(slice_code, original_code, dictionary):
    """
    Compute V2 features trên slice thay vì full function
    → Giảm noise, tăng alignment với context BiGRU nhìn
    """
    # Local features on slice
    slice_features = extract_vuln_features_v2(slice_code, dictionary)
    
    # Global features for context
    global_features = extract_vuln_features_v2(original_code, dictionary)
    
    # Combined: slice features + relative metrics
    combined = {
        # Slice-level
        **{f'slice_{k}': v for k, v in slice_features.items()},
        
        # Slice-to-global ratios
        'slice_loc_ratio': slice_features['loc'] / max(global_features['loc'], 1),
        'slice_danger_concentration': (
            slice_features['dangerous_call_count'] / 
            max(global_features['dangerous_call_count'], 1)
        ),
    }
    return combined
```

#### 2.3 Additional Graph-Level Features

```python
def compute_graph_complexity_features(cfg_stats, dfg_stats):
    """
    Thêm complexity metrics từ CFG/DFG stats có sẵn
    """
    cfg_blocks = cfg_stats.get('block_count', 0)
    cfg_edges = cfg_stats.get('edge_count', 0)
    dfg_nodes = dfg_stats.get('node_count', 0)
    dfg_edges = dfg_stats.get('edge_count', 0)
    dfg_defs = dfg_stats.get('def_count', 0)
    dfg_uses = dfg_stats.get('use_count', 0)
    
    return {
        # Cyclomatic complexity proxy
        'cyclomatic_complexity': cfg_edges - cfg_blocks + 2,
        
        # DFG density
        'dfg_avg_degree': (2 * dfg_edges) / max(dfg_nodes, 1),
        
        # Def-use ratio (high ratio = complex data flow)
        'dfg_def_use_ratio': dfg_defs / max(dfg_uses, 1),
        
        # CFG complexity
        'cfg_branching_factor': cfg_edges / max(cfg_blocks, 1),
    }
```

---

### Mức 3: Nâng cấp Model Architecture 🏗️

#### 3.1 Hierarchical Encoding (Statement-Level)

```python
class HierarchicalBiGRU(nn.Module):
    """
    Level 1: Encode mỗi statement riêng
    Level 2: BiGRU over statement embeddings
    """
    def __init__(self, vocab_size, embed_dim=64, stmt_hidden=128, doc_hidden=192):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        
        # Statement-level encoder (CNN hoặc BiGRU nhỏ)
        self.stmt_encoder = nn.LSTM(embed_dim, stmt_hidden//2, 
                                     bidirectional=True, batch_first=True)
        
        # Document-level encoder  
        self.doc_encoder = nn.GRU(stmt_hidden, doc_hidden//2,
                                   bidirectional=True, batch_first=True)
        
        # Attention pooling
        self.stmt_attention = nn.MultiheadAttention(stmt_hidden, 4)
        self.doc_attention = nn.MultiheadAttention(doc_hidden, 4)
        
    def forward(self, statements_batch):
        """
        statements_batch: [batch, max_stmts, max_tokens]
        """
        B, S, T = statements_batch.shape
        
        # Encode each statement
        stmt_embeds = []
        for s in range(S):
            tokens = statements_batch[:, s, :]  # [B, T]
            x = self.embedding(tokens)  # [B, T, E]
            h, _ = self.stmt_encoder(x)  # [B, T, H]
            # Attention pool over tokens
            pooled = self._attention_pool(h, self.stmt_attention)  # [B, H]
            stmt_embeds.append(pooled)
        
        # Stack statements: [B, S, H]
        stmt_seq = torch.stack(stmt_embeds, dim=1)
        
        # Encode statement sequence
        doc_h, _ = self.doc_encoder(stmt_seq)  # [B, S, D]
        
        # Final attention pool
        output = self._attention_pool(doc_h, self.doc_attention)  # [B, D]
        
        return output
```

**Lợi ích**:
- Match cấu trúc logic của AST/CFG tốt hơn
- Giảm sequence length (512 tokens → ~50 statements)
- Dễ học dependencies dài hơn

#### 3.2 Light-weight Transformer Encoder

```python
class LightTransformerEncoder(nn.Module):
    """
    2-3 layer Transformer, có thể dùng song song với BiGRU
    """
    def __init__(self, vocab_size, d_model=256, nhead=4, num_layers=3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_encoding = PositionalEncoding(d_model, max_len=512)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=d_model*4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Attention pooling
        self.pool = nn.Linear(d_model, 1)
        
    def forward(self, tokens, attention_mask):
        x = self.embedding(tokens)
        x = self.pos_encoding(x)
        
        # Create transformer mask
        mask = (attention_mask == 0)  # True where padded
        
        h = self.transformer(x, src_key_padding_mask=mask)
        
        # Attention pooling
        weights = self.pool(h).squeeze(-1)  # [B, T]
        weights = weights.masked_fill(mask, float('-inf'))
        weights = F.softmax(weights, dim=-1)
        
        output = (h * weights.unsqueeze(-1)).sum(dim=1)
        return output
```

#### 3.3 GNN Branch trên DFG (Branch thứ 3)

```python
import torch_geometric
from torch_geometric.nn import GCNConv, GATConv, global_attention_pool

class DFGGraphBranch(nn.Module):
    """
    Nhỏ gọn: 2 layer GCN/GAT, hidden 64-96
    """
    def __init__(self, node_features=32, hidden=64, output_dim=128):
        super().__init__()
        self.node_embed = nn.Linear(node_features, hidden)
        
        self.conv1 = GATConv(hidden, hidden, heads=2, concat=False, dropout=0.3)
        self.conv2 = GATConv(hidden, hidden, heads=2, concat=False, dropout=0.3)
        
        # Global attention pooling
        self.gate_nn = nn.Linear(hidden, 1)
        self.output = nn.Linear(hidden, output_dim)
        
    def forward(self, x, edge_index, batch):
        """
        x: node features [N, F]
        edge_index: [2, E]
        batch: batch assignment [N]
        """
        x = F.relu(self.node_embed(x))
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.3, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        
        # Global attention pooling
        pooled = global_attention_pool(x, batch, self.gate_nn)
        
        return self.output(pooled)

# Combined model
class HybridModelWithGNN(nn.Module):
    def __init__(self, ...):
        self.token_branch = BiGRUEncoder(...)
        self.v2_branch = MLPBranch(...)
        self.graph_branch = DFGGraphBranch(...)
        
        # Fusion
        total_dim = token_dim + v2_dim + graph_dim
        self.classifier = nn.Sequential(
            nn.Linear(total_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 2)
        )
        
    def forward(self, tokens, v2_features, dfg_data):
        token_embed = self.token_branch(tokens)
        v2_embed = self.v2_branch(v2_features)
        graph_embed = self.graph_branch(
            dfg_data.x, dfg_data.edge_index, dfg_data.batch
        )
        
        combined = torch.cat([token_embed, v2_embed, graph_embed], dim=-1)
        return self.classifier(combined)
```

---

### Mức 4: Training Strategy Improvements 📈

#### 4.1 K-Fold Cross-Validation Ensemble

```python
from sklearn.model_selection import StratifiedKFold

def train_kfold_ensemble(data, labels, n_folds=5):
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    fold_models = []
    oof_predictions = np.zeros(len(labels))
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(data, labels)):
        print(f"Training Fold {fold+1}/{n_folds}")
        
        train_data = data[train_idx]
        val_data = data[val_idx]
        
        model = build_model(config)
        train_fold(model, train_data, val_data, ...)
        
        # Out-of-fold predictions
        oof_predictions[val_idx] = model.predict_proba(val_data)[:, 1]
        
        fold_models.append(model)
    
    # Ensemble: average predictions
    return fold_models, oof_predictions
```

**Lợi ích**: Giảm variance do split, thường tăng AUC/F1 vài điểm

#### 4.2 Probability Calibration

```python
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression

def calibrate_model(model, val_loader):
    """Temperature scaling hoặc Isotonic regression"""
    
    # Collect logits and labels
    all_logits = []
    all_labels = []
    
    model.eval()
    with torch.no_grad():
        for batch in val_loader:
            logits = model(batch)
            all_logits.append(logits[:, 1].cpu().numpy())  # Positive class logit
            all_labels.append(batch['labels'].cpu().numpy())
    
    logits = np.concatenate(all_logits)
    labels = np.concatenate(all_labels)
    
    # Fit isotonic regression
    ir = IsotonicRegression(out_of_bounds='clip')
    probs = 1 / (1 + np.exp(-logits))  # Sigmoid
    ir.fit(probs, labels)
    
    return ir

def predict_calibrated(model, data, calibrator):
    logits = model(data)
    probs = torch.sigmoid(logits[:, 1]).cpu().numpy()
    calibrated_probs = calibrator.transform(probs)
    return calibrated_probs
```

**Lợi ích**: AUC tốt nhưng F1 chưa tốt thường do calibration kém → cải thiện 1-2 điểm F1

#### 4.3 Curriculum Learning

```python
def compute_sample_difficulty(v2_features):
    """
    Difficulty score dựa trên V2 features
    Samples với ratio cao = "dễ" (rõ ràng vulnerable)
    Samples với ratio thấp = "khó" (subtle)
    """
    danger_score = (
        v2_features['dangerous_call_without_check_ratio'] +
        v2_features['pointer_deref_without_null_check_ratio'] +
        v2_features['array_access_without_bounds_check_ratio']
    ) / 3
    
    # Invert: high danger = easy, low danger = hard
    difficulty = 1 - danger_score
    return difficulty

def curriculum_sampler(dataset, epoch, max_epochs):
    """
    Giai đoạn đầu: focus easy samples
    Giai đoạn sau: thêm dần hard samples
    """
    difficulties = [compute_sample_difficulty(s['v2_features']) for s in dataset]
    
    # Progress ratio
    progress = epoch / max_epochs
    
    # Sampling weights: easy samples get higher weight early
    weights = []
    for d in difficulties:
        if progress < 0.3:
            # Early: prefer easy (difficulty < 0.5)
            w = 1.0 if d < 0.5 else 0.3
        elif progress < 0.6:
            # Mid: balanced
            w = 1.0
        else:
            # Late: slight preference for hard samples
            w = 1.5 if d > 0.5 else 1.0
        weights.append(w)
    
    return WeightedRandomSampler(weights, len(weights))
```

---

### Mức 5: Advanced Techniques 🔬

#### 5.1 Self-Supervised Pretraining

```python
class MaskedLanguageModel(nn.Module):
    """
    Pretrain BiGRU/Transformer với Masked LM trên unlabeled code
    """
    def __init__(self, vocab_size, d_model=256):
        super().__init__()
        self.encoder = BiGRUEncoder(vocab_size, d_model)
        self.mlm_head = nn.Linear(d_model * 2, vocab_size)  # BiGRU hidden*2
        
    def forward(self, tokens, mask_positions):
        h = self.encoder(tokens)  # [B, T, H]
        masked_h = h[mask_positions]  # [N_masked, H]
        logits = self.mlm_head(masked_h)  # [N_masked, vocab_size]
        return logits

def pretrain_mlm(model, unlabeled_data, epochs=10):
    """
    Mask 15% tokens, predict original
    """
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    for epoch in range(epochs):
        for batch in unlabeled_data:
            tokens = batch['tokens']
            
            # Create mask (15% of non-padding tokens)
            mask = create_random_mask(tokens, mask_ratio=0.15)
            masked_tokens = tokens.clone()
            masked_tokens[mask] = MASK_TOKEN_ID
            
            # Forward
            logits = model(masked_tokens, mask)
            loss = F.cross_entropy(logits, tokens[mask])
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
```

#### 5.2 Knowledge Distillation từ CodeBERT

```python
def distill_from_codebert(student_model, codebert, train_loader):
    """
    Train student (BiGRU) để mimic CodeBERT embeddings
    """
    # Freeze CodeBERT
    codebert.eval()
    for p in codebert.parameters():
        p.requires_grad = False
    
    # Distillation loss
    def distill_loss(student_embed, teacher_embed, temperature=2.0):
        # Cosine similarity loss
        cos_loss = 1 - F.cosine_similarity(student_embed, teacher_embed).mean()
        
        # MSE loss
        mse_loss = F.mse_loss(student_embed, teacher_embed)
        
        return cos_loss + mse_loss
    
    optimizer = optim.Adam(student_model.parameters(), lr=5e-4)
    
    for batch in train_loader:
        # Teacher embedding
        with torch.no_grad():
            teacher_embed = codebert.encode(batch['raw_code'])
        
        # Student embedding
        student_embed = student_model.get_embedding(batch['tokens'])
        
        # Distillation + task loss
        d_loss = distill_loss(student_embed, teacher_embed)
        task_loss = F.cross_entropy(student_model(batch['tokens']), batch['labels'])
        
        total_loss = 0.5 * d_loss + 0.5 * task_loss
        
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
```

---

## 📋 III. ROADMAP ĐỀ XUẤT

### Phase 1: Quick Wins (1-2 tuần) ⚡

| Task | Expected Gain | Effort |
|------|---------------|--------|
| 1.1 Multi-slice + slice attention | +1-2% F1 | Medium |
| 1.2 Token distance-to-criterion | +0.5-1% F1 | Low |
| 2.1 Feature scaling (log-transform) | +0.5% F1 | Low |
| 4.2 Probability calibration | +1% F1 | Low |

### Phase 2: Structural Improvements (2-3 tuần) 🏗️

| Task | Expected Gain | Effort |
|------|---------------|--------|
| 2.2 Localized V2 features | +1-2% F1/AUC | Medium |
| 4.1 K-fold cross-validation | +1% F1/AUC | Medium |
| 3.1 Hierarchical encoding | +1-2% F1 | Medium-High |

### Phase 3: Advanced Upgrades (4+ tuần) 🚀

| Task | Expected Gain | Effort |
|------|---------------|--------|
| 3.2 Light Transformer encoder | +1-2% F1/AUC | High |
| 3.3 GNN branch trên DFG | +1-2% AUC | High |
| 5.1 Self-supervised pretraining | +2-3% F1/AUC | High |

---

## 📊 IV. TỔNG KẾT

### Nguyên nhân chính của plateau F1 ~72% / AUC ~82%

1. **Representation chưa đủ mạnh**: BiGRU + MLP không tận dụng hết cấu trúc đồ thị
2. **Single-view slicing**: Chỉ một slice, bỏ qua multi-perspective
3. **Global V2 features**: Không liên kết với vị trí trong sequence
4. **Vocabulary quá compact**: 266 tokens hạn chế phân biệt subtle patterns

### Khuyến nghị ưu tiên cao nhất

```
1. Multi-slice + slice-level attention (1.1) 
   + Token distance-to-criterion (1.2)
   → Đây là upgrade có tác động lớn nhất trong setting này

2. Feature scaling + V2 trên slice thay vì full function (2.1 + 2.2)
   → Tận dụng tri thức V2 tốt hơn, giảm noise

3. Cross-validation + calibration (4.1 + 4.2)
   → Cải thiện đáng kể F1/AUC mà không thay kiến trúc
```

### Target Achievement Forecast

| Metric | Current | After Phase 1 | After Phase 2 | After Phase 3 |
|--------|---------|---------------|---------------|---------------|
| F1 | 72% | 74-75% | 76-77% | 78-80% |
| AUC-ROC | 82% | 83-84% | 85-86% | 87-89% |

---

*Document generated by Oracle analysis - 22/12/2024*
