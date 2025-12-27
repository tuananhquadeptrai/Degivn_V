# C Vulnerability Detection CI/CD

![C Vulnerability Scan](https://github.com/<owner>/<repo>/actions/workflows/c-vuln-scan.yml/badge.svg?branch=main)

> ⚠️ **Thay `<owner>/<repo>` bằng tên GitHub repo thực tế của bạn**

Tự động phát hiện lỗ hổng bảo mật trong code C khi push lên GitHub.

## 🚀 Quick Start

### 1. Setup Git LFS (cho model file)

```bash
# Install Git LFS
sudo apt install git-lfs  # Ubuntu
# hoặc: brew install git-lfs  # macOS

# Setup trong repo
git lfs install
git lfs track "models/*.pt"
git add .gitattributes
```

### 2. Push code lên GitHub

```bash
git add .
git commit -m "Add C vulnerability detection CI/CD"
git push origin main
```

### 3. Sử dụng

Khi bạn push bất kỳ file `.c` hoặc `.h` nào, GitHub Actions sẽ tự động:
1. Chạy model BiGRU để phân tích code
2. Đánh dấu file là **VULNERABLE** hoặc **Clean**
3. Hiển thị kết quả trong tab Actions và Annotations

## 📁 Cấu trúc files

```
.github/workflows/c-vuln-scan.yml  # GitHub Actions workflow
devign_pipeline/
  ├── api/inference.py              # Model inference với HierarchicalBiGRU
  ├── cli/analyze_file.py           # CLI tool để scan file C
  └── src/models/
      └── hierarchical_bigru.py     # Model architecture (từ training)
models/
  ├── best_v2_seed42.pt            # Model 1 (Git LFS)
  ├── best_v2_seed1042.pt          # Model 2 (Git LFS) 
  ├── best_v2_seed2042.pt          # Model 3 (Git LFS)
  ├── config.json                   # Data config (vocab_size, max_len, etc.)
  ├── vocab.json                    # Vocabulary từ training
  └── feature_stats.json            # Feature normalization stats
```

## 🔧 Cách hoạt động

### 1. Tokenization & Normalization
- Tokenize C code thành tokens
- Normalize: `variable_name` → `VAR_0`, literals → `NUM`, `STR`
- Giữ nguyên C keywords và stdlib functions

### 2. Slicing
- Chia code thành 6 slices, mỗi slice tối đa 256 tokens
- Padding nếu code ngắn

### 3. Feature Extraction (26 features)
- `loc`, `stmt_count` - Code metrics
- `dangerous_call_count` - strcpy, memcpy, gets...
- `pointer_deref_*` - Pointer dereference analysis
- `array_access_*` - Array bounds checking
- `malloc_*`, `free_*` - Memory management
- `null_check_*`, `bounds_check_*` - Defense patterns

### 4. Inference
- **Ensemble 3 models** (seeds: 42, 1042, 2042) - average probabilities
- **HierarchicalBiGRU** (từ `03_training_v2.py`):
  - Global encoder: 2-layer BiGRU + Attention
  - Slice encoder: BiGRU + Slice-sequence BiGRU
  - Feature gating mechanism
- Optimal threshold: **0.37** (từ training với Focal Loss)
- Avg softmax probability > 0.37 → **VULNERABLE**

## 🧪 Test locally

```bash
# Scan một file
python -m devign_pipeline.cli.analyze_file --file test.c --json

# Output:
{
  "file": "test.c",
  "vulnerable": true,
  "score": 0.7234,
  "threshold": 0.37,
  "confidence": "medium"
}
```

## 📊 Model Performance

Từ `ensemble_config.json`:
- **F1 Score:** 0.7727
- **Precision:** 0.8022
- **Recall:** 0.7452
- **AUC-ROC:** 0.8783

## ⚠️ Lưu ý

1. **Git LFS required**: Model file ~50MB cần Git LFS
2. **Python 3.10+**: Workflow sử dụng Python 3.10
3. **Dependencies**: torch, pydantic, numpy (xem `requirements.txt`)

## 🔄 Workflow Triggers

- `push` với changes to `**/*.c` hoặc `**/*.h`
- `pull_request` với changes to C files
- Manual trigger: `workflow_dispatch` với optional file path

## 📝 Example Output trong GitHub Actions

```
Analyzing: src/vulnerable.c
::error file=src/vulnerable.c,line=1::VULNERABLE - Score: 0.7234, Confidence: medium

Analyzing: src/safe.c  
::notice file=src/safe.c,line=1::Clean - Score: 0.1523, Confidence: high
```
