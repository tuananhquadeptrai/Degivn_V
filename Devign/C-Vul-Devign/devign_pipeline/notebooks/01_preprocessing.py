# %% [markdown]
# # Devign Dataset Preprocessing Pipeline
# 
# This notebook runs the complete preprocessing pipeline for C/C++ vulnerability detection.
# 
# **Environment**: Kaggle with 2x NVIDIA T4 GPU (32GB total VRAM), 13GB RAM
# 
# **Steps**:
# 1. Install dependencies & Setup
# 2. Load and explore data
# 3. Extract vulnerability features
# 4. Parse AST
# 5. Build CFG/DFG
# 6. Slice code
# 7. Tokenize
# 8. Normalize
# 9. Build vocabulary
# 10. Vectorize

# %% [markdown]
# ## 1. Setup & Installation

# %%
# Copy pipeline code to working directory
!cp -r /kaggle/input/devign-pipeline/devign_pipeline /kaggle/working/

# Install dependencies
!pip install -q tree-sitter tree-sitter-c tree-sitter-cpp networkx tqdm joblib pyyaml

# %%
# Check GPU
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU count: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")

# %% [markdown]
# ## 2. Configuration

# %%
import sys
import os

# Kaggle paths
DATA_DIR = '/kaggle/input/devign'
WORKING_DIR = '/kaggle/working'
sys.path.insert(0, '/kaggle/working/devign_pipeline')

OUTPUT_DIR = os.path.join(WORKING_DIR, 'processed')
CHECKPOINT_DIR = os.path.join(WORKING_DIR, 'checkpoints')

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# Pipeline configuration
CONFIG = {
    'chunk_size': 2000,
    'n_jobs': 4,
    'max_seq_length': 512,
    'min_freq': 2,
    'max_vocab_size': 50000,
    'window_size': 15,
}

print(f"Data dir: {DATA_DIR}")
print(f"Output dir: {OUTPUT_DIR}")

# %% [markdown]
# ## 3. Data Loading & Exploration

# %%
from src.data.loader import DevignLoader
from src.data.explore import compute_statistics, generate_eda_report

loader = DevignLoader(DATA_DIR, chunk_size=CONFIG['chunk_size'])
splits = loader.get_splits()
print("Dataset splits:", splits)

# Load train data for EDA
train_df = loader.load_all(split='train')
print(f"\nTrain set: {len(train_df)} samples")

stats = compute_statistics(train_df)
print("\nLabel distribution:")
print(f"  Non-vulnerable: {stats['label_counts'].get(0, 0)}")
print(f"  Vulnerable: {stats['label_counts'].get(1, 0)}")
print(f"\nCode length stats:")
print(f"  Mean: {stats['code_length_mean']:.1f} chars")
print(f"  Max: {stats['code_length_max']} chars")

# %% [markdown]
# ## 4. Run Full Pipeline

# %%
from src.pipeline.preprocess import PreprocessPipeline, PipelineConfig

config = PipelineConfig(
    data_dir=DATA_DIR,
    output_dir=OUTPUT_DIR,
    checkpoint_dir=CHECKPOINT_DIR,
    **CONFIG
)

pipeline = PreprocessPipeline(config)

# Check current status
status = pipeline.get_status()
print("Pipeline status:", status)

# %%
# Run pipeline (will resume from last checkpoint if interrupted)
pipeline.run(split='train')

# %% [markdown]
# ## 5. Process Validation & Test Sets

# %%
# Use vocabulary from train set for val/test
pipeline.run(split='val')
pipeline.run(split='test')

# %% [markdown]
# ## 6. Verify Output

# %%
import numpy as np
from pathlib import Path

output_files = list(Path(OUTPUT_DIR).glob('**/*.npz'))
print(f"Output files: {len(output_files)}")

# Load sample and verify
if output_files:
    sample = np.load(output_files[0])
    print("\nSample file keys:", list(sample.keys()))
    print(f"input_ids shape: {sample['input_ids'].shape}")
    print(f"labels shape: {sample['labels'].shape}")

# %% [markdown]
# ## 7. Create DataLoader for Training

# %%
from src.pipeline.dataset import DevignTorchDataset, create_dataloader

train_paths = list(Path(OUTPUT_DIR).glob('train/*.npz'))
val_paths = list(Path(OUTPUT_DIR).glob('val/*.npz'))

print(f"Train chunks: {len(train_paths)}")
print(f"Val chunks: {len(val_paths)}")

if train_paths:
    train_dataset = DevignTorchDataset([str(p) for p in train_paths])
    train_loader = create_dataloader(
        train_dataset,
        batch_size=64,
        shuffle=True,
        num_workers=4
    )
    
    # Test batch
    batch = next(iter(train_loader))
    print("\nBatch shape:")
    print(f"  input_ids: {batch['input_ids'].shape}")
    print(f"  attention_mask: {batch['attention_mask'].shape}")
    print(f"  labels: {batch['labels'].shape}")

# %% [markdown]
# ## 8. Save Vocabulary & Config

# %%
from src.tokenization.vocab import Vocabulary

vocab_path = os.path.join(OUTPUT_DIR, 'vocab.json')
if os.path.exists(vocab_path):
    vocab = Vocabulary.load(vocab_path)
    print(f"Vocabulary size: {len(vocab)}")
    print(f"Most common tokens: {vocab.get_most_common(20)}")

config.save(os.path.join(OUTPUT_DIR, 'config.yaml'))
print("\nConfig and vocabulary saved!")

# %% [markdown]
# ## Next Steps
# 
# 1. Use `02_training.ipynb` to train vulnerability detection model
# 2. Models supported:
#    - Sequence-based: LSTM, Transformer, CodeBERT
#    - Graph-based: GCN, GAT, GGNN on CFG/DFG
