# ============================================
# DEVIGN PREPROCESSING - SIMPLE VERSION
# ============================================
# Chạy: !python /kaggle/input/devign-pipeline/devign_pipeline/kaggle_simple.py
# Input: devign (dataset), devign-pipeline (code)
# Output: processed dataset với vectors

# %% Cell 1: Setup
import subprocess
import sys

# Install dependencies
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 
                'tree-sitter', 'tree-sitter-c', 'tree-sitter-cpp', 
                'networkx', 'tqdm', 'joblib'], check=True)

# Add source path
sys.path.insert(0, '/kaggle/input/devign-pipeline/devign_pipeline')

import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from pathlib import Path

# Paths
DATA_DIR = '/kaggle/input/devign'
OUTPUT_DIR = '/kaggle/working/processed'
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("✓ Setup complete")

# %% Cell 2: Load Data
train_df = pd.read_parquet(f'{DATA_DIR}/train-00000-of-00001-396a063c42dfdb0a.parquet')
val_df = pd.read_parquet(f'{DATA_DIR}/validation-00000-of-00001-5d4ba937305086b9.parquet')
test_df = pd.read_parquet(f'{DATA_DIR}/test-00000-of-00001-e0e162fa10729371.parquet')

print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
print(f"Columns: {train_df.columns.tolist()}")

# %% Cell 3: Import Pipeline Modules
from src.tokenization.tokenizer import CTokenizer
from src.tokenization.normalization import CodeNormalizer
from src.tokenization.vocab import Vocabulary, VocabConfig
from src.slicing.slicer import CodeSlicer, SliceConfig

print("✓ Modules imported")

# %% Cell 4: Initialize Components
tokenizer = CTokenizer()
normalizer = CodeNormalizer()
slicer = CodeSlicer(SliceConfig(window_size=15))

print("✓ Components initialized")

# %% Cell 5: Process Function
def process_sample(row, idx):
    """Process single code sample"""
    try:
        # Get code - pandas Series uses [] not .get()
        code = None
        if 'normalized_func' in row.index and pd.notna(row['normalized_func']):
            code = str(row['normalized_func'])
        elif 'func_clean' in row.index and pd.notna(row['func_clean']):
            code = str(row['func_clean'])
        elif 'func' in row.index and pd.notna(row['func']):
            code = str(row['func'])
        
        if not code or len(code) < 10:
            return None
        
        # Simple tokenization using regex (faster, more reliable)
        import re
        # Split on whitespace and punctuation, keep tokens
        tokens_raw = re.findall(r'[a-zA-Z_][a-zA-Z0-9_]*|[0-9]+|[^\s\w]', code)
        
        if len(tokens_raw) < 3:
            return None
        
        # Simple normalization
        var_map = {}
        var_counter = 0
        norm_tokens = []
        
        for t in tokens_raw:
            # Skip very short tokens
            if len(t) < 1:
                continue
            # Keywords - keep as is
            if t in {'if', 'else', 'while', 'for', 'return', 'int', 'char', 'void', 
                     'struct', 'switch', 'case', 'break', 'continue', 'const', 'static',
                     'unsigned', 'signed', 'long', 'short', 'double', 'float', 'sizeof',
                     'NULL', 'true', 'false', 'typedef', 'enum', 'union', 'goto', 'extern'}:
                norm_tokens.append(t)
            # Numbers
            elif t.isdigit():
                norm_tokens.append('NUM')
            # Operators and punctuation
            elif not t[0].isalpha() and t[0] != '_':
                norm_tokens.append(t)
            # Identifiers - normalize
            else:
                if t not in var_map:
                    var_map[t] = f'VAR_{var_counter}'
                    var_counter += 1
                norm_tokens.append(var_map[t])
        
        # Get label
        label = 1 if ('target' in row.index and row['target']) else 0
        
        return {
            'id': idx,
            'tokens': norm_tokens,
            'label': label,
            'length': len(norm_tokens)
        }
    except Exception as e:
        print(f"Error processing sample {idx}: {e}")
        return None

# %% Cell 6: Process All Data
def process_dataset(df, name):
    print(f"\nProcessing {name}...")
    results = []
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        result = process_sample(row, idx)
        if result:
            results.append(result)
    print(f"  Processed: {len(results)}/{len(df)}")
    return results

train_data = process_dataset(train_df, "train")
val_data = process_dataset(val_df, "val")
test_data = process_dataset(test_df, "test")

# %% Cell 7: Build Vocabulary (from train only)
print("\nBuilding vocabulary...")
vocab = Vocabulary(VocabConfig(min_freq=2, max_vocab_size=50000))

all_tokens = [d['tokens'] for d in train_data]
vocab.build(iter(all_tokens), show_progress=True)

print(f"Vocabulary size: {len(vocab)}")
vocab.save(f'{OUTPUT_DIR}/vocab.json')

# %% Cell 8: Vectorize
MAX_LEN = 512

def vectorize_data(data, vocab, max_len=MAX_LEN):
    input_ids = []
    attention_masks = []
    labels = []
    
    for d in tqdm(data):
        # Convert tokens to ids
        ids = vocab.tokens_to_ids(d['tokens'])
        
        # Add BOS/EOS
        ids = [vocab.bos_id] + ids + [vocab.eos_id]
        
        # Truncate
        if len(ids) > max_len:
            ids = ids[:max_len]
        
        # Create attention mask
        mask = [1] * len(ids)
        
        # Pad
        pad_len = max_len - len(ids)
        ids = ids + [vocab.pad_id] * pad_len
        mask = mask + [0] * pad_len
        
        input_ids.append(ids)
        attention_masks.append(mask)
        labels.append(d['label'])
    
    return {
        'input_ids': np.array(input_ids, dtype=np.int32),
        'attention_mask': np.array(attention_masks, dtype=np.int32),
        'labels': np.array(labels, dtype=np.int32)
    }

print("\nVectorizing...")
train_vectors = vectorize_data(train_data, vocab)
val_vectors = vectorize_data(val_data, vocab)
test_vectors = vectorize_data(test_data, vocab)

# %% Cell 9: Save Output
print("\nSaving...")
np.savez_compressed(f'{OUTPUT_DIR}/train.npz', **train_vectors)
np.savez_compressed(f'{OUTPUT_DIR}/val.npz', **val_vectors)
np.savez_compressed(f'{OUTPUT_DIR}/test.npz', **test_vectors)

# Save config
import json
config = {
    'vocab_size': len(vocab),
    'max_len': MAX_LEN,
    'train_samples': len(train_data),
    'val_samples': len(val_data),
    'test_samples': len(test_data),
}
with open(f'{OUTPUT_DIR}/config.json', 'w') as f:
    json.dump(config, f, indent=2)

print(f"\n{'='*50}")
print("✓ DONE! Output saved to /kaggle/working/processed/")
print(f"  - train.npz: {train_vectors['input_ids'].shape}")
print(f"  - val.npz: {val_vectors['input_ids'].shape}")
print(f"  - test.npz: {test_vectors['input_ids'].shape}")
print(f"  - vocab.json: {len(vocab)} tokens")
print(f"  - config.json")
print(f"{'='*50}")

# %% Cell 10: Verify
sample = np.load(f'{OUTPUT_DIR}/train.npz')
print("\nSample verification:")
print(f"  input_ids[0][:20]: {sample['input_ids'][0][:20]}")
print(f"  Labels distribution: 0={sum(sample['labels']==0)}, 1={sum(sample['labels']==1)}")
