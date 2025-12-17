# ============================================
# DEVIGN PREPROCESSING - SIMPLE VERSION V2
# ============================================
# Improvements:
#   - C API whitelist for security-relevant functions
#   - Multi-char operator aware tokenization
#   - String/char literal normalization
#   - Vocabulary pruning to remove unused tokens
# Chạy: !python /kaggle/input/devign-pipeline/devign_pipeline/kaggle_simple_v2.py
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
import re
import json
import pandas as pd
import numpy as np
from tqdm import tqdm
from pathlib import Path

# Paths
DATA_DIR = '/kaggle/input/devign'
OUTPUT_DIR = '/kaggle/working/processed'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Core C keywords we want to preserve
C_KEYWORDS = {
    'if', 'else', 'while', 'for', 'return', 'int', 'char', 'void',
    'struct', 'switch', 'case', 'break', 'continue', 'const', 'static',
    'unsigned', 'signed', 'long', 'short', 'double', 'float', 'sizeof',
    'NULL', 'true', 'false', 'typedef', 'enum', 'union', 'goto', 'extern'
}

# Security-relevant C APIs to preserve (NOT normalize to VAR_x)
C_API_WHITELIST = {
    # memory management
    'malloc', 'calloc', 'realloc', 'free', 'alloca',
    # string / buffer functions
    'strcpy', 'strncpy', 'strcat', 'strncat', 'strlen', 'strcmp', 'strncmp',
    'sprintf', 'snprintf', 'vsprintf', 'vsnprintf',
    'memcpy', 'memmove', 'memset', 'memcmp', 'memchr',
    # input functions
    'gets', 'fgets', 'getc', 'getchar', 'fgetc',
    'scanf', 'sscanf', 'fscanf', 'fread',
    # output
    'printf', 'fprintf', 'vprintf', 'puts', 'fputs', 'fwrite',
    # file I/O
    'read', 'write', 'open', 'close', 'fopen', 'fclose',
}

# Multi-char operator aware tokenization pattern
TOKEN_PATTERN = re.compile(
    r'''
    # String literal (handles escapes)
    "(?:\\.|[^"\\])*"
    |
    # Char literal (handles escapes)
    '(?:\\.|[^'\\])*'
    |
    # Multi-character operators
    ==|!=|<=|>=|&&|\|\||->|\+\+|--|<<|>>|\+=|-=|\*=|/=|%=|&=|\|=|\^=|<<=|>>=
    |
    # Identifiers
    [A-Za-z_][A-Za-z0-9_]*
    |
    # Numbers (hex, decimal)
    0[xX][0-9a-fA-F]+|[0-9]+
    |
    # Any other single non-whitespace character
    [^\s]
    ''',
    re.VERBOSE
)

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
    """Process single code sample with improved normalization.
    
    Normalization rules:
    - String literals → 'STR'
    - Char literals → 'CHAR'
    - C_KEYWORDS → preserved as-is
    - C_API_WHITELIST → preserved as-is (security-relevant functions)
    - Numbers (decimal, hex) → 'NUM'
    - Other identifiers → VAR_0, VAR_1, etc. (consistent within sample)
    - Operators/punctuation → preserved as-is
    """
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
        
        # Tokenization using multi-char operator aware regex
        tokens_raw = TOKEN_PATTERN.findall(code)
        
        if len(tokens_raw) < 3:
            return None
        
        # Normalization with API whitelist
        var_map = {}
        var_counter = 0
        norm_tokens = []
        
        for t in tokens_raw:
            # Skip empty tokens
            if len(t) < 1:
                continue
            
            # String literals → 'STR'
            if t.startswith('"') and t.endswith('"'):
                norm_tokens.append('STR')
            # Char literals → 'CHAR'
            elif t.startswith("'") and t.endswith("'"):
                norm_tokens.append('CHAR')
            # C Keywords - keep as is
            elif t in C_KEYWORDS:
                norm_tokens.append(t)
            # Security-relevant C APIs - keep as is
            elif t in C_API_WHITELIST:
                norm_tokens.append(t)
            # Numbers (decimal or hex)
            elif t.isdigit() or (t.startswith('0x') or t.startswith('0X')):
                norm_tokens.append('NUM')
            # Operators and punctuation (starts with non-alpha, non-underscore)
            elif not t[0].isalpha() and t[0] != '_':
                norm_tokens.append(t)
            # Other identifiers - normalize to VAR_x
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

print(f"Vocabulary size (before pruning): {len(vocab)}")
vocab.save(f'{OUTPUT_DIR}/vocab.json')

# %% Cell 8: Vectorize
MAX_LEN = 512

def vectorize_data(data, vocab, max_len=MAX_LEN):
    """Convert token sequences to padded/truncated integer arrays.
    
    Each sequence is wrapped with BOS/EOS tokens, then:
    - Truncated to max_len if too long (including BOS/EOS)
    - Padded with PAD tokens to max_len if too short
    """
    input_ids = []
    attention_masks = []
    labels = []
    
    for d in tqdm(data):
        # Convert tokens to ids
        ids = vocab.tokens_to_ids(d['tokens'])
        
        # Add BOS/EOS
        ids = [vocab.bos_id] + ids + [vocab.eos_id]
        
        # Truncate (sequence already includes BOS/EOS, so truncate to max_len)
        if len(ids) > max_len:
            ids = ids[:max_len]
        
        # Create attention mask (1 for real tokens, 0 for padding)
        mask = [1] * len(ids)
        
        # Pad to max_len with PAD tokens
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

# %% Cell 9: Vocabulary Pruning
def prune_vocab_and_reindex(vocab, vector_sets):
    """Prune unused tokens from vocabulary and reindex all vectors.
    
    Args:
        vocab: Vocabulary object with token2id and id2token
        vector_sets: List of vector dicts, each with 'input_ids' array
        
    Returns:
        vocab: Updated vocabulary with compact token IDs
        vector_sets: List of updated vector dicts with reindexed input_ids
    """
    print("\nPruning vocabulary...")
    
    # Count which token IDs are actually used across all datasets
    used_ids = set()
    for vectors in vector_sets:
        unique_ids = np.unique(vectors['input_ids'])
        used_ids.update(unique_ids.tolist())
    
    print(f"  Used token IDs: {len(used_ids)} / {len(vocab)}")
    
    # Special tokens to always keep (by their IDs)
    special_ids = {vocab.pad_id, vocab.unk_id, vocab.bos_id, vocab.eos_id}
    used_ids.update(special_ids)
    
    # Build old_id -> new_id mapping
    # Sort used IDs to maintain consistent ordering
    sorted_used_ids = sorted(used_ids)
    old_to_new = {old_id: new_id for new_id, old_id in enumerate(sorted_used_ids)}
    
    # Build new token2id and id2token (id2token is a list in Vocabulary class)
    new_token2id = {}
    new_id2token = []
    new_token_freqs = {}
    
    # vocab.id2token is a list, access by index
    for old_id in sorted_used_ids:
        if old_id < len(vocab.id2token):
            token = vocab.id2token[old_id]
            new_id = old_to_new[old_id]
            new_token2id[token] = new_id
            new_id2token.append(token)
            if token in vocab.token_freqs:
                new_token_freqs[token] = vocab.token_freqs[token]
    
    # Store original special token IDs before updating
    orig_pad_id = vocab.pad_id
    orig_unk_id = vocab.unk_id
    orig_bos_id = vocab.bos_id
    orig_eos_id = vocab.eos_id
    orig_vocab_size = len(vocab.token2id)
    
    # Rewrite input_ids arrays with new compact IDs
    updated_vector_sets = []
    for vectors in vector_sets:
        new_input_ids = np.vectorize(lambda x: old_to_new.get(x, old_to_new[orig_unk_id]))(vectors['input_ids'])
        updated_vectors = {
            'input_ids': new_input_ids.astype(np.int32),
            'attention_mask': vectors['attention_mask'],
            'labels': vectors['labels']
        }
        updated_vector_sets.append(updated_vectors)
    
    # Update vocab object
    vocab.token2id = new_token2id
    vocab.id2token = new_id2token
    vocab.token_freqs = new_token_freqs
    
    print(f"  Pruned vocabulary size: {len(new_token2id)}")
    print(f"  Removed {orig_vocab_size - len(new_token2id)} unused tokens")
    
    return vocab, updated_vector_sets

# Apply pruning
vocab, (train_vectors, val_vectors, test_vectors) = prune_vocab_and_reindex(
    vocab, [train_vectors, val_vectors, test_vectors]
)

# Save pruned vocabulary
vocab.save(f'{OUTPUT_DIR}/vocab.json')
print(f"Vocabulary size (after pruning): {len(vocab)}")

# %% Cell 10: Save Output
print("\nSaving...")
np.savez_compressed(f'{OUTPUT_DIR}/train.npz', **train_vectors)
np.savez_compressed(f'{OUTPUT_DIR}/val.npz', **val_vectors)
np.savez_compressed(f'{OUTPUT_DIR}/test.npz', **test_vectors)

# Save config
config = {
    'vocab_size': len(vocab),
    'max_len': MAX_LEN,
    'train_samples': len(train_data),
    'val_samples': len(val_data),
    'test_samples': len(test_data),
    'special_tokens': {
        'pad_id': vocab.pad_id,
        'unk_id': vocab.unk_id,
        'bos_id': vocab.bos_id,
        'eos_id': vocab.eos_id,
    },
    'c_api_whitelist': list(C_API_WHITELIST),
    'c_keywords': list(C_KEYWORDS),
}
with open(f'{OUTPUT_DIR}/config.json', 'w') as f:
    json.dump(config, f, indent=2)

print(f"\n{'='*50}")
print("✓ DONE! Output saved to /kaggle/working/processed/")
print(f"  - train.npz: {train_vectors['input_ids'].shape}")
print(f"  - val.npz: {val_vectors['input_ids'].shape}")
print(f"  - test.npz: {test_vectors['input_ids'].shape}")
print(f"  - vocab.json: {len(vocab)} tokens (pruned)")
print(f"  - config.json")
print(f"{'='*50}")

# %% Cell 11: Verify
sample = np.load(f'{OUTPUT_DIR}/train.npz')
print("\nSample verification:")
print(f"  input_ids[0][:20]: {sample['input_ids'][0][:20]}")
print(f"  Labels distribution: 0={sum(sample['labels']==0)}, 1={sum(sample['labels']==1)}")

# Verify vocab is compact
max_id_in_data = max(
    train_vectors['input_ids'].max(),
    val_vectors['input_ids'].max(),
    test_vectors['input_ids'].max()
)
print(f"  Max token ID in data: {max_id_in_data}")
print(f"  Vocab size: {len(vocab)}")
print(f"  Vocab is compact: {max_id_in_data < len(vocab)}")
