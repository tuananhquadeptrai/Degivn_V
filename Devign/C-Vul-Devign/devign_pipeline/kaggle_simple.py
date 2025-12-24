# ============================================
# DEVIGN PREPROCESSING - KAGGLE SIMPLE
# ============================================
# Features:
#   - CFG/DFG-based backward slicing (not window-based fallback)
#   - Vulnerability risk scoring using rule-based analysis
#   - C API whitelist for security-relevant functions
#   - Vocabulary pruning
#
# Chạy: !python /kaggle/input/devign-pipeline/devign_pipeline/kaggle_simple.py
# Input: devign (dataset), devign-pipeline (code)
# Output: processed dataset với vectors và vulnerability features

# %% Cell 1: Setup & Install Dependencies
import subprocess
import sys

subprocess.run([sys.executable, '-m', 'pip', 'install', '-q',
                'tree-sitter', 'tree-sitter-c', 'tree-sitter-cpp',
                'networkx', 'tqdm', 'joblib', 'pyyaml'], check=True)

sys.path.insert(0, '/kaggle/input/devign-pipeline/devign_pipeline')

import os
import re
import json
import pandas as pd
import numpy as np
from tqdm import tqdm
from pathlib import Path
from collections import Counter
from typing import List, Dict, Any, Optional, Tuple

# Paths
DATA_DIR = '/kaggle/input/devign'
OUTPUT_DIR = '/kaggle/working/processed'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Core C keywords to preserve
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
    ==|!=|<=|>=|&&|\|\||\->|\+\+|--|\<\<|\>\>|\+=|-=|\*=|/=|%=|&=|\|=|\^=|\<\<=|\>\>=
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

# Vulnerability feature names for output (V2 - pattern-based with missing defense detection)
VULN_FEATURE_NAMES_V2 = [
    'loc', 'stmt_count',
    'dangerous_call_count', 'dangerous_call_without_check_count', 'dangerous_call_without_check_ratio',
    'pointer_deref_count', 'pointer_deref_without_null_check_count', 'pointer_deref_without_null_check_ratio',
    'array_access_count', 'array_access_without_bounds_check_count', 'array_access_without_bounds_check_ratio',
    'malloc_count', 'malloc_without_free_count', 'malloc_without_free_ratio',
    'free_count', 'free_without_null_check_count', 'free_without_null_check_ratio',
    'unchecked_return_value_count', 'unchecked_return_value_ratio',
    'null_check_count', 'bounds_check_count', 'defense_ratio',
    'dangerous_call_density', 'pointer_deref_density', 'array_access_density', 'null_check_density',
]

print("✓ Setup complete")

# %% Cell 2: Load Data from Parquet Files
train_df = pd.read_parquet(f'{DATA_DIR}/train-00000-of-00001-396a063c42dfdb0a.parquet')
val_df = pd.read_parquet(f'{DATA_DIR}/validation-00000-of-00001-5d4ba937305086b9.parquet')
test_df = pd.read_parquet(f'{DATA_DIR}/test-00000-of-00001-e0e162fa10729371.parquet')

print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
print(f"Columns: {train_df.columns.tolist()}")

# %% Cell 3: Import Pipeline Modules
from src.vuln.vuln_lines import extract_vul_line_numbers
from src.vuln.rules import get_vulnerability_summary, extract_vuln_features_v2, score_vulnerability_risk
from src.vuln.dictionary import get_default_dictionary
from src.slicing.slicer import CodeSlicer, SliceConfig, SliceType
from src.ast.parser import CFamilyParser
from src.graphs.cfg import CFGBuilder
from src.graphs.dfg import DFGBuilder
from src.tokenization.vocab import Vocabulary, VocabConfig

print("✓ Pipeline modules imported")

# %% Cell 4: Initialize Components
# Slicer with BACKWARD slicing (CFG/DFG-based, not window-based)
slice_config = SliceConfig(
    slice_type=SliceType.BACKWARD,
    window_size=15,  # Fallback only
    include_control_deps=True,
    include_data_deps=True,
    max_depth=5,
    remove_comments=True,
    normalize_output=True,
)
slicer = CodeSlicer(slice_config)

# Vulnerability dictionary for feature extraction
vuln_dict = get_default_dictionary()

# Parser for AST/CFG/DFG building
parser = CFamilyParser()
cfg_builder = CFGBuilder()
dfg_builder = DFGBuilder()

print("✓ Components initialized")
print(f"  Slice type: {slice_config.slice_type.value}")
print(f"  Vuln dictionary: {len(vuln_dict)} patterns")

# %% Cell 5: Process Function
def process_sample(row, idx: int) -> Optional[Dict[str, Any]]:
    """Process single code sample with backward slicing and vulnerability features.
    
    Steps:
    1. Extract vulnerability line numbers from vul_lines column
    2. Perform CFG/DFG-based backward slicing using those lines as criterion
    3. Extract vulnerability features from sliced code
    4. Tokenize and normalize
    5. Return tokens, label, and vuln features
    """
    try:
        # Get code
        code = None
        if 'normalized_func' in row.index and pd.notna(row['normalized_func']):
            code = str(row['normalized_func'])
        elif 'func_clean' in row.index and pd.notna(row['func_clean']):
            code = str(row['func_clean'])
        elif 'func' in row.index and pd.notna(row['func']):
            code = str(row['func'])
        
        if not code or len(code) < 10:
            return None
        
        # Extract vulnerability line numbers
        vul_lines_raw = row.get('vul_lines') if hasattr(row, 'get') else (
            row['vul_lines'] if 'vul_lines' in row.index else None
        )
        criterion_lines = extract_vul_line_numbers(vul_lines_raw)
        
        # If no vul_lines, use middle of function as criterion
        if not criterion_lines:
            num_lines = code.count('\n') + 1
            middle = max(1, num_lines // 2)
            criterion_lines = [middle]
        
        # Perform backward slicing
        try:
            code_slice = slicer.slice(code, criterion_lines)
            sliced_code = code_slice.code if code_slice.code else code
        except Exception:
            sliced_code = code
        
        # Extract vulnerability features from sliced code (V2 - pattern-based)
        vuln_features = extract_vuln_features_v2(sliced_code, vuln_dict)
        
        # Also get summary for risk_score and risk_level
        vuln_summary = get_vulnerability_summary(sliced_code, vuln_dict)
        risk_score = vuln_summary.get('risk_score', 0.0)
        risk_level = vuln_summary.get('risk_level', 'none')
        
        # Build feature vector using V2 features directly
        feature_vector = {name: vuln_features.get(name, 0.0) for name in VULN_FEATURE_NAMES_V2}
        
        # Tokenization using multi-char operator aware regex
        tokens_raw = TOKEN_PATTERN.findall(sliced_code)
        
        if len(tokens_raw) < 3:
            return None
        
        # Normalization with API whitelist
        var_map = {}
        var_counter = 0
        norm_tokens = []
        
        for t in tokens_raw:
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
            # Operators and punctuation
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
            'length': len(norm_tokens),
            'vuln_features': feature_vector,
            'risk_score': risk_score,
            'risk_level': risk_level,
            'num_criterion_lines': len(criterion_lines),
        }
    except Exception as e:
        print(f"Error processing sample {idx}: {e}")
        return None

# %% Cell 6: Process All Datasets (train/val/test)
def process_dataset(df: pd.DataFrame, name: str) -> List[Dict]:
    print(f"\nProcessing {name}...")
    results = []
    for idx, row in tqdm(df.iterrows(), total=len(df), desc=name):
        result = process_sample(row, idx)
        if result:
            results.append(result)
    
    # Statistics
    labels = [r['label'] for r in results]
    risk_scores = [r['risk_score'] for r in results]
    
    print(f"  Processed: {len(results)}/{len(df)}")
    print(f"  Labels: 0={labels.count(0)}, 1={labels.count(1)}")
    print(f"  Risk scores: min={min(risk_scores):.3f}, max={max(risk_scores):.3f}, mean={np.mean(risk_scores):.3f}")
    
    return results

train_data = process_dataset(train_df, "train")
val_data = process_dataset(val_df, "val")
test_data = process_dataset(test_df, "test")

# %% Cell 7: Build Vocabulary from Train
print("\nBuilding vocabulary...")
vocab = Vocabulary(VocabConfig(min_freq=2, max_vocab_size=50000))

all_tokens = [d['tokens'] for d in train_data]
vocab.build(iter(all_tokens), show_progress=True)

print(f"Vocabulary size (before pruning): {len(vocab)}")

# %% Cell 8: Vectorize Tokens to input_ids
MAX_LEN = 512

def vectorize_data(data: List[Dict], vocab: Vocabulary, max_len: int = MAX_LEN) -> Dict[str, np.ndarray]:
    """Convert token sequences to padded/truncated integer arrays."""
    input_ids = []
    attention_masks = []
    labels = []
    
    for d in tqdm(data, desc="Vectorizing"):
        ids = vocab.tokens_to_ids(d['tokens'])
        ids = [vocab.bos_id] + ids + [vocab.eos_id]
        
        if len(ids) > max_len:
            ids = ids[:max_len]
        
        mask = [1] * len(ids)
        
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

# %% Cell 9: Build Vulnerability Feature Matrix
def build_vuln_feature_matrix(data: List[Dict]) -> Dict[str, np.ndarray]:
    """Extract vulnerability features into a separate matrix."""
    feature_matrix = []
    
    for d in data:
        vf = d['vuln_features']
        row = [vf.get(name, 0.0) for name in VULN_FEATURE_NAMES_V2]
        feature_matrix.append(row)
    
    return {
        'features': np.array(feature_matrix, dtype=np.float32),
        'feature_names': VULN_FEATURE_NAMES_V2,
    }

print("\nBuilding vulnerability feature matrices...")
train_vuln = build_vuln_feature_matrix(train_data)
val_vuln = build_vuln_feature_matrix(val_data)
test_vuln = build_vuln_feature_matrix(test_data)

print(f"  Train vuln features: {train_vuln['features'].shape}")
print(f"  Val vuln features: {val_vuln['features'].shape}")
print(f"  Test vuln features: {test_vuln['features'].shape}")

# %% Cell 10: Prune Vocabulary
def prune_vocab_and_reindex(vocab: Vocabulary, 
                            vector_sets: List[Dict]) -> Tuple[Vocabulary, List[Dict]]:
    """Prune unused tokens from vocabulary and reindex all vectors."""
    print("\nPruning vocabulary...")
    
    used_ids = set()
    for vectors in vector_sets:
        unique_ids = np.unique(vectors['input_ids'])
        used_ids.update(unique_ids.tolist())
    
    print(f"  Used token IDs: {len(used_ids)} / {len(vocab)}")
    
    special_ids = {vocab.pad_id, vocab.unk_id, vocab.bos_id, vocab.eos_id}
    used_ids.update(special_ids)
    
    sorted_used_ids = sorted(used_ids)
    old_to_new = {old_id: new_id for new_id, old_id in enumerate(sorted_used_ids)}
    
    new_token2id = {}
    new_id2token = []
    new_token_freqs = {}
    
    for old_id in sorted_used_ids:
        if old_id < len(vocab.id2token):
            token = vocab.id2token[old_id]
            new_id = old_to_new[old_id]
            new_token2id[token] = new_id
            new_id2token.append(token)
            if token in vocab.token_freqs:
                new_token_freqs[token] = vocab.token_freqs[token]
    
    orig_unk_id = vocab.unk_id
    orig_vocab_size = len(vocab.token2id)
    
    updated_vector_sets = []
    for vectors in vector_sets:
        new_input_ids = np.vectorize(
            lambda x: old_to_new.get(x, old_to_new[orig_unk_id])
        )(vectors['input_ids'])
        updated_vectors = {
            'input_ids': new_input_ids.astype(np.int32),
            'attention_mask': vectors['attention_mask'],
            'labels': vectors['labels']
        }
        updated_vector_sets.append(updated_vectors)
    
    vocab.token2id = new_token2id
    vocab.id2token = new_id2token
    vocab.token_freqs = new_token_freqs
    
    print(f"  Pruned vocabulary size: {len(new_token2id)}")
    print(f"  Removed {orig_vocab_size - len(new_token2id)} unused tokens")
    
    return vocab, updated_vector_sets

vocab, (train_vectors, val_vectors, test_vectors) = prune_vocab_and_reindex(
    vocab, [train_vectors, val_vectors, test_vectors]
)

# %% Cell 11: Save Outputs
print("\nSaving outputs...")

# Save token vectors
np.savez_compressed(f'{OUTPUT_DIR}/train.npz', **train_vectors)
np.savez_compressed(f'{OUTPUT_DIR}/val.npz', **val_vectors)
np.savez_compressed(f'{OUTPUT_DIR}/test.npz', **test_vectors)

# Save vulnerability features
np.savez_compressed(f'{OUTPUT_DIR}/train_vuln.npz', **train_vuln)
np.savez_compressed(f'{OUTPUT_DIR}/val_vuln.npz', **val_vuln)
np.savez_compressed(f'{OUTPUT_DIR}/test_vuln.npz', **test_vuln)

# Save vocabulary
vocab.save(f'{OUTPUT_DIR}/vocab.json')

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
    'c_api_whitelist': sorted(list(C_API_WHITELIST)),
    'c_keywords': sorted(list(C_KEYWORDS)),
    'vuln_feature_names': list(VULN_FEATURE_NAMES_V2),
    'slice_config': {
        'slice_type': slice_config.slice_type.value,
        'window_size': slice_config.window_size,
        'include_control_deps': slice_config.include_control_deps,
        'include_data_deps': slice_config.include_data_deps,
        'max_depth': slice_config.max_depth,
    },
}

with open(f'{OUTPUT_DIR}/config.json', 'w') as f:
    json.dump(config, f, indent=2)

print(f"\n{'='*60}")
print("✓ DONE! Output saved to /kaggle/working/processed/")
print(f"  Token vectors:")
print(f"    - train.npz: {train_vectors['input_ids'].shape}")
print(f"    - val.npz: {val_vectors['input_ids'].shape}")
print(f"    - test.npz: {test_vectors['input_ids'].shape}")
print(f"  Vulnerability features:")
print(f"    - train_vuln.npz: {train_vuln['features'].shape}")
print(f"    - val_vuln.npz: {val_vuln['features'].shape}")
print(f"    - test_vuln.npz: {test_vuln['features'].shape}")
print(f"  Metadata:")
print(f"    - vocab.json: {len(vocab)} tokens (pruned)")
print(f"    - config.json")
print(f"{'='*60}")

# %% Cell 12: Verify Output
print("\n--- Output Verification ---")

# Verify token data
sample = np.load(f'{OUTPUT_DIR}/train.npz')
print(f"\nToken data (train.npz):")
print(f"  input_ids shape: {sample['input_ids'].shape}")
print(f"  input_ids[0][:20]: {sample['input_ids'][0][:20]}")
print(f"  Labels distribution: 0={sum(sample['labels']==0)}, 1={sum(sample['labels']==1)}")

# Verify vulnerability features
vuln_sample = np.load(f'{OUTPUT_DIR}/train_vuln.npz', allow_pickle=True)
print(f"\nVuln features (train_vuln.npz):")
print(f"  features shape: {vuln_sample['features'].shape}")
print(f"  feature_names: {list(vuln_sample['feature_names'])[:5]}...")
print(f"  Risk score stats: min={vuln_sample['features'][:,0].min():.3f}, "
      f"max={vuln_sample['features'][:,0].max():.3f}, "
      f"mean={vuln_sample['features'][:,0].mean():.3f}")

# Verify vocabulary compactness
max_id_in_data = max(
    train_vectors['input_ids'].max(),
    val_vectors['input_ids'].max(),
    test_vectors['input_ids'].max()
)
print(f"\nVocabulary verification:")
print(f"  Max token ID in data: {max_id_in_data}")
print(f"  Vocab size: {len(vocab)}")
print(f"  Vocab is compact: {max_id_in_data < len(vocab)}")

# Show config summary
with open(f'{OUTPUT_DIR}/config.json', 'r') as f:
    saved_config = json.load(f)
print(f"\nConfig summary:")
print(f"  Slice type: {saved_config['slice_config']['slice_type']}")
print(f"  Vuln features: {len(saved_config['vuln_feature_names'])} features")
print(f"  API whitelist: {len(saved_config['c_api_whitelist'])} functions")
