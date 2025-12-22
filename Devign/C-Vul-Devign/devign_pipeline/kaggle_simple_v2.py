# ============================================
# DEVIGN PREPROCESSING - KAGGLE SIMPLE V2
# ============================================
# Improvements over V1:
#   1. Multi-slice: backward + forward slicing with slice-level attention support
#   2. Slice-level V2 features: computed per slice + relative metrics
#   3. Backward compatible: old keys preserved for existing model
#
# Chạy: !python /kaggle/input/devign-pipeline/devign_pipeline/kaggle_simple_v2.py
# Input: devign (dataset), devign-pipeline (code)
# Output: processed dataset với vectors, slice vectors, và vulnerability features

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

# ============================================
# CONSTANTS
# ============================================

# Multi-slice config
MAX_SLICES = 6          # Max slices per sample (backward + forward)
SLICE_MAX_LEN = 256     # Max tokens per slice

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

# Relative feature names (slice vs global)
VULN_FEATURE_NAMES_REL = [
    f"{name}_rel_ratio" for name in VULN_FEATURE_NAMES_V2
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

# Backward slicer
backward_config = SliceConfig(
    slice_type=SliceType.BACKWARD,
    window_size=15,
    include_control_deps=True,
    include_data_deps=True,
    max_depth=5,
    remove_comments=True,
    normalize_output=True,
)
backward_slicer = CodeSlicer(backward_config)

# Forward slicer (NEW)
forward_config = SliceConfig(
    slice_type=SliceType.FORWARD,
    window_size=15,
    include_control_deps=True,
    include_data_deps=True,
    max_depth=5,
    remove_comments=True,
    normalize_output=True,
)
forward_slicer = CodeSlicer(forward_config)

# Vulnerability dictionary for feature extraction
vuln_dict = get_default_dictionary()

# Parser for AST/CFG/DFG building
parser = CFamilyParser()
cfg_builder = CFGBuilder()
dfg_builder = DFGBuilder()

print("✓ Components initialized")
print(f"  Backward slicer: {backward_config.slice_type.value}")
print(f"  Forward slicer: {forward_config.slice_type.value}")
print(f"  Max slices per sample: {MAX_SLICES}")
print(f"  Vuln dictionary: {len(vuln_dict)} patterns")

# %% Cell 5: Helper Functions

def normalize_tokens(tokens_raw: List[str]) -> List[str]:
    """Normalize tokens with API whitelist preservation."""
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
    
    return norm_tokens


def group_criterion_lines(criterion_lines: List[int], max_gap: int = 3) -> List[List[int]]:
    """Group nearby criterion lines together."""
    if not criterion_lines:
        return []
    
    groups = []
    current = []
    for l in sorted(criterion_lines):
        if not current or l - current[-1] <= max_gap:
            current.append(l)
        else:
            groups.append(current)
            current = [l]
    if current:
        groups.append(current)
    return groups


def compute_relative_features(slice_feats: List[float], global_feats: List[float]) -> List[float]:
    """Compute relative metrics: slice / global ratio."""
    eps = 1e-6
    rel = []
    for s_val, g_val in zip(slice_feats, global_feats):
        ratio = s_val / (g_val + eps) if g_val > eps else (1.0 if s_val > 0 else 0.0)
        rel.append(min(ratio, 10.0))  # Clip to avoid extreme values
    return rel


# %% Cell 6: Process Function (V2 - Multi-slice + Slice-level V2)

def process_sample_v2(row, idx: int) -> Optional[Dict[str, Any]]:
    """Process single code sample with multi-slice and slice-level V2 features.
    
    Improvements over V1:
    1. Multi-slice: backward + forward for each criterion line group
    2. Slice-level V2: features computed per slice
    3. Relative metrics: slice vs global comparison
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
        
        # ============================================
        # IMPROVEMENT 1: Multi-slice (backward + forward)
        # ============================================
        slices = []
        slice_types = []  # 'backward' or 'forward'
        
        # Group criterion lines
        groups = group_criterion_lines(criterion_lines)
        
        for group in groups[:3]:  # Limit groups to avoid too many slices
            # Backward slice
            try:
                backward_slice = backward_slicer.slice(code, group)
                if backward_slice.code and len(backward_slice.code.strip()) > 10:
                    slices.append(backward_slice.code)
                    slice_types.append('backward')
            except Exception:
                pass
            
            # Forward slice
            try:
                forward_slice = forward_slicer.slice(code, group)
                if forward_slice.code and len(forward_slice.code.strip()) > 10:
                    # Avoid duplicate if forward == backward
                    if not slices or forward_slice.code != slices[-1]:
                        slices.append(forward_slice.code)
                        slice_types.append('forward')
            except Exception:
                pass
        
        # Fallback: if no slices, use full code
        if not slices:
            slices = [code]
            slice_types = ['full']
        
        # Limit to MAX_SLICES
        slices = slices[:MAX_SLICES]
        slice_types = slice_types[:MAX_SLICES]
        
        # ============================================
        # IMPROVEMENT 2: Slice-level V2 features
        # ============================================
        
        # Global V2 features (from first slice, for backward compatibility)
        primary_slice = slices[0]
        vuln_features_global = extract_vuln_features_v2(primary_slice, vuln_dict)
        vuln_summary = get_vulnerability_summary(primary_slice, vuln_dict)
        risk_score = vuln_summary.get('risk_score', 0.0)
        risk_level = vuln_summary.get('risk_level', 'none')
        
        # Build global feature vector (for backward compatibility)
        feature_vector = {name: vuln_features_global.get(name, 0.0) for name in VULN_FEATURE_NAMES_V2}
        global_vec = [vuln_features_global.get(name, 0.0) for name in VULN_FEATURE_NAMES_V2]
        
        # Slice-level V2 features
        slice_vuln_features = []
        slice_relative_features = []
        
        for sl in slices:
            vf = extract_vuln_features_v2(sl, vuln_dict)
            slice_vec = [vf.get(name, 0.0) for name in VULN_FEATURE_NAMES_V2]
            slice_vuln_features.append(slice_vec)
            
            # Relative metrics
            rel_vec = compute_relative_features(slice_vec, global_vec)
            slice_relative_features.append(rel_vec)
        
        # ============================================
        # Tokenization for each slice
        # ============================================
        slice_tokens = []
        for sl in slices:
            tokens_raw = TOKEN_PATTERN.findall(sl)
            norm_tokens = normalize_tokens(tokens_raw)
            if len(norm_tokens) >= 3:
                slice_tokens.append(norm_tokens)
            else:
                slice_tokens.append(['<empty>'])
        
        # Primary tokens (backward compatible - from first slice)
        primary_tokens = slice_tokens[0] if slice_tokens else ['<empty>']
        
        if len(primary_tokens) < 3:
            return None
        
        # Get label
        label = 1 if ('target' in row.index and row['target']) else 0
        
        return {
            # Backward compatible fields
            'id': idx,
            'tokens': primary_tokens,
            'label': label,
            'length': len(primary_tokens),
            'vuln_features': feature_vector,
            'risk_score': risk_score,
            'risk_level': risk_level,
            'num_criterion_lines': len(criterion_lines),
            
            # NEW: Multi-slice data
            'slice_tokens': slice_tokens,
            'slice_types': slice_types,
            'slice_count': len(slice_tokens),
            
            # NEW: Slice-level V2 features
            'slice_vuln_features': slice_vuln_features,
            'slice_relative_features': slice_relative_features,
        }
    except Exception as e:
        print(f"Error processing sample {idx}: {e}")
        return None


# %% Cell 7: Process All Datasets (train/val/test)
def process_dataset(df: pd.DataFrame, name: str) -> List[Dict]:
    print(f"\nProcessing {name}...")
    results = []
    for idx, row in tqdm(df.iterrows(), total=len(df), desc=name):
        result = process_sample_v2(row, idx)
        if result:
            results.append(result)
    
    # Statistics
    labels = [r['label'] for r in results]
    risk_scores = [r['risk_score'] for r in results]
    slice_counts = [r['slice_count'] for r in results]
    
    print(f"  Processed: {len(results)}/{len(df)}")
    print(f"  Labels: 0={labels.count(0)}, 1={labels.count(1)}")
    print(f"  Risk scores: min={min(risk_scores):.3f}, max={max(risk_scores):.3f}, mean={np.mean(risk_scores):.3f}")
    print(f"  Slices per sample: min={min(slice_counts)}, max={max(slice_counts)}, mean={np.mean(slice_counts):.2f}")
    
    return results

train_data = process_dataset(train_df, "train")
val_data = process_dataset(val_df, "val")
test_data = process_dataset(test_df, "test")

# %% Cell 8: Build Vocabulary from Train
print("\nBuilding vocabulary...")
vocab = Vocabulary(VocabConfig(min_freq=2, max_vocab_size=50000))

# Build from all slice tokens (not just primary)
all_tokens = []
for d in train_data:
    for st in d['slice_tokens']:
        all_tokens.append(st)

vocab.build(iter(all_tokens), show_progress=True)
print(f"Vocabulary size (before pruning): {len(vocab)}")

# %% Cell 9: Vectorize Primary Tokens (backward compatible)
MAX_LEN = 512

def vectorize_data(data: List[Dict], vocab: Vocabulary, max_len: int = MAX_LEN) -> Dict[str, np.ndarray]:
    """Convert token sequences to padded/truncated integer arrays."""
    input_ids = []
    attention_masks = []
    labels = []
    
    for d in tqdm(data, desc="Vectorizing primary"):
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

print("\nVectorizing primary tokens...")
train_vectors = vectorize_data(train_data, vocab)
val_vectors = vectorize_data(val_data, vocab)
test_vectors = vectorize_data(test_data, vocab)

# %% Cell 10: Vectorize Slices (NEW)

def vectorize_slices(data: List[Dict], vocab: Vocabulary, 
                     max_slices: int = MAX_SLICES, 
                     max_len: int = SLICE_MAX_LEN) -> Dict[str, np.ndarray]:
    """Convert multi-slice tokens to padded arrays.
    
    Output shape: [num_samples, max_slices, max_len]
    """
    slice_input_ids = []
    slice_attention_masks = []
    slice_counts = []
    
    for d in tqdm(data, desc="Vectorizing slices"):
        sample_slice_ids = []
        sample_slice_masks = []
        
        for slice_tokens in d.get('slice_tokens', [])[:max_slices]:
            ids = vocab.tokens_to_ids(slice_tokens)
            ids = [vocab.bos_id] + ids + [vocab.eos_id]
            
            if len(ids) > max_len:
                ids = ids[:max_len]
            
            mask = [1] * len(ids)
            
            pad_len = max_len - len(ids)
            ids = ids + [vocab.pad_id] * pad_len
            mask = mask + [0] * pad_len
            
            sample_slice_ids.append(ids)
            sample_slice_masks.append(mask)
        
        # Pad slices to max_slices
        actual_count = len(sample_slice_ids)
        while len(sample_slice_ids) < max_slices:
            sample_slice_ids.append([vocab.pad_id] * max_len)
            sample_slice_masks.append([0] * max_len)
        
        slice_input_ids.append(sample_slice_ids)
        slice_attention_masks.append(sample_slice_masks)
        slice_counts.append(actual_count)
    
    return {
        'slice_input_ids': np.array(slice_input_ids, dtype=np.int32),
        'slice_attention_mask': np.array(slice_attention_masks, dtype=np.int32),
        'slice_count': np.array(slice_counts, dtype=np.int32),
    }

print("\nVectorizing slices...")
train_slice_vectors = vectorize_slices(train_data, vocab)
val_slice_vectors = vectorize_slices(val_data, vocab)
test_slice_vectors = vectorize_slices(test_data, vocab)

print(f"  Train slice shape: {train_slice_vectors['slice_input_ids'].shape}")
print(f"  Val slice shape: {val_slice_vectors['slice_input_ids'].shape}")
print(f"  Test slice shape: {test_slice_vectors['slice_input_ids'].shape}")

# %% Cell 11: Build Vulnerability Feature Matrices

def build_vuln_feature_matrix(data: List[Dict]) -> Dict[str, np.ndarray]:
    """Extract global vulnerability features (backward compatible)."""
    feature_matrix = []
    
    for d in data:
        vf = d['vuln_features']
        row = [vf.get(name, 0.0) for name in VULN_FEATURE_NAMES_V2]
        feature_matrix.append(row)
    
    return {
        'features': np.array(feature_matrix, dtype=np.float32),
        'feature_names': np.array(VULN_FEATURE_NAMES_V2, dtype=object),
    }

def build_slice_vuln_feature_tensor(data: List[Dict], max_slices: int = MAX_SLICES) -> Dict[str, np.ndarray]:
    """Build slice-level V2 features and relative metrics (NEW)."""
    slice_feature_tensor = []
    rel_feature_tensor = []
    
    num_features = len(VULN_FEATURE_NAMES_V2)
    num_rel_features = len(VULN_FEATURE_NAMES_REL)
    
    for d in data:
        svf = d.get('slice_vuln_features', [])
        rvf = d.get('slice_relative_features', [])
        
        svf = svf[:max_slices]
        rvf = rvf[:max_slices]
        
        # Padding
        while len(svf) < max_slices:
            svf.append([0.0] * num_features)
        while len(rvf) < max_slices:
            rvf.append([0.0] * num_features)  # rel features same length
        
        slice_feature_tensor.append(svf)
        rel_feature_tensor.append(rvf)
    
    return {
        'slice_vuln_features': np.array(slice_feature_tensor, dtype=np.float32),
        'slice_rel_features': np.array(rel_feature_tensor, dtype=np.float32),
        'slice_vuln_feature_names': np.array(VULN_FEATURE_NAMES_V2, dtype=object),
        'slice_rel_feature_names': np.array(VULN_FEATURE_NAMES_REL, dtype=object),
    }

print("\nBuilding vulnerability feature matrices...")

# Global features (backward compatible)
train_vuln = build_vuln_feature_matrix(train_data)
val_vuln = build_vuln_feature_matrix(val_data)
test_vuln = build_vuln_feature_matrix(test_data)

# Slice-level features (NEW)
train_slice_vuln = build_slice_vuln_feature_tensor(train_data)
val_slice_vuln = build_slice_vuln_feature_tensor(val_data)
test_slice_vuln = build_slice_vuln_feature_tensor(test_data)

print(f"  Train vuln features: {train_vuln['features'].shape}")
print(f"  Train slice vuln features: {train_slice_vuln['slice_vuln_features'].shape}")

# %% Cell 12: Prune Vocabulary
def prune_vocab_and_reindex(vocab: Vocabulary, 
                            vector_sets: List[Dict],
                            slice_vector_sets: List[Dict]) -> Tuple[Vocabulary, List[Dict], List[Dict]]:
    """Prune unused tokens from vocabulary and reindex all vectors."""
    print("\nPruning vocabulary...")
    
    used_ids = set()
    
    # From primary vectors
    for vectors in vector_sets:
        unique_ids = np.unique(vectors['input_ids'])
        used_ids.update(unique_ids.tolist())
    
    # From slice vectors
    for vectors in slice_vector_sets:
        unique_ids = np.unique(vectors['slice_input_ids'])
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
    
    # Reindex primary vectors
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
    
    # Reindex slice vectors
    updated_slice_sets = []
    for vectors in slice_vector_sets:
        new_slice_ids = np.vectorize(
            lambda x: old_to_new.get(x, old_to_new[orig_unk_id])
        )(vectors['slice_input_ids'])
        updated_vectors = {
            'slice_input_ids': new_slice_ids.astype(np.int32),
            'slice_attention_mask': vectors['slice_attention_mask'],
            'slice_count': vectors['slice_count']
        }
        updated_slice_sets.append(updated_vectors)
    
    vocab.token2id = new_token2id
    vocab.id2token = new_id2token
    vocab.token_freqs = new_token_freqs
    
    print(f"  Pruned vocabulary size: {len(new_token2id)}")
    print(f"  Removed {orig_vocab_size - len(new_token2id)} unused tokens")
    
    return vocab, updated_vector_sets, updated_slice_sets

vocab, (train_vectors, val_vectors, test_vectors), (train_slice_vectors, val_slice_vectors, test_slice_vectors) = prune_vocab_and_reindex(
    vocab, 
    [train_vectors, val_vectors, test_vectors],
    [train_slice_vectors, val_slice_vectors, test_slice_vectors]
)

# %% Cell 13: Save Outputs
print("\nSaving outputs...")

# Save primary token vectors (backward compatible)
np.savez_compressed(f'{OUTPUT_DIR}/train.npz', 
    **train_vectors,
    **train_slice_vectors,  # NEW: slice data included
)
np.savez_compressed(f'{OUTPUT_DIR}/val.npz', 
    **val_vectors,
    **val_slice_vectors,
)
np.savez_compressed(f'{OUTPUT_DIR}/test.npz', 
    **test_vectors,
    **test_slice_vectors,
)

# Save vulnerability features (backward compatible + NEW slice features)
np.savez_compressed(f'{OUTPUT_DIR}/train_vuln.npz', 
    **train_vuln,
    **train_slice_vuln,  # NEW
)
np.savez_compressed(f'{OUTPUT_DIR}/val_vuln.npz', 
    **val_vuln,
    **val_slice_vuln,
)
np.savez_compressed(f'{OUTPUT_DIR}/test_vuln.npz', 
    **test_vuln,
    **test_slice_vuln,
)

# Save vocabulary
vocab.save(f'{OUTPUT_DIR}/vocab.json')

# Save config
config = {
    'version': 'v2',  # NEW
    'vocab_size': len(vocab),
    'max_len': MAX_LEN,
    'max_slices': MAX_SLICES,  # NEW
    'slice_max_len': SLICE_MAX_LEN,  # NEW
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
    'vuln_rel_feature_names': list(VULN_FEATURE_NAMES_REL),  # NEW
    'slice_config': {
        'backward': {
            'slice_type': 'backward',
            'window_size': backward_config.window_size,
            'include_control_deps': backward_config.include_control_deps,
            'include_data_deps': backward_config.include_data_deps,
            'max_depth': backward_config.max_depth,
        },
        'forward': {  # NEW
            'slice_type': 'forward',
            'window_size': forward_config.window_size,
            'include_control_deps': forward_config.include_control_deps,
            'include_data_deps': forward_config.include_data_deps,
            'max_depth': forward_config.max_depth,
        },
    },
}

with open(f'{OUTPUT_DIR}/config.json', 'w') as f:
    json.dump(config, f, indent=2)

print(f"\n{'='*60}")
print("✓ DONE! Output saved to /kaggle/working/processed/")
print(f"  Token vectors (backward compatible + NEW slice data):")
print(f"    - train.npz: input_ids {train_vectors['input_ids'].shape}, slice_input_ids {train_slice_vectors['slice_input_ids'].shape}")
print(f"    - val.npz: input_ids {val_vectors['input_ids'].shape}, slice_input_ids {val_slice_vectors['slice_input_ids'].shape}")
print(f"    - test.npz: input_ids {test_vectors['input_ids'].shape}, slice_input_ids {test_slice_vectors['slice_input_ids'].shape}")
print(f"  Vulnerability features (backward compatible + NEW slice V2):")
print(f"    - train_vuln.npz: global {train_vuln['features'].shape}, slice {train_slice_vuln['slice_vuln_features'].shape}")
print(f"    - val_vuln.npz: global {val_vuln['features'].shape}, slice {val_slice_vuln['slice_vuln_features'].shape}")
print(f"    - test_vuln.npz: global {test_vuln['features'].shape}, slice {test_slice_vuln['slice_vuln_features'].shape}")
print(f"  Metadata:")
print(f"    - vocab.json: {len(vocab)} tokens (pruned)")
print(f"    - config.json (version: v2)")
print(f"{'='*60}")

# %% Cell 14: Verify Output
print("\n--- Output Verification ---")

# Verify token data
sample = np.load(f'{OUTPUT_DIR}/train.npz')
print(f"\nToken data (train.npz):")
print(f"  input_ids shape: {sample['input_ids'].shape}")
print(f"  slice_input_ids shape: {sample['slice_input_ids'].shape}")
print(f"  slice_count: min={sample['slice_count'].min()}, max={sample['slice_count'].max()}, mean={sample['slice_count'].mean():.2f}")
print(f"  Labels distribution: 0={sum(sample['labels']==0)}, 1={sum(sample['labels']==1)}")

# Verify vulnerability features
vuln_sample = np.load(f'{OUTPUT_DIR}/train_vuln.npz', allow_pickle=True)
print(f"\nVuln features (train_vuln.npz):")
print(f"  Global features shape: {vuln_sample['features'].shape}")
print(f"  Slice features shape: {vuln_sample['slice_vuln_features'].shape}")
print(f"  Slice rel features shape: {vuln_sample['slice_rel_features'].shape}")

# Verify vocabulary compactness
max_id_primary = train_vectors['input_ids'].max()
max_id_slices = train_slice_vectors['slice_input_ids'].max()
max_id_in_data = max(max_id_primary, max_id_slices)
print(f"\nVocabulary verification:")
print(f"  Max token ID in data: {max_id_in_data}")
print(f"  Vocab size: {len(vocab)}")
print(f"  Vocab is compact: {max_id_in_data < len(vocab)}")

# Show config summary
with open(f'{OUTPUT_DIR}/config.json', 'r') as f:
    saved_config = json.load(f)
print(f"\nConfig summary (V2):")
print(f"  Version: {saved_config['version']}")
print(f"  Max slices: {saved_config['max_slices']}")
print(f"  Slice max len: {saved_config['slice_max_len']}")
print(f"  Vuln features: {len(saved_config['vuln_feature_names'])} global + {len(saved_config['vuln_rel_feature_names'])} relative")
print(f"  API whitelist: {len(saved_config['c_api_whitelist'])} functions")
