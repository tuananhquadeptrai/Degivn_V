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
# IMPROVED: Reduced from 6 to 4 slices (analysis showed slices 5-6 rarely contribute)
MAX_SLICES = 4          # Max slices per sample (backward + forward)
SLICE_MAX_LEN = 256     # Max tokens per slice
MIN_FORWARD_LINES = 5   # Minimum lines for forward slice before window fallback

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
        # IMPROVED: Now ensures forward slice is never empty
        # ============================================
        slices = []
        slice_types = []  # 'backward', 'forward', or 'forward_window'
        
        # Group criterion lines
        groups = group_criterion_lines(criterion_lines)
        
        # Limit to 2 groups (will produce 4 slices max = 2 backward + 2 forward)
        for group in groups[:2]:
            # Backward slice
            try:
                backward_slice = backward_slicer.slice(code, group)
                if backward_slice.code and len(backward_slice.code.strip()) > 10:
                    slices.append(backward_slice.code)
                    slice_types.append('backward')
            except Exception:
                pass
            
            # Forward slice with fallback
            try:
                forward_slice = forward_slicer.slice(code, group)
                forward_lines = len(forward_slice.included_lines) - len(set(group))
                
                # Check if forward slice is too small
                if forward_lines < MIN_FORWARD_LINES or not forward_slice.code.strip():
                    # Fallback: window-based forward slice
                    forward_slice = backward_slicer.forward_window_slice(code, group)
                    slice_type = 'forward_window'
                else:
                    slice_type = 'forward'
                
                if forward_slice.code and len(forward_slice.code.strip()) > 10:
                    # Avoid duplicate if forward == backward
                    if not slices or forward_slice.code != slices[-1]:
                        slices.append(forward_slice.code)
                        slice_types.append(slice_type)
            except Exception:
                pass
        
        # Fallback: if no slices, use full code
        if not slices:
            slices = [code]
            slice_types = ['full']
        
        # Limit to MAX_SLICES (now 4)
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
PRUNE_STATS = {}  # Global variable to store prune statistics

def prune_vocab_and_reindex(vocab: Vocabulary, 
                            vector_sets: List[Dict],
                            slice_vector_sets: List[Dict]) -> Tuple[Vocabulary, List[Dict], List[Dict]]:
    """Prune unused tokens from vocabulary and reindex all vectors."""
    global PRUNE_STATS
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
    
    PRUNE_STATS = {
        "original_vocab_size": int(orig_vocab_size),
        "pruned_vocab_size": int(len(new_token2id)),
        "removed_tokens": int(orig_vocab_size - len(new_token2id)),
    }
    
    return vocab, updated_vector_sets, updated_slice_sets

vocab, (train_vectors, val_vectors, test_vectors), (train_slice_vectors, val_slice_vectors, test_slice_vectors) = prune_vocab_and_reindex(
    vocab, 
    [train_vectors, val_vectors, test_vectors],
    [train_slice_vectors, val_slice_vectors, test_slice_vectors]
)

# %% Cell 12b: Debug JSON/JSONL helpers
from collections import Counter
import math

def compute_stats(values):
    """Compute min, max, mean, percentiles for a list of values."""
    values = list(values)
    if not values:
        return {"min": 0, "max": 0, "mean": 0.0, "p50": 0, "p90": 0, "p95": 0, "p99": 0}
    v_sorted = sorted(values)
    n = len(v_sorted)

    def perc(p):
        idx = int(math.ceil(p * (n - 1)))
        return int(v_sorted[idx])

    return {
        "min": int(v_sorted[0]),
        "max": int(v_sorted[-1]),
        "mean": round(float(sum(v_sorted) / n), 2),
        "p50": perc(0.50),
        "p90": perc(0.90),
        "p95": perc(0.95),
        "p99": perc(0.99),
    }


def save_jsonl(path, records):
    """Save list of dicts to JSONL file."""
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def dump_split_debug(split_name, vectors, slice_vectors, vocab, output_dir):
    """
    Generate 4 debug JSON files for each split:
      1. {split}_overview.json        - High-level metadata and tensor shapes
      2. {split}_sequences_full.jsonl - Full 512-length sequences with padding
      3. {split}_slices_full.jsonl    - All 6×256 slices with padding  
      4. {split}_padding_stats.json   - Aggregate stats and consistency checks
    Returns stats for vectorization_stats.json.
    """
    input_ids = vectors["input_ids"]
    attn_mask = vectors["attention_mask"]
    labels = vectors["labels"]

    slice_ids = slice_vectors["slice_input_ids"]
    slice_attn = slice_vectors["slice_attention_mask"]
    slice_count_arr = slice_vectors["slice_count"]

    num_samples = input_ids.shape[0]
    max_len = input_ids.shape[1]
    max_slices = slice_ids.shape[1]
    slice_max_len = slice_ids.shape[2]

    pad_id = vocab.pad_id
    unk_id = vocab.unk_id

    # Collectors for stats
    seq_lengths = []
    slice_lengths = []
    num_slices_per_sample = []
    slice_count_dist = Counter()
    
    token_counter = Counter()
    id_counter = Counter()
    total_token_slots = int(num_samples * max_len)
    total_non_pad_tokens = 0
    total_pad_tokens = 0
    total_unk_tokens = 0

    num_truncated_seq = 0
    num_truncated_slices = 0
    
    # Mask consistency checks
    mask_mismatch_id_zero_mask_one = 0
    mask_mismatch_id_nonzero_mask_zero = 0
    
    # Per-position padding stats
    seq_nonzero_per_pos = np.zeros(max_len, dtype=np.int64)
    slice_nonzero_per_pos = np.zeros(slice_max_len, dtype=np.int64)
    total_real_slices = 0

    sequences_jsonl = []
    slices_jsonl = []

    for idx in range(num_samples):
        # === SEQUENCE DATA ===
        seq_ids_all = input_ids[idx]
        seq_mask_all = attn_mask[idx]
        seq_len = int(seq_mask_all.sum())
        seq_lengths.append(seq_len)
        pad_len = max_len - seq_len
        
        if seq_len == max_len:
            num_truncated_seq += 1

        # Convert to tokens (full length including padding)
        seq_tokens_full = [
            vocab.id2token[i] if i < len(vocab.id2token) else "<invalid>"
            for i in seq_ids_all
        ]
        seq_ids_full = [int(i) for i in seq_ids_all]
        seq_mask_full = [int(m) for m in seq_mask_all]

        # Count tokens (only non-pad for frequency)
        for pos, (i, m) in enumerate(zip(seq_ids_all, seq_mask_all)):
            if i == pad_id:
                total_pad_tokens += 1
            else:
                total_non_pad_tokens += 1
                seq_nonzero_per_pos[pos] += 1
                if i == unk_id:
                    total_unk_tokens += 1
                token_counter[vocab.id2token[i] if i < len(vocab.id2token) else "<invalid>"] += 1
                id_counter[int(i)] += 1
            
            # Mask consistency
            if i == pad_id and m == 1:
                mask_mismatch_id_zero_mask_one += 1
            if i != pad_id and m == 0:
                mask_mismatch_id_nonzero_mask_zero += 1

        # === SLICE DATA ===
        sample_slice_count = int(slice_count_arr[idx])
        num_slices_per_sample.append(sample_slice_count)
        slice_count_dist[sample_slice_count] += 1

        sample_slices = []
        for s in range(max_slices):
            s_ids_all = slice_ids[idx, s]
            s_mask_all = slice_attn[idx, s]
            s_len = int(s_mask_all.sum())
            s_pad_len = slice_max_len - s_len
            
            is_real = s < sample_slice_count
            
            if is_real:
                slice_lengths.append(s_len)
                total_real_slices += 1
                if s_len == slice_max_len:
                    num_truncated_slices += 1
                # Per-position stats for real slices
                for pos, i in enumerate(s_ids_all):
                    if i != pad_id:
                        slice_nonzero_per_pos[pos] += 1

            s_tokens_full = [
                vocab.id2token[i] if i < len(vocab.id2token) else "<invalid>"
                for i in s_ids_all
            ]
            s_ids_full = [int(i) for i in s_ids_all]
            s_mask_full = [int(m) for m in s_mask_all]

            sample_slices.append({
                "slice_id": s,
                "is_real_slice": is_real,
                "max_len": slice_max_len,
                "nonpad_len": s_len if is_real else 0,
                "pad_len": s_pad_len if is_real else slice_max_len,
                "input_ids": s_ids_full,
                "attention_mask": s_mask_full,
                "tokens": s_tokens_full,
            })

        # Build sequence record
        seq_record = {
            "sample_id": idx,
            "label": int(labels[idx]),
            "sequence": {
                "max_len": max_len,
                "nonpad_len": seq_len,
                "pad_len": pad_len,
                "input_ids": seq_ids_full,
                "attention_mask": seq_mask_full,
                "tokens": seq_tokens_full,
            },
        }
        sequences_jsonl.append(seq_record)

        # Build slices record
        slices_record = {
            "sample_id": idx,
            "label": int(labels[idx]),
            "slice_count": sample_slice_count,
            "slices": sample_slices,
        }
        slices_jsonl.append(slices_record)

    # === FILE 1: Overview JSON ===
    overview = {
        "split": split_name,
        "num_samples": num_samples,
        "max_sequence_length": max_len,
        "max_slices_per_sample": max_slices,
        "max_slice_length": slice_max_len,
        "labels": {
            "num_classes": 2,
            "mapping": {"0": "non_vulnerable", "1": "vulnerable"},
            "distribution": {
                "0": int((labels == 0).sum()),
                "1": int((labels == 1).sum()),
            },
        },
        "tensors": {
            "input_ids": {
                "shape": list(input_ids.shape),
                "dtype": str(input_ids.dtype),
                "description": "Token IDs for full sequence; padded with 0.",
            },
            "attention_mask": {
                "shape": list(attn_mask.shape),
                "dtype": str(attn_mask.dtype),
                "description": "1 for real tokens, 0 for padding.",
            },
            "labels": {
                "shape": list(labels.shape),
                "dtype": str(labels.dtype),
            },
            "slice_input_ids": {
                "shape": list(slice_ids.shape),
                "dtype": str(slice_ids.dtype),
            },
            "slice_attention_mask": {
                "shape": list(slice_attn.shape),
                "dtype": str(slice_attn.dtype),
            },
            "slice_count": {
                "shape": list(slice_count_arr.shape),
                "dtype": str(slice_count_arr.dtype),
                "description": f"Number of real slices per sample (≤ {max_slices}).",
            },
        },
        "special_tokens": {
            "pad": {"id": int(pad_id), "token": "<PAD>"},
            "unk": {"id": int(unk_id), "token": "<UNK>"},
            "bos": {"id": int(vocab.bos_id), "token": "<BOS>"},
            "eos": {"id": int(vocab.eos_id), "token": "<EOS>"},
        },
    }

    with open(os.path.join(output_dir, f"{split_name}_overview.json"), "w", encoding="utf-8") as f:
        json.dump(overview, f, indent=2, ensure_ascii=False)

    # === FILE 2: Sequences JSONL ===
    save_jsonl(os.path.join(output_dir, f"{split_name}_sequences_full.jsonl"), sequences_jsonl)

    # === FILE 3: Slices JSONL ===
    save_jsonl(os.path.join(output_dir, f"{split_name}_slices_full.jsonl"), slices_jsonl)

    # === FILE 4: Padding Stats JSON ===
    # Compute histogram bins for sequence lengths
    hist_bins = [0, 64, 128, 192, 256, 320, 384, 448, 512]
    seq_hist_counts = [0] * (len(hist_bins) - 1)
    for sl in seq_lengths:
        for i in range(len(hist_bins) - 1):
            if hist_bins[i] <= sl < hist_bins[i + 1]:
                seq_hist_counts[i] += 1
                break
        else:
            seq_hist_counts[-1] += 1

    unk_ratio = float(total_unk_tokens) / max(total_non_pad_tokens, 1)
    
    padding_stats = {
        "split": split_name,
        "sequence_lengths": {
            "min": int(min(seq_lengths)) if seq_lengths else 0,
            "max": int(max(seq_lengths)) if seq_lengths else 0,
            "mean": round(float(np.mean(seq_lengths)), 2) if seq_lengths else 0,
            "median": int(np.median(seq_lengths)) if seq_lengths else 0,
            "histogram": {
                "bins": hist_bins,
                "counts": seq_hist_counts,
            },
        },
        "slice_lengths": {
            "real_slices_only": {
                "count": len(slice_lengths),
                "min": int(min(slice_lengths)) if slice_lengths else 0,
                "max": int(max(slice_lengths)) if slice_lengths else 0,
                "mean": round(float(np.mean(slice_lengths)), 2) if slice_lengths else 0,
            },
        },
        "slice_count_distribution": {str(k): v for k, v in sorted(slice_count_dist.items())},
        "padding_summary": {
            "total_sequence_slots": total_token_slots,
            "total_non_pad_tokens": total_non_pad_tokens,
            "total_pad_tokens": total_pad_tokens,
            "sequence_pad_fraction": round(float(total_pad_tokens / max(total_token_slots, 1)), 4),
            "total_unk_tokens": total_unk_tokens,
            "unk_ratio": round(unk_ratio, 6),
        },
        "mask_consistency": {
            "id_zero_but_mask_one": mask_mismatch_id_zero_mask_one,
            "id_nonzero_but_mask_zero": mask_mismatch_id_nonzero_mask_zero,
            "is_consistent": mask_mismatch_id_zero_mask_one == 0 and mask_mismatch_id_nonzero_mask_zero == 0,
        },
        "truncation": {
            "num_truncated_sequences": num_truncated_seq,
            "num_truncated_slices": num_truncated_slices,
        },
        "token_frequency_top20": [
            {"rank": i+1, "token": tok, "count": cnt}
            for i, (tok, cnt) in enumerate(token_counter.most_common(20))
        ],
        "id_frequency_top20": [
            {"rank": i+1, "id": int(i_id), "token": vocab.id2token[i_id] if i_id < len(vocab.id2token) else "<invalid>", "count": cnt}
            for i, (i_id, cnt) in enumerate(id_counter.most_common(20))
        ],
    }

    with open(os.path.join(output_dir, f"{split_name}_padding_stats.json"), "w", encoding="utf-8") as f:
        json.dump(padding_stats, f, indent=2, ensure_ascii=False)

    # Return stats for vectorization_stats.json
    split_vec_stats = {
        "num_samples": num_samples,
        "seq_length": compute_stats(seq_lengths),
        "num_truncated_seq": num_truncated_seq,
        "seq_pad_fraction": round(float(total_pad_tokens / max(total_token_slots, 1)), 4),
        "slice_length": compute_stats(slice_lengths),
        "num_truncated_slices": num_truncated_slices,
        "slice_pad_fraction": round(float(
            (slice_ids.size - int(np.count_nonzero(slice_attn)))
            / max(slice_ids.size, 1)
        ), 4),
    }
    return split_vec_stats


def compute_feature_stats(arr: np.ndarray) -> Dict[str, Any]:
    """Compute statistics for a 1D feature array."""
    arr = arr.astype(float)
    return {
        "mean": round(float(np.mean(arr)), 4),
        "std": round(float(np.std(arr)), 4),
        "min": round(float(np.min(arr)), 4),
        "p25": round(float(np.percentile(arr, 25)), 4),
        "median": round(float(np.median(arr)), 4),
        "p75": round(float(np.percentile(arr, 75)), 4),
        "max": round(float(np.max(arr)), 4),
        "num_nonzero": int(np.count_nonzero(arr)),
        "num_zero": int(np.sum(arr == 0)),
    }


def dump_vuln_debug(split_name: str, vuln_data: Dict, slice_vuln_data: Dict, output_dir: str):
    """
    Generate 4 debug JSON files for vulnerability features:
      1. {split}_vuln_features_stats.json     - Stats for global features
      2. {split}_vuln_slice_features_stats.json - Stats for slice vuln features
      3. {split}_vuln_slice_rel_stats.json    - Stats for slice relative features
      4. {split}_vuln_debug_samples.jsonl     - Per-sample feature values
    """
    features = vuln_data["features"]  # (num_samples, 26)
    feature_names = list(vuln_data["feature_names"])
    
    slice_vuln_features = slice_vuln_data["slice_vuln_features"]  # (num_samples, 6, 26)
    slice_rel_features = slice_vuln_data["slice_rel_features"]    # (num_samples, 6, 26)
    slice_vuln_names = list(slice_vuln_data["slice_vuln_feature_names"])
    slice_rel_names = list(slice_vuln_data["slice_rel_feature_names"])
    
    num_samples = features.shape[0]
    num_slices = slice_vuln_features.shape[1]
    num_features = features.shape[1]
    
    # === FILE 1: Global features stats ===
    features_stats = {
        "split": split_name,
        "num_samples": num_samples,
        "num_features": num_features,
        "array": "features",
        "feature_stats": {}
    }
    for i, fname in enumerate(feature_names):
        features_stats["feature_stats"][fname] = compute_feature_stats(features[:, i])
    
    with open(os.path.join(output_dir, f"{split_name}_vuln_features_stats.json"), "w", encoding="utf-8") as f:
        json.dump(features_stats, f, indent=2, ensure_ascii=False)
    
    # === FILE 2: Slice vuln features stats ===
    slice_vuln_stats = {
        "split": split_name,
        "num_samples": num_samples,
        "num_slices": num_slices,
        "num_features": num_features,
        "array": "slice_vuln_features",
        "feature_stats": {}
    }
    for i, fname in enumerate(slice_vuln_names):
        # Overall stats (all slices pooled)
        all_vals = slice_vuln_features[:, :, i].flatten()
        overall = compute_feature_stats(all_vals)
        
        # Per-slice stats
        per_slice = {}
        for s in range(num_slices):
            per_slice[str(s)] = compute_feature_stats(slice_vuln_features[:, s, i])
        
        slice_vuln_stats["feature_stats"][fname] = {
            "overall": overall,
            "per_slice": per_slice,
        }
    
    with open(os.path.join(output_dir, f"{split_name}_vuln_slice_features_stats.json"), "w", encoding="utf-8") as f:
        json.dump(slice_vuln_stats, f, indent=2, ensure_ascii=False)
    
    # === FILE 3: Slice relative features stats ===
    slice_rel_stats = {
        "split": split_name,
        "num_samples": num_samples,
        "num_slices": num_slices,
        "num_features": num_features,
        "array": "slice_rel_features",
        "feature_stats": {}
    }
    for i, fname in enumerate(slice_rel_names):
        all_vals = slice_rel_features[:, :, i].flatten()
        overall = compute_feature_stats(all_vals)
        
        per_slice = {}
        for s in range(num_slices):
            per_slice[str(s)] = compute_feature_stats(slice_rel_features[:, s, i])
        
        slice_rel_stats["feature_stats"][fname] = {
            "overall": overall,
            "per_slice": per_slice,
        }
    
    with open(os.path.join(output_dir, f"{split_name}_vuln_slice_rel_stats.json"), "w", encoding="utf-8") as f:
        json.dump(slice_rel_stats, f, indent=2, ensure_ascii=False)
    
    # === FILE 4: Per-sample debug JSONL ===
    samples_jsonl = []
    for idx in range(num_samples):
        # Global features
        global_feats = {fname: round(float(features[idx, i]), 4) for i, fname in enumerate(feature_names)}
        
        # Slice vuln features
        slice_vuln_list = []
        for s in range(num_slices):
            s_feats = {"slice_id": s}
            for i, fname in enumerate(slice_vuln_names):
                s_feats[fname] = round(float(slice_vuln_features[idx, s, i]), 4)
            slice_vuln_list.append(s_feats)
        
        # Slice rel features
        slice_rel_list = []
        for s in range(num_slices):
            s_feats = {"slice_id": s}
            for i, fname in enumerate(slice_rel_names):
                s_feats[fname] = round(float(slice_rel_features[idx, s, i]), 4)
            slice_rel_list.append(s_feats)
        
        samples_jsonl.append({
            "sample_id": idx,
            "features": global_feats,
            "slice_vuln_features": slice_vuln_list,
            "slice_rel_features": slice_rel_list,
        })
    
    save_jsonl(os.path.join(output_dir, f"{split_name}_vuln_debug_samples.jsonl"), samples_jsonl)
    
    print(f"  ✓ {split_name}_vuln: 4 debug files saved")


def build_vocab_debug(vocab, prune_stats):
    """Build vocab_debug.json content."""
    freqs = vocab.token_freqs if hasattr(vocab, "token_freqs") else {}
    freq_items = list(freqs.items())
    freq_items.sort(key=lambda x: x[1], reverse=True)
    most = freq_items[:100]
    least = freq_items[-100:]

    return {
        "version": "v2",
        "vocab_size": int(len(vocab)),
        "prune_stats": prune_stats,
        "special_tokens": {
            "pad": {"id": int(vocab.pad_id), "token": vocab.id2token[vocab.pad_id]},
            "unk": {"id": int(vocab.unk_id), "token": vocab.id2token[vocab.unk_id]},
            "bos": {"id": int(vocab.bos_id), "token": vocab.id2token[vocab.bos_id]},
            "eos": {"id": int(vocab.eos_id), "token": vocab.id2token[vocab.eos_id]},
        },
        "most_frequent_tokens": [
            {"token": tok, "id": int(vocab.token2id.get(tok, -1)), "freq": int(c)}
            for tok, c in most
        ],
        "least_frequent_tokens": [
            {"token": tok, "id": int(vocab.token2id.get(tok, -1)), "freq": int(c)}
            for tok, c in least
        ],
    }


# %% Cell 12c: Dump debug JSON/JSONL (before creating .npz)
print("\nSaving intermediate debug JSON/JSONL...")

vectorization_stats = {
    "version": "v2",
    "max_len": int(MAX_LEN),
    "max_slices": int(MAX_SLICES),
    "slice_max_len": int(SLICE_MAX_LEN),
    "splits": {},
}

for split_name, vec, s_vec in [
    ("train", train_vectors, train_slice_vectors),
    ("val",   val_vectors,   val_slice_vectors),
    ("test",  test_vectors,  test_slice_vectors),
]:
    print(f"  Processing {split_name}...")
    split_stats = dump_split_debug(split_name, vec, s_vec, vocab, OUTPUT_DIR)
    vectorization_stats["splits"][split_name] = split_stats

with open(os.path.join(OUTPUT_DIR, "vectorization_stats.json"), "w", encoding="utf-8") as f:
    json.dump(vectorization_stats, f, indent=2, ensure_ascii=False)

vocab_debug = build_vocab_debug(vocab, PRUNE_STATS)
with open(os.path.join(OUTPUT_DIR, "vocab_debug.json"), "w", encoding="utf-8") as f:
    json.dump(vocab_debug, f, indent=2, ensure_ascii=False)

print("✓ Debug files saved:")
print(f"  - {{train,val,test}}_overview.json")
print(f"  - {{train,val,test}}_sequences_full.jsonl")
print(f"  - {{train,val,test}}_slices_full.jsonl")
print(f"  - {{train,val,test}}_padding_stats.json")
print(f"  - vectorization_stats.json")
print(f"  - vocab_debug.json")

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

# NEW: Dump vuln debug files
print("\nSaving vuln debug files...")
dump_vuln_debug("train", train_vuln, train_slice_vuln, OUTPUT_DIR)
dump_vuln_debug("val", val_vuln, val_slice_vuln, OUTPUT_DIR)
dump_vuln_debug("test", test_vuln, test_slice_vuln, OUTPUT_DIR)

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
