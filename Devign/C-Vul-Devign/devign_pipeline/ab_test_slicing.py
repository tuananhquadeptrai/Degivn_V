#!/usr/bin/env python3
"""
A/B Test Script for Slicing Improvements

Compares baseline preprocessing vs improved preprocessing to measure
the impact of:
1. Post-dominator forward control dependencies
2. Increased DFG scope (50 -> 150)
3. Forward slice fallback (window-based when empty)
4. Reduced MAX_SLICES (6 -> 4)

Usage:
    python3 ab_test_slicing.py --data_dir /path/to/devign --output_dir /path/to/output

Output:
    - Preprocessed data for both configs
    - Trained models for both configs
    - Comparison report (JSON + console)
"""

import os
import sys
import json
import time
import argparse
import gc
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime

import numpy as np
import pandas as pd

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

# ============================================
# CONFIGURATION
# ============================================

@dataclass
class SlicingConfig:
    """Configuration for slicing experiment"""
    name: str
    use_post_dominator: bool = True
    dfg_scope_limit: int = 150
    max_slices: int = 4
    min_forward_lines: int = 5
    slice_max_len: int = 256
    max_seq_length: int = 512
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# Define A/B configs
CONFIG_A = SlicingConfig(
    name="baseline",
    use_post_dominator=False,  # Old: BFS reachability
    dfg_scope_limit=50,        # Old: smaller scope
    max_slices=6,              # Old: more slices
    min_forward_lines=0,       # Old: no fallback
)

CONFIG_B = SlicingConfig(
    name="improved",
    use_post_dominator=True,   # New: post-dominator
    dfg_scope_limit=150,       # New: larger scope
    max_slices=4,              # New: fewer slices
    min_forward_lines=5,       # New: fallback when forward < 5 lines
)


# ============================================
# PREPROCESSING
# ============================================

def preprocess_with_config(
    data_dir: str,
    output_dir: str,
    config: SlicingConfig,
    sample_size: Optional[int] = None
) -> Dict[str, Any]:
    """Run preprocessing with given config and return statistics."""
    
    from src.slicing.slicer import CodeSlicer, SliceConfig, SliceType
    from src.graphs.dfg import DFGBuilder
    from src.graphs.cfg import CFGBuilder
    from src.ast.parser import CFamilyParser
    from src.tokenization.tokenizer import CTokenizer
    from src.tokenization.normalization import CodeNormalizer
    from src.vuln.vuln_lines import extract_vul_line_numbers
    
    print(f"\n{'='*60}")
    print(f"Preprocessing with config: {config.name}")
    print(f"{'='*60}")
    print(f"  use_post_dominator: {config.use_post_dominator}")
    print(f"  dfg_scope_limit: {config.dfg_scope_limit}")
    print(f"  max_slices: {config.max_slices}")
    print(f"  min_forward_lines: {config.min_forward_lines}")
    
    # Setup output directory
    config_output = Path(output_dir) / config.name
    config_output.mkdir(parents=True, exist_ok=True)
    
    # Initialize components with config
    slice_config = SliceConfig(
        slice_type=SliceType.BACKWARD,
        use_post_dominator=config.use_post_dominator,
        window_size=15,
    )
    
    backward_slicer = CodeSlicer(slice_config)
    
    forward_config = SliceConfig(
        slice_type=SliceType.FORWARD,
        use_post_dominator=config.use_post_dominator,
        window_size=15,
    )
    forward_slicer = CodeSlicer(forward_config)
    
    dfg_builder = DFGBuilder(scope_limit=config.dfg_scope_limit)
    cfg_builder = CFGBuilder()
    parser = CFamilyParser()
    tokenizer = CTokenizer()
    normalizer = CodeNormalizer()
    
    # Load data
    data_path = Path(data_dir)
    train_file = data_path / "train.parquet"
    
    if not train_file.exists():
        # Try JSON format
        train_file = data_path / "function.json"
        if train_file.exists():
            df = pd.read_json(train_file)
        else:
            raise FileNotFoundError(f"No data found in {data_dir}")
    else:
        df = pd.read_parquet(train_file)
    
    if sample_size:
        df = df.sample(n=min(sample_size, len(df)), random_state=42)
    
    print(f"Loaded {len(df)} samples")
    
    # Statistics tracking
    stats = {
        'total_samples': len(df),
        'slice_type_counts': {'backward': 0, 'forward': 0, 'forward_window': 0, 'window': 0},
        'samples_with_1_slice': 0,
        'samples_with_2_slices': 0,
        'total_slices': 0,
        'avg_slice_lines': [],
        'avg_tokens_per_slice': [],
        'processing_time': 0,
    }
    
    start_time = time.time()
    
    processed_samples = []
    
    for idx, row in df.iterrows():
        code = row.get('func', '') or row.get('code', '')
        label = int(row.get('target', 0))
        
        if not code or len(code.strip()) < 10:
            continue
        
        # Get vulnerability lines
        vul_lines_raw = row.get('vul_lines', {})
        if isinstance(vul_lines_raw, str):
            try:
                vul_lines_raw = json.loads(vul_lines_raw)
            except:
                vul_lines_raw = {}
        
        criterion_lines = extract_vul_line_numbers(vul_lines_raw)
        
        if not criterion_lines:
            num_lines = code.count('\n') + 1
            criterion_lines = [max(1, num_lines // 2)]
        
        # Generate slices
        slices = []
        slice_types = []
        
        try:
            # Parse and build graphs
            parse_result = parser.parse_with_fallback(code)
            cfg = cfg_builder.build(parse_result) if parse_result else None
            dfg = dfg_builder.build(parse_result, criterion_lines) if parse_result else None
            
            # Backward slice
            backward = backward_slicer.slice(code, criterion_lines, cfg, dfg)
            if backward.code and len(backward.code.strip()) > 10:
                slices.append(backward.code)
                slice_types.append('backward')
                stats['slice_type_counts']['backward'] += 1
            
            # Forward slice with potential fallback
            forward = forward_slicer.slice(code, criterion_lines, cfg, dfg)
            forward_lines = len(forward.included_lines) - len(set(criterion_lines))
            
            if forward_lines < config.min_forward_lines or not forward.code.strip():
                # Fallback to window-based forward
                forward = backward_slicer.forward_window_slice(code, criterion_lines)
                slice_type = 'forward_window'
            else:
                slice_type = 'forward'
            
            if forward.code and len(forward.code.strip()) > 10:
                if not slices or forward.code != slices[-1]:
                    slices.append(forward.code)
                    slice_types.append(slice_type)
                    stats['slice_type_counts'][slice_type] += 1
        
        except Exception as e:
            # Fallback to window
            window = backward_slicer.window_slice(code, criterion_lines)
            slices = [window.code]
            slice_types = ['window']
            stats['slice_type_counts']['window'] += 1
        
        # Limit slices
        slices = slices[:config.max_slices]
        slice_types = slice_types[:config.max_slices]
        
        # Track statistics
        stats['total_slices'] += len(slices)
        if len(slices) == 1:
            stats['samples_with_1_slice'] += 1
        elif len(slices) >= 2:
            stats['samples_with_2_slices'] += 1
        
        for sl in slices:
            stats['avg_slice_lines'].append(sl.count('\n') + 1)
            tokens = tokenizer.tokenize(sl)
            stats['avg_tokens_per_slice'].append(len(tokens))
        
        # Tokenize and normalize
        all_tokens = []
        for sl in slices:
            tokens = tokenizer.tokenize(sl)
            token_texts = [t.text for t in tokens]
            normalized, _ = normalizer.normalize_tokens(
                [type('Token', (), {'value': t, 'type': None})() for t in token_texts]
            )
            all_tokens.extend(normalized[:config.slice_max_len])
        
        # Truncate to max length
        all_tokens = all_tokens[:config.max_seq_length]
        
        processed_samples.append({
            'tokens': all_tokens,
            'label': label,
            'num_slices': len(slices),
            'slice_types': slice_types,
        })
        
        if (idx + 1) % 500 == 0:
            print(f"  Processed {idx + 1}/{len(df)} samples...")
    
    stats['processing_time'] = time.time() - start_time
    stats['avg_slice_lines'] = float(np.mean(stats['avg_slice_lines'])) if stats['avg_slice_lines'] else 0
    stats['avg_tokens_per_slice'] = float(np.mean(stats['avg_tokens_per_slice'])) if stats['avg_tokens_per_slice'] else 0
    stats['samples_processed'] = len(processed_samples)
    
    # Calculate percentages
    total = stats['samples_processed']
    stats['pct_1_slice'] = stats['samples_with_1_slice'] / max(1, total) * 100
    stats['pct_2_slices'] = stats['samples_with_2_slices'] / max(1, total) * 100
    
    total_slice_types = sum(stats['slice_type_counts'].values())
    stats['pct_forward_window'] = stats['slice_type_counts']['forward_window'] / max(1, total_slice_types) * 100
    stats['pct_window_fallback'] = stats['slice_type_counts']['window'] / max(1, total_slice_types) * 100
    
    # Save processed data
    save_path = config_output / 'processed_samples.json'
    with open(save_path, 'w') as f:
        json.dump(processed_samples, f)
    
    # Save stats
    stats_path = config_output / 'preprocessing_stats.json'
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"\n  Preprocessing complete!")
    print(f"  Time: {stats['processing_time']:.1f}s")
    print(f"  Samples: {stats['samples_processed']}")
    print(f"  1-slice rate: {stats['pct_1_slice']:.1f}%")
    print(f"  2-slice rate: {stats['pct_2_slices']:.1f}%")
    print(f"  Window fallback: {stats['pct_window_fallback']:.1f}%")
    
    gc.collect()
    
    return stats, processed_samples


# ============================================
# MODEL TRAINING
# ============================================

def build_vocab(samples: List[Dict], min_freq: int = 2) -> Dict[str, int]:
    """Build vocabulary from samples."""
    from collections import Counter
    
    token_counts = Counter()
    for sample in samples:
        token_counts.update(sample['tokens'])
    
    vocab = {'<PAD>': 0, '<UNK>': 1, '<BOS>': 2, '<EOS>': 3}
    idx = 4
    
    for token, count in token_counts.most_common():
        if count >= min_freq:
            vocab[token] = idx
            idx += 1
    
    return vocab


def tokens_to_ids(tokens: List[str], vocab: Dict[str, int], max_len: int) -> np.ndarray:
    """Convert tokens to IDs with padding."""
    ids = [vocab.get('<BOS>', 2)]
    for t in tokens[:max_len - 2]:
        ids.append(vocab.get(t, vocab.get('<UNK>', 1)))
    ids.append(vocab.get('<EOS>', 3))
    
    # Pad
    while len(ids) < max_len:
        ids.append(vocab.get('<PAD>', 0))
    
    return np.array(ids[:max_len], dtype=np.int32)


def train_and_evaluate(
    samples: List[Dict],
    config: SlicingConfig,
    output_dir: Path,
    epochs: int = 10,
    batch_size: int = 32,
) -> Dict[str, float]:
    """Train BiGRU model and evaluate."""
    
    print(f"\n{'='*60}")
    print(f"Training model for config: {config.name}")
    print(f"{'='*60}")
    
    # Build vocab
    vocab = build_vocab(samples)
    print(f"  Vocab size: {len(vocab)}")
    
    # Prepare data
    X = np.array([
        tokens_to_ids(s['tokens'], vocab, config.max_seq_length)
        for s in samples
    ])
    y = np.array([s['label'] for s in samples])
    
    print(f"  Data shape: {X.shape}")
    print(f"  Label distribution: {np.bincount(y)}")
    
    # Split
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Try to use PyTorch
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"  Using device: {device}")
        
        # Simple BiGRU model
        class BiGRUClassifier(nn.Module):
            def __init__(self, vocab_size, embed_dim=128, hidden_dim=128):
                super().__init__()
                self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
                self.gru = nn.GRU(embed_dim, hidden_dim, bidirectional=True, batch_first=True)
                self.fc = nn.Linear(hidden_dim * 2, 1)
                self.dropout = nn.Dropout(0.3)
            
            def forward(self, x):
                emb = self.embedding(x)
                emb = self.dropout(emb)
                _, hidden = self.gru(emb)
                hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)
                out = self.fc(hidden)
                return out.squeeze(-1)
        
        model = BiGRUClassifier(len(vocab)).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.BCEWithLogitsLoss()
        
        # Create dataloaders
        train_ds = TensorDataset(
            torch.LongTensor(X_train),
            torch.FloatTensor(y_train)
        )
        val_ds = TensorDataset(
            torch.LongTensor(X_val),
            torch.FloatTensor(y_val)
        )
        
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size)
        
        # Training loop
        best_f1 = 0
        history = {'train_loss': [], 'val_loss': [], 'val_f1': []}
        
        for epoch in range(epochs):
            model.train()
            train_loss = 0
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                optimizer.zero_grad()
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            # Validation
            model.eval()
            val_loss = 0
            all_preds = []
            all_labels = []
            
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                    outputs = model(batch_x)
                    val_loss += criterion(outputs, batch_y).item()
                    preds = (torch.sigmoid(outputs) > 0.5).cpu().numpy()
                    all_preds.extend(preds)
                    all_labels.extend(batch_y.cpu().numpy())
            
            from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
            
            f1 = f1_score(all_labels, all_preds)
            acc = accuracy_score(all_labels, all_preds)
            
            history['train_loss'].append(train_loss / len(train_loader))
            history['val_loss'].append(val_loss / len(val_loader))
            history['val_f1'].append(f1)
            
            if f1 > best_f1:
                best_f1 = f1
                torch.save(model.state_dict(), output_dir / f'{config.name}_best_model.pt')
            
            if (epoch + 1) % 2 == 0:
                print(f"  Epoch {epoch+1}/{epochs}: loss={train_loss/len(train_loader):.4f}, val_f1={f1:.4f}")
        
        # Final evaluation
        model.load_state_dict(torch.load(output_dir / f'{config.name}_best_model.pt'))
        model.eval()
        
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                outputs = model(batch_x)
                preds = (torch.sigmoid(outputs) > 0.5).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(batch_y.numpy())
        
        metrics = {
            'accuracy': accuracy_score(all_labels, all_preds),
            'f1': f1_score(all_labels, all_preds),
            'precision': precision_score(all_labels, all_preds),
            'recall': recall_score(all_labels, all_preds),
        }
        
    except ImportError:
        print("  PyTorch not available, using sklearn fallback")
        
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
        
        # Flatten to bag of words style
        X_train_flat = (X_train > 0).astype(float)
        X_val_flat = (X_val > 0).astype(float)
        
        clf = LogisticRegression(max_iter=1000, random_state=42)
        clf.fit(X_train_flat, y_train)
        
        preds = clf.predict(X_val_flat)
        
        metrics = {
            'accuracy': accuracy_score(y_val, preds),
            'f1': f1_score(y_val, preds),
            'precision': precision_score(y_val, preds),
            'recall': recall_score(y_val, preds),
        }
    
    print(f"\n  Final metrics:")
    print(f"    Accuracy:  {metrics['accuracy']:.4f}")
    print(f"    F1:        {metrics['f1']:.4f}")
    print(f"    Precision: {metrics['precision']:.4f}")
    print(f"    Recall:    {metrics['recall']:.4f}")
    
    # Save metrics
    metrics_path = output_dir / f'{config.name}_metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    return metrics


# ============================================
# COMPARISON
# ============================================

def generate_comparison_report(
    stats_a: Dict, metrics_a: Dict,
    stats_b: Dict, metrics_b: Dict,
    config_a: SlicingConfig, config_b: SlicingConfig,
    output_dir: Path
) -> Dict[str, Any]:
    """Generate comparison report."""
    
    print(f"\n{'='*60}")
    print("A/B TEST COMPARISON REPORT")
    print(f"{'='*60}")
    
    report = {
        'timestamp': datetime.now().isoformat(),
        'configs': {
            'A': config_a.to_dict(),
            'B': config_b.to_dict(),
        },
        'preprocessing': {
            'A': stats_a,
            'B': stats_b,
        },
        'model_metrics': {
            'A': metrics_a,
            'B': metrics_b,
        },
        'improvements': {},
    }
    
    # Calculate improvements
    for metric in ['accuracy', 'f1', 'precision', 'recall']:
        a_val = metrics_a.get(metric, 0)
        b_val = metrics_b.get(metric, 0)
        diff = b_val - a_val
        pct = diff / max(a_val, 0.0001) * 100
        
        report['improvements'][metric] = {
            'baseline': a_val,
            'improved': b_val,
            'difference': diff,
            'percent_change': pct,
        }
    
    # Print comparison table
    print("\n┌─────────────┬────────────┬────────────┬────────────┬──────────┐")
    print("│ Metric      │ Baseline A │ Improved B │ Difference │ Change % │")
    print("├─────────────┼────────────┼────────────┼────────────┼──────────┤")
    
    for metric in ['accuracy', 'f1', 'precision', 'recall']:
        imp = report['improvements'][metric]
        sign = '+' if imp['difference'] > 0 else ''
        print(f"│ {metric:11} │ {imp['baseline']:10.4f} │ {imp['improved']:10.4f} │ {sign}{imp['difference']:10.4f} │ {sign}{imp['percent_change']:7.2f}% │")
    
    print("└─────────────┴────────────┴────────────┴────────────┴──────────┘")
    
    # Preprocessing comparison
    print("\n┌──────────────────────┬────────────┬────────────┐")
    print("│ Preprocessing Metric │ Baseline A │ Improved B │")
    print("├──────────────────────┼────────────┼────────────┤")
    print(f"│ 1-slice rate (%)     │ {stats_a.get('pct_1_slice', 0):10.1f} │ {stats_b.get('pct_1_slice', 0):10.1f} │")
    print(f"│ 2-slice rate (%)     │ {stats_a.get('pct_2_slices', 0):10.1f} │ {stats_b.get('pct_2_slices', 0):10.1f} │")
    print(f"│ Window fallback (%)  │ {stats_a.get('pct_window_fallback', 0):10.1f} │ {stats_b.get('pct_window_fallback', 0):10.1f} │")
    print(f"│ Avg tokens/slice     │ {stats_a.get('avg_tokens_per_slice', 0):10.1f} │ {stats_b.get('avg_tokens_per_slice', 0):10.1f} │")
    print(f"│ Processing time (s)  │ {stats_a.get('processing_time', 0):10.1f} │ {stats_b.get('processing_time', 0):10.1f} │")
    print("└──────────────────────┴────────────┴────────────┘")
    
    # Conclusion
    f1_diff = report['improvements']['f1']['difference']
    if f1_diff > 0.01:
        conclusion = f"✅ IMPROVED: F1 increased by {f1_diff:.4f} ({report['improvements']['f1']['percent_change']:.2f}%)"
    elif f1_diff < -0.01:
        conclusion = f"❌ DEGRADED: F1 decreased by {abs(f1_diff):.4f} ({abs(report['improvements']['f1']['percent_change']):.2f}%)"
    else:
        conclusion = f"➖ NO CHANGE: F1 difference is negligible ({f1_diff:.4f})"
    
    report['conclusion'] = conclusion
    print(f"\n{conclusion}")
    
    # Save report
    report_path = output_dir / 'ab_test_report.json'
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\nReport saved to: {report_path}")
    
    return report


# ============================================
# MAIN
# ============================================

def main():
    parser = argparse.ArgumentParser(description='A/B Test for Slicing Improvements')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to Devign dataset')
    parser.add_argument('--output_dir', type=str, default='./ab_test_output', help='Output directory')
    parser.add_argument('--sample_size', type=int, default=None, help='Limit samples for quick test')
    parser.add_argument('--epochs', type=int, default=10, help='Training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*60)
    print("A/B TEST: SLICING IMPROVEMENTS")
    print("="*60)
    print(f"Data directory: {args.data_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Sample size: {args.sample_size or 'all'}")
    
    # Run preprocessing for both configs
    stats_a, samples_a = preprocess_with_config(
        args.data_dir, str(output_dir), CONFIG_A, args.sample_size
    )
    
    stats_b, samples_b = preprocess_with_config(
        args.data_dir, str(output_dir), CONFIG_B, args.sample_size
    )
    
    # Train and evaluate both
    metrics_a = train_and_evaluate(
        samples_a, CONFIG_A, output_dir / CONFIG_A.name,
        epochs=args.epochs, batch_size=args.batch_size
    )
    
    metrics_b = train_and_evaluate(
        samples_b, CONFIG_B, output_dir / CONFIG_B.name,
        epochs=args.epochs, batch_size=args.batch_size
    )
    
    # Generate comparison report
    report = generate_comparison_report(
        stats_a, metrics_a,
        stats_b, metrics_b,
        CONFIG_A, CONFIG_B,
        output_dir
    )
    
    print("\n" + "="*60)
    print("A/B TEST COMPLETE")
    print("="*60)


if __name__ == '__main__':
    main()
