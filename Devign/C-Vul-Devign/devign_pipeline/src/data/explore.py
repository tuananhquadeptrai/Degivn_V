"""Data exploration and EDA utilities for Devign dataset"""

import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional, TYPE_CHECKING

import numpy as np
import pandas as pd

from src.vuln.vuln_lines import extract_vul_line_numbers

if TYPE_CHECKING:
    from .loader import DevignLoader


def compute_statistics(df: pd.DataFrame) -> Dict[str, Any]:
    """Tính stats: label distribution, code length, vul_lines count
    
    Args:
        df: DataFrame with Devign data
        
    Returns:
        Dict containing computed statistics
    """
    stats: Dict[str, Any] = {
        'total_samples': len(df),
        'label_distribution': {},
        'code_length': {},
        'vul_lines': {},
        'lines_per_sample': {},
    }
    
    if 'target' in df.columns:
        target_counts = df['target'].value_counts().to_dict()
        stats['label_distribution'] = {
            'vulnerable': int(target_counts.get(True, target_counts.get(1, 0))),
            'safe': int(target_counts.get(False, target_counts.get(0, 0))),
        }
        total = stats['label_distribution']['vulnerable'] + stats['label_distribution']['safe']
        if total > 0:
            stats['label_distribution']['vulnerable_ratio'] = round(
                stats['label_distribution']['vulnerable'] / total, 4
            )
            stats['label_distribution']['safe_ratio'] = round(
                stats['label_distribution']['safe'] / total, 4
            )
    
    func_col = 'func' if 'func' in df.columns else 'func_clean' if 'func_clean' in df.columns else None
    if func_col:
        lengths = df[func_col].astype(str).str.len()
        stats['code_length'] = {
            'min': int(lengths.min()),
            'max': int(lengths.max()),
            'mean': round(float(lengths.mean()), 2),
            'median': round(float(lengths.median()), 2),
            'std': round(float(lengths.std()), 2),
        }
    
    if 'vul_lines' in df.columns:
        def count_vul_lines(val: Any) -> int:
            if val is None or (isinstance(val, float) and pd.isna(val)):
                return 0
            if isinstance(val, str):
                try:
                    val = json.loads(val)
                except:
                    return 0
            return len(extract_vul_line_numbers(val))
        
        vul_counts = df['vul_lines'].apply(count_vul_lines)
        stats['vul_lines'] = {
            'total_vul_line_entries': int(vul_counts.sum()),
            'samples_with_vul_lines': int((vul_counts > 0).sum()),
            'mean_per_sample': round(float(vul_counts.mean()), 2),
            'max_per_sample': int(vul_counts.max()),
        }
    
    if 'lines' in df.columns:
        def count_lines(val: Any) -> int:
            if val is None:
                return 0
            try:
                if pd.isna(val):
                    return 0
            except (ValueError, TypeError):
                pass
            if isinstance(val, (list, np.ndarray)):
                return len(val)
            if isinstance(val, str):
                try:
                    parsed = json.loads(val)
                    return len(parsed) if isinstance(parsed, list) else 0
                except json.JSONDecodeError:
                    return 0
            return 0
        
        line_counts = df['lines'].apply(count_lines)
        stats['lines_per_sample'] = {
            'min': int(line_counts.min()),
            'max': int(line_counts.max()),
            'mean': round(float(line_counts.mean()), 2),
            'median': round(float(line_counts.median()), 2),
        }
    
    if 'project' in df.columns:
        project_counts = df['project'].value_counts().to_dict()
        stats['projects'] = {
            'unique_count': len(project_counts),
            'distribution': {str(k): int(v) for k, v in project_counts.items()},
        }
    
    return stats


def generate_eda_report(
    loader: 'DevignLoader', 
    output_path: str,
    include_per_split: bool = True
) -> None:
    """Generate full EDA report as JSON
    
    Args:
        loader: DevignLoader instance
        output_path: Path to save JSON report
        include_per_split: Whether to include per-split statistics
    """
    report: Dict[str, Any] = {
        'dataset_path': str(loader.data_path),
        'chunk_size': loader.chunk_size,
        'splits': {},
        'overall': {},
    }
    
    splits = loader.get_splits()
    report['available_splits'] = {
        split: [str(f) for f in files] 
        for split, files in splits.items()
    }
    
    all_dfs = []
    
    if include_per_split:
        for split_name in ['train', 'validation', 'test']:
            try:
                df = loader.load_all(split=split_name)
                if len(df) > 0:
                    report['splits'][split_name] = compute_statistics(df)
                    all_dfs.append(df)
            except FileNotFoundError:
                report['splits'][split_name] = {'error': 'No files found'}
    
    if all_dfs:
        combined_df = pd.concat(all_dfs, ignore_index=True)
        report['overall'] = compute_statistics(combined_df)
    else:
        try:
            df = loader.load_all()
            report['overall'] = compute_statistics(df)
        except FileNotFoundError:
            report['overall'] = {'error': 'No data found'}
    
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)


def plot_distributions(
    df: pd.DataFrame, 
    save_dir: str,
    prefix: str = ''
) -> None:
    """Save distribution plots (skip if matplotlib not available)
    
    Args:
        df: DataFrame with Devign data
        save_dir: Directory to save plots
        prefix: Prefix for plot filenames
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return
    
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    if 'target' in df.columns:
        fig, ax = plt.subplots(figsize=(8, 6))
        target_counts = df['target'].value_counts()
        labels = ['Safe (0)', 'Vulnerable (1)']
        values = [
            target_counts.get(False, target_counts.get(0, 0)),
            target_counts.get(True, target_counts.get(1, 0)),
        ]
        colors = ['#2ecc71', '#e74c3c']
        ax.bar(labels, values, color=colors)
        ax.set_ylabel('Count')
        ax.set_title('Label Distribution')
        for i, v in enumerate(values):
            ax.text(i, v + 0.01 * max(values), str(v), ha='center')
        plt.tight_layout()
        plt.savefig(save_path / f'{prefix}label_distribution.png', dpi=150)
        plt.close()
    
    func_col = 'func' if 'func' in df.columns else 'func_clean' if 'func_clean' in df.columns else None
    if func_col:
        fig, ax = plt.subplots(figsize=(10, 6))
        lengths = df[func_col].astype(str).str.len()
        ax.hist(lengths, bins=50, edgecolor='black', alpha=0.7)
        ax.set_xlabel('Code Length (characters)')
        ax.set_ylabel('Frequency')
        ax.set_title('Code Length Distribution')
        ax.axvline(lengths.mean(), color='red', linestyle='--', label=f'Mean: {lengths.mean():.0f}')
        ax.axvline(lengths.median(), color='orange', linestyle='--', label=f'Median: {lengths.median():.0f}')
        ax.legend()
        plt.tight_layout()
        plt.savefig(save_path / f'{prefix}code_length_distribution.png', dpi=150)
        plt.close()
    
    if 'project' in df.columns:
        fig, ax = plt.subplots(figsize=(12, 6))
        project_counts = df['project'].value_counts()
        ax.bar(range(len(project_counts)), project_counts.values, edgecolor='black', alpha=0.7)
        ax.set_xticks(range(len(project_counts)))
        ax.set_xticklabels(project_counts.index, rotation=45, ha='right')
        ax.set_xlabel('Project')
        ax.set_ylabel('Count')
        ax.set_title('Samples per Project')
        plt.tight_layout()
        plt.savefig(save_path / f'{prefix}project_distribution.png', dpi=150)
        plt.close()


class DataExplorer:
    """Explore and analyze dataset - legacy class for backward compatibility"""
    
    def __init__(self, data: List[Dict[str, Any]]):
        self.data = data
    
    def get_label_distribution(self) -> Dict[int, int]:
        """Get distribution of labels"""
        return dict(Counter(item.get('target', 0) for item in self.data))
    
    def get_code_length_stats(self) -> Dict[str, float]:
        """Get statistics about code lengths"""
        lengths = [len(item.get('func', '')) for item in self.data]
        return {
            'min': min(lengths) if lengths else 0,
            'max': max(lengths) if lengths else 0,
            'mean': sum(lengths) / len(lengths) if lengths else 0,
            'count': len(lengths),
        }
    
    def sample(self, n: int = 5, label: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get sample of data"""
        data = self.data
        if label is not None:
            data = [item for item in data if item.get('target') == label]
        return data[:n]


def analyze_distribution(data: List[Dict[str, Any]]) -> Tuple[int, int, float]:
    """Analyze label distribution: returns (vulnerable, safe, ratio)"""
    vuln = sum(1 for item in data if item.get('target') == 1)
    safe = len(data) - vuln
    ratio = vuln / safe if safe > 0 else 0
    return vuln, safe, ratio
