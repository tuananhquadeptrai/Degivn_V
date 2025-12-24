"""Data loading utilities for Devign dataset"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Dict, Any, Optional, Union

import numpy as np
import pandas as pd


@dataclass
class DevignSample:
    """Single sample from Devign dataset"""
    id: int
    func: str
    func_clean: str
    normalized_func: str
    target: bool
    project: str
    commit_id: str
    vul_lines: dict
    lines: list
    label: list
    line_no: list


class DevignLoader:
    """Load and iterate over Devign dataset from Parquet files"""
    
    SPLIT_PATTERNS = {
        'train': 'train-*.parquet',
        'validation': 'validation-*.parquet',
        'test': 'test-*.parquet',
    }
    
    def __init__(self, data_path: str, chunk_size: int = 2000):
        self.data_path = Path(data_path)
        self.chunk_size = chunk_size
        self._validate_path()
    
    def _validate_path(self) -> None:
        """Validate data path exists"""
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data path not found: {self.data_path}")
    
    def _get_parquet_files(self, split: Optional[str] = None) -> List[Path]:
        """Get list of parquet files, optionally filtered by split"""
        if split:
            if split not in self.SPLIT_PATTERNS:
                raise ValueError(f"Unknown split: {split}. Expected one of {list(self.SPLIT_PATTERNS.keys())}")
            pattern = self.SPLIT_PATTERNS[split]
            files = list(self.data_path.glob(pattern))
        else:
            files = list(self.data_path.glob('*.parquet'))
        return sorted(files)
    
    def load_all(
        self, 
        columns: Optional[List[str]] = None,
        split: Optional[str] = None
    ) -> pd.DataFrame:
        """Load toàn bộ dataset hoặc một split cụ thể
        
        Args:
            columns: List of columns to load, None for all
            split: 'train', 'validation', 'test', or None for all
            
        Returns:
            DataFrame with loaded data
        """
        files = self._get_parquet_files(split)
        if not files:
            raise FileNotFoundError(f"No parquet files found in {self.data_path}")
        
        dfs = []
        for f in files:
            try:
                df = pd.read_parquet(f, columns=columns)
                dfs.append(df)
            except Exception as e:
                raise IOError(f"Failed to read {f}: {e}") from e
        
        return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
    
    def iter_chunks(
        self, 
        columns: Optional[List[str]] = None,
        split: Optional[str] = None
    ) -> Iterator[pd.DataFrame]:
        """Iterate over chunks để tiết kiệm RAM
        
        Args:
            columns: List of columns to load, None for all
            split: 'train', 'validation', 'test', or None for all
            
        Yields:
            DataFrame chunks of size chunk_size
        """
        files = self._get_parquet_files(split)
        if not files:
            return
        
        buffer = pd.DataFrame()
        
        for f in files:
            try:
                df = pd.read_parquet(f, columns=columns)
            except Exception as e:
                raise IOError(f"Failed to read {f}: {e}") from e
            
            buffer = pd.concat([buffer, df], ignore_index=True)
            
            while len(buffer) >= self.chunk_size:
                yield buffer.iloc[:self.chunk_size].copy()
                buffer = buffer.iloc[self.chunk_size:].reset_index(drop=True)
        
        if len(buffer) > 0:
            yield buffer
    
    def get_splits(self) -> Dict[str, List[Path]]:
        """Return paths cho train/val/test parquet files
        
        Returns:
            Dict mapping split name to list of file paths
        """
        return {
            split: self._get_parquet_files(split)
            for split in self.SPLIT_PATTERNS.keys()
        }
    
    def sample_to_dataclass(self, row: pd.Series) -> DevignSample:
        """Convert DataFrame row to DevignSample
        
        Args:
            row: Pandas Series from DataFrame row
            
        Returns:
            DevignSample dataclass instance
        """
        def safe_get(key: str, default: Any = None) -> Any:
            val = row.get(key, default)
            if val is None:
                return default
            try:
                if pd.isna(val):
                    return default
            except (ValueError, TypeError):
                pass
            return val
        
        def parse_json_field(val: Any, default: Any) -> Any:
            if val is None:
                return default
            try:
                if pd.isna(val):
                    return default
            except (ValueError, TypeError):
                pass
            if isinstance(val, str):
                try:
                    return json.loads(val)
                except json.JSONDecodeError:
                    return default
            if isinstance(val, (list, dict, np.ndarray)):
                if isinstance(val, np.ndarray):
                    return val.tolist()
                return val
            return default
        
        return DevignSample(
            id=int(safe_get('id', 0)),
            func=str(safe_get('func', '')),
            func_clean=str(safe_get('func_clean', '')),
            normalized_func=str(safe_get('normalized_func', '')),
            target=bool(safe_get('target', False)),
            project=str(safe_get('project', '')),
            commit_id=str(safe_get('commit_id', '')),
            vul_lines=parse_json_field(safe_get('vul_lines'), {}),
            lines=parse_json_field(safe_get('lines'), []),
            label=parse_json_field(safe_get('label'), []),
            line_no=parse_json_field(safe_get('line_no'), []),
        )
    
    def iter_samples(
        self, 
        split: Optional[str] = None
    ) -> Iterator[DevignSample]:
        """Iterate over samples as dataclass instances
        
        Args:
            split: 'train', 'validation', 'test', or None for all
            
        Yields:
            DevignSample instances
        """
        for chunk in self.iter_chunks(split=split):
            for _, row in chunk.iterrows():
                yield self.sample_to_dataclass(row)
    
    def __len__(self) -> int:
        """Get total number of samples across all splits"""
        total = 0
        for files in self.get_splits().values():
            for f in files:
                pf = pd.read_parquet(f, columns=[])
                total += len(pf)
        return total


class DataLoader:
    """Legacy loader for JSONL files - kept for backward compatibility"""
    
    def __init__(self, data_path: str, chunk_size: int = 2000):
        self.data_path = Path(data_path)
        self.chunk_size = chunk_size
    
    def load(self) -> List[Dict[str, Any]]:
        """Load entire dataset into memory"""
        return load_jsonl(self.data_path)
    
    def iterate_chunks(self) -> Iterator[List[Dict[str, Any]]]:
        """Iterate over dataset in chunks"""
        chunk = []
        for item in self._iterate_items():
            chunk.append(item)
            if len(chunk) >= self.chunk_size:
                yield chunk
                chunk = []
        if chunk:
            yield chunk
    
    def _iterate_items(self) -> Iterator[Dict[str, Any]]:
        """Iterate over individual items"""
        with open(self.data_path, 'r') as f:
            for line in f:
                if line.strip():
                    yield json.loads(line)


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Load JSONL file"""
    data = []
    with open(path, 'r') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def load_dataset(path: str, split: Optional[str] = None) -> List[Dict[str, Any]]:
    """Load dataset from path, optionally filtering by split"""
    data = load_jsonl(Path(path))
    if split:
        data = [item for item in data if item.get('split') == split]
    return data
