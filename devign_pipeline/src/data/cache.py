"""Caching utilities for processed data"""

import pickle
import hashlib
from pathlib import Path
from typing import Any, Callable, Optional, Dict, Union

import numpy as np
import pandas as pd
from functools import wraps


def save_chunk(
    df: pd.DataFrame, 
    path: str, 
    format: str = 'parquet',
    compression: Optional[str] = 'snappy'
) -> None:
    """Save DataFrame chunk to file
    
    Args:
        df: DataFrame to save
        path: Output file path
        format: 'parquet' or 'csv'
        compression: Compression for parquet ('snappy', 'gzip', 'brotli', None)
    """
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if format == 'parquet':
        df.to_parquet(output_path, compression=compression, index=False)
    elif format == 'csv':
        df.to_csv(output_path, index=False)
    else:
        raise ValueError(f"Unsupported format: {format}. Use 'parquet' or 'csv'")


def load_chunk(path: str) -> pd.DataFrame:
    """Load DataFrame chunk from file
    
    Args:
        path: Input file path (parquet or csv)
        
    Returns:
        Loaded DataFrame
    """
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Chunk file not found: {path}")
    
    suffix = file_path.suffix.lower()
    
    if suffix == '.parquet':
        return pd.read_parquet(file_path)
    elif suffix == '.csv':
        return pd.read_csv(file_path)
    else:
        raise ValueError(f"Unsupported file format: {suffix}. Use .parquet or .csv")


def save_numpy(data: Dict[str, np.ndarray], path: str) -> None:
    """Save dict of numpy arrays to npz file
    
    Args:
        data: Dict mapping names to numpy arrays
        path: Output file path (will add .npz extension if needed)
    """
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    np.savez_compressed(output_path, **data)


def load_numpy(path: str) -> Dict[str, np.ndarray]:
    """Load numpy arrays from npz file
    
    Args:
        path: Input file path
        
    Returns:
        Dict mapping names to numpy arrays
    """
    file_path = Path(path)
    
    if not file_path.exists():
        npz_path = file_path.with_suffix('.npz')
        if npz_path.exists():
            file_path = npz_path
        else:
            raise FileNotFoundError(f"Numpy file not found: {path}")
    
    with np.load(file_path, allow_pickle=True) as npz:
        return {key: npz[key] for key in npz.files}


def chunk_path(
    base_dir: str, 
    step_name: str, 
    chunk_idx: int, 
    ext: str = 'parquet'
) -> str:
    """Generate standardized chunk file path
    
    Args:
        base_dir: Base directory for chunks
        step_name: Processing step name (e.g., 'tokenized', 'normalized')
        chunk_idx: Chunk index
        ext: File extension without dot
        
    Returns:
        Full path string for the chunk file
    """
    base = Path(base_dir)
    filename = f"{step_name}_chunk_{chunk_idx:05d}.{ext}"
    return str(base / step_name / filename)


def list_chunks(base_dir: str, step_name: str, ext: str = 'parquet') -> list:
    """List all chunk files for a processing step
    
    Args:
        base_dir: Base directory for chunks
        step_name: Processing step name
        ext: File extension
        
    Returns:
        Sorted list of chunk file paths
    """
    base = Path(base_dir) / step_name
    if not base.exists():
        return []
    
    pattern = f"{step_name}_chunk_*.{ext}"
    return sorted([str(f) for f in base.glob(pattern)])


def save_pickle(data: Any, path: str) -> None:
    """Save any Python object using pickle
    
    Args:
        data: Object to save
        path: Output file path
    """
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_pickle(path: str) -> Any:
    """Load pickled Python object
    
    Args:
        path: Input file path
        
    Returns:
        Loaded object
    """
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Pickle file not found: {path}")
    
    with open(file_path, 'rb') as f:
        return pickle.load(f)


class CacheManager:
    """Manage cache for processed data"""
    
    def __init__(self, cache_dir: str = ".cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def get_cache_path(self, key: str) -> Path:
        """Get path for cache file"""
        hash_key = hashlib.md5(key.encode()).hexdigest()
        return self.cache_dir / f"{hash_key}.pkl"
    
    def exists(self, key: str) -> bool:
        """Check if cache exists"""
        return self.get_cache_path(key).exists()
    
    def load(self, key: str) -> Optional[Any]:
        """Load from cache"""
        path = self.get_cache_path(key)
        if path.exists():
            with open(path, 'rb') as f:
                return pickle.load(f)
        return None
    
    def save(self, key: str, data: Any) -> None:
        """Save to cache"""
        path = self.get_cache_path(key)
        with open(path, 'wb') as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    def clear(self) -> None:
        """Clear all cache"""
        for f in self.cache_dir.glob("*.pkl"):
            f.unlink()
    
    def get_size(self) -> int:
        """Get total cache size in bytes"""
        return sum(f.stat().st_size for f in self.cache_dir.glob("*.pkl"))


class ChunkCache:
    """Specialized cache for managing processed data chunks"""
    
    def __init__(self, base_dir: str):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
    
    def save_chunk(
        self, 
        step_name: str, 
        chunk_idx: int, 
        data: Union[pd.DataFrame, Dict[str, np.ndarray]],
        format: str = 'parquet'
    ) -> str:
        """Save a processing chunk
        
        Args:
            step_name: Name of processing step
            chunk_idx: Chunk index
            data: DataFrame or dict of numpy arrays
            format: 'parquet', 'csv', or 'npz'
            
        Returns:
            Path where chunk was saved
        """
        path = chunk_path(str(self.base_dir), step_name, chunk_idx, format)
        
        if format in ('parquet', 'csv'):
            if not isinstance(data, pd.DataFrame):
                raise TypeError("DataFrame expected for parquet/csv format")
            save_chunk(data, path, format=format)
        elif format == 'npz':
            if not isinstance(data, dict):
                raise TypeError("Dict of numpy arrays expected for npz format")
            save_numpy(data, path)
        else:
            raise ValueError(f"Unknown format: {format}")
        
        return path
    
    def load_chunk(
        self, 
        step_name: str, 
        chunk_idx: int,
        format: str = 'parquet'
    ) -> Union[pd.DataFrame, Dict[str, np.ndarray]]:
        """Load a processing chunk
        
        Args:
            step_name: Name of processing step
            chunk_idx: Chunk index
            format: 'parquet', 'csv', or 'npz'
            
        Returns:
            Loaded data
        """
        path = chunk_path(str(self.base_dir), step_name, chunk_idx, format)
        
        if format in ('parquet', 'csv'):
            return load_chunk(path)
        elif format == 'npz':
            return load_numpy(path)
        else:
            raise ValueError(f"Unknown format: {format}")
    
    def list_chunks(self, step_name: str, format: str = 'parquet') -> list:
        """List all chunks for a step"""
        return list_chunks(str(self.base_dir), step_name, format)
    
    def chunk_exists(self, step_name: str, chunk_idx: int, format: str = 'parquet') -> bool:
        """Check if a specific chunk exists"""
        path = chunk_path(str(self.base_dir), step_name, chunk_idx, format)
        return Path(path).exists()


def cache_result(cache_key: str, cache_dir: str = ".cache") -> Callable:
    """Decorator to cache function results
    
    Args:
        cache_key: Key to identify cached result
        cache_dir: Directory for cache storage
        
    Returns:
        Decorator function
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            manager = CacheManager(cache_dir)
            if manager.exists(cache_key):
                return manager.load(cache_key)
            result = func(*args, **kwargs)
            manager.save(cache_key, result)
            return result
        return wrapper
    return decorator
