"""Multiprocessing utilities"""

from typing import Callable, List, Any, Iterable, TypeVar
from multiprocessing import Pool, cpu_count
from functools import partial

T = TypeVar('T')
R = TypeVar('R')


def parallel_map(
    func: Callable[[T], R],
    items: Iterable[T],
    num_workers: int = None,
    chunk_size: int = 100
) -> List[R]:
    """Apply function to items in parallel"""
    num_workers = num_workers or max(1, cpu_count() - 1)
    items_list = list(items)
    
    if num_workers == 1 or len(items_list) < chunk_size:
        return [func(item) for item in items_list]
    
    with Pool(num_workers) as pool:
        results = pool.map(func, items_list, chunksize=chunk_size)
    
    return results


class ChunkProcessor:
    """Process data in chunks with optional parallelism"""
    
    def __init__(
        self,
        chunk_size: int = 2000,
        num_workers: int = None
    ):
        self.chunk_size = chunk_size
        self.num_workers = num_workers or max(1, cpu_count() - 1)
    
    def process(
        self,
        items: List[T],
        func: Callable[[T], R],
        progress_callback: Callable[[int, int], None] = None
    ) -> List[R]:
        """Process items in chunks"""
        results = []
        total_chunks = (len(items) + self.chunk_size - 1) // self.chunk_size
        
        for i in range(0, len(items), self.chunk_size):
            chunk = items[i:i + self.chunk_size]
            chunk_results = parallel_map(
                func, chunk, 
                num_workers=self.num_workers
            )
            results.extend(chunk_results)
            
            if progress_callback:
                chunk_num = i // self.chunk_size + 1
                progress_callback(chunk_num, total_chunks)
        
        return results
