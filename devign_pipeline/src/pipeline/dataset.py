"""
PyTorch Dataset Module for Devign Vulnerability Detection

Provides memory-efficient dataset classes for training with:
- Lazy loading for large datasets
- Multi-GPU support
- Class weight computation for imbalanced data
- Graph data support for GNN models
"""

from typing import List, Dict, Optional, Union, Tuple, Any, Iterator
import numpy as np
from pathlib import Path
import json
from collections import Counter

try:
    import torch
    from torch.utils.data import Dataset, DataLoader, Sampler
    from torch.utils.data.distributed import DistributedSampler
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    Dataset = object

from ..tokenization.vectorizer import EncodedSample, CodeVectorizer


class DevignDataset:
    """
    Dataset class that works with or without PyTorch.
    Loads pre-vectorized data from npz files.
    
    Features:
    - Lazy loading for memory efficiency
    - Index-based access across multiple files
    - Class weight computation
    - Statistics and analysis
    """
    
    def __init__(self, 
                 vector_paths: Union[List[str], str],
                 lazy_load: bool = True,
                 cache_size: int = 5):
        """
        Args:
            vector_paths: List of paths to npz files containing vectorized data,
                         or single path/glob pattern
            lazy_load: If True, load data on-demand; if False, load all into RAM
            cache_size: Number of files to keep in LRU cache (lazy mode only)
        """
        if isinstance(vector_paths, str):
            path = Path(vector_paths)
            if path.is_dir():
                vector_paths = sorted([str(p) for p in path.glob("*.npz")])
            elif '*' in vector_paths:
                vector_paths = sorted([str(p) for p in Path('.').glob(vector_paths)])
            else:
                vector_paths = [vector_paths]
                
        self.vector_paths = [str(p) for p in vector_paths]
        self.lazy_load = lazy_load
        self.cache_size = cache_size
        
        self.data: Optional[Dict[str, np.ndarray]] = None
        self._length: int = 0
        self._file_indices: List[Tuple[str, int, int]] = []
        self._file_cache: Dict[str, Dict[str, np.ndarray]] = {}
        self._cache_order: List[str] = []
        self._labels_cache: Optional[np.ndarray] = None
        
        self._build_index()
        
        if not lazy_load:
            self._load_all()
        
    def _build_index(self) -> None:
        """Build index mapping global idx to file + local idx."""
        self._file_indices = []
        current_idx = 0
        
        for path in self.vector_paths:
            data = np.load(path, allow_pickle=True)
            n_samples = len(data['input_ids'])
            
            self._file_indices.append((path, current_idx, current_idx + n_samples))
            current_idx += n_samples
            
        self._length = current_idx
        
    def _load_all(self) -> None:
        """Load all data into memory."""
        all_input_ids = []
        all_attention_masks = []
        all_labels = []
        all_lengths = []
        all_sample_ids = []
        
        for path in self.vector_paths:
            data = np.load(path, allow_pickle=True)
            all_input_ids.append(data['input_ids'])
            all_attention_masks.append(data['attention_mask'])
            all_lengths.append(data['lengths'])
            
            if 'labels' in data:
                all_labels.append(data['labels'])
            if 'sample_ids' in data:
                all_sample_ids.append(data['sample_ids'])
                
        self.data = {
            'input_ids': np.concatenate(all_input_ids),
            'attention_mask': np.concatenate(all_attention_masks),
            'lengths': np.concatenate(all_lengths),
        }
        
        if all_labels:
            self.data['labels'] = np.concatenate(all_labels)
        if all_sample_ids:
            self.data['sample_ids'] = np.concatenate(all_sample_ids)
            
    def _get_file_data(self, path: str) -> Dict[str, np.ndarray]:
        """Get file data with LRU caching."""
        if path in self._file_cache:
            self._cache_order.remove(path)
            self._cache_order.append(path)
            return self._file_cache[path]
        
        data = dict(np.load(path, allow_pickle=True))
        
        if len(self._file_cache) >= self.cache_size:
            oldest = self._cache_order.pop(0)
            del self._file_cache[oldest]
            
        self._file_cache[path] = data
        self._cache_order.append(path)
        
        return data
        
    def _find_file_for_idx(self, idx: int) -> Tuple[str, int]:
        """Find which file contains the given global index."""
        for path, start, end in self._file_indices:
            if start <= idx < end:
                return path, idx - start
        raise IndexError(f"Index {idx} out of range [0, {self._length})")
    
    def __len__(self) -> int:
        return self._length
    
    def __getitem__(self, idx: int) -> Dict[str, np.ndarray]:
        """Return {'input_ids': ..., 'attention_mask': ..., 'label': ...}"""
        if idx < 0:
            idx = self._length + idx
        if idx < 0 or idx >= self._length:
            raise IndexError(f"Index {idx} out of range")
            
        if not self.lazy_load and self.data is not None:
            result = {
                'input_ids': self.data['input_ids'][idx],
                'attention_mask': self.data['attention_mask'][idx],
            }
            if 'labels' in self.data:
                result['label'] = self.data['labels'][idx]
            if 'lengths' in self.data:
                result['length'] = self.data['lengths'][idx]
            return result
            
        path, local_idx = self._find_file_for_idx(idx)
        file_data = self._get_file_data(path)
        
        result = {
            'input_ids': file_data['input_ids'][local_idx],
            'attention_mask': file_data['attention_mask'][local_idx],
        }
        if 'labels' in file_data:
            result['label'] = file_data['labels'][local_idx]
        if 'lengths' in file_data:
            result['length'] = file_data['lengths'][local_idx]
            
        return result
        
    def get_batch(self, indices: List[int]) -> Dict[str, np.ndarray]:
        """Get batch of samples by indices."""
        items = [self[idx] for idx in indices]
        
        batch = {
            'input_ids': np.stack([item['input_ids'] for item in items]),
            'attention_mask': np.stack([item['attention_mask'] for item in items]),
        }
        
        if 'label' in items[0]:
            batch['labels'] = np.array([item['label'] for item in items])
        if 'length' in items[0]:
            batch['lengths'] = np.array([item['length'] for item in items])
            
        return batch
        
    def get_labels(self) -> np.ndarray:
        """Get all labels (for class weight computation)."""
        if self._labels_cache is not None:
            return self._labels_cache
            
        if not self.lazy_load and self.data is not None:
            self._labels_cache = self.data.get('labels', np.zeros(self._length, dtype=np.int32))
            return self._labels_cache
            
        all_labels = []
        for path in self.vector_paths:
            data = np.load(path, allow_pickle=True)
            if 'labels' in data:
                all_labels.append(data['labels'])
                
        if all_labels:
            self._labels_cache = np.concatenate(all_labels)
        else:
            self._labels_cache = np.zeros(self._length, dtype=np.int32)
            
        return self._labels_cache
        
    def get_class_weights(self, method: str = 'balanced') -> np.ndarray:
        """
        Compute class weights for imbalanced data.
        
        Args:
            method: 'balanced', 'inverse', or 'sqrt_inverse'
            
        Returns:
            Array of class weights [weight_class_0, weight_class_1]
        """
        labels = self.get_labels()
        counter = Counter(labels)
        n_samples = len(labels)
        n_classes = len(counter)
        
        if method == 'balanced':
            weights = np.array([
                n_samples / (n_classes * counter[c])
                for c in sorted(counter.keys())
            ])
        elif method == 'inverse':
            total = sum(counter.values())
            weights = np.array([
                total / counter[c]
                for c in sorted(counter.keys())
            ])
            weights = weights / weights.sum() * n_classes
        elif method == 'sqrt_inverse':
            total = sum(counter.values())
            weights = np.array([
                np.sqrt(total / counter[c])
                for c in sorted(counter.keys())
            ])
            weights = weights / weights.sum() * n_classes
        else:
            raise ValueError(f"Unknown method: {method}")
            
        return weights.astype(np.float32)
    
    def get_sample_weights(self) -> np.ndarray:
        """Get per-sample weights for WeightedRandomSampler."""
        labels = self.get_labels()
        class_weights = self.get_class_weights()
        return class_weights[labels]
        
    def get_stats(self) -> Dict[str, Any]:
        """Get dataset statistics."""
        labels = self.get_labels()
        counter = Counter(labels)
        
        lengths = []
        for path in self.vector_paths:
            data = np.load(path, allow_pickle=True)
            if 'lengths' in data:
                lengths.extend(data['lengths'].tolist())
                
        stats = {
            'num_samples': self._length,
            'num_files': len(self.vector_paths),
            'class_distribution': dict(counter),
            'class_ratio': counter[1] / counter[0] if 0 in counter and counter[0] > 0 else 0,
        }
        
        if lengths:
            stats.update({
                'avg_length': np.mean(lengths),
                'median_length': np.median(lengths),
                'max_length': max(lengths),
                'min_length': min(lengths),
            })
            
        return stats
    
    def split(self, train_ratio: float = 0.8, 
              stratify: bool = True,
              seed: int = 42) -> Tuple['DevignDataset', 'DevignDataset']:
        """
        Split dataset into train/val.
        
        Note: Only works with lazy_load=False or single file datasets.
        For multi-file datasets, split at file level instead.
        """
        if self.lazy_load and len(self.vector_paths) > 1:
            n_train = int(len(self.vector_paths) * train_ratio)
            train_paths = self.vector_paths[:n_train]
            val_paths = self.vector_paths[n_train:]
            
            return (
                DevignDataset(train_paths, lazy_load=True),
                DevignDataset(val_paths, lazy_load=True)
            )
        
        np.random.seed(seed)
        indices = np.arange(self._length)
        
        if stratify:
            labels = self.get_labels()
            train_indices = []
            val_indices = []
            
            for label in np.unique(labels):
                label_indices = indices[labels == label]
                np.random.shuffle(label_indices)
                split_idx = int(len(label_indices) * train_ratio)
                train_indices.extend(label_indices[:split_idx])
                val_indices.extend(label_indices[split_idx:])
                
            train_indices = np.array(train_indices)
            val_indices = np.array(val_indices)
        else:
            np.random.shuffle(indices)
            split_idx = int(len(indices) * train_ratio)
            train_indices = indices[:split_idx]
            val_indices = indices[split_idx:]
        
        return _SubsetDataset(self, train_indices), _SubsetDataset(self, val_indices)
    
    def __iter__(self) -> Iterator[Dict[str, np.ndarray]]:
        """Iterate over all samples."""
        for i in range(len(self)):
            yield self[i]


class _SubsetDataset:
    """Subset wrapper for split datasets."""
    
    def __init__(self, parent: DevignDataset, indices: np.ndarray):
        self.parent = parent
        self.indices = indices
        
    def __len__(self) -> int:
        return len(self.indices)
    
    def __getitem__(self, idx: int) -> Dict[str, np.ndarray]:
        return self.parent[self.indices[idx]]
    
    def get_labels(self) -> np.ndarray:
        return self.parent.get_labels()[self.indices]
    
    def get_class_weights(self, method: str = 'balanced') -> np.ndarray:
        labels = self.get_labels()
        counter = Counter(labels)
        n_samples = len(labels)
        n_classes = len(counter)
        
        if method == 'balanced':
            weights = np.array([
                n_samples / (n_classes * counter[c])
                for c in sorted(counter.keys())
            ])
        else:
            total = sum(counter.values())
            weights = np.array([total / counter[c] for c in sorted(counter.keys())])
            weights = weights / weights.sum() * n_classes
            
        return weights.astype(np.float32)


if TORCH_AVAILABLE:
    class DevignTorchDataset(Dataset):
        """
        PyTorch Dataset wrapper for DevignDataset.
        
        Optimized for:
        - Kaggle T4 GPU (16GB VRAM)
        - DataParallel/DistributedDataParallel
        - Mixed precision training
        """
        
        def __init__(self, 
                     vector_paths: Union[List[str], str],
                     device: str = 'cpu',
                     lazy_load: bool = False,
                     return_tensors: bool = True):
            """
            Args:
                vector_paths: Paths to npz files
                device: Target device for tensors
                lazy_load: Use lazy loading (slower but memory efficient)
                return_tensors: Return PyTorch tensors instead of numpy
            """
            self.dataset = DevignDataset(vector_paths, lazy_load=lazy_load)
            self.device = device
            self.return_tensors = return_tensors
            
        def __len__(self) -> int:
            return len(self.dataset)
        
        def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
            item = self.dataset[idx]
            
            if self.return_tensors:
                result = {}
                for k, v in item.items():
                    if isinstance(v, np.ndarray):
                        result[k] = torch.from_numpy(v.copy())
                    else:
                        result[k] = torch.tensor(v)
                return result
            return item
        
        def get_class_weights(self, method: str = 'balanced') -> torch.Tensor:
            """Get class weights as PyTorch tensor."""
            weights = self.dataset.get_class_weights(method)
            return torch.from_numpy(weights).float()
        
        def get_sample_weights(self) -> torch.Tensor:
            """Get sample weights for WeightedRandomSampler."""
            weights = self.dataset.get_sample_weights()
            return torch.from_numpy(weights).float()

    def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Custom collate function for DataLoader."""
        result = {}
        
        for key in batch[0].keys():
            values = [item[key] for item in batch]
            if isinstance(values[0], torch.Tensor):
                result[key] = torch.stack(values)
            else:
                result[key] = torch.tensor(values)
                
        return result

    def create_dataloader(
        dataset: Union[DevignTorchDataset, DevignDataset, str, List[str]],
        batch_size: int = 32,
        shuffle: bool = True,
        num_workers: int = 4,
        pin_memory: bool = True,
        drop_last: bool = False,
        prefetch_factor: int = 2,
        use_weighted_sampler: bool = False
    ) -> DataLoader:
        """
        Create DataLoader with optimal settings for Kaggle T4.
        
        Args:
            dataset: Dataset or paths to load
            batch_size: Samples per batch (32 works well for T4)
            shuffle: Shuffle data each epoch
            num_workers: Parallel data loading workers
            pin_memory: Pin memory for faster GPU transfer
            drop_last: Drop incomplete last batch
            prefetch_factor: Batches to prefetch per worker
            use_weighted_sampler: Use weighted sampling for imbalanced data
            
        Returns:
            Configured DataLoader
        """
        if isinstance(dataset, (str, list)):
            dataset = DevignTorchDataset(dataset, lazy_load=False)
        elif isinstance(dataset, DevignDataset):
            dataset = DevignTorchDataset(dataset.vector_paths, lazy_load=False)
            
        sampler = None
        if use_weighted_sampler:
            sample_weights = dataset.get_sample_weights()
            sampler = torch.utils.data.WeightedRandomSampler(
                weights=sample_weights,
                num_samples=len(sample_weights),
                replacement=True
            )
            shuffle = False
            
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle if sampler is None else False,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last,
            prefetch_factor=prefetch_factor if num_workers > 0 else None,
            collate_fn=collate_fn,
            persistent_workers=num_workers > 0,
        )

    def create_distributed_dataloaders(
        train_paths: Union[List[str], str],
        val_paths: Union[List[str], str],
        batch_size: int = 32,
        num_workers: int = 4,
        world_size: int = None,
        rank: int = None
    ) -> Tuple[DataLoader, DataLoader]:
        """
        Create DataLoaders for multi-GPU training with DistributedDataParallel.
        
        Args:
            train_paths: Training data paths
            val_paths: Validation data paths
            batch_size: Per-GPU batch size
            num_workers: Workers per process
            world_size: Total number of processes
            rank: Current process rank
            
        Returns:
            Tuple of (train_loader, val_loader)
        """
        train_dataset = DevignTorchDataset(train_paths, lazy_load=False)
        val_dataset = DevignTorchDataset(val_paths, lazy_load=False)
        
        if world_size is None:
            world_size = torch.cuda.device_count() if torch.cuda.is_available() else 1
        if rank is None:
            rank = 0
            
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True
        )
        
        val_sampler = DistributedSampler(
            val_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=train_sampler,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
            collate_fn=collate_fn,
            persistent_workers=num_workers > 0,
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            sampler=val_sampler,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False,
            collate_fn=collate_fn,
            persistent_workers=num_workers > 0,
        )
        
        return train_loader, val_loader
    
    class BalancedBatchSampler(Sampler):
        """
        Sampler that yields balanced batches (equal class representation).
        Useful for highly imbalanced datasets.
        """
        
        def __init__(self, labels: np.ndarray, batch_size: int, 
                     drop_last: bool = False):
            self.labels = labels
            self.batch_size = batch_size
            self.drop_last = drop_last
            
            self.class_indices = {}
            for idx, label in enumerate(labels):
                if label not in self.class_indices:
                    self.class_indices[label] = []
                self.class_indices[label].append(idx)
                
            self.n_classes = len(self.class_indices)
            self.samples_per_class = batch_size // self.n_classes
            
        def __iter__(self) -> Iterator[List[int]]:
            class_iters = {
                c: iter(np.random.permutation(indices))
                for c, indices in self.class_indices.items()
            }
            
            while True:
                batch = []
                try:
                    for c in self.class_indices.keys():
                        for _ in range(self.samples_per_class):
                            batch.append(next(class_iters[c]))
                except StopIteration:
                    if not self.drop_last and batch:
                        yield batch
                    break
                    
                if len(batch) == self.batch_size:
                    np.random.shuffle(batch)
                    yield batch
                    
        def __len__(self) -> int:
            min_class_size = min(len(v) for v in self.class_indices.values())
            n_batches = min_class_size // self.samples_per_class
            return n_batches


class GraphDataset:
    """
    Dataset for graph-based models (GNN).
    Loads CFG/DFG representations with node features.
    
    Expected npz format for graphs:
    - node_features: (N, D) node feature matrix
    - edge_index: (2, E) edge indices (source, target)
    - edge_type: (E,) edge type labels (optional)
    - num_nodes: scalar, number of nodes
    - label: scalar, graph label
    """
    
    def __init__(self, 
                 graph_paths: Union[List[str], str],
                 vector_paths: Optional[Union[List[str], str]] = None,
                 lazy_load: bool = True):
        """
        Args:
            graph_paths: Paths to npz files with graph data (nodes, edges)
            vector_paths: Optional paths to npz files with additional node features
            lazy_load: Load on-demand or preload all
        """
        if isinstance(graph_paths, str):
            path = Path(graph_paths)
            if path.is_dir():
                graph_paths = sorted([str(p) for p in path.glob("*.npz")])
            else:
                graph_paths = [graph_paths]
                
        self.graph_paths = graph_paths
        self.vector_paths = vector_paths if vector_paths else []
        self.lazy_load = lazy_load
        
        self._length = 0
        self._file_indices: List[Tuple[str, int, int]] = []
        self._graph_cache: Dict[str, List[Dict]] = {}
        
        self._build_index()
        
        if not lazy_load:
            self._load_all()
            
    def _build_index(self) -> None:
        """Build index for graph files."""
        current_idx = 0
        
        for path in self.graph_paths:
            data = np.load(path, allow_pickle=True)
            
            if 'graphs' in data:
                n_graphs = len(data['graphs'])
            else:
                n_graphs = 1
                
            self._file_indices.append((path, current_idx, current_idx + n_graphs))
            current_idx += n_graphs
            
        self._length = current_idx
        
    def _load_all(self) -> None:
        """Load all graphs into memory."""
        for path in self.graph_paths:
            self._graph_cache[path] = self._load_graphs_from_file(path)
            
    def _load_graphs_from_file(self, path: str) -> List[Dict[str, np.ndarray]]:
        """Load graphs from a single file."""
        data = np.load(path, allow_pickle=True)
        
        if 'graphs' in data:
            return list(data['graphs'])
        
        return [{
            'node_features': data['node_features'],
            'edge_index': data['edge_index'],
            'edge_type': data.get('edge_type'),
            'num_nodes': data.get('num_nodes', len(data['node_features'])),
            'label': data.get('label', 0),
        }]
        
    def _find_graph(self, idx: int) -> Tuple[str, int]:
        """Find file and local index for global index."""
        for path, start, end in self._file_indices:
            if start <= idx < end:
                return path, idx - start
        raise IndexError(f"Index {idx} out of range")
        
    def __len__(self) -> int:
        return self._length
        
    def __getitem__(self, idx: int) -> Dict[str, np.ndarray]:
        """Return node_features, edge_index, label."""
        if idx < 0:
            idx = self._length + idx
            
        path, local_idx = self._find_graph(idx)
        
        if path in self._graph_cache:
            graphs = self._graph_cache[path]
        else:
            graphs = self._load_graphs_from_file(path)
            if not self.lazy_load:
                self._graph_cache[path] = graphs
                
        return graphs[local_idx]
    
    def get_labels(self) -> np.ndarray:
        """Get all graph labels."""
        labels = []
        for i in range(len(self)):
            labels.append(self[i].get('label', 0))
        return np.array(labels)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get dataset statistics."""
        n_nodes = []
        n_edges = []
        labels = []
        
        for i in range(len(self)):
            graph = self[i]
            n_nodes.append(graph.get('num_nodes', len(graph['node_features'])))
            n_edges.append(graph['edge_index'].shape[1] if graph['edge_index'].ndim > 1 else 0)
            labels.append(graph.get('label', 0))
            
        return {
            'num_graphs': len(self),
            'avg_nodes': np.mean(n_nodes),
            'max_nodes': max(n_nodes),
            'avg_edges': np.mean(n_edges),
            'max_edges': max(n_edges),
            'class_distribution': dict(Counter(labels)),
        }


if TORCH_AVAILABLE:
    class GraphTorchDataset(Dataset):
        """PyTorch Dataset wrapper for GraphDataset."""
        
        def __init__(self, 
                     graph_paths: Union[List[str], str],
                     vector_paths: Optional[Union[List[str], str]] = None,
                     lazy_load: bool = False):
            self.dataset = GraphDataset(graph_paths, vector_paths, lazy_load)
            
        def __len__(self) -> int:
            return len(self.dataset)
        
        def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
            item = self.dataset[idx]
            
            result = {
                'node_features': torch.from_numpy(item['node_features'].astype(np.float32)),
                'edge_index': torch.from_numpy(item['edge_index'].astype(np.int64)),
                'label': torch.tensor(item.get('label', 0), dtype=torch.long),
                'num_nodes': torch.tensor(item.get('num_nodes', len(item['node_features']))),
            }
            
            if item.get('edge_type') is not None:
                result['edge_type'] = torch.from_numpy(item['edge_type'].astype(np.int64))
                
            return result

    def graph_collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """
        Collate function for graph batching.
        Concatenates graphs with adjusted edge indices.
        """
        node_features = []
        edge_indices = []
        edge_types = []
        labels = []
        batch_indices = []
        
        node_offset = 0
        
        for i, item in enumerate(batch):
            n_nodes = item['num_nodes'].item()
            
            node_features.append(item['node_features'])
            
            edges = item['edge_index'].clone()
            edges += node_offset
            edge_indices.append(edges)
            
            if 'edge_type' in item:
                edge_types.append(item['edge_type'])
                
            labels.append(item['label'])
            batch_indices.append(torch.full((n_nodes,), i, dtype=torch.long))
            
            node_offset += n_nodes
            
        result = {
            'node_features': torch.cat(node_features, dim=0),
            'edge_index': torch.cat(edge_indices, dim=1),
            'labels': torch.stack(labels),
            'batch': torch.cat(batch_indices),
        }
        
        if edge_types:
            result['edge_type'] = torch.cat(edge_types)
            
        return result

    def create_graph_dataloader(
        dataset: Union[GraphTorchDataset, str, List[str]],
        batch_size: int = 32,
        shuffle: bool = True,
        num_workers: int = 4,
        pin_memory: bool = True
    ) -> DataLoader:
        """Create DataLoader for graph data."""
        if isinstance(dataset, (str, list)):
            dataset = GraphTorchDataset(dataset, lazy_load=False)
            
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=graph_collate_fn,
            persistent_workers=num_workers > 0,
        )


def get_optimal_batch_size(
    seq_length: int = 512,
    model_size: str = 'base',
    gpu_memory_gb: float = 16.0,
    mixed_precision: bool = True
) -> int:
    """
    Estimate optimal batch size for given configuration.
    
    Args:
        seq_length: Sequence length
        model_size: 'small', 'base', 'large'
        gpu_memory_gb: Available GPU memory
        mixed_precision: Using FP16/BF16
        
    Returns:
        Recommended batch size
    """
    base_memory = {
        'small': 0.5,
        'base': 1.0,
        'large': 2.5,
    }
    
    memory_per_sample = base_memory.get(model_size, 1.0) * (seq_length / 512)
    
    if mixed_precision:
        memory_per_sample *= 0.6
        
    available = gpu_memory_gb * 0.8
    
    batch_size = int(available / memory_per_sample)
    
    return max(1, min(batch_size, 128))
