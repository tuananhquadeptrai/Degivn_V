"""
Code Vectorization Module

Converts token sequences to numerical vectors for model training.
Supports padding, truncation, attention masks, and batch processing.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Union, Iterator
import numpy as np
from pathlib import Path
import json

from .vocab import Vocabulary, PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, UNK_TOKEN


@dataclass
class VectorizerConfig:
    """Configuration for vectorization."""
    max_length: int = 512
    add_bos: bool = True
    add_eos: bool = True
    truncation: str = 'tail'      # 'head', 'tail', 'center'
    padding: str = 'right'        # 'left', 'right'
    return_attention_mask: bool = True
    return_length: bool = False
    dtype: str = 'int32'          # 'int32', 'int64'
    
    def to_dict(self) -> dict:
        return {
            'max_length': self.max_length,
            'add_bos': self.add_bos,
            'add_eos': self.add_eos,
            'truncation': self.truncation,
            'padding': self.padding,
            'return_attention_mask': self.return_attention_mask,
            'return_length': self.return_length,
            'dtype': self.dtype,
        }
    
    @classmethod
    def from_dict(cls, d: dict) -> 'VectorizerConfig':
        return cls(**d)


@dataclass
class EncodedSample:
    """Encoded sample with IDs and metadata."""
    input_ids: np.ndarray       # Shape: (max_length,)
    attention_mask: np.ndarray  # Shape: (max_length,)
    length: int                 # Actual length before padding
    label: Optional[int] = None
    sample_id: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Union[np.ndarray, int, None]]:
        return {
            'input_ids': self.input_ids,
            'attention_mask': self.attention_mask,
            'length': self.length,
            'label': self.label,
            'sample_id': self.sample_id,
        }
    
    @classmethod
    def from_dict(cls, d: dict) -> 'EncodedSample':
        return cls(
            input_ids=d['input_ids'],
            attention_mask=d['attention_mask'],
            length=d['length'],
            label=d.get('label'),
            sample_id=d.get('sample_id'),
        )


class CodeVectorizer:
    """
    Vectorizer for converting token sequences to model input.
    
    Features:
    - Configurable padding (left/right)
    - Configurable truncation (head/tail/center)
    - BOS/EOS token insertion
    - Attention mask generation
    - Batch processing
    - Save/load functionality
    """
    
    def __init__(self, vocab: Vocabulary, config: VectorizerConfig = None):
        self.vocab = vocab
        self.config = config or VectorizerConfig()
        self._dtype = np.int32 if self.config.dtype == 'int32' else np.int64
        
    @property
    def pad_id(self) -> int:
        return self.vocab.pad_id
    
    @property
    def bos_id(self) -> int:
        return self.vocab.bos_id
    
    @property
    def eos_id(self) -> int:
        return self.vocab.eos_id
    
    def _compute_effective_max_length(self) -> int:
        """Compute max length accounting for special tokens."""
        effective = self.config.max_length
        if self.config.add_bos:
            effective -= 1
        if self.config.add_eos:
            effective -= 1
        return effective
        
    def encode(self, tokens: List[str], label: Optional[int] = None, 
               sample_id: Optional[int] = None) -> EncodedSample:
        """
        Encode token sequence to IDs with padding/truncation.
        
        Args:
            tokens: List of tokens to encode
            label: Optional label for the sample
            sample_id: Optional sample identifier
            
        Returns:
            EncodedSample with input_ids, attention_mask, and metadata
        """
        ids = self.vocab.tokens_to_ids(tokens)
        
        if self.config.add_bos:
            ids = [self.bos_id] + ids
        if self.config.add_eos:
            ids = ids + [self.eos_id]
        
        original_length = len(ids)
        
        if len(ids) > self.config.max_length:
            ids = self._truncate(ids)
        
        actual_length = len(ids)
        
        padded_ids, attention_mask = self._pad(ids)
        
        return EncodedSample(
            input_ids=np.array(padded_ids, dtype=self._dtype),
            attention_mask=np.array(attention_mask, dtype=self._dtype),
            length=actual_length,
            label=label,
            sample_id=sample_id,
        )
        
    def encode_batch(self, token_lists: List[List[str]], 
                     labels: Optional[List[int]] = None,
                     sample_ids: Optional[List[int]] = None,
                     show_progress: bool = False) -> List[EncodedSample]:
        """
        Encode multiple samples.
        
        Args:
            token_lists: List of token sequences
            labels: Optional list of labels
            sample_ids: Optional list of sample IDs
            show_progress: Show tqdm progress bar
            
        Returns:
            List of EncodedSample objects
        """
        if labels is None:
            labels = [None] * len(token_lists)
        if sample_ids is None:
            sample_ids = list(range(len(token_lists)))
            
        samples = []
        iterator = zip(token_lists, labels, sample_ids)
        
        if show_progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(list(iterator), desc="Vectorizing")
            except ImportError:
                pass
        
        for tokens, label, sid in iterator:
            samples.append(self.encode(tokens, label, sid))
            
        return samples
        
    def decode(self, ids: Union[List[int], np.ndarray], 
               skip_special: bool = True) -> List[str]:
        """
        Convert IDs back to tokens (for debugging).
        
        Args:
            ids: Sequence of token IDs
            skip_special: Skip PAD, BOS, EOS tokens
            
        Returns:
            List of token strings
        """
        if isinstance(ids, np.ndarray):
            ids = ids.tolist()
            
        tokens = []
        special_ids = {self.pad_id, self.bos_id, self.eos_id}
        
        for idx in ids:
            if skip_special and idx in special_ids:
                continue
            tokens.append(self.vocab.id_to_token(idx))
            
        return tokens
        
    def _truncate(self, ids: List[int]) -> List[int]:
        """
        Truncate to max_length based on strategy.
        
        Strategies:
        - 'tail': Keep first max_length tokens (remove from end)
        - 'head': Keep last max_length tokens (remove from beginning)
        - 'center': Keep beginning and end, remove middle
        """
        max_len = self.config.max_length
        
        if len(ids) <= max_len:
            return ids
            
        if self.config.truncation == 'tail':
            return ids[:max_len]
            
        elif self.config.truncation == 'head':
            return ids[-max_len:]
            
        elif self.config.truncation == 'center':
            half = max_len // 2
            remainder = max_len % 2
            return ids[:half + remainder] + ids[-half:]
            
        else:
            raise ValueError(f"Unknown truncation strategy: {self.config.truncation}")
        
    def _pad(self, ids: List[int]) -> Tuple[List[int], List[int]]:
        """
        Pad to max_length, return (padded_ids, attention_mask).
        
        Args:
            ids: List of token IDs
            
        Returns:
            Tuple of (padded_ids, attention_mask)
        """
        max_len = self.config.max_length
        current_len = len(ids)
        
        if current_len >= max_len:
            return ids[:max_len], [1] * max_len
            
        pad_length = max_len - current_len
        padding = [self.pad_id] * pad_length
        mask_pad = [0] * pad_length
        mask_real = [1] * current_len
        
        if self.config.padding == 'right':
            padded_ids = ids + padding
            attention_mask = mask_real + mask_pad
        elif self.config.padding == 'left':
            padded_ids = padding + ids
            attention_mask = mask_pad + mask_real
        else:
            raise ValueError(f"Unknown padding strategy: {self.config.padding}")
            
        return padded_ids, attention_mask
        
    def to_numpy_batch(self, samples: List[EncodedSample]) -> Dict[str, np.ndarray]:
        """
        Convert list of samples to batched numpy arrays.
        
        Args:
            samples: List of EncodedSample objects
            
        Returns:
            Dict with keys: 'input_ids', 'attention_mask', 'labels', 'lengths', 'sample_ids'
            Shapes: (B, L) for input_ids/attention_mask, (B,) for others
        """
        batch = {
            'input_ids': np.stack([s.input_ids for s in samples]),
            'attention_mask': np.stack([s.attention_mask for s in samples]),
            'lengths': np.array([s.length for s in samples], dtype=self._dtype),
        }
        
        if samples[0].label is not None:
            batch['labels'] = np.array([s.label for s in samples], dtype=self._dtype)
            
        if samples[0].sample_id is not None:
            batch['sample_ids'] = np.array([s.sample_id for s in samples], dtype=self._dtype)
            
        return batch
        
    def save_vectors(self, samples: List[EncodedSample], path: str,
                     compress: bool = True) -> None:
        """
        Save encoded samples to npz file.
        
        Args:
            samples: List of EncodedSample objects
            path: Output file path
            compress: Use compressed npz format
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        batch = self.to_numpy_batch(samples)
        
        batch['config'] = np.array([json.dumps(self.config.to_dict())])
        
        if compress:
            np.savez_compressed(path, **batch)
        else:
            np.savez(path, **batch)
            
    @staticmethod
    def load_vectors(path: str) -> Tuple[List[EncodedSample], Optional[VectorizerConfig]]:
        """
        Load encoded samples from npz file.
        
        Args:
            path: Path to npz file
            
        Returns:
            Tuple of (List of EncodedSample, VectorizerConfig or None)
        """
        data = np.load(path, allow_pickle=True)
        
        input_ids = data['input_ids']
        attention_mask = data['attention_mask']
        lengths = data['lengths']
        labels = data.get('labels', None)
        sample_ids = data.get('sample_ids', None)
        
        config = None
        if 'config' in data:
            config_str = str(data['config'][0])
            config = VectorizerConfig.from_dict(json.loads(config_str))
        
        samples = []
        for i in range(len(input_ids)):
            sample = EncodedSample(
                input_ids=input_ids[i],
                attention_mask=attention_mask[i],
                length=int(lengths[i]),
                label=int(labels[i]) if labels is not None else None,
                sample_id=int(sample_ids[i]) if sample_ids is not None else i,
            )
            samples.append(sample)
            
        return samples, config
    
    def get_stats(self, samples: List[EncodedSample]) -> Dict[str, Union[int, float]]:
        """Get statistics about encoded samples."""
        lengths = [s.length for s in samples]
        
        return {
            'num_samples': len(samples),
            'max_length': max(lengths),
            'min_length': min(lengths),
            'avg_length': sum(lengths) / len(lengths),
            'median_length': float(np.median(lengths)),
            'truncated_count': sum(1 for l in lengths if l >= self.config.max_length),
            'truncated_ratio': sum(1 for l in lengths if l >= self.config.max_length) / len(lengths),
            'config_max_length': self.config.max_length,
        }


def vectorize_chunk(tokens_list: List[List[str]], vocab: Vocabulary,
                    labels: List[int], sample_ids: List[int],
                    config: VectorizerConfig = None) -> Dict[str, np.ndarray]:
    """
    Vectorize a chunk of samples, return numpy arrays.
    
    Designed for parallel processing of large datasets.
    
    Args:
        tokens_list: List of token sequences
        vocab: Vocabulary for encoding
        labels: List of labels
        sample_ids: List of sample IDs
        config: Vectorizer configuration
        
    Returns:
        Dict with batched numpy arrays
    """
    vectorizer = CodeVectorizer(vocab, config)
    samples = vectorizer.encode_batch(tokens_list, labels, sample_ids)
    return vectorizer.to_numpy_batch(samples)


def vectorize_dataset_parallel(
    tokens_iterator: Iterator[Tuple[List[str], int, int]],
    vocab: Vocabulary,
    output_dir: str,
    config: VectorizerConfig = None,
    chunk_size: int = 10000,
    num_workers: int = 4,
    show_progress: bool = True
) -> List[str]:
    """
    Vectorize large dataset in parallel chunks.
    
    Args:
        tokens_iterator: Iterator yielding (tokens, label, sample_id)
        vocab: Vocabulary for encoding
        output_dir: Directory to save chunk files
        config: Vectorizer configuration
        chunk_size: Number of samples per chunk file
        num_workers: Number of parallel workers
        show_progress: Show progress bar
        
    Returns:
        List of paths to saved chunk files
    """
    from concurrent.futures import ProcessPoolExecutor, as_completed
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    vectorizer = CodeVectorizer(vocab, config)
    
    chunk_buffer = []
    chunk_idx = 0
    saved_paths = []
    
    iterator = tokens_iterator
    if show_progress:
        try:
            from tqdm import tqdm
            iterator = tqdm(tokens_iterator, desc="Vectorizing dataset")
        except ImportError:
            pass
    
    for tokens, label, sample_id in iterator:
        chunk_buffer.append((tokens, label, sample_id))
        
        if len(chunk_buffer) >= chunk_size:
            tokens_list = [t for t, _, _ in chunk_buffer]
            labels = [l for _, l, _ in chunk_buffer]
            sample_ids = [s for _, _, s in chunk_buffer]
            
            samples = vectorizer.encode_batch(tokens_list, labels, sample_ids)
            
            chunk_path = output_dir / f"chunk_{chunk_idx:04d}.npz"
            vectorizer.save_vectors(samples, str(chunk_path))
            saved_paths.append(str(chunk_path))
            
            chunk_buffer = []
            chunk_idx += 1
    
    if chunk_buffer:
        tokens_list = [t for t, _, _ in chunk_buffer]
        labels = [l for _, l, _ in chunk_buffer]
        sample_ids = [s for _, _, s in chunk_buffer]
        
        samples = vectorizer.encode_batch(tokens_list, labels, sample_ids)
        
        chunk_path = output_dir / f"chunk_{chunk_idx:04d}.npz"
        vectorizer.save_vectors(samples, str(chunk_path))
        saved_paths.append(str(chunk_path))
    
    return saved_paths


class StreamingVectorizer:
    """
    Memory-efficient vectorizer for large datasets.
    
    Processes data in chunks and saves to disk incrementally.
    """
    
    def __init__(self, vocab: Vocabulary, config: VectorizerConfig = None,
                 output_dir: str = None, chunk_size: int = 10000):
        self.vocab = vocab
        self.config = config or VectorizerConfig()
        self.output_dir = Path(output_dir) if output_dir else None
        self.chunk_size = chunk_size
        self.vectorizer = CodeVectorizer(vocab, config)
        
        self._buffer: List[EncodedSample] = []
        self._chunk_idx = 0
        self._saved_paths: List[str] = []
        
        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def add(self, tokens: List[str], label: Optional[int] = None,
            sample_id: Optional[int] = None) -> None:
        """Add a sample to the buffer."""
        sample = self.vectorizer.encode(tokens, label, sample_id)
        self._buffer.append(sample)
        
        if len(self._buffer) >= self.chunk_size:
            self._flush()
            
    def add_batch(self, token_lists: List[List[str]],
                  labels: Optional[List[int]] = None,
                  sample_ids: Optional[List[int]] = None) -> None:
        """Add multiple samples."""
        samples = self.vectorizer.encode_batch(token_lists, labels, sample_ids)
        self._buffer.extend(samples)
        
        while len(self._buffer) >= self.chunk_size:
            self._flush()
            
    def _flush(self) -> None:
        """Save buffer to disk and clear."""
        if not self._buffer or not self.output_dir:
            return
            
        chunk_samples = self._buffer[:self.chunk_size]
        self._buffer = self._buffer[self.chunk_size:]
        
        chunk_path = self.output_dir / f"chunk_{self._chunk_idx:04d}.npz"
        self.vectorizer.save_vectors(chunk_samples, str(chunk_path))
        self._saved_paths.append(str(chunk_path))
        self._chunk_idx += 1
        
    def finalize(self) -> List[str]:
        """Flush remaining buffer and return all saved paths."""
        if self._buffer and self.output_dir:
            chunk_path = self.output_dir / f"chunk_{self._chunk_idx:04d}.npz"
            self.vectorizer.save_vectors(self._buffer, str(chunk_path))
            self._saved_paths.append(str(chunk_path))
            self._buffer = []
            
        return self._saved_paths
    
    def get_all_samples(self) -> List[EncodedSample]:
        """Get all buffered samples (for small datasets)."""
        return self._buffer.copy()
