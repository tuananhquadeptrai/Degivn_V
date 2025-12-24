"""
Preprocessing Pipeline Orchestrator for Devign Dataset.

10-step pipeline with:
- Checkpointing for resume on session timeout
- Memory-efficient chunk processing with gc.collect()
- Parallel processing with joblib
- Config via YAML or CLI args
- Status reporting
- Disk space management
"""

from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any, Callable
from pathlib import Path
from datetime import datetime
from enum import Enum
import yaml
import json
import gc
import time
import shutil

import pandas as pd
import numpy as np

import sys
sys.path.insert(0, '/media/hdi/Hdii/Work/C Vul Devign/devign_pipeline')

from src.data.loader import DevignLoader
from src.data.cache import (
    ChunkCache, save_chunk, load_chunk, save_numpy, load_numpy,
    save_pickle, load_pickle, chunk_path, list_chunks
)
from src.vuln.dictionary import VulnDictionary, get_default_dictionary
from src.vuln.rules import extract_vuln_features, get_vulnerability_summary
from src.ast.parser import CFamilyParser, ParseResult
from src.graphs.cfg import CFGBuilder, CFG, serialize_cfg, deserialize_cfg
from src.graphs.dfg import DFGBuilder, DFG, serialize_dfg, deserialize_dfg
from src.slicing.slicer import CodeSlicer, SliceConfig, SliceType
from src.tokenization.tokenizer import CTokenizer
from src.tokenization.normalization import CodeNormalizer, NormalizationMaps
from src.tokenization.vocab import Vocabulary, VocabBuilder, VocabConfig
from src.utils.checkpoint import CheckpointManager
from src.utils.logging import get_logger
from src.vuln.vuln_lines import extract_vul_line_numbers


class StepStatus(Enum):
    PENDING = 'pending'
    IN_PROGRESS = 'in_progress'
    COMPLETED = 'completed'
    FAILED = 'failed'
    SKIPPED = 'skipped'


@dataclass
class SliceStatistics:
    """Track and aggregate slicing statistics to monitor fallback rates.
    
    This helps identify issues with AST parsing or graph building:
    - High WINDOW rate (>30%) indicates parser problems
    - High truncation rate indicates slice_max_len may be too small
    - High 1-slice rate indicates forward slicing issues
    """
    
    # Slice type counts
    backward_count: int = 0
    forward_count: int = 0
    both_count: int = 0
    window_count: int = 0  # FALLBACK - indicates parse/graph failure
    
    # Additional metrics
    total_samples: int = 0
    total_slices: int = 0
    samples_with_1_slice: int = 0
    truncated_slices: int = 0
    empty_slices: int = 0
    dropped_slices_due_to_max: int = 0
    
    # Parse/graph failure tracking
    parse_failures: int = 0
    cfg_failures: int = 0
    dfg_failures: int = 0
    
    def add_slice(self, slice_type: str, num_lines: int, max_lines: int = 256,
                  is_empty: bool = False) -> None:
        """Record a single slice."""
        self.total_slices += 1
        
        if slice_type == 'backward':
            self.backward_count += 1
        elif slice_type == 'forward':
            self.forward_count += 1
        elif slice_type == 'both':
            self.both_count += 1
        elif slice_type == 'window':
            self.window_count += 1
        
        if is_empty or num_lines == 0:
            self.empty_slices += 1
        
        # Check truncation (approximate: if slice has max lines)
        if num_lines >= max_lines * 0.95:  # 95% of max = likely truncated
            self.truncated_slices += 1
    
    def add_sample(self, num_slices: int, num_kept: Optional[int] = None) -> None:
        """Record a sample's slice count."""
        self.total_samples += 1
        if num_slices <= 1:
            self.samples_with_1_slice += 1
        if num_kept is not None and num_slices > num_kept:
            self.dropped_slices_due_to_max += (num_slices - num_kept)
    
    def record_failure(self, failure_type: str) -> None:
        """Record a parse/graph building failure."""
        if failure_type == 'parse':
            self.parse_failures += 1
        elif failure_type == 'cfg':
            self.cfg_failures += 1
        elif failure_type == 'dfg':
            self.dfg_failures += 1
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics with percentages."""
        total_typed = self.backward_count + self.forward_count + self.both_count + self.window_count
        
        def pct(val: int, total: int) -> float:
            return round(val / max(1, total) * 100, 2)
        
        return {
            'total_samples': self.total_samples,
            'total_slices': self.total_slices,
            
            # Slice type distribution
            'slice_types': {
                'backward': self.backward_count,
                'forward': self.forward_count,
                'both': self.both_count,
                'window_fallback': self.window_count,
            },
            'slice_type_percentages': {
                'backward': pct(self.backward_count, total_typed),
                'forward': pct(self.forward_count, total_typed),
                'both': pct(self.both_count, total_typed),
                'window_fallback': pct(self.window_count, total_typed),
            },
            
            # Quality metrics
            'quality_metrics': {
                'window_fallback_rate': pct(self.window_count, total_typed),
                'samples_with_1_slice': self.samples_with_1_slice,
                'samples_with_1_slice_rate': pct(self.samples_with_1_slice, self.total_samples),
                'truncated_slices': self.truncated_slices,
                'truncated_rate': pct(self.truncated_slices, self.total_slices),
                'empty_slices': self.empty_slices,
                'empty_rate': pct(self.empty_slices, self.total_slices),
            },
            
            # Failure tracking
            'failures': {
                'parse_failures': self.parse_failures,
                'cfg_failures': self.cfg_failures,
                'dfg_failures': self.dfg_failures,
                'total_failures': self.parse_failures + self.cfg_failures + self.dfg_failures,
            },
            
            # Health assessment
            'health': self._assess_health(),
        }
    
    def _assess_health(self) -> Dict[str, Any]:
        """Assess overall slicing health."""
        total_typed = self.backward_count + self.forward_count + self.both_count + self.window_count
        window_rate = self.window_count / max(1, total_typed) * 100
        one_slice_rate = self.samples_with_1_slice / max(1, self.total_samples) * 100
        truncate_rate = self.truncated_slices / max(1, self.total_slices) * 100
        
        issues = []
        if window_rate > 30:
            issues.append(f"HIGH window fallback rate ({window_rate:.1f}%) - check AST parser")
        elif window_rate > 10:
            issues.append(f"Moderate window fallback rate ({window_rate:.1f}%)")
        
        if one_slice_rate > 30:
            issues.append(f"HIGH 1-slice rate ({one_slice_rate:.1f}%) - check forward slicing")
        elif one_slice_rate > 15:
            issues.append(f"Moderate 1-slice rate ({one_slice_rate:.1f}%)")
        
        if truncate_rate > 10:
            issues.append(f"HIGH truncation rate ({truncate_rate:.1f}%) - consider increasing slice_max_len")
        
        status = 'GOOD' if not issues else ('WARNING' if len(issues) <= 1 else 'NEEDS_ATTENTION')
        
        return {
            'status': status,
            'issues': issues,
            'recommendations': self._get_recommendations(window_rate, one_slice_rate, truncate_rate),
        }
    
    def _get_recommendations(self, window_rate: float, one_slice_rate: float, 
                             truncate_rate: float) -> List[str]:
        """Get actionable recommendations."""
        recs = []
        
        if window_rate > 10:
            recs.append("Consider improving AST parser fallback handling")
            recs.append("Check if code samples have unusual syntax")
        
        if one_slice_rate > 15:
            recs.append("Implement window-based forward slice when forward slice is empty")
            recs.append("Review forward slicing algorithm for edge cases")
        
        if truncate_rate > 5:
            recs.append("Consider increasing slice_max_len from 256 to 384 or 512")
        
        if not recs:
            recs.append("Slicing quality looks good - no immediate action needed")
        
        return recs
    
    def to_dict(self) -> Dict[str, int]:
        """Convert to simple dict for serialization."""
        return {
            'backward_count': self.backward_count,
            'forward_count': self.forward_count,
            'both_count': self.both_count,
            'window_count': self.window_count,
            'total_samples': self.total_samples,
            'total_slices': self.total_slices,
            'samples_with_1_slice': self.samples_with_1_slice,
            'truncated_slices': self.truncated_slices,
            'empty_slices': self.empty_slices,
            'dropped_slices_due_to_max': self.dropped_slices_due_to_max,
            'parse_failures': self.parse_failures,
            'cfg_failures': self.cfg_failures,
            'dfg_failures': self.dfg_failures,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, int]) -> 'SliceStatistics':
        """Create from dict."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass 
class PipelineConfig:
    # Paths
    data_dir: str = '/kaggle/input/devign'
    output_dir: str = '/kaggle/working/processed'
    checkpoint_dir: str = '/kaggle/working/checkpoints'
    vuln_patterns_path: Optional[str] = None
    
    # Processing
    chunk_size: int = 2000
    n_jobs: int = 4
    
    # Slicing
    slice_type: str = 'backward'
    window_size: int = 15
    include_control_deps: bool = True
    include_data_deps: bool = True
    max_slice_depth: int = 5
    max_slices: int = 4
    slice_max_len: int = 256
    
    # Tokenization
    include_comments: bool = False
    
    # Normalization
    normalize_vars: bool = True
    normalize_funcs: bool = True
    normalize_literals: bool = True
    normalize_types: bool = False
    
    # Vocabulary
    min_freq: int = 2
    max_vocab_size: int = 50000
    
    # Vectorization
    max_seq_length: int = 512
    add_bos_eos: bool = True
    
    # Memory management
    gc_after_chunk: bool = True
    
    # JSONL export for demo/debugging
    export_jsonl: bool = False
    jsonl_subdir: str = 'jsonl'
    
    @classmethod
    def from_yaml(cls, path: str) -> 'PipelineConfig':
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
    
    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            yaml.dump(asdict(self), f, default_flow_style=False)
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class StepState:
    step: str
    status: StepStatus
    split: str
    chunk_idx: int = 0
    total_chunks: int = 0
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    error_message: Optional[str] = None
    samples_processed: int = 0


@dataclass
class PipelineState:
    current_step: str = 'load'
    current_split: str = 'train'
    steps: Dict[str, StepState] = field(default_factory=dict)
    config: Optional[Dict[str, Any]] = None
    started_at: Optional[str] = None
    last_updated: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'current_step': self.current_step,
            'current_split': self.current_split,
            'steps': {
                k: {
                    'step': v.step,
                    'status': v.status.value,
                    'split': v.split,
                    'chunk_idx': v.chunk_idx,
                    'total_chunks': v.total_chunks,
                    'start_time': v.start_time,
                    'end_time': v.end_time,
                    'error_message': v.error_message,
                    'samples_processed': v.samples_processed,
                }
                for k, v in self.steps.items()
            },
            'config': self.config,
            'started_at': self.started_at,
            'last_updated': self.last_updated,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PipelineState':
        state = cls(
            current_step=data.get('current_step', 'load'),
            current_split=data.get('current_split', 'train'),
            config=data.get('config'),
            started_at=data.get('started_at'),
            last_updated=data.get('last_updated'),
        )
        for k, v in data.get('steps', {}).items():
            state.steps[k] = StepState(
                step=v['step'],
                status=StepStatus(v['status']),
                split=v['split'],
                chunk_idx=v.get('chunk_idx', 0),
                total_chunks=v.get('total_chunks', 0),
                start_time=v.get('start_time'),
                end_time=v.get('end_time'),
                error_message=v.get('error_message'),
                samples_processed=v.get('samples_processed', 0),
            )
        return state


class JsonlExporter:
    """Export pipeline data to JSONL format for demo/debugging purposes.
    
    Creates human-readable JSONL files showing:
    - Original vs sliced code
    - Tokenization and normalization
    - Token IDs from vocabulary
    - Vulnerability features
    """
    
    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.stats = {
            'splits': {},
            'tokens': {},
            'tokens_with_ids': {},
        }
    
    def _append_line(self, path: Path, obj: dict) -> None:
        """Append a single JSON line to file"""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(obj, ensure_ascii=False) + '\n')
    
    def append_metadata_chunk(self, split: str, df: 'pd.DataFrame') -> None:
        """Export sample metadata to train/val/test.jsonl
        
        Shows: original code, sliced code, labels, vulnerability features
        """
        out_path = self.base_dir / f'{split}.jsonl'
        s_stats = self.stats['splits'].setdefault(split, {
            'num_samples': 0, 
            'num_vulnerable': 0,
            'num_with_vuln_issues': 0
        })
        
        for idx, row in df.iterrows():
            vuln_features = {}
            for k in df.columns:
                if k.startswith('vf_'):
                    val = row[k]
                    if pd.notna(val):
                        vuln_features[k[3:]] = float(val) if isinstance(val, (int, float)) else val
            
            original_code = row.get('func', '') or row.get('func_clean', '')
            sliced_code = row.get('sliced_code', '')
            target = bool(row.get('target', False))
            has_vuln_issues = bool(row.get('vuln_has_issues', False))
            
            obj = {
                'id': int(row.get('id', idx)) if pd.notna(row.get('id')) else int(idx),
                'split': split,
                'project': str(row.get('project', '')) if pd.notna(row.get('project')) else '',
                'commit_id': str(row.get('commit_id', '')) if pd.notna(row.get('commit_id')) else '',
                'target': target,
                'vuln_risk_score': float(row.get('vuln_risk_score', 0.0)) if pd.notna(row.get('vuln_risk_score')) else 0.0,
                'vuln_risk_level': str(row.get('vuln_risk_level', 'none')) if pd.notna(row.get('vuln_risk_level')) else 'none',
                'vuln_has_issues': has_vuln_issues,
                'vuln_features': vuln_features,
                'original_code': original_code,
                'sliced_code': sliced_code,
                'slice_stats': {
                    'original_lines': int(row.get('slice_original_lines', 0)) if pd.notna(row.get('slice_original_lines')) else len(original_code.split('\n')),
                    'slice_lines': int(row.get('slice_slice_lines', 0)) if pd.notna(row.get('slice_slice_lines')) else 0,
                    'slice_ratio': float(row.get('slice_slice_ratio', 0.0)) if pd.notna(row.get('slice_slice_ratio')) else 0.0,
                    'slice_type': str(row.get('slice_slice_type', 'backward')) if pd.notna(row.get('slice_slice_type')) else 'backward',
                },
            }
            
            self._append_line(out_path, obj)
            s_stats['num_samples'] += 1
            if target:
                s_stats['num_vulnerable'] += 1
            if has_vuln_issues:
                s_stats['num_with_vuln_issues'] += 1
    
    def append_tokens_chunk(
        self,
        split: str,
        df: 'pd.DataFrame',
        tokens_list: list,
        normalized_list: list,
    ) -> None:
        """Export tokens to tokens.jsonl
        
        Shows: raw tokens, normalized tokens for each sample
        """
        out_path = self.base_dir / 'tokens.jsonl'
        t_stats = self.stats['tokens'].setdefault(split, {
            'num_samples': 0,
            'total_tokens': 0,
            'total_normalized_tokens': 0,
        })
        
        for (idx, row), toks, norm in zip(df.iterrows(), tokens_list, normalized_list):
            obj = {
                'id': int(row.get('id', idx)) if pd.notna(row.get('id')) else int(idx),
                'split': split,
                'sliced_code': row.get('sliced_code', '') or row.get('func', ''),
                'tokens': toks,
                'normalized_tokens': norm,
                'token_count': len(toks),
                'normalized_token_count': len(norm),
            }
            
            self._append_line(out_path, obj)
            t_stats['num_samples'] += 1
            t_stats['total_tokens'] += len(toks)
            t_stats['total_normalized_tokens'] += len(norm)
    
    def append_tokens_with_ids_chunk(
        self,
        split: str,
        df: 'pd.DataFrame',
        normalized_list: list,
        ids_list: list,
        seq_lengths: list,
    ) -> None:
        """Export tokens with IDs to tokens_with_ids.jsonl
        
        Shows: normalized tokens, vocabulary IDs, sequence lengths
        """
        out_path = self.base_dir / 'tokens_with_ids.jsonl'
        twi_stats = self.stats['tokens_with_ids'].setdefault(split, {
            'num_samples': 0,
            'total_ids': 0,
        })
        
        for (idx, row), norm, ids, sl in zip(df.iterrows(), normalized_list, ids_list, seq_lengths):
            oov_rate = float(row.get('oov_rate', 0.0)) if pd.notna(row.get('oov_rate')) else 0.0
            
            obj = {
                'id': int(row.get('id', idx)) if pd.notna(row.get('id')) else int(idx),
                'split': split,
                'normalized_tokens': norm,
                'token_ids': ids,
                'seq_length': sl,
                'oov_rate': oov_rate,
            }
            
            self._append_line(out_path, obj)
            twi_stats['num_samples'] += 1
            twi_stats['total_ids'] += len(ids)
    
    def save_stats(self, output_dir: Path, vocab: 'Vocabulary' = None) -> None:
        """Save build_stats.json and split_info.json"""
        split_info = self.stats['splits']
        split_info_path = output_dir / 'split_info.json'
        with open(split_info_path, 'w', encoding='utf-8') as f:
            json.dump(split_info, f, indent=2)
        
        build_stats = {
            'splits': split_info,
            'tokens': self.stats['tokens'],
            'tokens_with_ids': self.stats['tokens_with_ids'],
        }
        if vocab is not None:
            build_stats['vocab'] = vocab.get_stats()
        
        build_stats_path = output_dir / 'build_stats.json'
        with open(build_stats_path, 'w', encoding='utf-8') as f:
            json.dump(build_stats, f, indent=2)


class PreprocessPipeline:
    """
    End-to-end preprocessing pipeline with checkpointing.
    
    10 Steps:
    0. load         - Load raw data from parquet files
    1. vuln_features - Extract vulnerability features using rules
    2. ast          - Parse AST using tree-sitter
    3. cfg          - Build Control Flow Graphs
    4. dfg          - Build Data Flow Graphs
    5. slice        - Code slicing based on vulnerability lines
    6. tokenize     - Tokenize sliced code
    7. normalize    - Normalize tokens (vars, funcs, literals)
    8. vocab        - Build vocabulary from training set
    9. vectorize    - Convert tokens to integer indices
    """
    
    STEPS = [
        'load',           # Step 0: Load raw data
        'vuln_features',  # Step 1: Extract vulnerability features
        'ast',            # Step 2: Parse AST
        'cfg',            # Step 3: Build CFG
        'dfg',            # Step 4: Build DFG
        'slice',          # Step 5: Code slicing
        'tokenize',       # Step 6: Tokenization
        'normalize',      # Step 7: Normalization
        'vocab',          # Step 8: Build vocabulary
        'vectorize',      # Step 9: Vectorization
    ]
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.logger = get_logger('pipeline')
        
        # Setup paths
        self.output_dir = Path(config.output_dir)
        self.checkpoint_dir = Path(config.checkpoint_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize managers
        self.checkpoint_mgr = CheckpointManager(str(self.checkpoint_dir))
        self.cache = ChunkCache(str(self.output_dir))
        
        # Initialize components (lazy loading)
        self._components_initialized = False
        
        # Load or create pipeline state
        self.state = self._load_state()
        
        # Initialize JSONL exporter if enabled
        if self.config.export_jsonl:
            self.jsonl_dir = self.output_dir / self.config.jsonl_subdir
            self.jsonl_exporter = JsonlExporter(self.jsonl_dir)
            self.logger.info(f"JSONL export enabled. Output: {self.jsonl_dir}")
        else:
            self.jsonl_exporter = None
        
    def _init_components(self) -> None:
        """Initialize all pipeline components (lazy loading)"""
        if self._components_initialized:
            return
            
        self.logger.info("Initializing pipeline components...")
        
        self.loader = DevignLoader(self.config.data_dir, self.config.chunk_size)
        
        if self.config.vuln_patterns_path:
            self.vuln_dict = VulnDictionary(config_path=self.config.vuln_patterns_path)
        else:
            self.vuln_dict = get_default_dictionary()
        
        self.parser = CFamilyParser()
        self.cfg_builder = CFGBuilder()
        self.dfg_builder = DFGBuilder()
        
        slice_type = SliceType.BACKWARD
        if self.config.slice_type == 'forward':
            slice_type = SliceType.FORWARD
        elif self.config.slice_type == 'both':
            slice_type = SliceType.BOTH
        elif self.config.slice_type == 'window':
            slice_type = SliceType.WINDOW
            
        self.slice_config = SliceConfig(
            slice_type=slice_type,
            window_size=self.config.window_size,
            include_control_deps=self.config.include_control_deps,
            include_data_deps=self.config.include_data_deps,
            max_depth=self.config.max_slice_depth,
        )
        self.slicer = CodeSlicer(self.slice_config)
        
        self.tokenizer = CTokenizer(include_comments=self.config.include_comments)
        self.normalizer = CodeNormalizer(
            normalize_vars=self.config.normalize_vars,
            normalize_funcs=self.config.normalize_funcs,
            normalize_literals=self.config.normalize_literals,
            normalize_types=self.config.normalize_types,
        )
        
        self.vocab_builder = VocabBuilder(VocabConfig(
            min_freq=self.config.min_freq,
            max_vocab_size=self.config.max_vocab_size,
        ))
        
        self.vocab: Optional[Vocabulary] = None
        
        self._components_initialized = True
        self.logger.info("Components initialized successfully")
    
    def _load_state(self) -> PipelineState:
        """Load pipeline state from checkpoint"""
        state_path = self.checkpoint_dir / 'pipeline_state.json'
        if state_path.exists():
            try:
                with open(state_path) as f:
                    data = json.load(f)
                self.logger.info(f"Loaded existing pipeline state from {state_path}")
                return PipelineState.from_dict(data)
            except Exception as e:
                self.logger.warning(f"Failed to load state: {e}, starting fresh")
        
        return PipelineState(
            started_at=datetime.now().isoformat(),
            config=self.config.to_dict(),
        )
    
    def _save_state(self) -> None:
        """Save pipeline state to checkpoint"""
        self.state.last_updated = datetime.now().isoformat()
        state_path = self.checkpoint_dir / 'pipeline_state.json'
        with open(state_path, 'w') as f:
            json.dump(self.state.to_dict(), f, indent=2)
    
    def _get_step_key(self, step: str, split: str) -> str:
        """Generate unique key for step+split combination"""
        return f"{step}_{split}"
    
    def _update_step_state(self, step: str, split: str, **kwargs) -> None:
        """Update step state and save checkpoint"""
        key = self._get_step_key(step, split)
        if key not in self.state.steps:
            self.state.steps[key] = StepState(step=step, status=StepStatus.PENDING, split=split)
        
        state = self.state.steps[key]
        for k, v in kwargs.items():
            if hasattr(state, k):
                setattr(state, k, v)
        
        self._save_state()
    
    def _is_step_completed(self, step: str, split: str) -> bool:
        """Check if a step is completed for a split"""
        key = self._get_step_key(step, split)
        if key in self.state.steps:
            return self.state.steps[key].status == StepStatus.COMPLETED
        return False
    
    def _get_resume_chunk(self, step: str, split: str) -> int:
        """Get chunk index to resume from"""
        key = self._get_step_key(step, split)
        if key in self.state.steps:
            state = self.state.steps[key]
            if state.status == StepStatus.IN_PROGRESS:
                return state.chunk_idx
        return 0
    
    def _count_chunks(self, split: str) -> int:
        """Count total chunks for a split"""
        chunks_list = list(self.loader.iter_chunks(split=split))
        return len(chunks_list)
    
    def run(self, start_step: Optional[str] = None, 
            end_step: Optional[str] = None,
            split: str = 'train') -> None:
        """
        Run pipeline from start_step to end_step.
        Resumes from checkpoint if available.
        """
        self._init_components()
        
        # Determine step range
        if start_step is None:
            # Find first incomplete step
            for step in self.STEPS:
                if not self._is_step_completed(step, split):
                    start_step = step
                    break
            if start_step is None:
                self.logger.info(f"All steps completed for split '{split}'")
                return
        
        if end_step is None:
            end_step = self.STEPS[-1]
        
        start_idx = self.STEPS.index(start_step)
        end_idx = self.STEPS.index(end_step)
        
        self.logger.info(f"Running pipeline: {start_step} -> {end_step} for split '{split}'")
        
        for step_idx in range(start_idx, end_idx + 1):
            step = self.STEPS[step_idx]
            
            if self._is_step_completed(step, split):
                self.logger.info(f"Step '{step}' already completed, skipping")
                continue
            
            try:
                self.run_step(step, split)
            except Exception as e:
                self.logger.error(f"Step '{step}' failed: {e}")
                self._update_step_state(
                    step, split,
                    status=StepStatus.FAILED,
                    error_message=str(e),
                    end_time=datetime.now().isoformat()
                )
                raise
        
        self.logger.info(f"Pipeline completed for split '{split}'")
    
    def run_step(self, step: str, split: str = 'train') -> None:
        """Run a single pipeline step"""
        self._init_components()
        
        if step not in self.STEPS:
            raise ValueError(f"Unknown step: {step}. Valid steps: {self.STEPS}")
        
        self.logger.info(f"Running step: {step} for split: {split}")
        
        self._update_step_state(
            step, split,
            status=StepStatus.IN_PROGRESS,
            start_time=datetime.now().isoformat()
        )
        
        step_method = getattr(self, f'_run_step_{step}')
        step_method(split)
        
        self._update_step_state(
            step, split,
            status=StepStatus.COMPLETED,
            end_time=datetime.now().isoformat()
        )
        
        self.logger.info(f"Step '{step}' completed")
    
    def _run_step_load(self, split: str) -> None:
        """Step 0: Load raw data and save as chunks"""
        self.logger.info(f"Loading data for split: {split}")
        
        chunk_idx = self._get_resume_chunk('load', split)
        total_processed = 0
        
        for i, chunk_df in enumerate(self.loader.iter_chunks(split=split)):
            if i < chunk_idx:
                continue
            
            # Save chunk
            path = chunk_path(str(self.output_dir), 'raw', i, 'parquet')
            save_chunk(chunk_df, path)
            
            total_processed += len(chunk_df)
            self._update_step_state(
                'load', split,
                chunk_idx=i + 1,
                samples_processed=total_processed
            )
            
            self.logger.info(f"Saved chunk {i}: {len(chunk_df)} samples")
            
            if self.config.gc_after_chunk:
                del chunk_df
                gc.collect()
        
        self.logger.info(f"Loaded {total_processed} samples for split '{split}'")
    
    def _run_step_vuln_features(self, split: str) -> None:
        """Step 1: Extract vulnerability features for each chunk
        
        Uses get_vulnerability_summary for rich analysis including:
        - risk_score (0-1 float)
        - risk_level (none/low/medium/high)
        - has_vulnerabilities flag
        - Individual feature counts as vf_* columns
        """
        raw_chunks = list_chunks(str(self.output_dir), 'raw', 'parquet')
        
        if not raw_chunks:
            raise RuntimeError("No raw chunks found. Run 'load' step first.")
        
        chunk_idx = self._get_resume_chunk('vuln_features', split)
        total_processed = 0
        
        vuln_feature_keys = None
        
        for i, chunk_path_str in enumerate(raw_chunks):
            if i < chunk_idx:
                continue
            
            chunk_df = load_chunk(chunk_path_str)
            
            risk_scores = []
            risk_levels = []
            has_vulns = []
            feature_values = {}
            
            for _, row in chunk_df.iterrows():
                code = row.get('func', '') or row.get('func_clean', '')
                
                summary = get_vulnerability_summary(code, self.vuln_dict)
                
                risk_scores.append(summary.get('risk_score', 0.0))
                risk_levels.append(summary.get('risk_level', 'none'))
                has_vulns.append(summary.get('has_vulnerabilities', False))
                
                features = summary.get('features', {})
                
                if vuln_feature_keys is None:
                    vuln_feature_keys = list(features.keys())
                    for key in vuln_feature_keys:
                        feature_values[key] = []
                
                for key in vuln_feature_keys:
                    feature_values[key].append(features.get(key, 0))
            
            chunk_df['vuln_risk_score'] = risk_scores
            chunk_df['vuln_risk_level'] = risk_levels
            chunk_df['vuln_has_issues'] = has_vulns
            
            for key in (vuln_feature_keys or []):
                chunk_df[f'vf_{key}'] = feature_values[key]
            
            out_path = chunk_path(str(self.output_dir), 'vuln_features', i, 'parquet')
            save_chunk(chunk_df, out_path)
            
            total_processed += len(chunk_df)
            self._update_step_state(
                'vuln_features', split,
                chunk_idx=i + 1,
                total_chunks=len(raw_chunks),
                samples_processed=total_processed
            )
            
            self.logger.info(f"Processed vuln features for chunk {i}: {sum(has_vulns)}/{len(has_vulns)} with issues")
            
            if self.config.gc_after_chunk:
                del chunk_df, risk_scores, risk_levels, has_vulns, feature_values
                gc.collect()
    
    def _run_step_ast(self, split: str) -> None:
        """Step 2: Parse AST for each chunk"""
        input_chunks = list_chunks(str(self.output_dir), 'vuln_features', 'parquet')
        
        if not input_chunks:
            input_chunks = list_chunks(str(self.output_dir), 'raw', 'parquet')
        
        if not input_chunks:
            raise RuntimeError("No input chunks found for AST parsing")
        
        chunk_idx = self._get_resume_chunk('ast', split)
        total_processed = 0
        
        for i, chunk_path_str in enumerate(input_chunks):
            if i < chunk_idx:
                continue
            
            chunk_df = load_chunk(chunk_path_str)
            
            # Parse AST for each sample
            ast_results = []
            for _, row in chunk_df.iterrows():
                code = row.get('func', '') or row.get('func_clean', '')
                parse_result = self.parser.parse_with_fallback(code)
                
                ast_data = {
                    'has_errors': parse_result.has_errors if parse_result else True,
                    'error_count': parse_result.error_count if parse_result else 0,
                    'node_count': len(parse_result.nodes) if parse_result else 0,
                }
                ast_results.append(ast_data)
            
            # Add AST metadata
            ast_df = pd.DataFrame(ast_results)
            for col in ast_df.columns:
                chunk_df[f'ast_{col}'] = ast_df[col].values
            
            # Save parsed AST data separately (pickle for complex objects)
            ast_objects = []
            for _, row in chunk_df.iterrows():
                code = row.get('func', '') or row.get('func_clean', '')
                parse_result = self.parser.parse_with_fallback(code)
                if parse_result:
                    ast_objects.append({
                        'nodes': [(n.node_type, n.start_line, n.end_line) for n in parse_result.nodes],
                        'root_index': parse_result.root_index,
                    })
                else:
                    ast_objects.append(None)
            
            ast_pkl_path = chunk_path(str(self.output_dir), 'ast_objects', i, 'pkl')
            save_pickle(ast_objects, ast_pkl_path)
            
            # Save metadata chunk
            out_path = chunk_path(str(self.output_dir), 'ast', i, 'parquet')
            save_chunk(chunk_df, out_path)
            
            total_processed += len(chunk_df)
            self._update_step_state(
                'ast', split,
                chunk_idx=i + 1,
                total_chunks=len(input_chunks),
                samples_processed=total_processed
            )
            
            self.logger.info(f"Parsed AST for chunk {i}: {len(ast_objects)} samples")
            
            if self.config.gc_after_chunk:
                del chunk_df, ast_df, ast_objects
                gc.collect()
    
    def _run_step_cfg(self, split: str) -> None:
        """Step 3: Build CFG (requires AST)"""
        input_chunks = list_chunks(str(self.output_dir), 'ast', 'parquet')
        
        if not input_chunks:
            raise RuntimeError("No AST chunks found. Run 'ast' step first.")
        
        chunk_idx = self._get_resume_chunk('cfg', split)
        total_processed = 0
        
        for i, chunk_path_str in enumerate(input_chunks):
            if i < chunk_idx:
                continue
            
            chunk_df = load_chunk(chunk_path_str)
            
            cfg_objects = []
            cfg_stats = []
            
            for _, row in chunk_df.iterrows():
                code = row.get('func', '') or row.get('func_clean', '')
                
                parse_result = self.parser.parse_with_fallback(code)
                if parse_result and parse_result.nodes:
                    cfg = self.cfg_builder.build(parse_result)
                    if cfg:
                        cfg_objects.append(serialize_cfg(cfg))
                        cfg_stats.append({
                            'block_count': len(cfg.blocks),
                            'edge_count': len(cfg.edges),
                        })
                    else:
                        cfg_objects.append(None)
                        cfg_stats.append({'block_count': 0, 'edge_count': 0})
                else:
                    cfg_objects.append(None)
                    cfg_stats.append({'block_count': 0, 'edge_count': 0})
            
            # Save CFG objects
            cfg_pkl_path = chunk_path(str(self.output_dir), 'cfg_objects', i, 'pkl')
            save_pickle(cfg_objects, cfg_pkl_path)
            
            # Add stats to dataframe
            stats_df = pd.DataFrame(cfg_stats)
            for col in stats_df.columns:
                chunk_df[f'cfg_{col}'] = stats_df[col].values
            
            out_path = chunk_path(str(self.output_dir), 'cfg', i, 'parquet')
            save_chunk(chunk_df, out_path)
            
            total_processed += len(chunk_df)
            self._update_step_state(
                'cfg', split,
                chunk_idx=i + 1,
                total_chunks=len(input_chunks),
                samples_processed=total_processed
            )
            
            self.logger.info(f"Built CFG for chunk {i}")
            
            if self.config.gc_after_chunk:
                del chunk_df, cfg_objects, cfg_stats
                gc.collect()
    
    def _run_step_dfg(self, split: str) -> None:
        """Step 4: Build DFG (requires AST)"""
        input_chunks = list_chunks(str(self.output_dir), 'cfg', 'parquet')
        
        if not input_chunks:
            input_chunks = list_chunks(str(self.output_dir), 'ast', 'parquet')
        
        if not input_chunks:
            raise RuntimeError("No input chunks found for DFG building")
        
        chunk_idx = self._get_resume_chunk('dfg', split)
        total_processed = 0
        
        for i, chunk_path_str in enumerate(input_chunks):
            if i < chunk_idx:
                continue
            
            chunk_df = load_chunk(chunk_path_str)
            
            dfg_objects = []
            dfg_stats = []
            
            for _, row in chunk_df.iterrows():
                code = row.get('func', '') or row.get('func_clean', '')
                
                # Get vulnerability lines as focus
                vul_lines_raw = row.get('vul_lines', {})
                if isinstance(vul_lines_raw, str):
                    import json as jn
                    try:
                        vul_lines_raw = jn.loads(vul_lines_raw)
                    except:
                        vul_lines_raw = {}
                
                focus_lines = extract_vul_line_numbers(vul_lines_raw)
                
                parse_result = self.parser.parse_with_fallback(code)
                if parse_result and parse_result.nodes:
                    dfg = self.dfg_builder.build(parse_result, focus_lines=focus_lines or None)
                    if dfg:
                        dfg_objects.append(serialize_dfg(dfg))
                        dfg_stats.append({
                            'node_count': len(dfg.nodes),
                            'edge_count': len(dfg.edges),
                            'def_count': sum(len(v) for v in dfg.var_defs.values()),
                            'use_count': sum(len(v) for v in dfg.var_uses.values()),
                        })
                    else:
                        dfg_objects.append(None)
                        dfg_stats.append({'node_count': 0, 'edge_count': 0, 'def_count': 0, 'use_count': 0})
                else:
                    dfg_objects.append(None)
                    dfg_stats.append({'node_count': 0, 'edge_count': 0, 'def_count': 0, 'use_count': 0})
            
            # Save DFG objects
            dfg_pkl_path = chunk_path(str(self.output_dir), 'dfg_objects', i, 'pkl')
            save_pickle(dfg_objects, dfg_pkl_path)
            
            # Add stats
            stats_df = pd.DataFrame(dfg_stats)
            for col in stats_df.columns:
                chunk_df[f'dfg_{col}'] = stats_df[col].values
            
            out_path = chunk_path(str(self.output_dir), 'dfg', i, 'parquet')
            save_chunk(chunk_df, out_path)
            
            total_processed += len(chunk_df)
            self._update_step_state(
                'dfg', split,
                chunk_idx=i + 1,
                total_chunks=len(input_chunks),
                samples_processed=total_processed
            )
            
            self.logger.info(f"Built DFG for chunk {i}")
            
            if self.config.gc_after_chunk:
                del chunk_df, dfg_objects, dfg_stats
                gc.collect()
    
    def _run_step_slice(self, split: str) -> None:
        """Step 5: Slice code (requires CFG, DFG)
        
        Now includes SliceStatistics tracking for monitoring:
        - Window fallback rate (indicates parser/graph issues)
        - 1-slice sample rate (indicates forward slicing issues)
        - Truncation rate (indicates slice_max_len issues)
        """
        input_chunks = list_chunks(str(self.output_dir), 'dfg', 'parquet')
        
        if not input_chunks:
            raise RuntimeError("No DFG chunks found. Run 'dfg' step first.")
        
        chunk_idx = self._get_resume_chunk('slice', split)
        total_processed = 0
        
        # Initialize slice statistics tracker
        slice_statistics = SliceStatistics()
        
        # Try to load existing statistics if resuming
        stats_path = self.output_dir / f'slice_statistics_{split}.json'
        if chunk_idx > 0 and stats_path.exists():
            try:
                with open(stats_path) as f:
                    saved_stats = json.load(f)
                if 'raw_counts' in saved_stats:
                    slice_statistics = SliceStatistics.from_dict(saved_stats['raw_counts'])
                    self.logger.info(f"Resumed slice statistics from chunk {chunk_idx}")
            except Exception as e:
                self.logger.warning(f"Could not load saved slice statistics: {e}")
        
        for i, chunk_path_str in enumerate(input_chunks):
            if i < chunk_idx:
                continue
            
            chunk_df = load_chunk(chunk_path_str)
            
            # Try to load precomputed graphs
            cfg_objects = None
            dfg_objects = None
            try:
                cfg_pkl = chunk_path(str(self.output_dir), 'cfg_objects', i, 'pkl')
                dfg_pkl = chunk_path(str(self.output_dir), 'dfg_objects', i, 'pkl')
                if Path(cfg_pkl).exists():
                    cfg_objects = load_pickle(cfg_pkl)
                    self.logger.debug(f"Loaded precomputed CFG objects for chunk {i}")
                if Path(dfg_pkl).exists():
                    dfg_objects = load_pickle(dfg_pkl)
                    self.logger.debug(f"Loaded precomputed DFG objects for chunk {i}")
            except Exception as e:
                self.logger.warning(f"Could not load precomputed graphs for chunk {i}: {e}")
            
            sliced_codes = []
            slice_stats = []
            
            for row_idx, (_, row) in enumerate(chunk_df.iterrows()):
                code = row.get('func', '') or row.get('func_clean', '')
                
                # Get vulnerability lines
                vul_lines_raw = row.get('vul_lines', {})
                if isinstance(vul_lines_raw, str):
                    import json as jn
                    try:
                        vul_lines_raw = jn.loads(vul_lines_raw)
                    except:
                        vul_lines_raw = {}
                
                criterion_lines = extract_vul_line_numbers(vul_lines_raw)
                
                # Default to all lines if no vuln lines
                if not criterion_lines:
                    lines = code.split('\n')
                    criterion_lines = list(range(1, len(lines) + 1))
                
                # Get precomputed graphs if available
                cfg = None
                dfg = None
                cfg_failed = False
                dfg_failed = False
                
                if cfg_objects and row_idx < len(cfg_objects):
                    if cfg_objects[row_idx]:
                        cfg = deserialize_cfg(cfg_objects[row_idx])
                    else:
                        cfg_failed = True
                        slice_statistics.record_failure('cfg')
                
                if dfg_objects and row_idx < len(dfg_objects):
                    if dfg_objects[row_idx]:
                        dfg = deserialize_dfg(dfg_objects[row_idx])
                    else:
                        dfg_failed = True
                        slice_statistics.record_failure('dfg')
                
                # Perform slicing with precomputed graphs (or rebuild if not available)
                slice_result = self.slicer.slice(code, criterion_lines, cfg=cfg, dfg=dfg)
                
                slice_type_str = slice_result.slice_type.value
                num_slice_lines = len(slice_result.included_lines)
                is_empty = not slice_result.code.strip()
                
                # Track slice statistics
                slice_statistics.add_slice(
                    slice_type=slice_type_str,
                    num_lines=num_slice_lines,
                    max_lines=256,
                    is_empty=is_empty
                )
                
                sliced_codes.append(slice_result.code)
                slice_stats.append({
                    'slice_lines': num_slice_lines,
                    'original_lines': len(code.split('\n')),
                    'slice_ratio': num_slice_lines / max(1, len(code.split('\n'))),
                    'slice_type': slice_type_str,
                    'is_window_fallback': slice_type_str == 'window',
                    'cfg_available': cfg is not None,
                    'dfg_available': dfg is not None,
                })
            
            # Track samples (1 slice per sample in this version)
            for _ in range(len(chunk_df)):
                slice_statistics.add_sample(num_slices=1)
            
            # Add sliced code and stats
            chunk_df['sliced_code'] = sliced_codes
            stats_df = pd.DataFrame(slice_stats)
            for col in stats_df.columns:
                chunk_df[f'slice_{col}'] = stats_df[col].values
            
            out_path = chunk_path(str(self.output_dir), 'sliced', i, 'parquet')
            save_chunk(chunk_df, out_path)
            
            # Export to JSONL if enabled
            if self.jsonl_exporter is not None:
                self.jsonl_exporter.append_metadata_chunk(split, chunk_df)
            
            total_processed += len(chunk_df)
            self._update_step_state(
                'slice', split,
                chunk_idx=i + 1,
                total_chunks=len(input_chunks),
                samples_processed=total_processed
            )
            
            # Log chunk progress with slice type breakdown
            chunk_window_count = sum(1 for s in slice_stats if s['slice_type'] == 'window')
            chunk_window_pct = chunk_window_count / max(1, len(slice_stats)) * 100
            self.logger.info(
                f"Sliced chunk {i}: {len(chunk_df)} samples, "
                f"window_fallback={chunk_window_count} ({chunk_window_pct:.1f}%)"
            )
            
            # Save intermediate statistics (for resume capability)
            self._save_slice_statistics(slice_statistics, split)
            
            if self.config.gc_after_chunk:
                del chunk_df, sliced_codes, slice_stats
                gc.collect()
        
        # Save final statistics summary
        self._save_slice_statistics(slice_statistics, split, final=True)
        
        # Log final summary
        summary = slice_statistics.get_summary()
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"SLICE STATISTICS SUMMARY ({split})")
        self.logger.info(f"{'='*60}")
        self.logger.info(f"Total samples: {summary['total_samples']}")
        self.logger.info(f"Total slices: {summary['total_slices']}")
        self.logger.info(f"Slice types: {summary['slice_types']}")
        self.logger.info(f"Window fallback rate: {summary['slice_type_percentages']['window_fallback']:.1f}%")
        self.logger.info(f"1-slice rate: {summary['quality_metrics']['samples_with_1_slice_rate']:.1f}%")
        self.logger.info(f"Truncation rate: {summary['quality_metrics']['truncated_rate']:.1f}%")
        self.logger.info(f"Health status: {summary['health']['status']}")
        if summary['health']['issues']:
            self.logger.warning(f"Issues: {summary['health']['issues']}")
        self.logger.info(f"{'='*60}\n")
    
    def _save_slice_statistics(self, stats: SliceStatistics, split: str, 
                                final: bool = False) -> None:
        """Save slice statistics to JSON file."""
        stats_path = self.output_dir / f'slice_statistics_{split}.json'
        
        output = {
            'split': split,
            'final': final,
            'timestamp': datetime.now().isoformat(),
            'raw_counts': stats.to_dict(),
            'summary': stats.get_summary(),
        }
        
        with open(stats_path, 'w') as f:
            json.dump(output, f, indent=2)
        
        if final:
            self.logger.info(f"Saved final slice statistics to {stats_path}")
    
    def _run_step_tokenize(self, split: str) -> None:
        """Step 6: Tokenize sliced code"""
        input_chunks = list_chunks(str(self.output_dir), 'sliced', 'parquet')
        
        if not input_chunks:
            raise RuntimeError("No sliced chunks found. Run 'slice' step first.")
        
        chunk_idx = self._get_resume_chunk('tokenize', split)
        total_processed = 0
        
        for i, chunk_path_str in enumerate(input_chunks):
            if i < chunk_idx:
                continue
            
            chunk_df = load_chunk(chunk_path_str)
            
            tokenized_results = []
            token_stats = []
            
            for _, row in chunk_df.iterrows():
                code = row.get('sliced_code', '') or row.get('func', '')
                
                tokens = self.tokenizer.tokenize(code)
                token_texts = [t.text for t in tokens]
                
                tokenized_results.append(token_texts)
                token_stats.append({
                    'token_count': len(token_texts),
                    'unique_tokens': len(set(token_texts)),
                })
            
            # Save tokens as pickle (lists don't fit well in parquet)
            tokens_pkl_path = chunk_path(str(self.output_dir), 'tokens', i, 'pkl')
            save_pickle(tokenized_results, tokens_pkl_path)
            
            # Add stats
            stats_df = pd.DataFrame(token_stats)
            for col in stats_df.columns:
                chunk_df[f'token_{col}'] = stats_df[col].values
            
            out_path = chunk_path(str(self.output_dir), 'tokenized', i, 'parquet')
            save_chunk(chunk_df, out_path)
            
            total_processed += len(chunk_df)
            self._update_step_state(
                'tokenize', split,
                chunk_idx=i + 1,
                total_chunks=len(input_chunks),
                samples_processed=total_processed
            )
            
            self.logger.info(f"Tokenized chunk {i}")
            
            if self.config.gc_after_chunk:
                del chunk_df, tokenized_results, token_stats
                gc.collect()
    
    def _run_step_normalize(self, split: str) -> None:
        """Step 7: Normalize tokens"""
        input_chunks = list_chunks(str(self.output_dir), 'tokenized', 'parquet')
        tokens_chunks = list_chunks(str(self.output_dir), 'tokens', 'pkl')
        
        if not input_chunks or not tokens_chunks:
            raise RuntimeError("No tokenized chunks found. Run 'tokenize' step first.")
        
        chunk_idx = self._get_resume_chunk('normalize', split)
        total_processed = 0
        
        for i, (chunk_path_str, tokens_path) in enumerate(zip(input_chunks, tokens_chunks)):
            if i < chunk_idx:
                continue
            
            chunk_df = load_chunk(chunk_path_str)
            tokenized_results = load_pickle(tokens_path)
            
            normalized_results = []
            norm_maps_list = []
            
            for tokens in tokenized_results:
                # Create Token objects for normalizer
                from src.tokenization.normalization import Token as NormToken, TokenType as NormTokenType
                token_objs = [NormToken(value=t, type=NormTokenType.IDENTIFIER) for t in tokens]
                
                normalized, maps = self.normalizer.normalize_tokens(token_objs)
                normalized_results.append(normalized)
                norm_maps_list.append({
                    'var_map': maps.var_map,
                    'func_map': maps.func_map,
                })
            
            # Save normalized tokens
            norm_pkl_path = chunk_path(str(self.output_dir), 'normalized', i, 'pkl')
            save_pickle(normalized_results, norm_pkl_path)
            
            # Save normalization maps
            maps_pkl_path = chunk_path(str(self.output_dir), 'norm_maps', i, 'pkl')
            save_pickle(norm_maps_list, maps_pkl_path)
            
            # Add stats
            norm_stats = [{
                'normalized_token_count': len(tokens),
                'var_count': len(m['var_map']),
                'func_count': len(m['func_map']),
            } for tokens, m in zip(normalized_results, norm_maps_list)]
            
            stats_df = pd.DataFrame(norm_stats)
            for col in stats_df.columns:
                chunk_df[f'norm_{col}'] = stats_df[col].values
            
            out_path = chunk_path(str(self.output_dir), 'normalized_meta', i, 'parquet')
            save_chunk(chunk_df, out_path)
            
            # Export tokens to JSONL if enabled
            if self.jsonl_exporter is not None:
                self.jsonl_exporter.append_tokens_chunk(
                    split, chunk_df, tokenized_results, normalized_results
                )
            
            total_processed += len(chunk_df)
            self._update_step_state(
                'normalize', split,
                chunk_idx=i + 1,
                total_chunks=len(input_chunks),
                samples_processed=total_processed
            )
            
            self.logger.info(f"Normalized chunk {i}")
            
            if self.config.gc_after_chunk:
                del chunk_df, tokenized_results, normalized_results
                gc.collect()
    
    def _run_step_vocab(self, split: str = 'train') -> None:
        """Step 8: Build vocabulary from all training chunks"""
        if split != 'train':
            self.logger.info("Vocabulary is built from train split only. Loading existing vocab.")
            vocab_path = self.output_dir / 'vocab.json'
            if vocab_path.exists():
                self.vocab = Vocabulary.load(str(vocab_path))
                self._update_step_state('vocab', split, status=StepStatus.SKIPPED)
                return
            else:
                raise RuntimeError("No vocabulary found. Run vocab step on train split first.")
        
        norm_chunks = list_chunks(str(self.output_dir), 'normalized', 'pkl')
        
        if not norm_chunks:
            raise RuntimeError("No normalized chunks found. Run 'normalize' step first.")
        
        self.logger.info(f"Building vocabulary from {len(norm_chunks)} chunks")
        
        # Reset vocab builder
        self.vocab_builder = VocabBuilder(VocabConfig(
            min_freq=self.config.min_freq,
            max_vocab_size=self.config.max_vocab_size,
        ))
        
        total_tokens = 0
        for i, chunk_path_str in enumerate(norm_chunks):
            normalized_results = load_pickle(chunk_path_str)
            
            for tokens in normalized_results:
                self.vocab_builder.add_tokens(tokens)
                total_tokens += len(tokens)
            
            self.logger.info(f"Added tokens from chunk {i}")
            
            if self.config.gc_after_chunk:
                del normalized_results
                gc.collect()
        
        # Build vocabulary
        self.vocab = self.vocab_builder.build_vocab()
        
        # Save vocabulary
        vocab_path = self.output_dir / 'vocab.json'
        self.vocab.save(str(vocab_path))
        
        # Save vocab stats
        stats = self.vocab.get_stats()
        stats['total_tokens_processed'] = total_tokens
        
        stats_path = self.output_dir / 'vocab_stats.json'
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        
        self.logger.info(f"Built vocabulary: {len(self.vocab)} tokens")
        
        self._update_step_state(
            'vocab', split,
            samples_processed=total_tokens
        )
    
    def _run_step_vectorize(self, split: str) -> None:
        """Step 9: Vectorize normalized tokens"""
        # Load vocabulary
        vocab_path = self.output_dir / 'vocab.json'
        if not vocab_path.exists():
            raise RuntimeError("No vocabulary found. Run 'vocab' step first.")
        
        self.vocab = Vocabulary.load(str(vocab_path))
        
        norm_chunks = list_chunks(str(self.output_dir), 'normalized', 'pkl')
        meta_chunks = list_chunks(str(self.output_dir), 'normalized_meta', 'parquet')
        
        if not norm_chunks:
            raise RuntimeError("No normalized chunks found. Run 'normalize' step first.")
        
        chunk_idx = self._get_resume_chunk('vectorize', split)
        total_processed = 0
        
        for i, (norm_path, meta_path) in enumerate(zip(norm_chunks, meta_chunks)):
            if i < chunk_idx:
                continue
            
            normalized_results = load_pickle(norm_path)
            chunk_df = load_chunk(meta_path)
            
            vectors = []
            attention_masks = []
            logical_ids_per_sample = []
            logical_seq_lengths = []
            
            for tokens in normalized_results:
                # Convert to indices
                indices = self.vocab.tokens_to_ids(tokens)
                
                # Add BOS/EOS if configured
                if self.config.add_bos_eos:
                    indices = [self.vocab.bos_id] + indices + [self.vocab.eos_id]
                
                # Truncate
                if len(indices) > self.config.max_seq_length:
                    indices = indices[:self.config.max_seq_length]
                
                # Save logical (unpadded) ids for JSONL export
                logical_ids = list(indices)
                logical_ids_per_sample.append(logical_ids)
                logical_seq_lengths.append(len(logical_ids))
                
                # Create attention mask before padding
                mask = [1] * len(indices)
                
                # Pad
                pad_len = self.config.max_seq_length - len(indices)
                indices = indices + [self.vocab.pad_id] * pad_len
                mask = mask + [0] * pad_len
                
                vectors.append(np.array(indices, dtype=np.int32))
                attention_masks.append(np.array(mask, dtype=np.int32))
            
            # Save as numpy arrays
            vectors_arr = np.stack(vectors)
            masks_arr = np.stack(attention_masks)
            
            # Get labels
            labels = chunk_df['target'].values.astype(np.int32) if 'target' in chunk_df.columns else np.zeros(len(vectors), dtype=np.int32)
            
            npz_path = chunk_path(str(self.output_dir), 'vectors', i, 'npz')
            save_numpy({
                'input_ids': vectors_arr,
                'attention_mask': masks_arr,
                'labels': labels,
            }, npz_path)
            
            # Add stats to metadata
            chunk_df['seq_length'] = [min(len(t) + 2, self.config.max_seq_length) for t in normalized_results]
            chunk_df['oov_rate'] = [self.vocab.get_oov_rate(t) for t in normalized_results]
            
            out_path = chunk_path(str(self.output_dir), 'vectorized', i, 'parquet')
            save_chunk(chunk_df, out_path)
            
            # Export tokens with IDs to JSONL if enabled
            if self.jsonl_exporter is not None:
                self.jsonl_exporter.append_tokens_with_ids_chunk(
                    split, chunk_df, normalized_results,
                    logical_ids_per_sample, logical_seq_lengths
                )
            
            total_processed += len(chunk_df)
            self._update_step_state(
                'vectorize', split,
                chunk_idx=i + 1,
                total_chunks=len(norm_chunks),
                samples_processed=total_processed
            )
            
            self.logger.info(f"Vectorized chunk {i}: shape={vectors_arr.shape}")
            
            if self.config.gc_after_chunk:
                del normalized_results, vectors, attention_masks, vectors_arr, masks_arr
                del logical_ids_per_sample, logical_seq_lengths
                gc.collect()
        
        self.logger.info(f"Vectorization complete: {total_processed} samples")
        
        # Save JSONL stats if enabled
        if self.jsonl_exporter is not None:
            self.jsonl_exporter.save_stats(self.output_dir, self.vocab)
            self.logger.info(f"JSONL export stats saved to {self.output_dir}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current pipeline status"""
        status = {
            'pipeline_state': self.state.to_dict(),
            'steps': {},
            'disk_usage': {},
        }
        
        for step in self.STEPS:
            step_dirs = [
                self.output_dir / step,
                self.output_dir / f"{step}_objects",
            ]
            
            chunks_found = 0
            size_mb = 0
            
            for d in step_dirs:
                if d.exists():
                    for f in d.iterdir():
                        chunks_found += 1
                        size_mb += f.stat().st_size / (1024 * 1024)
            
            status['steps'][step] = {
                'chunks': chunks_found,
                'size_mb': round(size_mb, 2),
            }
        
        # Total disk usage
        total_size = sum(
            f.stat().st_size 
            for f in self.output_dir.rglob('*') 
            if f.is_file()
        )
        status['disk_usage']['total_mb'] = round(total_size / (1024 * 1024), 2)
        status['disk_usage']['total_gb'] = round(total_size / (1024 * 1024 * 1024), 2)
        
        return status
    
    def clean_checkpoints(self, keep_steps: List[str] = None) -> None:
        """Remove intermediate checkpoints to save disk space"""
        if keep_steps is None:
            keep_steps = ['raw', 'vectorized', 'vectors']
        
        cleaned_size = 0
        
        for step in self.STEPS:
            if step in keep_steps:
                continue
            
            step_dirs = [
                self.output_dir / step,
                self.output_dir / f"{step}_objects",
                self.output_dir / f"{step}_meta",
            ]
            
            for d in step_dirs:
                if d.exists():
                    size = sum(f.stat().st_size for f in d.rglob('*') if f.is_file())
                    cleaned_size += size
                    shutil.rmtree(d)
                    self.logger.info(f"Removed {d}")
        
        self.logger.info(f"Cleaned {cleaned_size / (1024*1024):.2f} MB of intermediate data")
    
    def reset(self, step: Optional[str] = None, split: Optional[str] = None) -> None:
        """Reset pipeline state for a step or entirely"""
        if step and split:
            key = self._get_step_key(step, split)
            if key in self.state.steps:
                del self.state.steps[key]
        elif step:
            keys_to_remove = [k for k in self.state.steps.keys() if k.startswith(f"{step}_")]
            for k in keys_to_remove:
                del self.state.steps[k]
        else:
            self.state = PipelineState(
                started_at=datetime.now().isoformat(),
                config=self.config.to_dict(),
            )
        
        self._save_state()
        self.logger.info("Pipeline state reset")


def run_pipeline(config_path: str = None, **kwargs) -> None:
    """Convenience function to run full pipeline"""
    if config_path:
        config = PipelineConfig.from_yaml(config_path)
        for k, v in kwargs.items():
            if hasattr(config, k):
                setattr(config, k, v)
    else:
        config = PipelineConfig(**kwargs)
    
    pipeline = PreprocessPipeline(config)
    
    # Run for each split
    for split in ['train', 'validation', 'test']:
        try:
            pipeline.run(split=split)
        except Exception as e:
            pipeline.logger.error(f"Pipeline failed for split '{split}': {e}")
            raise


if __name__ == '__main__':
    # Quick test
    config = PipelineConfig(
        data_dir='/kaggle/input/devign',
        output_dir='/kaggle/working/processed',
        chunk_size=100,
        n_jobs=2,
    )
    
    pipeline = PreprocessPipeline(config)
    print("Pipeline status:")
    print(json.dumps(pipeline.get_status(), indent=2))
