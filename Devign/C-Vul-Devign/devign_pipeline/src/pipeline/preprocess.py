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
        """Step 5: Slice code (requires CFG, DFG)"""
        input_chunks = list_chunks(str(self.output_dir), 'dfg', 'parquet')
        
        if not input_chunks:
            raise RuntimeError("No DFG chunks found. Run 'dfg' step first.")
        
        chunk_idx = self._get_resume_chunk('slice', split)
        total_processed = 0
        
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
                if cfg_objects and row_idx < len(cfg_objects) and cfg_objects[row_idx]:
                    cfg = deserialize_cfg(cfg_objects[row_idx])
                if dfg_objects and row_idx < len(dfg_objects) and dfg_objects[row_idx]:
                    dfg = deserialize_dfg(dfg_objects[row_idx])
                
                # Perform slicing with precomputed graphs (or rebuild if not available)
                slice_result = self.slicer.slice(code, criterion_lines, cfg=cfg, dfg=dfg)
                
                sliced_codes.append(slice_result.code)
                slice_stats.append({
                    'slice_lines': len(slice_result.included_lines),
                    'original_lines': len(code.split('\n')),
                    'slice_ratio': len(slice_result.included_lines) / max(1, len(code.split('\n'))),
                    'slice_type': slice_result.slice_type.value,
                })
            
            # Add sliced code and stats
            chunk_df['sliced_code'] = sliced_codes
            stats_df = pd.DataFrame(slice_stats)
            for col in stats_df.columns:
                chunk_df[f'slice_{col}'] = stats_df[col].values
            
            out_path = chunk_path(str(self.output_dir), 'sliced', i, 'parquet')
            save_chunk(chunk_df, out_path)
            
            total_processed += len(chunk_df)
            self._update_step_state(
                'slice', split,
                chunk_idx=i + 1,
                total_chunks=len(input_chunks),
                samples_processed=total_processed
            )
            
            self.logger.info(f"Sliced chunk {i}")
            
            if self.config.gc_after_chunk:
                del chunk_df, sliced_codes, slice_stats
                gc.collect()
    
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
            
            for tokens in normalized_results:
                # Convert to indices
                indices = self.vocab.tokens_to_ids(tokens)
                
                # Add BOS/EOS if configured
                if self.config.add_bos_eos:
                    indices = [self.vocab.bos_id] + indices + [self.vocab.eos_id]
                
                # Truncate
                if len(indices) > self.config.max_seq_length:
                    indices = indices[:self.config.max_seq_length]
                
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
                gc.collect()
        
        self.logger.info(f"Vectorization complete: {total_processed} samples")
    
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
