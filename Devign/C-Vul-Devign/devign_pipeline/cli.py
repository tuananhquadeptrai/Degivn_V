#!/usr/bin/env python3
"""
Devign Preprocessing Pipeline CLI

Usage:
    python cli.py run --data-dir /path/to/data --output-dir /path/to/output
    python cli.py run --config config.yaml
    python cli.py step ast --config config.yaml --split train
    python cli.py status --checkpoint-dir /path/to/checkpoints
    python cli.py clean --keep raw vectorized
    python cli.py info
"""

import argparse
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.pipeline.preprocess import (
    PreprocessPipeline, 
    PipelineConfig, 
    run_pipeline
)
from src.utils.logging import setup_logging, get_logger


def cmd_run(args):
    """Run full preprocessing pipeline"""
    logger = get_logger('cli')
    
    if args.config:
        config = PipelineConfig.from_yaml(args.config)
    else:
        config = PipelineConfig()
    
    # Override with CLI args
    if args.data_dir:
        config.data_dir = args.data_dir
    if args.output_dir:
        config.output_dir = args.output_dir
    if args.checkpoint_dir:
        config.checkpoint_dir = args.checkpoint_dir
    if args.chunk_size:
        config.chunk_size = args.chunk_size
    if args.n_jobs:
        config.n_jobs = args.n_jobs
    if args.slice_type:
        config.slice_type = args.slice_type
    if args.max_seq_length:
        config.max_seq_length = args.max_seq_length
    if args.min_freq:
        config.min_freq = args.min_freq
    if args.max_vocab_size:
        config.max_vocab_size = args.max_vocab_size
    
    # Save config for reproducibility
    config_save_path = Path(config.output_dir) / 'pipeline_config.yaml'
    config_save_path.parent.mkdir(parents=True, exist_ok=True)
    config.save(str(config_save_path))
    logger.info(f"Config saved to {config_save_path}")
    
    pipeline = PreprocessPipeline(config)
    
    splits = [args.split] if args.split != 'all' else ['train', 'validation', 'test']
    
    for split in splits:
        logger.info(f"Processing split: {split}")
        try:
            pipeline.run(
                start_step=args.start_step,
                end_step=args.end_step,
                split=split
            )
        except Exception as e:
            logger.error(f"Pipeline failed for split '{split}': {e}")
            if not args.continue_on_error:
                raise
    
    # Print final status
    status = pipeline.get_status()
    logger.info(f"Pipeline completed. Total disk usage: {status['disk_usage']['total_gb']:.2f} GB")


def cmd_step(args):
    """Run a single pipeline step"""
    logger = get_logger('cli')
    
    if args.config:
        config = PipelineConfig.from_yaml(args.config)
    else:
        config = PipelineConfig()
    
    if args.output_dir:
        config.output_dir = args.output_dir
    if args.checkpoint_dir:
        config.checkpoint_dir = args.checkpoint_dir
    
    pipeline = PreprocessPipeline(config)
    
    logger.info(f"Running step: {args.step_name} for split: {args.split}")
    
    try:
        pipeline.run_step(args.step_name, args.split)
        logger.info(f"Step '{args.step_name}' completed successfully")
    except Exception as e:
        logger.error(f"Step '{args.step_name}' failed: {e}")
        raise


def cmd_status(args):
    """Check pipeline status"""
    if args.checkpoint_dir:
        config = PipelineConfig(checkpoint_dir=args.checkpoint_dir)
        if args.output_dir:
            config.output_dir = args.output_dir
    elif args.config:
        config = PipelineConfig.from_yaml(args.config)
    else:
        config = PipelineConfig()
    
    pipeline = PreprocessPipeline(config)
    status = pipeline.get_status()
    
    if args.json:
        print(json.dumps(status, indent=2))
    else:
        print("\n" + "=" * 60)
        print("PIPELINE STATUS")
        print("=" * 60)
        
        state = status['pipeline_state']
        print(f"\nCurrent step: {state['current_step']}")
        print(f"Current split: {state['current_split']}")
        print(f"Started: {state['started_at']}")
        print(f"Last updated: {state['last_updated']}")
        
        print("\n" + "-" * 40)
        print("STEP STATUS:")
        print("-" * 40)
        
        for step_key, step_data in state.get('steps', {}).items():
            status_str = step_data['status']
            emoji = {
                'completed': '✓',
                'in_progress': '⟳',
                'pending': '○',
                'failed': '✗',
                'skipped': '→'
            }.get(status_str, '?')
            
            chunk_info = ""
            if step_data.get('total_chunks', 0) > 0:
                chunk_info = f" [{step_data['chunk_idx']}/{step_data['total_chunks']}]"
            
            samples = step_data.get('samples_processed', 0)
            print(f"  {emoji} {step_key}: {status_str}{chunk_info} ({samples} samples)")
        
        print("\n" + "-" * 40)
        print("DISK USAGE:")
        print("-" * 40)
        
        for step, info in status['steps'].items():
            if info['chunks'] > 0:
                print(f"  {step}: {info['chunks']} chunks, {info['size_mb']:.1f} MB")
        
        print(f"\n  TOTAL: {status['disk_usage']['total_gb']:.2f} GB")
        print("=" * 60 + "\n")


def cmd_clean(args):
    """Clean intermediate checkpoints"""
    logger = get_logger('cli')
    
    if args.config:
        config = PipelineConfig.from_yaml(args.config)
    else:
        config = PipelineConfig()
    
    if args.output_dir:
        config.output_dir = args.output_dir
    if args.checkpoint_dir:
        config.checkpoint_dir = args.checkpoint_dir
    
    pipeline = PreprocessPipeline(config)
    
    keep_steps = args.keep if args.keep else ['raw', 'vectorized', 'vectors']
    
    if not args.yes:
        status = pipeline.get_status()
        print("\nSteps to be cleaned:")
        for step, info in status['steps'].items():
            if step not in keep_steps and info['chunks'] > 0:
                print(f"  - {step}: {info['size_mb']:.1f} MB")
        
        print(f"\nSteps to keep: {keep_steps}")
        response = input("\nProceed? [y/N]: ")
        if response.lower() != 'y':
            print("Aborted.")
            return
    
    pipeline.clean_checkpoints(keep_steps=keep_steps)
    logger.info("Cleanup completed")


def cmd_reset(args):
    """Reset pipeline state"""
    logger = get_logger('cli')
    
    if args.config:
        config = PipelineConfig.from_yaml(args.config)
    else:
        config = PipelineConfig()
    
    if args.checkpoint_dir:
        config.checkpoint_dir = args.checkpoint_dir
    
    pipeline = PreprocessPipeline(config)
    
    if not args.yes:
        if args.step:
            print(f"This will reset state for step '{args.step}'")
        else:
            print("This will reset ALL pipeline state")
        
        response = input("Proceed? [y/N]: ")
        if response.lower() != 'y':
            print("Aborted.")
            return
    
    pipeline.reset(step=args.step, split=args.split)
    logger.info("Pipeline state reset")


def cmd_info(args):
    """Show pipeline information"""
    print("\n" + "=" * 60)
    print("DEVIGN PREPROCESSING PIPELINE")
    print("=" * 60)
    
    print("\nSTEPS:")
    steps_info = [
        ("0. load", "Load raw data from parquet files"),
        ("1. vuln_features", "Extract vulnerability features using rules"),
        ("2. ast", "Parse AST using tree-sitter"),
        ("3. cfg", "Build Control Flow Graphs"),
        ("4. dfg", "Build Data Flow Graphs"),
        ("5. slice", "Code slicing based on vulnerability lines"),
        ("6. tokenize", "Tokenize sliced code"),
        ("7. normalize", "Normalize tokens (vars, funcs, literals)"),
        ("8. vocab", "Build vocabulary from training set"),
        ("9. vectorize", "Convert tokens to integer indices"),
    ]
    
    for step, desc in steps_info:
        print(f"  {step}: {desc}")
    
    print("\nFEATURES:")
    print("  - Checkpointing for resume on session timeout")
    print("  - Memory-efficient chunk processing with gc.collect()")
    print("  - Parallel processing support")
    print("  - Config via YAML or CLI args")
    print("  - Status reporting")
    print("  - Disk space management")
    
    print("\nEXAMPLES:")
    print("  # Run full pipeline")
    print("  python cli.py run --data-dir /data --output-dir /output")
    print("")
    print("  # Run with config file")
    print("  python cli.py run --config config.yaml")
    print("")
    print("  # Run single step")
    print("  python cli.py step ast --split train")
    print("")
    print("  # Check status")
    print("  python cli.py status --checkpoint-dir /output/checkpoints")
    print("")
    print("  # Clean intermediate files")
    print("  python cli.py clean --keep raw vectorized")
    print("=" * 60 + "\n")


def cmd_config(args):
    """Generate or show config"""
    if args.generate:
        config = PipelineConfig()
        output_path = args.output or 'pipeline_config.yaml'
        config.save(output_path)
        print(f"Config saved to {output_path}")
    elif args.show:
        if args.config:
            config = PipelineConfig.from_yaml(args.config)
        else:
            config = PipelineConfig()
        
        if args.json:
            print(json.dumps(config.to_dict(), indent=2))
        else:
            import yaml
            print(yaml.dump(config.to_dict(), default_flow_style=False))


def main():
    parser = argparse.ArgumentParser(
        description='Devign Preprocessing Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s run --data-dir /data --output-dir /output
  %(prog)s run --config config.yaml --split train
  %(prog)s step ast --split train
  %(prog)s status --json
  %(prog)s clean --keep raw vectorized --yes
  %(prog)s info
        """
    )
    
    parser.add_argument('-v', '--verbose', action='store_true', help='Verbose output')
    
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # ========== run command ==========
    run_parser = subparsers.add_parser('run', help='Run preprocessing pipeline')
    run_parser.add_argument('--config', type=str, help='Path to config YAML')
    run_parser.add_argument('--data-dir', type=str, help='Input data directory')
    run_parser.add_argument('--output-dir', type=str, help='Output directory')
    run_parser.add_argument('--checkpoint-dir', type=str, help='Checkpoint directory')
    run_parser.add_argument('--split', type=str, default='all',
                           choices=['train', 'validation', 'test', 'all'],
                           help='Data split to process')
    run_parser.add_argument('--start-step', type=str, help='Step to start from')
    run_parser.add_argument('--end-step', type=str, help='Step to end at')
    run_parser.add_argument('--chunk-size', type=int, help='Chunk size for processing')
    run_parser.add_argument('--n-jobs', type=int, help='Number of parallel jobs')
    run_parser.add_argument('--slice-type', type=str, 
                           choices=['backward', 'forward', 'both', 'window'],
                           help='Slicing type')
    run_parser.add_argument('--max-seq-length', type=int, help='Max sequence length')
    run_parser.add_argument('--min-freq', type=int, help='Min token frequency for vocab')
    run_parser.add_argument('--max-vocab-size', type=int, help='Max vocabulary size')
    run_parser.add_argument('--continue-on-error', action='store_true',
                           help='Continue to next split on error')
    run_parser.set_defaults(func=cmd_run)
    
    # ========== step command ==========
    step_parser = subparsers.add_parser('step', help='Run single pipeline step')
    step_parser.add_argument('step_name', type=str,
                            choices=['load', 'vuln_features', 'ast', 'cfg', 'dfg',
                                    'slice', 'tokenize', 'normalize', 'vocab', 'vectorize'],
                            help='Step to run')
    step_parser.add_argument('--config', type=str, help='Path to config YAML')
    step_parser.add_argument('--output-dir', type=str, help='Output directory')
    step_parser.add_argument('--checkpoint-dir', type=str, help='Checkpoint directory')
    step_parser.add_argument('--split', type=str, default='train',
                            choices=['train', 'validation', 'test'],
                            help='Data split to process')
    step_parser.set_defaults(func=cmd_step)
    
    # ========== status command ==========
    status_parser = subparsers.add_parser('status', help='Check pipeline status')
    status_parser.add_argument('--config', type=str, help='Path to config YAML')
    status_parser.add_argument('--checkpoint-dir', type=str, help='Checkpoint directory')
    status_parser.add_argument('--output-dir', type=str, help='Output directory')
    status_parser.add_argument('--json', action='store_true', help='Output as JSON')
    status_parser.set_defaults(func=cmd_status)
    
    # ========== clean command ==========
    clean_parser = subparsers.add_parser('clean', help='Clean intermediate checkpoints')
    clean_parser.add_argument('--config', type=str, help='Path to config YAML')
    clean_parser.add_argument('--output-dir', type=str, help='Output directory')
    clean_parser.add_argument('--checkpoint-dir', type=str, help='Checkpoint directory')
    clean_parser.add_argument('--keep', nargs='+', 
                             help='Steps to keep (default: raw vectorized vectors)')
    clean_parser.add_argument('-y', '--yes', action='store_true',
                             help='Skip confirmation prompt')
    clean_parser.set_defaults(func=cmd_clean)
    
    # ========== reset command ==========
    reset_parser = subparsers.add_parser('reset', help='Reset pipeline state')
    reset_parser.add_argument('--config', type=str, help='Path to config YAML')
    reset_parser.add_argument('--checkpoint-dir', type=str, help='Checkpoint directory')
    reset_parser.add_argument('--step', type=str, help='Step to reset (default: all)')
    reset_parser.add_argument('--split', type=str, help='Split to reset')
    reset_parser.add_argument('-y', '--yes', action='store_true',
                             help='Skip confirmation prompt')
    reset_parser.set_defaults(func=cmd_reset)
    
    # ========== info command ==========
    info_parser = subparsers.add_parser('info', help='Show pipeline information')
    info_parser.set_defaults(func=cmd_info)
    
    # ========== config command ==========
    config_parser = subparsers.add_parser('config', help='Generate or show config')
    config_parser.add_argument('--generate', action='store_true',
                              help='Generate default config file')
    config_parser.add_argument('--show', action='store_true',
                              help='Show current config')
    config_parser.add_argument('--config', type=str, help='Config file to show')
    config_parser.add_argument('--output', '-o', type=str, help='Output path for generated config')
    config_parser.add_argument('--json', action='store_true', help='Output as JSON')
    config_parser.set_defaults(func=cmd_config)
    
    args = parser.parse_args()
    
    if args.verbose:
        import logging
        setup_logging(level=logging.DEBUG)
    else:
        setup_logging()
    
    if args.command is None:
        parser.print_help()
        sys.exit(1)
    
    try:
        args.func(args)
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        sys.exit(130)
    except Exception as e:
        logger = get_logger('cli')
        logger.error(f"Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
