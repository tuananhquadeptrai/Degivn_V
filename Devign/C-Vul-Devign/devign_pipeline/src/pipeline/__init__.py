"""Pipeline orchestration"""

from .preprocess import (
    PreprocessPipeline, 
    PipelineConfig, 
    PipelineState,
    StepStatus,
    run_pipeline,
)
from .dataset import (
    DevignDataset,
    GraphDataset,
    get_optimal_batch_size,
)

try:
    from .dataset import (
        DevignTorchDataset,
        GraphTorchDataset,
        create_dataloader,
        create_distributed_dataloaders,
        create_graph_dataloader,
        BalancedBatchSampler,
        collate_fn,
        graph_collate_fn,
    )
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

__all__ = [
    "PreprocessPipeline",
    "PipelineConfig",
    "PipelineState",
    "StepStatus",
    "run_pipeline",
    "DevignDataset",
    "GraphDataset",
    "get_optimal_batch_size",
]

if TORCH_AVAILABLE:
    __all__.extend([
        "DevignTorchDataset",
        "GraphTorchDataset",
        "create_dataloader",
        "create_distributed_dataloaders",
        "create_graph_dataloader",
        "BalancedBatchSampler",
        "collate_fn",
        "graph_collate_fn",
    ])
