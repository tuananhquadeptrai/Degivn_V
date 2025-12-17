"""Checkpoint management utilities"""

import pickle
import json
from pathlib import Path
from typing import Any, Dict, Optional
from datetime import datetime


class CheckpointManager:
    """Manage processing checkpoints"""
    
    def __init__(self, checkpoint_dir: str = "checkpoints"):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    def save(self, name: str, data: Any, metadata: Dict = None) -> Path:
        """Save checkpoint"""
        checkpoint = {
            'data': data,
            'metadata': metadata or {},
            'timestamp': datetime.now().isoformat(),
        }
        
        path = self.checkpoint_dir / f"{name}.pkl"
        with open(path, 'wb') as f:
            pickle.dump(checkpoint, f)
        
        return path
    
    def load(self, name: str) -> Optional[Dict]:
        """Load checkpoint"""
        path = self.checkpoint_dir / f"{name}.pkl"
        if not path.exists():
            return None
        
        with open(path, 'rb') as f:
            return pickle.load(f)
    
    def exists(self, name: str) -> bool:
        """Check if checkpoint exists"""
        return (self.checkpoint_dir / f"{name}.pkl").exists()
    
    def list_checkpoints(self) -> list:
        """List all checkpoints"""
        return [p.stem for p in self.checkpoint_dir.glob("*.pkl")]
    
    def delete(self, name: str) -> bool:
        """Delete checkpoint"""
        path = self.checkpoint_dir / f"{name}.pkl"
        if path.exists():
            path.unlink()
            return True
        return False


def save_checkpoint(name: str, data: Any, checkpoint_dir: str = "checkpoints") -> Path:
    """Convenience function to save checkpoint"""
    return CheckpointManager(checkpoint_dir).save(name, data)


def load_checkpoint(name: str, checkpoint_dir: str = "checkpoints") -> Optional[Any]:
    """Convenience function to load checkpoint"""
    result = CheckpointManager(checkpoint_dir).load(name)
    return result['data'] if result else None
