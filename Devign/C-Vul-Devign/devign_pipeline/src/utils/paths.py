"""Path utilities"""

from pathlib import Path
from typing import Optional
import os


def get_project_root() -> Path:
    """Get project root directory"""
    current = Path(__file__).resolve()
    # Go up from utils -> src -> project_root
    return current.parent.parent.parent


def get_config_path(config_name: str = "config.yaml") -> Path:
    """Get path to config file"""
    return get_project_root() / "config" / config_name


def ensure_dir(path: str) -> Path:
    """Ensure directory exists and return Path"""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def get_kaggle_paths() -> tuple:
    """Get Kaggle input/output paths if running on Kaggle"""
    if os.path.exists('/kaggle/input'):
        return Path('/kaggle/input'), Path('/kaggle/working')
    return None, None


def resolve_path(path: str, base: Optional[Path] = None) -> Path:
    """Resolve path relative to base or project root"""
    p = Path(path)
    if p.is_absolute():
        return p
    
    base = base or get_project_root()
    return base / p
