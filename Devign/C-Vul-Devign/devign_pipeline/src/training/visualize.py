"""Visualization module for training evaluation metrics.

Generates plots for:
- Training/Validation Loss
- Training/Validation F1 Score  
- Training/Validation AUC
- Precision/Recall curves
- Confusion Matrix
- ROC Curve
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np

try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    from sklearn.metrics import roc_curve, precision_recall_curve, auc, confusion_matrix
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


def plot_training_history(
    history: Dict[str, list],
    output_dir: str,
    show: bool = False
) -> None:
    """Plot training history metrics.
    
    Args:
        history: Dictionary with training history (from history.json)
        output_dir: Directory to save plots
    """
    if not MATPLOTLIB_AVAILABLE:
        print("matplotlib not available. Install with: pip install matplotlib")
        return
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Set style
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # 1. Loss plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    ax.plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'loss_curve.png', dpi=150)
    if show:
        plt.show()
    plt.close()
    
    # 2. F1 Score plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, history['train_f1'], 'b-', label='Train F1', linewidth=2)
    ax.plot(epochs, history['val_f1'], 'r-', label='Val F1', linewidth=2)
    
    # Mark best F1
    best_epoch = np.argmax(history['val_f1']) + 1
    best_f1 = max(history['val_f1'])
    ax.axvline(x=best_epoch, color='g', linestyle='--', alpha=0.7, label=f'Best (epoch {best_epoch})')
    ax.scatter([best_epoch], [best_f1], color='g', s=100, zorder=5)
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('F1 Score', fontsize=12)
    ax.set_title('Training and Validation F1 Score', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'f1_curve.png', dpi=150)
    if show:
        plt.show()
    plt.close()
    
    # 3. AUC plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, history['train_auc'], 'b-', label='Train AUC', linewidth=2)
    ax.plot(epochs, history['val_auc'], 'r-', label='Val AUC', linewidth=2)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('AUC', fontsize=12)
    ax.set_title('Training and Validation AUC', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'auc_curve.png', dpi=150)
    if show:
        plt.show()
    plt.close()
    
    # 4. Precision/Recall plot
    if 'val_precision' in history and 'val_recall' in history:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(epochs, history['val_precision'], 'g-', label='Precision', linewidth=2)
        ax.plot(epochs, history['val_recall'], 'm-', label='Recall', linewidth=2)
        ax.plot(epochs, history['val_f1'], 'b--', label='F1', linewidth=2, alpha=0.7)
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title('Validation Precision, Recall & F1', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / 'precision_recall_curve.png', dpi=150)
        if show:
            plt.show()
        plt.close()
    
    # 5. Combined metrics plot (2x2 grid)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Loss
    axes[0, 0].plot(epochs, history['train_loss'], 'b-', label='Train', linewidth=2)
    axes[0, 0].plot(epochs, history['val_loss'], 'r-', label='Val', linewidth=2)
    axes[0, 0].set_title('Loss', fontsize=12, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # F1
    axes[0, 1].plot(epochs, history['train_f1'], 'b-', label='Train', linewidth=2)
    axes[0, 1].plot(epochs, history['val_f1'], 'r-', label='Val', linewidth=2)
    axes[0, 1].set_title('F1 Score', fontsize=12, fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # AUC
    axes[1, 0].plot(epochs, history['train_auc'], 'b-', label='Train', linewidth=2)
    axes[1, 0].plot(epochs, history['val_auc'], 'r-', label='Val', linewidth=2)
    axes[1, 0].set_title('AUC', fontsize=12, fontweight='bold')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Precision/Recall
    if 'val_precision' in history:
        axes[1, 1].plot(epochs, history['val_precision'], 'g-', label='Precision', linewidth=2)
        axes[1, 1].plot(epochs, history['val_recall'], 'm-', label='Recall', linewidth=2)
        axes[1, 1].set_title('Precision & Recall', fontsize=12, fontweight='bold')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    
    for ax in axes.flat:
        ax.set_xlabel('Epoch')
    
    plt.suptitle('Training Metrics Summary', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / 'metrics_summary.png', dpi=150)
    if show:
        plt.show()
    plt.close()
    
    print(f"Saved plots to {output_dir}")


def plot_confusion_matrix(
    cm: np.ndarray,
    output_dir: str,
    labels: list = ['Non-Vulnerable', 'Vulnerable'],
    show: bool = False
) -> None:
    """Plot confusion matrix heatmap.
    
    Args:
        cm: Confusion matrix array (2x2)
        output_dir: Directory to save plot
        labels: Class labels
    """
    if not MATPLOTLIB_AVAILABLE:
        return
    
    output_dir = Path(output_dir)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    ax.figure.colorbar(im, ax=ax)
    
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=labels,
        yticklabels=labels,
        ylabel='True Label',
        xlabel='Predicted Label',
        title='Confusion Matrix'
    )
    
    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'confusion_matrix.png', dpi=150)
    if show:
        plt.show()
    plt.close()


def plot_roc_curve(
    labels: np.ndarray,
    probs: np.ndarray,
    output_dir: str,
    show: bool = False
) -> None:
    """Plot ROC curve.
    
    Args:
        labels: True labels
        probs: Predicted probabilities
        output_dir: Directory to save plot
    """
    if not MATPLOTLIB_AVAILABLE or not SKLEARN_AVAILABLE:
        return
    
    output_dir = Path(output_dir)
    
    fpr, tpr, _ = roc_curve(labels, probs)
    roc_auc = auc(fpr, tpr)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', alpha=0.5)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('Receiver Operating Characteristic (ROC) Curve', fontsize=14, fontweight='bold')
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'roc_curve.png', dpi=150)
    if show:
        plt.show()
    plt.close()


def plot_precision_recall_curve(
    labels: np.ndarray,
    probs: np.ndarray,
    output_dir: str,
    show: bool = False
) -> None:
    """Plot Precision-Recall curve.
    
    Args:
        labels: True labels
        probs: Predicted probabilities
        output_dir: Directory to save plot
    """
    if not MATPLOTLIB_AVAILABLE or not SKLEARN_AVAILABLE:
        return
    
    output_dir = Path(output_dir)
    
    precision, recall, _ = precision_recall_curve(labels, probs)
    pr_auc = auc(recall, precision)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(recall, precision, color='green', lw=2, label=f'PR curve (AUC = {pr_auc:.4f})')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title('Precision-Recall Curve', fontsize=14, fontweight='bold')
    ax.legend(loc="lower left", fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'pr_curve.png', dpi=150)
    if show:
        plt.show()
    plt.close()


def generate_all_plots(
    history_path: str,
    output_dir: str,
    test_results: Optional[Dict[str, Any]] = None,
    test_probs: Optional[np.ndarray] = None,
    test_labels: Optional[np.ndarray] = None,
    show: bool = False
) -> None:
    """Generate all evaluation plots.
    
    Args:
        history_path: Path to history.json file
        output_dir: Directory to save plots
        test_results: Optional test results dict (with confusion_matrix)
        test_probs: Optional test probabilities for ROC/PR curves
        test_labels: Optional test labels for ROC/PR curves
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load history
    with open(history_path) as f:
        history = json.load(f)
    
    # Plot training history
    plot_training_history(history, output_dir, show)
    
    # Plot confusion matrix if available
    if test_results and 'confusion_matrix' in test_results:
        cm = np.array(test_results['confusion_matrix'])
        plot_confusion_matrix(cm, output_dir, show=show)
    
    # Plot ROC and PR curves if probs/labels available
    if test_probs is not None and test_labels is not None:
        plot_roc_curve(test_labels, test_probs, output_dir, show)
        plot_precision_recall_curve(test_labels, test_probs, output_dir, show)
    
    print(f"\nAll plots saved to: {output_dir}")
    print("Generated plots:")
    for f in output_dir.glob('*.png'):
        print(f"  - {f.name}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate training evaluation plots")
    parser.add_argument('--history', type=str, default='checkpoints/history.json',
                        help='Path to history.json file')
    parser.add_argument('--output_dir', type=str, default='checkpoints/plots',
                        help='Directory to save plots')
    parser.add_argument('--show', action='store_true',
                        help='Display plots interactively')
    
    args = parser.parse_args()
    
    generate_all_plots(args.history, args.output_dir, show=args.show)
