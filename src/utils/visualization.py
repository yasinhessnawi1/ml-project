"""
Visualization utilities for model training and evaluation.

This module provides plotting functions for:
- Training curves (loss, metrics)
- Confusion matrices
- ROC curves
- Precision-Recall curves
- Class distribution
- Attention weights visualization
- Prediction analysis

Functions:
    plot_training_history: Plot training and validation curves
    plot_confusion_matrix: Plot confusion matrix for a class
    plot_all_confusion_matrices: Plot confusion matrices for all classes
    plot_roc_curves: Plot ROC curves for all classes
    plot_precision_recall_curves: Plot PR curves for all classes
    plot_class_distribution: Plot distribution of genres
    plot_attention_weights: Visualize attention weights
    plot_prediction_comparison: Compare predictions vs ground truth
    plot_per_class_metrics: Bar plot of per-class metrics
    save_figure: Save figure with proper formatting
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score
)

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


def plot_training_history(
    history: Dict[str, List[float]],
    save_path: Optional[Union[str, Path]] = None,
    show: bool = True
) -> plt.Figure:
    """
    Plot training history (loss and metrics over epochs).

    Args:
        history (Dict[str, List[float]]): Training history with keys:
            - 'train_loss': Training loss per epoch
            - 'val_loss': Validation loss per epoch
            - 'val_metric': Validation metric per epoch
            - 'learning_rates': Learning rates per epoch (optional)
        save_path (Optional[Union[str, Path]], optional): Path to save figure.
            Defaults to None.
        show (bool, optional): Display figure. Defaults to True.

    Returns:
        plt.Figure: Matplotlib figure

    Example:
        >>> history = {
        ...     'train_loss': [0.5, 0.4, 0.3],
        ...     'val_loss': [0.6, 0.5, 0.4],
        ...     'val_metric': [0.7, 0.75, 0.8]
        ... }
        >>> fig = plot_training_history(history, save_path='training_curves.png')
    """
    num_plots = 3 if 'learning_rates' in history else 2
    fig, axes = plt.subplots(1, num_plots, figsize=(6 * num_plots, 5))

    if num_plots == 2:
        axes = list(axes)
    else:
        axes = list(axes)

    epochs = range(1, len(history['train_loss']) + 1)

    # Plot 1: Loss curves
    axes[0].plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    axes[0].plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Validation metric
    axes[1].plot(epochs, history['val_metric'], 'g-', linewidth=2)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Metric')
    axes[1].set_title('Validation Metric')
    axes[1].grid(True, alpha=0.3)

    # Plot 3: Learning rate (if available)
    if num_plots == 3 and 'learning_rates' in history:
        axes[2].plot(epochs, history['learning_rates'], 'm-', linewidth=2)
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('Learning Rate')
        axes[2].set_title('Learning Rate Schedule')
        axes[2].set_yscale('log')
        axes[2].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        save_figure(fig, save_path)

    if show:
        plt.show()
    else:
        plt.close()

    return fig


def plot_confusion_matrix(
    cm: np.ndarray,
    class_name: str,
    normalize: bool = False,
    save_path: Optional[Union[str, Path]] = None,
    show: bool = True
) -> plt.Figure:
    """
    Plot confusion matrix for a single class.

    Args:
        cm (np.ndarray): Confusion matrix, shape (2, 2) - [[TN, FP], [FN, TP]]
        class_name (str): Name of the class
        normalize (bool, optional): Normalize to percentages. Defaults to False.
        save_path (Optional[Union[str, Path]], optional): Path to save figure.
            Defaults to None.
        show (bool, optional): Display figure. Defaults to True.

    Returns:
        plt.Figure: Matplotlib figure

    Example:
        >>> cm = np.array([[80, 10], [5, 5]])
        >>> fig = plot_confusion_matrix(cm, 'Action', save_path='cm_action.png')
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
        fmt = '.2%'
    else:
        fmt = 'd'

    sns.heatmap(
        cm,
        annot=True,
        fmt=fmt,
        cmap='Blues',
        xticklabels=['Negative', 'Positive'],
        yticklabels=['Negative', 'Positive'],
        ax=ax,
        cbar_kws={'label': 'Percentage' if normalize else 'Count'}
    )

    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    ax.set_title(f'Confusion Matrix: {class_name}')

    plt.tight_layout()

    if save_path:
        save_figure(fig, save_path)

    if show:
        plt.show()
    else:
        plt.close()

    return fig


def plot_all_confusion_matrices(
    confusion_matrices: np.ndarray,
    class_names: List[str],
    normalize: bool = False,
    save_path: Optional[Union[str, Path]] = None,
    show: bool = True
) -> plt.Figure:
    """
    Plot confusion matrices for all classes in a grid.

    Args:
        confusion_matrices (np.ndarray): Array of confusion matrices,
            shape (num_classes, 2, 2)
        class_names (List[str]): Names of classes
        normalize (bool, optional): Normalize to percentages. Defaults to False.
        save_path (Optional[Union[str, Path]], optional): Path to save figure.
            Defaults to None.
        show (bool, optional): Display figure. Defaults to True.

    Returns:
        plt.Figure: Matplotlib figure

    Example:
        >>> cms = np.random.randint(0, 100, (18, 2, 2))
        >>> class_names = ['Action', 'Comedy', ...]
        >>> fig = plot_all_confusion_matrices(cms, class_names)
    """
    num_classes = len(class_names)
    ncols = 6
    nrows = (num_classes + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 3 * nrows))
    axes = axes.flatten()

    for i, (cm, class_name) in enumerate(zip(confusion_matrices, class_names)):
        if normalize:
            cm_norm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
            cm_norm = np.nan_to_num(cm_norm)  # Handle division by zero
            fmt = '.2f'
        else:
            cm_norm = cm
            fmt = 'd'

        sns.heatmap(
            cm_norm,
            annot=True,
            fmt=fmt,
            cmap='Blues',
            xticklabels=['N', 'P'],
            yticklabels=['N', 'P'],
            ax=axes[i],
            cbar=False,
            square=True
        )
        axes[i].set_title(class_name, fontsize=10)

    # Hide extra subplots
    for i in range(num_classes, len(axes)):
        axes[i].axis('off')

    plt.suptitle('Confusion Matrices for All Classes', fontsize=14, y=1.0)
    plt.tight_layout()

    if save_path:
        save_figure(fig, save_path)

    if show:
        plt.show()
    else:
        plt.close()

    return fig


def plot_roc_curves(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    class_names: List[str],
    save_path: Optional[Union[str, Path]] = None,
    show: bool = True,
    max_classes: int = 18
) -> plt.Figure:
    """
    Plot ROC curves for all classes.

    Args:
        predictions (torch.Tensor): Prediction probabilities,
            shape (batch_size, num_classes)
        targets (torch.Tensor): Ground truth binary labels,
            shape (batch_size, num_classes)
        class_names (List[str]): Names of classes
        save_path (Optional[Union[str, Path]], optional): Path to save figure.
            Defaults to None.
        show (bool, optional): Display figure. Defaults to True.
        max_classes (int, optional): Maximum number of classes to plot.
            Defaults to 18.

    Returns:
        plt.Figure: Matplotlib figure

    Example:
        >>> preds = torch.rand(100, 18)
        >>> targets = torch.randint(0, 2, (100, 18)).float()
        >>> class_names = ['Action', 'Comedy', ...]
        >>> fig = plot_roc_curves(preds, targets, class_names)
    """
    # Convert to numpy
    preds_np = predictions.cpu().numpy()
    targets_np = targets.cpu().numpy()

    num_classes = min(len(class_names), max_classes)
    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot ROC curve for each class
    for i in range(num_classes):
        try:
            fpr, tpr, _ = roc_curve(targets_np[:, i], preds_np[:, i])
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, lw=2, label=f'{class_names[i]} (AUC = {roc_auc:.2f})')
        except ValueError:
            # Skip if class has no positive samples
            continue

    # Plot diagonal
    ax.plot([0, 1], [0, 1], 'k--', lw=2, label='Random (AUC = 0.50)')

    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curves for All Classes')
    ax.legend(loc='lower right', fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        save_figure(fig, save_path)

    if show:
        plt.show()
    else:
        plt.close()

    return fig


def plot_precision_recall_curves(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    class_names: List[str],
    save_path: Optional[Union[str, Path]] = None,
    show: bool = True,
    max_classes: int = 18
) -> plt.Figure:
    """
    Plot Precision-Recall curves for all classes.

    Args:
        predictions (torch.Tensor): Prediction probabilities,
            shape (batch_size, num_classes)
        targets (torch.Tensor): Ground truth binary labels,
            shape (batch_size, num_classes)
        class_names (List[str]): Names of classes
        save_path (Optional[Union[str, Path]], optional): Path to save figure.
            Defaults to None.
        show (bool, optional): Display figure. Defaults to True.
        max_classes (int, optional): Maximum number of classes to plot.
            Defaults to 18.

    Returns:
        plt.Figure: Matplotlib figure
    """
    # Convert to numpy
    preds_np = predictions.cpu().numpy()
    targets_np = targets.cpu().numpy()

    num_classes = min(len(class_names), max_classes)
    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot PR curve for each class
    for i in range(num_classes):
        try:
            precision, recall, _ = precision_recall_curve(targets_np[:, i], preds_np[:, i])
            avg_precision = average_precision_score(targets_np[:, i], preds_np[:, i])
            ax.plot(recall, precision, lw=2,
                   label=f'{class_names[i]} (AP = {avg_precision:.2f})')
        except ValueError:
            # Skip if class has no positive samples
            continue

    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curves for All Classes')
    ax.legend(loc='best', fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])

    plt.tight_layout()

    if save_path:
        save_figure(fig, save_path)

    if show:
        plt.show()
    else:
        plt.close()

    return fig


def plot_class_distribution(
    targets: torch.Tensor,
    class_names: List[str],
    save_path: Optional[Union[str, Path]] = None,
    show: bool = True
) -> plt.Figure:
    """
    Plot distribution of classes in dataset.

    Args:
        targets (torch.Tensor): Binary labels, shape (num_samples, num_classes)
        class_names (List[str]): Names of classes
        save_path (Optional[Union[str, Path]], optional): Path to save figure.
            Defaults to None.
        show (bool, optional): Display figure. Defaults to True.

    Returns:
        plt.Figure: Matplotlib figure

    Example:
        >>> targets = torch.randint(0, 2, (1000, 18)).float()
        >>> class_names = ['Action', 'Comedy', ...]
        >>> fig = plot_class_distribution(targets, class_names)
    """
    # Convert to numpy and count
    targets_np = targets.cpu().numpy()
    counts = targets_np.sum(axis=0)

    # Sort by count
    sorted_indices = np.argsort(counts)[::-1]
    sorted_counts = counts[sorted_indices]
    sorted_names = [class_names[i] for i in sorted_indices]

    # Create bar plot
    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(range(len(sorted_names)), sorted_counts, color='steelblue', alpha=0.8)

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height,
               f'{int(height)}',
               ha='center', va='bottom', fontsize=9)

    ax.set_xlabel('Genre')
    ax.set_ylabel('Count')
    ax.set_title('Distribution of Genres in Dataset')
    ax.set_xticks(range(len(sorted_names)))
    ax.set_xticklabels(sorted_names, rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    if save_path:
        save_figure(fig, save_path)

    if show:
        plt.show()
    else:
        plt.close()

    return fig


def plot_attention_weights(
    attention_weights: torch.Tensor,
    tokens: Optional[List[str]] = None,
    max_tokens: int = 50,
    save_path: Optional[Union[str, Path]] = None,
    show: bool = True
) -> plt.Figure:
    """
    Visualize attention weights as heatmap.

    Args:
        attention_weights (torch.Tensor): Attention weights, shape (seq_len,) or (1, seq_len)
        tokens (Optional[List[str]], optional): Token strings for labels. Defaults to None.
        max_tokens (int, optional): Maximum number of tokens to display. Defaults to 50.
        save_path (Optional[Union[str, Path]], optional): Path to save figure.
            Defaults to None.
        show (bool, optional): Display figure. Defaults to True.

    Returns:
        plt.Figure: Matplotlib figure

    Example:
        >>> attention = torch.rand(100)
        >>> tokens = ['word1', 'word2', ...]
        >>> fig = plot_attention_weights(attention, tokens)
    """
    # Convert to numpy and reshape
    weights = attention_weights.cpu().numpy()
    if weights.ndim == 1:
        weights = weights.reshape(1, -1)

    # Limit number of tokens
    if weights.shape[1] > max_tokens:
        weights = weights[:, :max_tokens]
        if tokens:
            tokens = tokens[:max_tokens]

    fig, ax = plt.subplots(figsize=(15, 2))

    sns.heatmap(
        weights,
        cmap='YlOrRd',
        xticklabels=tokens if tokens else range(weights.shape[1]),
        yticklabels=False,
        ax=ax,
        cbar_kws={'label': 'Attention Weight'}
    )

    ax.set_xlabel('Tokens')
    ax.set_title('Attention Weights Visualization')

    if tokens:
        plt.xticks(rotation=45, ha='right')

    plt.tight_layout()

    if save_path:
        save_figure(fig, save_path)

    if show:
        plt.show()
    else:
        plt.close()

    return fig


def plot_per_class_metrics(
    per_class_metrics: Dict[str, Dict[str, float]],
    metric_name: str = 'f1',
    save_path: Optional[Union[str, Path]] = None,
    show: bool = True
) -> plt.Figure:
    """
    Plot per-class metrics as bar chart.

    Args:
        per_class_metrics (Dict[str, Dict[str, float]]): Dictionary mapping
            class name to metrics dictionary
        metric_name (str, optional): Metric to plot ('f1', 'precision', 'recall').
            Defaults to 'f1'.
        save_path (Optional[Union[str, Path]], optional): Path to save figure.
            Defaults to None.
        show (bool, optional): Display figure. Defaults to True.

    Returns:
        plt.Figure: Matplotlib figure

    Example:
        >>> metrics = {
        ...     'Action': {'f1': 0.75, 'precision': 0.80, 'recall': 0.70},
        ...     'Comedy': {'f1': 0.82, 'precision': 0.85, 'recall': 0.79}
        ... }
        >>> fig = plot_per_class_metrics(metrics, metric_name='f1')
    """
    # Extract metric values
    class_names = list(per_class_metrics.keys())
    metric_values = [per_class_metrics[c][metric_name] for c in class_names]

    # Sort by metric value
    sorted_indices = np.argsort(metric_values)[::-1]
    sorted_names = [class_names[i] for i in sorted_indices]
    sorted_values = [metric_values[i] for i in sorted_indices]

    # Create bar plot
    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(range(len(sorted_names)), sorted_values, color='teal', alpha=0.7)

    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height,
               f'{height:.3f}',
               ha='center', va='bottom', fontsize=8)

    ax.set_xlabel('Genre')
    ax.set_ylabel(metric_name.capitalize())
    ax.set_title(f'Per-Class {metric_name.upper()} Scores')
    ax.set_xticks(range(len(sorted_names)))
    ax.set_xticklabels(sorted_names, rotation=45, ha='right')
    ax.set_ylim([0, 1.0])
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    if save_path:
        save_figure(fig, save_path)

    if show:
        plt.show()
    else:
        plt.close()

    return fig


def plot_prediction_comparison(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    class_names: List[str],
    sample_idx: int = 0,
    threshold: float = 0.5,
    save_path: Optional[Union[str, Path]] = None,
    show: bool = True
) -> plt.Figure:
    """
    Compare predictions vs ground truth for a single sample.

    Args:
        predictions (torch.Tensor): Prediction probabilities,
            shape (batch_size, num_classes)
        targets (torch.Tensor): Ground truth binary labels,
            shape (batch_size, num_classes)
        class_names (List[str]): Names of classes
        sample_idx (int, optional): Sample index to visualize. Defaults to 0.
        threshold (float, optional): Classification threshold. Defaults to 0.5.
        save_path (Optional[Union[str, Path]], optional): Path to save figure.
            Defaults to None.
        show (bool, optional): Display figure. Defaults to True.

    Returns:
        plt.Figure: Matplotlib figure
    """
    # Extract sample
    pred_probs = predictions[sample_idx].cpu().numpy()
    true_labels = targets[sample_idx].cpu().numpy()

    # Sort by prediction probability
    sorted_indices = np.argsort(pred_probs)[::-1]

    fig, ax = plt.subplots(figsize=(10, 8))

    y_pos = np.arange(len(class_names))

    # Plot predictions
    bars = ax.barh(y_pos, pred_probs[sorted_indices], alpha=0.6, label='Prediction Probability')

    # Add threshold line
    ax.axvline(x=threshold, color='r', linestyle='--', linewidth=2, label=f'Threshold ({threshold})')

    # Highlight true positives
    for i, idx in enumerate(sorted_indices):
        if true_labels[idx] == 1:
            ax.barh(i, pred_probs[idx], color='green', alpha=0.8)

    ax.set_yticks(y_pos)
    ax.set_yticklabels([class_names[i] for i in sorted_indices])
    ax.set_xlabel('Prediction Probability')
    ax.set_title(f'Prediction Comparison for Sample {sample_idx}\n(Green = True Positive)')
    ax.set_xlim([0, 1.0])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()

    if save_path:
        save_figure(fig, save_path)

    if show:
        plt.show()
    else:
        plt.close()

    return fig


def save_figure(
    fig: plt.Figure,
    save_path: Union[str, Path],
    dpi: int = 300,
    bbox_inches: str = 'tight'
) -> None:
    """
    Save matplotlib figure with proper formatting.

    Args:
        fig (plt.Figure): Matplotlib figure to save
        save_path (Union[str, Path]): Path to save figure
        dpi (int, optional): Resolution in dots per inch. Defaults to 300.
        bbox_inches (str, optional): Bounding box setting. Defaults to 'tight'.

    Example:
        >>> fig, ax = plt.subplots()
        >>> ax.plot([1, 2, 3], [1, 4, 9])
        >>> save_figure(fig, 'my_plot.png')
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(
        save_path,
        dpi=dpi,
        bbox_inches=bbox_inches,
        facecolor='white',
        edgecolor='none'
    )
    print(f"Figure saved to: {save_path}")


# Example usage
if __name__ == "__main__":
    print("Testing Visualization Utilities...\n")

    # Create dummy data
    num_samples, num_classes = 100, 18
    predictions = torch.rand(num_samples, num_classes)
    targets = torch.randint(0, 2, (num_samples, num_classes)).float()

    class_names = [
        'Action', 'Adventure', 'Animation', 'Biography', 'Comedy',
        'Crime', 'Documentary', 'Drama', 'Family', 'Fantasy',
        'History', 'Horror', 'Music', 'Mystery', 'Romance',
        'Sci-Fi', 'Sport', 'Thriller'
    ]

    # Test training history plot
    print("1. Testing training history plot...")
    history = {
        'train_loss': [0.5, 0.4, 0.35, 0.3, 0.28],
        'val_loss': [0.55, 0.45, 0.42, 0.38, 0.37],
        'val_metric': [0.6, 0.65, 0.68, 0.72, 0.74],
        'learning_rates': [0.001, 0.001, 0.0008, 0.0005, 0.0003]
    }
    plot_training_history(history, show=False)

    # Test class distribution
    print("2. Testing class distribution plot...")
    plot_class_distribution(targets, class_names, show=False)

    # Test per-class metrics
    print("3. Testing per-class metrics plot...")
    per_class_metrics = {
        name: {
            'f1': np.random.rand(),
            'precision': np.random.rand(),
            'recall': np.random.rand()
        }
        for name in class_names
    }
    plot_per_class_metrics(per_class_metrics, metric_name='f1', show=False)

    # Test prediction comparison
    print("4. Testing prediction comparison plot...")
    plot_prediction_comparison(predictions, targets, class_names, sample_idx=0, show=False)

    print("\nAll visualization utilities tested successfully!")
