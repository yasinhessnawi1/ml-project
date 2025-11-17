"""
Utilities package.

This package provides utility functions for:
- Configuration management
- Visualization and plotting
- General helper functions

Modules:
    config: Configuration loading and device setup
    visualization: Plotting and visualization utilities
"""

from .config import (
    load_config,
    set_seed,
    setup_device,
    validate_config
)

from .visualization import (
    plot_training_history,
    plot_confusion_matrix,
    plot_all_confusion_matrices,
    plot_roc_curves,
    plot_precision_recall_curves,
    plot_class_distribution,
    plot_attention_weights,
    plot_per_class_metrics,
    plot_prediction_comparison,
    save_figure
)

__all__ = [
    # Config
    'load_config',
    'set_seed',
    'setup_device',
    'validate_config',

    # Visualization
    'plot_training_history',
    'plot_confusion_matrix',
    'plot_all_confusion_matrices',
    'plot_roc_curves',
    'plot_precision_recall_curves',
    'plot_class_distribution',
    'plot_attention_weights',
    'plot_per_class_metrics',
    'plot_prediction_comparison',
    'save_figure',
]
