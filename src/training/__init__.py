"""
Training infrastructure package.

This package provides complete training functionality including:
- Loss functions (BCE, Focal, Weighted BCE, Label Smoothing)
- Trainer class with checkpointing, early stopping, and logging
- Optimizer and scheduler factories

Modules:
    losses: Loss functions for multi-label classification
    trainer: Training infrastructure and utilities
"""

from .losses import (
    FocalLoss,
    WeightedBCELoss,
    LabelSmoothingBCELoss,
    get_loss_function,
    compute_class_weights
)

from .trainer import (
    Trainer,
    EarlyStopping,
    create_optimizer,
    create_scheduler
)

__all__ = [
    # Loss functions
    'FocalLoss',
    'WeightedBCELoss',
    'LabelSmoothingBCELoss',
    'get_loss_function',
    'compute_class_weights',

    # Trainer
    'Trainer',
    'EarlyStopping',
    'create_optimizer',
    'create_scheduler',
]
