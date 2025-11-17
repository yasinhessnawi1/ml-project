"""
Evaluation package for multi-label classification.

This package provides comprehensive evaluation metrics including:
- F1-score (macro, micro, weighted, per-class)
- Precision and Recall
- Hamming Loss
- Subset Accuracy
- ROC-AUC
- Confusion matrices

Modules:
    metrics: Evaluation metrics and utilities
"""

from .metrics import (
    apply_threshold,
    compute_f1_scores,
    compute_precision_recall,
    compute_hamming_loss,
    compute_subset_accuracy,
    compute_roc_auc,
    compute_confusion_matrices,
    compute_per_class_metrics,
    compute_all_metrics,
    get_classification_report
)

__all__ = [
    'apply_threshold',
    'compute_f1_scores',
    'compute_precision_recall',
    'compute_hamming_loss',
    'compute_subset_accuracy',
    'compute_roc_auc',
    'compute_confusion_matrices',
    'compute_per_class_metrics',
    'compute_all_metrics',
    'get_classification_report',
]
