"""
Evaluation metrics for multi-label genre classification.

This module implements various metrics for evaluating multi-label classification:
- F1-score (macro, micro, weighted, per-class)
- Precision and Recall
- Hamming Loss
- Subset Accuracy
- ROC-AUC
- Confusion Matrix per class

Functions:
    compute_f1_scores: Compute F1 scores (macro, micro, weighted, per-class)
    compute_precision_recall: Compute precision and recall
    compute_hamming_loss: Compute Hamming loss
    compute_subset_accuracy: Compute subset accuracy (exact match)
    compute_roc_auc: Compute ROC-AUC scores
    compute_confusion_matrices: Compute confusion matrix per class
    compute_all_metrics: Compute all metrics at once
    apply_threshold: Convert probabilities to binary predictions
"""

import torch
import numpy as np
from typing import Dict, Optional, Tuple, List
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    hamming_loss,
    roc_auc_score,
    confusion_matrix,
    classification_report
)


def apply_threshold(
    predictions: torch.Tensor,
    threshold: float = 0.5
) -> torch.Tensor:
    """
    Apply threshold to convert probabilities to binary predictions.

    Args:
        predictions (torch.Tensor): Prediction probabilities, shape (batch_size, num_classes)
        threshold (float, optional): Classification threshold. Defaults to 0.5.

    Returns:
        torch.Tensor: Binary predictions, shape (batch_size, num_classes)

    Example:
        >>> probs = torch.tensor([[0.7, 0.3, 0.9], [0.2, 0.6, 0.4]])
        >>> binary = apply_threshold(probs, threshold=0.5)
        >>> binary
        tensor([[1., 0., 1.],
                [0., 1., 0.]])
    """
    return (predictions >= threshold).float()


def compute_f1_scores(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
    average: Optional[str] = None
) -> Dict[str, float]:
    """
    Compute F1 scores for multi-label classification.

    Args:
        predictions (torch.Tensor): Prediction probabilities, shape (batch_size, num_classes)
        targets (torch.Tensor): Ground truth binary labels, shape (batch_size, num_classes)
        threshold (float, optional): Classification threshold. Defaults to 0.5.
        average (Optional[str], optional): Averaging strategy. If None, computes all.
            Options: 'micro', 'macro', 'weighted', 'samples', None. Defaults to None.

    Returns:
        Dict[str, float]: Dictionary with F1 scores:
            - 'f1_micro': Micro-averaged F1
            - 'f1_macro': Macro-averaged F1
            - 'f1_weighted': Weighted F1
            - 'f1_per_class': Per-class F1 scores (if average=None)

    Example:
        >>> preds = torch.rand(100, 18)
        >>> targets = torch.randint(0, 2, (100, 18)).float()
        >>> f1_scores = compute_f1_scores(preds, targets)
        >>> f1_scores.keys()
        dict_keys(['f1_micro', 'f1_macro', 'f1_weighted', 'f1_per_class'])
    """
    # Convert to numpy
    preds_np = predictions.cpu().numpy()
    targets_np = targets.cpu().numpy()

    # Apply threshold
    binary_preds = (preds_np >= threshold).astype(int)

    results = {}

    if average is None:
        # Compute all averages
        results['f1_micro'] = f1_score(targets_np, binary_preds, average='micro', zero_division=0)
        results['f1_macro'] = f1_score(targets_np, binary_preds, average='macro', zero_division=0)
        results['f1_weighted'] = f1_score(targets_np, binary_preds, average='weighted', zero_division=0)
        results['f1_per_class'] = f1_score(targets_np, binary_preds, average=None, zero_division=0)
    else:
        # Compute specific average
        results[f'f1_{average}'] = f1_score(targets_np, binary_preds, average=average, zero_division=0)

    return results


def compute_precision_recall(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
    average: Optional[str] = None
) -> Dict[str, float]:
    """
    Compute precision and recall for multi-label classification.

    Args:
        predictions (torch.Tensor): Prediction probabilities, shape (batch_size, num_classes)
        targets (torch.Tensor): Ground truth binary labels, shape (batch_size, num_classes)
        threshold (float, optional): Classification threshold. Defaults to 0.5.
        average (Optional[str], optional): Averaging strategy. If None, computes all.
            Options: 'micro', 'macro', 'weighted', 'samples', None. Defaults to None.

    Returns:
        Dict[str, float]: Dictionary with precision and recall scores

    Example:
        >>> preds = torch.rand(100, 18)
        >>> targets = torch.randint(0, 2, (100, 18)).float()
        >>> metrics = compute_precision_recall(preds, targets)
    """
    # Convert to numpy
    preds_np = predictions.cpu().numpy()
    targets_np = targets.cpu().numpy()

    # Apply threshold
    binary_preds = (preds_np >= threshold).astype(int)

    results = {}

    if average is None:
        # Compute all averages
        for avg in ['micro', 'macro', 'weighted']:
            results[f'precision_{avg}'] = precision_score(
                targets_np, binary_preds, average=avg, zero_division=0
            )
            results[f'recall_{avg}'] = recall_score(
                targets_np, binary_preds, average=avg, zero_division=0
            )

        results['precision_per_class'] = precision_score(
            targets_np, binary_preds, average=None, zero_division=0
        )
        results['recall_per_class'] = recall_score(
            targets_np, binary_preds, average=None, zero_division=0
        )
    else:
        # Compute specific average
        results[f'precision_{average}'] = precision_score(
            targets_np, binary_preds, average=average, zero_division=0
        )
        results[f'recall_{average}'] = recall_score(
            targets_np, binary_preds, average=average, zero_division=0
        )

    return results


def compute_hamming_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5
) -> float:
    """
    Compute Hamming loss (fraction of incorrect labels).

    Hamming loss is the fraction of labels that are incorrectly predicted.
    Lower is better (0 = perfect, 1 = all wrong).

    Args:
        predictions (torch.Tensor): Prediction probabilities, shape (batch_size, num_classes)
        targets (torch.Tensor): Ground truth binary labels, shape (batch_size, num_classes)
        threshold (float, optional): Classification threshold. Defaults to 0.5.

    Returns:
        float: Hamming loss in range [0, 1]

    Example:
        >>> preds = torch.rand(100, 18)
        >>> targets = torch.randint(0, 2, (100, 18)).float()
        >>> loss = compute_hamming_loss(preds, targets)
    """
    # Convert to numpy
    preds_np = predictions.cpu().numpy()
    targets_np = targets.cpu().numpy()

    # Apply threshold
    binary_preds = (preds_np >= threshold).astype(int)

    return hamming_loss(targets_np, binary_preds)


def compute_subset_accuracy(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5
) -> float:
    """
    Compute subset accuracy (exact match ratio).

    Subset accuracy is the percentage of samples where all labels are
    correctly predicted. Very strict metric for multi-label classification.

    Args:
        predictions (torch.Tensor): Prediction probabilities, shape (batch_size, num_classes)
        targets (torch.Tensor): Ground truth binary labels, shape (batch_size, num_classes)
        threshold (float, optional): Classification threshold. Defaults to 0.5.

    Returns:
        float: Subset accuracy in range [0, 1]

    Example:
        >>> preds = torch.tensor([[0.9, 0.1, 0.8], [0.3, 0.7, 0.2]])
        >>> targets = torch.tensor([[1., 0., 1.], [0., 1., 0.]])
        >>> acc = compute_subset_accuracy(preds, targets, threshold=0.5)
        >>> acc
        1.0  # Both samples perfectly match
    """
    # Convert to numpy
    preds_np = predictions.cpu().numpy()
    targets_np = targets.cpu().numpy()

    # Apply threshold
    binary_preds = (preds_np >= threshold).astype(int)

    # Check exact match for each sample
    exact_matches = np.all(binary_preds == targets_np, axis=1)
    return exact_matches.mean()


def compute_roc_auc(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    average: Optional[str] = None
) -> Dict[str, float]:
    """
    Compute ROC-AUC scores.

    Note: ROC-AUC uses probabilities, not binary predictions.

    Args:
        predictions (torch.Tensor): Prediction probabilities, shape (batch_size, num_classes)
        targets (torch.Tensor): Ground truth binary labels, shape (batch_size, num_classes)
        average (Optional[str], optional): Averaging strategy. If None, computes all.
            Options: 'micro', 'macro', 'weighted', 'samples', None. Defaults to None.

    Returns:
        Dict[str, float]: Dictionary with ROC-AUC scores

    Example:
        >>> preds = torch.rand(100, 18)
        >>> targets = torch.randint(0, 2, (100, 18)).float()
        >>> auc_scores = compute_roc_auc(preds, targets)
    """
    # Convert to numpy
    preds_np = predictions.cpu().numpy()
    targets_np = targets.cpu().numpy()

    results = {}

    try:
        if average is None:
            # Compute all averages
            results['roc_auc_micro'] = roc_auc_score(targets_np, preds_np, average='micro')
            results['roc_auc_macro'] = roc_auc_score(targets_np, preds_np, average='macro')
            results['roc_auc_weighted'] = roc_auc_score(targets_np, preds_np, average='weighted')
            results['roc_auc_per_class'] = roc_auc_score(targets_np, preds_np, average=None)
        else:
            # Compute specific average
            results[f'roc_auc_{average}'] = roc_auc_score(targets_np, preds_np, average=average)
    except ValueError as e:
        # Handle case where a class has no positive samples
        print(f"Warning: Could not compute ROC-AUC: {e}")
        if average is None:
            results['roc_auc_micro'] = 0.0
            results['roc_auc_macro'] = 0.0
            results['roc_auc_weighted'] = 0.0
        else:
            results[f'roc_auc_{average}'] = 0.0

    return results


def compute_confusion_matrices(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5
) -> np.ndarray:
    """
    Compute confusion matrix for each class.

    For multi-label classification, computes binary confusion matrix
    for each class separately.

    Args:
        predictions (torch.Tensor): Prediction probabilities, shape (batch_size, num_classes)
        targets (torch.Tensor): Ground truth binary labels, shape (batch_size, num_classes)
        threshold (float, optional): Classification threshold. Defaults to 0.5.

    Returns:
        np.ndarray: Array of confusion matrices, shape (num_classes, 2, 2)
            Each matrix is [[TN, FP], [FN, TP]]

    Example:
        >>> preds = torch.rand(100, 18)
        >>> targets = torch.randint(0, 2, (100, 18)).float()
        >>> cms = compute_confusion_matrices(preds, targets)
        >>> cms.shape
        (18, 2, 2)
    """
    # Convert to numpy
    preds_np = predictions.cpu().numpy()
    targets_np = targets.cpu().numpy()

    # Apply threshold
    binary_preds = (preds_np >= threshold).astype(int)

    num_classes = targets_np.shape[1]
    confusion_matrices = np.zeros((num_classes, 2, 2))

    for i in range(num_classes):
        cm = confusion_matrix(targets_np[:, i], binary_preds[:, i], labels=[0, 1])
        confusion_matrices[i] = cm

    return confusion_matrices


def compute_per_class_metrics(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    class_names: Optional[List[str]] = None,
    threshold: float = 0.5
) -> Dict[str, Dict[str, float]]:
    """
    Compute detailed metrics for each class.

    Args:
        predictions (torch.Tensor): Prediction probabilities, shape (batch_size, num_classes)
        targets (torch.Tensor): Ground truth binary labels, shape (batch_size, num_classes)
        class_names (Optional[List[str]], optional): Names of classes. If None, uses indices.
        threshold (float, optional): Classification threshold. Defaults to 0.5.

    Returns:
        Dict[str, Dict[str, float]]: Dictionary mapping class name to metrics:
            - 'precision': Precision for this class
            - 'recall': Recall for this class
            - 'f1': F1-score for this class
            - 'support': Number of true positive samples
            - 'roc_auc': ROC-AUC for this class (if computable)

    Example:
        >>> preds = torch.rand(100, 3)
        >>> targets = torch.randint(0, 2, (100, 3)).float()
        >>> class_names = ['Action', 'Comedy', 'Drama']
        >>> metrics = compute_per_class_metrics(preds, targets, class_names)
        >>> metrics['Action']
        {'precision': 0.75, 'recall': 0.80, 'f1': 0.77, 'support': 45, 'roc_auc': 0.82}
    """
    # Convert to numpy
    preds_np = predictions.cpu().numpy()
    targets_np = targets.cpu().numpy()

    # Apply threshold
    binary_preds = (preds_np >= threshold).astype(int)

    num_classes = targets_np.shape[1]

    # Use class names or indices
    if class_names is None:
        class_names = [f"Class_{i}" for i in range(num_classes)]

    # Compute metrics for each class
    precision_per_class = precision_score(targets_np, binary_preds, average=None, zero_division=0)
    recall_per_class = recall_score(targets_np, binary_preds, average=None, zero_division=0)
    f1_per_class = f1_score(targets_np, binary_preds, average=None, zero_division=0)

    # Support (number of true samples)
    support_per_class = targets_np.sum(axis=0).astype(int)

    # ROC-AUC per class (if possible)
    try:
        roc_auc_per_class = roc_auc_score(targets_np, preds_np, average=None)
    except ValueError:
        roc_auc_per_class = np.zeros(num_classes)

    # Organize into dictionary
    results = {}
    for i, class_name in enumerate(class_names):
        results[class_name] = {
            'precision': float(precision_per_class[i]),
            'recall': float(recall_per_class[i]),
            'f1': float(f1_per_class[i]),
            'support': int(support_per_class[i]),
            'roc_auc': float(roc_auc_per_class[i])
        }

    return results


def compute_all_metrics(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    class_names: Optional[List[str]] = None,
    threshold: float = 0.5,
    verbose: bool = True
) -> Dict[str, any]:
    """
    Compute all evaluation metrics at once.

    Comprehensive evaluation including F1, precision, recall, Hamming loss,
    subset accuracy, ROC-AUC, and per-class metrics.

    Args:
        predictions (torch.Tensor): Prediction probabilities, shape (batch_size, num_classes)
        targets (torch.Tensor): Ground truth binary labels, shape (batch_size, num_classes)
        class_names (Optional[List[str]], optional): Names of classes. Defaults to None.
        threshold (float, optional): Classification threshold. Defaults to 0.5.
        verbose (bool, optional): Print summary. Defaults to True.

    Returns:
        Dict[str, any]: Dictionary with all metrics

    Example:
        >>> preds = torch.rand(100, 18)
        >>> targets = torch.randint(0, 2, (100, 18)).float()
        >>> all_metrics = compute_all_metrics(preds, targets)
    """
    results = {}

    # F1 scores
    results.update(compute_f1_scores(predictions, targets, threshold))

    # Precision and Recall
    results.update(compute_precision_recall(predictions, targets, threshold))

    # Hamming loss
    results['hamming_loss'] = compute_hamming_loss(predictions, targets, threshold)

    # Subset accuracy
    results['subset_accuracy'] = compute_subset_accuracy(predictions, targets, threshold)

    # ROC-AUC
    results.update(compute_roc_auc(predictions, targets))

    # Per-class metrics
    results['per_class_metrics'] = compute_per_class_metrics(
        predictions, targets, class_names, threshold
    )

    # Confusion matrices
    results['confusion_matrices'] = compute_confusion_matrices(predictions, targets, threshold)

    # Print summary if verbose
    if verbose:
        print("\n" + "="*80)
        print("EVALUATION METRICS SUMMARY")
        print("="*80)
        print(f"\nOverall Metrics:")
        print(f"  F1-Score (Macro):     {results['f1_macro']:.4f}")
        print(f"  F1-Score (Micro):     {results['f1_micro']:.4f}")
        print(f"  F1-Score (Weighted):  {results['f1_weighted']:.4f}")
        print(f"  Precision (Macro):    {results['precision_macro']:.4f}")
        print(f"  Recall (Macro):       {results['recall_macro']:.4f}")
        print(f"  ROC-AUC (Macro):      {results['roc_auc_macro']:.4f}")
        print(f"  Hamming Loss:         {results['hamming_loss']:.4f}")
        print(f"  Subset Accuracy:      {results['subset_accuracy']:.4f}")

        if class_names:
            print(f"\nTop 5 Classes by F1-Score:")
            per_class = results['per_class_metrics']
            sorted_classes = sorted(
                per_class.items(),
                key=lambda x: x[1]['f1'],
                reverse=True
            )
            for class_name, metrics in sorted_classes[:5]:
                print(f"  {class_name:20s}: F1={metrics['f1']:.3f}, "
                      f"P={metrics['precision']:.3f}, R={metrics['recall']:.3f}, "
                      f"Support={metrics['support']}")

        print("="*80 + "\n")

    return results


def get_classification_report(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    class_names: Optional[List[str]] = None,
    threshold: float = 0.5
) -> str:
    """
    Generate sklearn-style classification report.

    Args:
        predictions (torch.Tensor): Prediction probabilities, shape (batch_size, num_classes)
        targets (torch.Tensor): Ground truth binary labels, shape (batch_size, num_classes)
        class_names (Optional[List[str]], optional): Names of classes. Defaults to None.
        threshold (float, optional): Classification threshold. Defaults to 0.5.

    Returns:
        str: Formatted classification report

    Example:
        >>> preds = torch.rand(100, 18)
        >>> targets = torch.randint(0, 2, (100, 18)).float()
        >>> report = get_classification_report(preds, targets)
        >>> print(report)
    """
    # Convert to numpy
    preds_np = predictions.cpu().numpy()
    targets_np = targets.cpu().numpy()

    # Apply threshold
    binary_preds = (preds_np >= threshold).astype(int)

    # Generate report
    report = classification_report(
        targets_np,
        binary_preds,
        target_names=class_names,
        zero_division=0
    )

    return report


# Example usage and testing
if __name__ == "__main__":
    print("Testing Evaluation Metrics...\n")

    # Create dummy data
    batch_size, num_classes = 100, 18
    predictions = torch.rand(batch_size, num_classes)
    targets = torch.randint(0, 2, (batch_size, num_classes)).float()

    # Define class names (MM-IMDb genres)
    class_names = [
        'Action', 'Adventure', 'Animation', 'Biography', 'Comedy',
        'Crime', 'Documentary', 'Drama', 'Family', 'Fantasy',
        'History', 'Horror', 'Music', 'Mystery', 'Romance',
        'Sci-Fi', 'Sport', 'Thriller'
    ]

    # Compute all metrics
    print("Computing all metrics...")
    all_metrics = compute_all_metrics(
        predictions,
        targets,
        class_names=class_names,
        threshold=0.5,
        verbose=True
    )

    # Test individual metric functions
    print("\nTesting individual functions:")

    print("\n1. F1 Scores:")
    f1_scores = compute_f1_scores(predictions, targets)
    print(f"   Macro F1: {f1_scores['f1_macro']:.4f}")
    print(f"   Micro F1: {f1_scores['f1_micro']:.4f}")

    print("\n2. Hamming Loss:")
    hamming = compute_hamming_loss(predictions, targets)
    print(f"   {hamming:.4f}")

    print("\n3. Subset Accuracy:")
    subset_acc = compute_subset_accuracy(predictions, targets)
    print(f"   {subset_acc:.4f}")

    print("\n4. Classification Report:")
    report = get_classification_report(predictions, targets, class_names)
    print(report)

    print("\nAll evaluation metrics tested successfully!")
