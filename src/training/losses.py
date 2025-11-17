"""
Loss functions for multi-label genre classification.

This module implements various loss functions for training:
- Binary Cross-Entropy Loss (BCE)
- Focal Loss (addresses class imbalance)
- Weighted BCE Loss (handles imbalanced datasets)
- Label Smoothing BCE Loss

Classes:
    FocalLoss: Focal loss for addressing class imbalance
    WeightedBCELoss: BCE with class weights
    LabelSmoothingBCELoss: BCE with label smoothing

Functions:
    get_loss_function: Factory function to create loss functions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any


class FocalLoss(nn.Module):
    """
    Focal Loss for multi-label classification.

    Focal loss focuses training on hard examples by down-weighting
    easy examples. Useful for addressing class imbalance.

    Reference:
        Lin et al. "Focal Loss for Dense Object Detection" (2017)
        https://arxiv.org/abs/1708.02002

    Attributes:
        alpha (float): Weighting factor for positive class
        gamma (float): Focusing parameter (higher = more focus on hard examples)
        reduction (str): Reduction method ('none', 'mean', 'sum')

    Example:
        >>> loss_fn = FocalLoss(alpha=0.25, gamma=2.0)
        >>> logits = torch.randn(32, 18)
        >>> targets = torch.randint(0, 2, (32, 18)).float()
        >>> loss = loss_fn(logits, targets)
    """

    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        reduction: str = 'mean'
    ):
        """
        Initialize Focal Loss.

        Args:
            alpha (float, optional): Weighting factor in range [0,1] for positive class.
                Higher alpha gives more weight to positive examples. Defaults to 0.25.
            gamma (float, optional): Focusing parameter (gamma >= 0). Higher gamma
                increases focus on hard examples. Defaults to 2.0.
            reduction (str, optional): Specifies reduction to apply to output:
                'none' | 'mean' | 'sum'. Defaults to 'mean'.
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute focal loss.

        Args:
            inputs (torch.Tensor): Logits from model, shape (batch_size, num_classes)
            targets (torch.Tensor): Binary ground truth labels,
                shape (batch_size, num_classes), values in {0, 1}

        Returns:
            torch.Tensor: Focal loss value (scalar if reduction != 'none')
        """
        # Compute BCE loss without reduction
        bce_loss = F.binary_cross_entropy_with_logits(
            inputs, targets, reduction='none'
        )

        # Compute probabilities
        probs = torch.sigmoid(inputs)

        # Compute pt (probability of correct class)
        # pt = p if target = 1, else pt = 1 - p
        pt = targets * probs + (1 - targets) * (1 - probs)

        # Compute focal weight: (1 - pt)^gamma
        focal_weight = (1 - pt) ** self.gamma

        # Apply alpha weighting
        # alpha_t = alpha if target = 1, else alpha_t = 1 - alpha
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)

        # Compute focal loss
        focal_loss = alpha_t * focal_weight * bce_loss

        # Apply reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:  # 'none'
            return focal_loss


class WeightedBCELoss(nn.Module):
    """
    Weighted Binary Cross-Entropy Loss.

    Applies class weights to BCE loss to handle class imbalance.
    Weights can be provided manually or computed from class frequencies.

    Attributes:
        pos_weight (Optional[torch.Tensor]): Weight for positive class per label
        reduction (str): Reduction method ('none', 'mean', 'sum')

    Example:
        >>> # Compute weights from training data
        >>> pos_weight = torch.tensor([1.5, 2.0, 0.8, ...])  # 18 weights
        >>> loss_fn = WeightedBCELoss(pos_weight=pos_weight)
        >>> logits = torch.randn(32, 18)
        >>> targets = torch.randint(0, 2, (32, 18)).float()
        >>> loss = loss_fn(logits, targets)
    """

    def __init__(
        self,
        pos_weight: Optional[torch.Tensor] = None,
        reduction: str = 'mean'
    ):
        """
        Initialize Weighted BCE Loss.

        Args:
            pos_weight (Optional[torch.Tensor], optional): Weight for positive class
                for each label. Shape (num_classes,). If None, uses equal weights.
                Defaults to None.
            reduction (str, optional): Specifies reduction to apply to output:
                'none' | 'mean' | 'sum'. Defaults to 'mean'.
        """
        super(WeightedBCELoss, self).__init__()
        self.pos_weight = pos_weight
        self.reduction = reduction

    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute weighted BCE loss.

        Args:
            inputs (torch.Tensor): Logits from model, shape (batch_size, num_classes)
            targets (torch.Tensor): Binary ground truth labels,
                shape (batch_size, num_classes), values in {0, 1}

        Returns:
            torch.Tensor: Weighted BCE loss value (scalar if reduction != 'none')
        """
        return F.binary_cross_entropy_with_logits(
            inputs,
            targets,
            pos_weight=self.pos_weight,
            reduction=self.reduction
        )


class LabelSmoothingBCELoss(nn.Module):
    """
    Binary Cross-Entropy Loss with Label Smoothing.

    Label smoothing prevents the model from becoming over-confident
    by smoothing the target labels from {0, 1} to {smoothing, 1-smoothing}.

    Attributes:
        smoothing (float): Label smoothing factor
        reduction (str): Reduction method ('none', 'mean', 'sum')

    Example:
        >>> loss_fn = LabelSmoothingBCELoss(smoothing=0.1)
        >>> logits = torch.randn(32, 18)
        >>> targets = torch.randint(0, 2, (32, 18)).float()
        >>> loss = loss_fn(logits, targets)
    """

    def __init__(
        self,
        smoothing: float = 0.1,
        reduction: str = 'mean'
    ):
        """
        Initialize Label Smoothing BCE Loss.

        Args:
            smoothing (float, optional): Label smoothing factor in range [0, 1].
                0 = no smoothing, higher values = more smoothing. Defaults to 0.1.
            reduction (str, optional): Specifies reduction to apply to output:
                'none' | 'mean' | 'sum'. Defaults to 'mean'.
        """
        super(LabelSmoothingBCELoss, self).__init__()
        assert 0 <= smoothing < 1, "smoothing must be in [0, 1)"
        self.smoothing = smoothing
        self.reduction = reduction

    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute BCE loss with label smoothing.

        Args:
            inputs (torch.Tensor): Logits from model, shape (batch_size, num_classes)
            targets (torch.Tensor): Binary ground truth labels,
                shape (batch_size, num_classes), values in {0, 1}

        Returns:
            torch.Tensor: Label smoothing BCE loss value (scalar if reduction != 'none')
        """
        # Apply label smoothing
        # 0 -> smoothing, 1 -> 1 - smoothing
        smoothed_targets = targets * (1 - self.smoothing) + self.smoothing * 0.5

        return F.binary_cross_entropy_with_logits(
            inputs,
            smoothed_targets,
            reduction=self.reduction
        )


# ============================================================================
# Loss Function Factory
# ============================================================================

def get_loss_function(
    loss_config: Dict[str, Any],
    device: Optional[torch.device] = None
) -> nn.Module:
    """
    Factory function to create loss functions based on configuration.

    Args:
        loss_config (Dict[str, Any]): Loss configuration dictionary with keys:
            - 'type': Loss type ('bce', 'focal', 'weighted_bce', 'label_smoothing_bce')
            - Additional parameters specific to each loss type
        device (Optional[torch.device], optional): Device to move loss parameters to.
            Defaults to None.

    Returns:
        nn.Module: Initialized loss function

    Raises:
        ValueError: If loss type is unknown

    Example:
        >>> config = {
        ...     'type': 'focal',
        ...     'alpha': 0.25,
        ...     'gamma': 2.0,
        ...     'reduction': 'mean'
        ... }
        >>> loss_fn = get_loss_function(config, device=torch.device('cuda'))

        >>> config = {
        ...     'type': 'weighted_bce',
        ...     'pos_weight': torch.ones(18) * 2.0,
        ...     'reduction': 'mean'
        ... }
        >>> loss_fn = get_loss_function(config)
    """
    loss_type = loss_config.get('type', 'bce').lower()

    if loss_type == 'bce' or loss_type == 'binary_cross_entropy':
        # Standard BCE with logits
        loss_fn = nn.BCEWithLogitsLoss(
            reduction=loss_config.get('reduction', 'mean')
        )

    elif loss_type == 'focal':
        loss_fn = FocalLoss(
            alpha=loss_config.get('alpha', 0.25),
            gamma=loss_config.get('gamma', 2.0),
            reduction=loss_config.get('reduction', 'mean')
        )

    elif loss_type == 'weighted_bce':
        pos_weight = loss_config.get('pos_weight', None)
        if pos_weight is not None and device is not None:
            pos_weight = pos_weight.to(device)

        loss_fn = WeightedBCELoss(
            pos_weight=pos_weight,
            reduction=loss_config.get('reduction', 'mean')
        )

    elif loss_type == 'label_smoothing_bce':
        loss_fn = LabelSmoothingBCELoss(
            smoothing=loss_config.get('smoothing', 0.1),
            reduction=loss_config.get('reduction', 'mean')
        )

    else:
        raise ValueError(f"Unknown loss type: {loss_type}")

    return loss_fn


def compute_class_weights(
    targets: torch.Tensor,
    method: str = 'inverse_freq',
    smooth: float = 1.0
) -> torch.Tensor:
    """
    Compute class weights from target distribution.

    Useful for creating pos_weight parameter for WeightedBCELoss.

    Args:
        targets (torch.Tensor): Binary target labels, shape (num_samples, num_classes)
        method (str, optional): Weight computation method:
            - 'inverse_freq': weight = (total - pos_count) / pos_count
            - 'balanced': weight = total / (2 * pos_count)
            Defaults to 'inverse_freq'.
        smooth (float, optional): Smoothing factor to avoid division by zero.
            Defaults to 1.0.

    Returns:
        torch.Tensor: Class weights, shape (num_classes,)

    Example:
        >>> targets = torch.randint(0, 2, (1000, 18)).float()
        >>> weights = compute_class_weights(targets, method='inverse_freq')
        >>> weights.shape
        torch.Size([18])
    """
    # Count positive samples per class
    pos_count = targets.sum(dim=0) + smooth  # Shape: (num_classes,)
    total = targets.size(0)

    if method == 'inverse_freq':
        # weight = (total - pos_count) / pos_count
        neg_count = total - pos_count + smooth
        weights = neg_count / pos_count

    elif method == 'balanced':
        # weight = total / (2 * pos_count)
        weights = total / (2 * pos_count)

    else:
        raise ValueError(f"Unknown weight computation method: {method}")

    return weights


# Example usage and testing
if __name__ == "__main__":
    print("Testing Loss Functions...\n")

    # Create dummy data
    batch_size, num_classes = 32, 18
    logits = torch.randn(batch_size, num_classes)
    targets = torch.randint(0, 2, (batch_size, num_classes)).float()

    # Test standard BCE
    print("1. Binary Cross-Entropy Loss")
    bce_loss = nn.BCEWithLogitsLoss()
    loss = bce_loss(logits, targets)
    print(f"   Loss: {loss.item():.4f}\n")

    # Test Focal Loss
    print("2. Focal Loss")
    focal_loss = FocalLoss(alpha=0.25, gamma=2.0)
    loss = focal_loss(logits, targets)
    print(f"   Loss: {loss.item():.4f}\n")

    # Test Weighted BCE
    print("3. Weighted BCE Loss")
    pos_weights = compute_class_weights(targets, method='inverse_freq')
    print(f"   Computed weights (first 5): {pos_weights[:5]}")
    weighted_bce = WeightedBCELoss(pos_weight=pos_weights)
    loss = weighted_bce(logits, targets)
    print(f"   Loss: {loss.item():.4f}\n")

    # Test Label Smoothing BCE
    print("4. Label Smoothing BCE Loss")
    label_smooth_bce = LabelSmoothingBCELoss(smoothing=0.1)
    loss = label_smooth_bce(logits, targets)
    print(f"   Loss: {loss.item():.4f}\n")

    # Test factory function
    print("5. Factory Function")
    config = {
        'type': 'focal',
        'alpha': 0.25,
        'gamma': 2.0,
        'reduction': 'mean'
    }
    loss_fn = get_loss_function(config)
    loss = loss_fn(logits, targets)
    print(f"   Created: {type(loss_fn).__name__}")
    print(f"   Loss: {loss.item():.4f}\n")

    print("All loss functions tested successfully!")
