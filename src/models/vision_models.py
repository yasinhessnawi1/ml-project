"""
Vision-based models for genre classification from movie posters.

This module implements vision-only models including:
- ResNet (pretrained on ImageNet)
- Custom CNN

Classes:
    ResNetVisionModel: ResNet-based image classifier
    CustomCNNModel: Custom CNN architecture
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from typing import Optional, List


class ResNetVisionModel(nn.Module):
    """
    ResNet-based vision model for genre classification.

    Uses pretrained ResNet as backbone with custom classification head.

    Architecture:
        ResNet Backbone → Global Average Pooling → FC → Dropout → Output

    Attributes:
        backbone (nn.Module): ResNet backbone
        num_classes (int): Number of output classes
        pretrained (bool): Whether backbone is pretrained

    Example:
        >>> model = ResNetVisionModel(
        ...     architecture='resnet18',
        ...     num_classes=18,
        ...     pretrained=True
        ... )
        >>> images = torch.randn(32, 3, 224, 224)
        >>> output = model(images)
        >>> output.shape
        torch.Size([32, 18])
    """

    def __init__(
        self,
        architecture: str = "resnet18",
        num_classes: int = 18,
        pretrained: bool = True,
        fine_tune_strategy: str = "all",
        classifier_hidden_dim: int = 256,
        dropout: float = 0.5,
    ):
        """
        Initialize ResNet vision model.

        Args:
            architecture (str, optional): ResNet architecture ('resnet18', 'resnet34',
                'resnet50', 'resnet101', 'resnet152'). Defaults to 'resnet18'.
            num_classes (int, optional): Number of output classes. Defaults to 18.
            pretrained (bool, optional): Use ImageNet pretrained weights. Defaults to True.
            fine_tune_strategy (str, optional): Fine-tuning strategy:
                - 'all': Fine-tune all layers
                - 'last_layer': Only train classifier, freeze backbone
                - 'unfreeze_layer3_layer4': Unfreeze layer3 and layer4, freeze earlier
                Defaults to 'all'.
            classifier_hidden_dim (int, optional): Hidden dimension in classifier head.
                Defaults to 256.
            dropout (float, optional): Dropout probability. Defaults to 0.5.

        Raises:
            ValueError: If architecture is not supported
        """
        super(ResNetVisionModel, self).__init__()

        self.architecture = architecture
        self.num_classes = num_classes
        self.pretrained = pretrained
        self.fine_tune_strategy = fine_tune_strategy

        # Load pretrained ResNet
        if architecture == "resnet18":
            self.backbone = models.resnet18(pretrained=pretrained)
            backbone_output_dim = 512
        elif architecture == "resnet34":
            self.backbone = models.resnet34(pretrained=pretrained)
            backbone_output_dim = 512
        elif architecture == "resnet50":
            self.backbone = models.resnet50(pretrained=pretrained)
            backbone_output_dim = 2048
        elif architecture == "resnet101":
            self.backbone = models.resnet101(pretrained=pretrained)
            backbone_output_dim = 2048
        elif architecture == "resnet152":
            self.backbone = models.resnet152(pretrained=pretrained)
            backbone_output_dim = 2048
        else:
            raise ValueError(f"Unsupported architecture: {architecture}")

        # Remove original classification head
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])

        # Apply fine-tuning strategy
        self._apply_fine_tune_strategy()

        # Custom classification head
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(backbone_output_dim, classifier_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(classifier_hidden_dim, num_classes),
        )

    def _apply_fine_tune_strategy(self):
        """Apply the specified fine-tuning strategy to the backbone."""
        if self.fine_tune_strategy == "last_layer":
            # Freeze all backbone parameters
            for param in self.backbone.parameters():
                param.requires_grad = False

        elif self.fine_tune_strategy == "unfreeze_layer3_layer4":
            # Freeze early layers, unfreeze layer3 and layer4
            # ResNet structure: conv1, bn1, relu, maxpool, layer1, layer2, layer3, layer4
            freeze_until = 6  # Freeze up to layer2 (index 5)

            for idx, child in enumerate(self.backbone.children()):
                if idx < freeze_until:
                    for param in child.parameters():
                        param.requires_grad = False
                else:
                    for param in child.parameters():
                        param.requires_grad = True

        elif self.fine_tune_strategy == "all":
            # Fine-tune all layers (default, all params already trainable)
            pass

        else:
            raise ValueError(
                f"Unknown fine_tune_strategy: {self.fine_tune_strategy}"
            )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            images (torch.Tensor): Batch of images, shape (batch, 3, 224, 224)

        Returns:
            torch.Tensor: Logits, shape (batch, num_classes)
        """
        # Extract features: (batch, 3, 224, 224) → (batch, feature_dim, 1, 1)
        features = self.backbone(images)

        # Classify: (batch, feature_dim, 1, 1) → (batch, num_classes)
        logits = self.classifier(features)

        return logits

    def get_features(self, images: torch.Tensor) -> torch.Tensor:
        """
        Extract feature representations (without classification).

        Useful for visualization, transfer learning, or fusion models.

        Args:
            images (torch.Tensor): Batch of images, shape (batch, 3, 224, 224)

        Returns:
            torch.Tensor: Feature vectors, shape (batch, feature_dim)
        """
        features = self.backbone(images)
        features = features.view(features.size(0), -1)  # Flatten
        return features


class CustomCNNModel(nn.Module):
    """
    Custom CNN architecture for genre classification.

    Lightweight CNN trained from scratch (or with custom pretraining).

    Architecture:
        Conv Blocks (Conv→BN→ReLU→MaxPool) × N → Global Avg Pool → FC → Output

    Attributes:
        channels (List[int]): Channel progression
        num_classes (int): Number of output classes

    Example:
        >>> model = CustomCNNModel(
        ...     channels=[64, 128, 256, 512],
        ...     num_classes=18
        ... )
        >>> images = torch.randn(32, 3, 224, 224)
        >>> output = model(images)
        >>> output.shape
        torch.Size([32, 18])
    """

    def __init__(
        self,
        channels: List[int] = [64, 128, 256, 512],
        kernel_sizes: List[int] = [7, 3, 3, 3],
        num_classes: int = 18,
        dropout: float = 0.5,
    ):
        """
        Initialize custom CNN model.

        Args:
            channels (List[int], optional): Output channels for each conv block.
                Defaults to [64, 128, 256, 512].
            kernel_sizes (List[int], optional): Kernel size for each conv layer.
                Defaults to [7, 3, 3, 3].
            num_classes (int, optional): Number of output classes. Defaults to 18.
            dropout (float, optional): Dropout probability. Defaults to 0.5.

        Raises:
            ValueError: If channels and kernel_sizes have different lengths
        """
        super(CustomCNNModel, self).__init__()

        if len(channels) != len(kernel_sizes):
            raise ValueError(
                "channels and kernel_sizes must have same length"
            )

        self.channels = channels
        self.num_classes = num_classes

        # Build convolutional blocks
        conv_blocks = []
        in_channels = 3  # RGB images

        for idx, (out_channels, kernel_size) in enumerate(
            zip(channels, kernel_sizes)
        ):
            # Calculate padding to maintain spatial dimensions before pooling
            padding = kernel_size // 2

            block = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride=1,
                    padding=padding,
                ),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(
                    kernel_size=2, stride=2
                ),  # Reduce spatial dimensions by 2
            )

            conv_blocks.append(block)
            in_channels = out_channels

        self.conv_blocks = nn.Sequential(*conv_blocks)

        # Global average pooling (reduces to (batch, channels, 1, 1))
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Classification head
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels[-1], 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            images (torch.Tensor): Batch of images, shape (batch, 3, 224, 224)

        Returns:
            torch.Tensor: Logits, shape (batch, num_classes)
        """
        # Convolutional feature extraction
        features = self.conv_blocks(images)

        # Global pooling
        pooled = self.global_avg_pool(features)

        # Classification
        logits = self.classifier(pooled)

        return logits

    def get_features(self, images: torch.Tensor) -> torch.Tensor:
        """
        Extract feature representations (without classification).

        Args:
            images (torch.Tensor): Batch of images, shape (batch, 3, 224, 224)

        Returns:
            torch.Tensor: Feature vectors, shape (batch, channels[-1])
        """
        features = self.conv_blocks(images)
        pooled = self.global_avg_pool(features)
        pooled = pooled.view(pooled.size(0), -1)  # Flatten
        return pooled


# ============================================================================
# Model Factory Function
# ============================================================================


def create_vision_model(config: dict) -> nn.Module:
    """
    Factory function to create vision models based on configuration.

    Args:
        config (dict): Model configuration dictionary

    Returns:
        nn.Module: Initialized vision model

    Raises:
        ValueError: If model type is unknown

    Example:
        >>> config = {
        ...     'type': 'resnet_vision',
        ...     'architecture': 'resnet18',
        ...     'pretrained': True,
        ...     'num_classes': 18
        ... }
        >>> model = create_vision_model(config)
    """
    model_type = config.get("type", "").lower()

    if model_type == "resnet_vision" or model_type == "resnet":
        model = ResNetVisionModel(
            architecture=config.get("architecture", "resnet18"),
            num_classes=config.get("num_classes", 18),
            pretrained=config.get("pretrained", True),
            fine_tune_strategy=config.get("fine_tune_strategy", "all"),
            classifier_hidden_dim=config.get("classifier_hidden_dim", 256),
            dropout=config.get("dropout", 0.5),
        )

    elif model_type == "custom_cnn" or model_type == "cnn":
        model = CustomCNNModel(
            channels=config.get("channels", [64, 128, 256, 512]),
            kernel_sizes=config.get("kernel_sizes", [7, 3, 3, 3]),
            num_classes=config.get("num_classes", 18),
            dropout=config.get("dropout", 0.5),
        )

    else:
        raise ValueError(f"Unknown vision model type: {model_type}")

    return model


# Example usage and testing
if __name__ == "__main__":
    print("Testing Vision Models...\n")

    # Test ResNet model
    print("1. ResNet-18 Vision Model")
    resnet_model = ResNetVisionModel(
        architecture="resnet18",
        num_classes=18,
        pretrained=False,  # Use False for testing (faster)
        fine_tune_strategy="all",
    )
    # Dummy input: batch_size=4, 3 channels, 224x224 images
    dummy_images = torch.randn(4, 3, 224, 224)
    resnet_output = resnet_model(dummy_images)
    resnet_features = resnet_model.get_features(dummy_images)
    print(f"   Input shape: {dummy_images.shape}")
    print(f"   Output shape: {resnet_output.shape}")
    print(f"   Feature shape: {resnet_features.shape}")
    print(
        f"   Parameters: {sum(p.numel() for p in resnet_model.parameters()):,}\n"
    )

    # Test Custom CNN
    print("2. Custom CNN Model")
    cnn_model = CustomCNNModel(
        channels=[64, 128, 256, 512],
        kernel_sizes=[7, 3, 3, 3],
        num_classes=18,
    )
    cnn_output = cnn_model(dummy_images)
    cnn_features = cnn_model.get_features(dummy_images)
    print(f"   Input shape: {dummy_images.shape}")
    print(f"   Output shape: {cnn_output.shape}")
    print(f"   Feature shape: {cnn_features.shape}")
    print(
        f"   Parameters: {sum(p.numel() for p in cnn_model.parameters()):,}\n"
    )

    print("All vision models tested successfully!")
