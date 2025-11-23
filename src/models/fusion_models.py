"""
Multimodal fusion models for genre classification.

This module implements various fusion strategies to combine text and image modalities:
- Early Fusion: Concatenate embeddings before classification
- Late Fusion: Combine predictions from separate models
- Attention Fusion: Cross-attention between modalities

Classes:
    EarlyFusionModel: Concatenates text and image embeddings
    LateFusionModel: Combines predictions from separate models
    AttentionFusionModel: Cross-attention between text and image features
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Union

from src.models.text_models import LSTMTextModel, DistilBERTTextModel
from src.models.vision_models import ResNetVisionModel, CustomCNNModel


class EarlyFusionModel(nn.Module):
    """
    Early fusion multimodal model.

    Combines text and image embeddings through concatenation, then processes
    through a fusion network for classification.

    Architecture:
        Text Model → Text Embedding
                                    → Concatenate → Fusion MLP → Output
        Image Model → Image Embedding

    Attributes:
        text_model (nn.Module): Text encoder model
        vision_model (nn.Module): Vision encoder model
        fusion_network (nn.Module): MLP for fused features

    Example:
        >>> model = EarlyFusionModel(
        ...     text_model_config={'type': 'lstm_text', ...},
        ...     vision_model_config={'type': 'resnet_vision', ...},
        ...     num_classes=18
        ... )
        >>> text = torch.randint(0, 10000, (32, 256))
        >>> images = torch.randn(32, 3, 224, 224)
        >>> output = model(text, images)
        >>> output.shape
        torch.Size([32, 18])
    """

    def __init__(
        self,
        text_model: nn.Module,
        vision_model: nn.Module,
        text_output_dim: int,
        vision_output_dim: int,
        text_projection_dim: int = 512,
        vision_projection_dim: int = 512,
        fusion_hidden_dims: list = [1024, 512, 256],
        num_classes: int = 18,
        dropout: list = [0.5, 0.3],
        activation: str = "relu",
    ):
        """
        Initialize early fusion model.

        Args:
            text_model (nn.Module): Pretrained or initialized text model
            vision_model (nn.Module): Pretrained or initialized vision model
            text_output_dim (int): Output dimension of text model features
            vision_output_dim (int): Output dimension of vision model features
            text_projection_dim (int, optional): Project text features to this dim.
                Defaults to 512.
            vision_projection_dim (int, optional): Project vision features to this dim.
                Defaults to 512.
            fusion_hidden_dims (list, optional): Hidden dimensions for fusion network.
                Defaults to [1024, 512, 256].
            num_classes (int, optional): Number of output classes. Defaults to 18.
            dropout (list, optional): Dropout rates for fusion layers.
                Defaults to [0.5, 0.3].
            activation (str, optional): Activation function ('relu', 'gelu', 'mish').
                Defaults to 'relu'.
        """
        super(EarlyFusionModel, self).__init__()

        self.text_model = text_model
        self.vision_model = vision_model

        # Remove classification heads from base models (use only as feature extractors)
        # For text models, we'll extract features before the final layer
        # For vision models, we'll use get_features() method

        # Projection layers to normalize embedding dimensions
        self.text_projection = nn.Linear(
            text_output_dim, text_projection_dim
        )
        self.vision_projection = nn.Linear(
            vision_output_dim, vision_projection_dim
        )

        # Fused embedding dimension
        fused_dim = text_projection_dim + vision_projection_dim

        # Build fusion network
        fusion_layers = []
        in_dim = fused_dim

        for idx, hidden_dim in enumerate(fusion_hidden_dims):
            fusion_layers.append(nn.Linear(in_dim, hidden_dim))

            # Activation
            if activation == "relu":
                fusion_layers.append(nn.ReLU())
            elif activation == "gelu":
                fusion_layers.append(nn.GELU())
            elif activation == "mish":
                fusion_layers.append(nn.Mish())

            # Dropout
            if idx < len(dropout):
                fusion_layers.append(nn.Dropout(dropout[idx]))
            else:
                fusion_layers.append(nn.Dropout(dropout[-1]))

            in_dim = hidden_dim

        # Output layer
        fusion_layers.append(nn.Linear(fusion_hidden_dims[-1], num_classes))

        self.fusion_network = nn.Sequential(*fusion_layers)

    def forward(
        self,
        text_input: torch.Tensor,
        image_input: torch.Tensor,
        text_attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            text_input (torch.Tensor): Text input (token IDs), shape (batch, seq_len)
            image_input (torch.Tensor): Image input, shape (batch, 3, 224, 224)
            text_attention_mask (Optional[torch.Tensor], optional): Attention mask
                for BERT models. Defaults to None.

        Returns:
            torch.Tensor: Logits, shape (batch, num_classes)
        """
        # Extract text features
        if isinstance(self.text_model, DistilBERTTextModel):
            # For BERT, get [CLS] embedding before final classifier
            # Ensure attention_mask is provided to avoid transformer warnings
            if text_attention_mask is None:
                # Create attention mask (1 for non-padding, 0 for padding)
                text_attention_mask = (text_input != 0).long()
            outputs = self.text_model.bert(
                input_ids=text_input, attention_mask=text_attention_mask
            )
            text_features = outputs.last_hidden_state[
                :, 0, :
            ]  # [CLS] token
        elif isinstance(self.text_model, LSTMTextModel):
            # For LSTM, forward through embedding and LSTM, get context vector
            embedded = self.text_model.embedding(text_input)
            lstm_output, (hidden, cell) = self.text_model.lstm(embedded)

            if self.text_model.use_attention:
                mask = (text_input != 0).long()
                text_features, _ = self.text_model.attention(
                    lstm_output, mask
                )
            else:
                if self.text_model.bidirectional:
                    text_features = torch.cat(
                        (hidden[-2], hidden[-1]), dim=1
                    )
                else:
                    text_features = hidden[-1]
        else:
            # Generic: assume model has a method to get features
            text_features = self.text_model(text_input)

        # Extract vision features
        if hasattr(self.vision_model, "get_features"):
            vision_features = self.vision_model.get_features(image_input)
        else:
            # If no get_features method, use forward pass and extract
            vision_features = self.vision_model.backbone(image_input)
            vision_features = vision_features.view(
                vision_features.size(0), -1
            )

        # Project to common dimensions
        text_proj = self.text_projection(text_features)
        vision_proj = self.vision_projection(vision_features)

        # Concatenate (early fusion)
        fused = torch.cat([text_proj, vision_proj], dim=1)

        # Fusion network
        logits = self.fusion_network(fused)

        return logits


class LateFusionModel(nn.Module):
    """
    Late fusion multimodal model.

    Trains separate text and image models, then combines their predictions.

    Architecture:
        Text Model → Text Logits
                                 → Weighted Average → Final Predictions
        Image Model → Image Logits

    Attributes:
        text_model (nn.Module): Complete text classification model
        vision_model (nn.Module): Complete vision classification model
        alpha (nn.Parameter): Learnable weight for combining predictions

    Example:
        >>> model = LateFusionModel(
        ...     text_model=text_classifier,
        ...     vision_model=vision_classifier,
        ...     fusion_strategy='learned_weighted_average'
        ... )
        >>> text = torch.randint(0, 10000, (32, 256))
        >>> images = torch.randn(32, 3, 224, 224)
        >>> output = model(text, images)
    """

    def __init__(
        self,
        text_model: nn.Module,
        vision_model: nn.Module,
        fusion_strategy: str = "average",
        initial_alpha: float = 0.5,
    ):
        """
        Initialize late fusion model.

        Args:
            text_model (nn.Module): Complete text model with classifier
            vision_model (nn.Module): Complete vision model with classifier
            fusion_strategy (str, optional): Fusion strategy:
                - 'average': Simple average of predictions
                - 'learned_weighted_average': Learnable weight parameter
                Defaults to 'average'.
            initial_alpha (float, optional): Initial weight for text model
                (if using learned_weighted_average). Defaults to 0.5.
        """
        super(LateFusionModel, self).__init__()

        self.text_model = text_model
        self.vision_model = vision_model
        self.fusion_strategy = fusion_strategy

        if fusion_strategy == "learned_weighted_average":
            # Learnable weight (alpha for text, 1-alpha for vision)
            self.alpha = nn.Parameter(torch.tensor(initial_alpha))
        else:
            self.alpha = None

    def forward(
        self,
        text_input: torch.Tensor,
        image_input: torch.Tensor,
        text_attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            text_input (torch.Tensor): Text input, shape (batch, seq_len)
            image_input (torch.Tensor): Image input, shape (batch, 3, 224, 224)
            text_attention_mask (Optional[torch.Tensor], optional): Attention mask.
                Defaults to None.

        Returns:
            torch.Tensor: Combined logits, shape (batch, num_classes)
        """
        # Get predictions from both models
        if isinstance(self.text_model, DistilBERTTextModel):
            text_logits = self.text_model(text_input, text_attention_mask)
        else:
            text_logits = self.text_model(text_input)

        vision_logits = self.vision_model(image_input)

        # Combine predictions
        if self.fusion_strategy == "average":
            # Simple average
            combined_logits = (text_logits + vision_logits) / 2

        elif self.fusion_strategy == "learned_weighted_average":
            # Learnable weighted average
            # Ensure alpha is in [0, 1] using sigmoid
            alpha = torch.sigmoid(self.alpha)
            combined_logits = (
                alpha * text_logits + (1 - alpha) * vision_logits
            )

        else:
            raise ValueError(
                f"Unknown fusion_strategy: {self.fusion_strategy}"
            )

        return combined_logits

    def get_fusion_weight(self) -> Optional[float]:
        """
        Get the current fusion weight (alpha).

        Returns:
            Optional[float]: Current alpha value, or None if not using learned weights
        """
        if self.alpha is not None:
            return torch.sigmoid(self.alpha).item()
        return None


class CrossAttentionLayer(nn.Module):
    """
    Cross-attention layer for multimodal fusion.

    Allows one modality to attend to the other.

    Attributes:
        query_dim (int): Dimension of query vectors
        key_value_dim (int): Dimension of key and value vectors
        num_heads (int): Number of attention heads
    """

    def __init__(
        self,
        query_dim: int,
        key_value_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        """
        Initialize cross-attention layer.

        Args:
            query_dim (int): Dimension of query vectors
            key_value_dim (int): Dimension of key and value vectors
            num_heads (int, optional): Number of attention heads. Defaults to 8.
            dropout (float, optional): Dropout probability. Defaults to 0.1.
        """
        super(CrossAttentionLayer, self).__init__()

        self.num_heads = num_heads
        self.head_dim = query_dim // num_heads

        assert (
            query_dim % num_heads == 0
        ), "query_dim must be divisible by num_heads"

        # Linear projections for Q, K, V
        self.query_proj = nn.Linear(query_dim, query_dim)
        self.key_proj = nn.Linear(key_value_dim, query_dim)
        self.value_proj = nn.Linear(key_value_dim, query_dim)

        # Output projection
        self.out_proj = nn.Linear(query_dim, query_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self, query: torch.Tensor, key_value: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply cross-attention.

        Args:
            query (torch.Tensor): Query tensor, shape (batch, query_len, query_dim)
            key_value (torch.Tensor): Key/Value tensor, shape (batch, kv_len, kv_dim)

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - attended_output: Shape (batch, query_len, query_dim)
                - attention_weights: Shape (batch, num_heads, query_len, kv_len)
        """
        batch_size = query.size(0)

        # Project Q, K, V
        Q = self.query_proj(query)  # (batch, query_len, query_dim)
        K = self.key_proj(key_value)  # (batch, kv_len, query_dim)
        V = self.value_proj(key_value)  # (batch, kv_len, query_dim)

        # Reshape for multi-head attention
        # (batch, seq_len, query_dim) → (batch, num_heads, seq_len, head_dim)
        Q = Q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(
            1, 2
        )
        K = K.view(batch_size, -1, self.num_heads, self.head_dim).transpose(
            1, 2
        )
        V = V.view(batch_size, -1, self.num_heads, self.head_dim).transpose(
            1, 2
        )

        # Scaled dot-product attention
        # (batch, num_heads, query_len, head_dim) x (batch, num_heads, head_dim, kv_len)
        # → (batch, num_heads, query_len, kv_len)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (
            self.head_dim**0.5
        )
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # Apply attention to values
        # (batch, num_heads, query_len, kv_len) x (batch, num_heads, kv_len, head_dim)
        # → (batch, num_heads, query_len, head_dim)
        attended = torch.matmul(attention_weights, V)

        # Concatenate heads
        # (batch, num_heads, query_len, head_dim) → (batch, query_len, query_dim)
        attended = (
            attended.transpose(1, 2)
            .contiguous()
            .view(batch_size, -1, self.num_heads * self.head_dim)
        )

        # Output projection
        output = self.out_proj(attended)

        return output, attention_weights


class AttentionFusionModel(nn.Module):
    """
    Attention-based fusion model with cross-attention.

    Uses cross-attention to allow text and image modalities to attend to each other.

    Architecture:
        Text Features → Cross-Attention ← Image Features
                             ↓
                       Fused Features → Classifier

    Example:
        >>> model = AttentionFusionModel(
        ...     text_model=text_model,
        ...     vision_model=vision_model,
        ...     text_dim=768,
        ...     vision_dim=512,
        ...     num_classes=18
        ... )
    """

    def __init__(
        self,
        text_model: nn.Module,
        vision_model: nn.Module,
        text_dim: int,
        vision_dim: int,
        num_heads: int = 8,
        attention_dropout: float = 0.1,
        fusion_hidden_dim: int = 512,
        num_classes: int = 18,
        dropout: float = 0.3,
    ):
        """
        Initialize attention fusion model.

        Args:
            text_model (nn.Module): Text encoder model
            vision_model (nn.Module): Vision encoder model
            text_dim (int): Text feature dimension
            vision_dim (int): Vision feature dimension
            num_heads (int, optional): Number of attention heads. Defaults to 8.
            attention_dropout (float, optional): Attention dropout. Defaults to 0.1.
            fusion_hidden_dim (int, optional): Hidden dimension for classifier.
                Defaults to 512.
            num_classes (int, optional): Number of output classes. Defaults to 18.
            dropout (float, optional): Classifier dropout. Defaults to 0.3.
        """
        super(AttentionFusionModel, self).__init__()

        self.text_model = text_model
        self.vision_model = vision_model

        # Project vision features to text dimension for attention compatibility
        self.vision_proj = nn.Linear(vision_dim, text_dim)

        # Cross-attention: text attends to image
        self.text_to_image_attention = CrossAttentionLayer(
            query_dim=text_dim,
            key_value_dim=text_dim,  # After projection
            num_heads=num_heads,
            dropout=attention_dropout,
        )

        # Cross-attention: image attends to text
        self.image_to_text_attention = CrossAttentionLayer(
            query_dim=text_dim,
            key_value_dim=text_dim,
            num_heads=num_heads,
            dropout=attention_dropout,
        )

        # Classifier on fused features
        self.classifier = nn.Sequential(
            nn.Linear(
                text_dim * 2, fusion_hidden_dim
            ),  # *2 because we concatenate
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden_dim, num_classes),
        )

    def forward(
        self,
        text_input: torch.Tensor,
        image_input: torch.Tensor,
        text_attention_mask: Optional[torch.Tensor] = None,
        return_attention: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict]]:
        """
        Forward pass.

        Args:
            text_input (torch.Tensor): Text input, shape (batch, seq_len)
            image_input (torch.Tensor): Image input, shape (batch, 3, 224, 224)
            text_attention_mask (Optional[torch.Tensor], optional): Attention mask.
                Defaults to None.
            return_attention (bool, optional): Return attention weights. Defaults to False.

        Returns:
            Union[torch.Tensor, Tuple[torch.Tensor, Dict]]:
                - logits: Shape (batch, num_classes)
                - attention_weights (if return_attention=True): Dictionary with keys
                  'text_to_image' and 'image_to_text'
        """
        # Extract text features (sequence representation)
        if isinstance(self.text_model, DistilBERTTextModel):
            outputs = self.text_model.bert(
                input_ids=text_input, attention_mask=text_attention_mask
            )
            text_features = (
                outputs.last_hidden_state
            )  # (batch, seq_len, text_dim)
        else:
            # For LSTM, get full sequence output
            embedded = self.text_model.embedding(text_input)
            text_features, _ = self.text_model.lstm(
                embedded
            )  # (batch, seq_len, lstm_dim)

        # Extract vision features
        vision_features = self.vision_model.get_features(image_input)
        vision_features = self.vision_proj(
            vision_features
        )  # Project to text_dim
        vision_features = vision_features.unsqueeze(
            1
        )  # (batch, 1, text_dim)

        # Cross-attention
        text_attended, text_attn_weights = self.text_to_image_attention(
            query=text_features, key_value=vision_features
        )
        # Pool text attended features (mean pooling)
        text_attended = text_attended.mean(dim=1)  # (batch, text_dim)

        vision_attended, image_attn_weights = self.image_to_text_attention(
            query=vision_features, key_value=text_features
        )
        vision_attended = vision_attended.squeeze(1)  # (batch, text_dim)

        # Concatenate attended features
        fused = torch.cat([text_attended, vision_attended], dim=1)

        # Classify
        logits = self.classifier(fused)

        if return_attention:
            attention_weights = {
                "text_to_image": text_attn_weights,
                "image_to_text": image_attn_weights,
            }
            return logits, attention_weights
        else:
            return logits


# Example usage and testing
if __name__ == "__main__":
    print("Testing Fusion Models...\n")

    # This would require actual text and vision models to be instantiated
    # See test_implementation.py for complete testing

    print("Fusion models defined successfully!")
    print("Models available:")
    print("  - EarlyFusionModel")
    print("  - LateFusionModel")
    print("  - AttentionFusionModel")
