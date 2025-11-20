"""
Text-based models for genre classification.

This module implements text-only models including:
- LSTM with attention
- DistilBERT fine-tuning

Classes:
    LSTMTextModel: Bidirectional LSTM with attention mechanism
    DistilBERTTextModel: Fine-tuned DistilBERT for classification
    AttentionLayer: Custom attention mechanism for LSTM
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import DistilBertModel
from typing import Optional, Tuple, Union


class AttentionLayer(nn.Module):
    """
    Attention mechanism for sequence models.

    Computes attention weights over sequence and returns weighted sum.

    Attributes:
        hidden_dim (int): Dimension of hidden states

    Example:
        >>> attention = AttentionLayer(hidden_dim=512)
        >>> lstm_output = torch.randn(32, 100, 512)  # (batch, seq_len, hidden)
        >>> context, weights = attention(lstm_output)
        >>> context.shape, weights.shape
        (torch.Size([32, 512]), torch.Size([32, 100]))
    """

    def __init__(self, hidden_dim: int):
        """
        Initialize attention layer.

        Args:
            hidden_dim (int): Dimension of hidden states from LSTM
        """
        super(AttentionLayer, self).__init__()
        self.hidden_dim = hidden_dim

        # Learnable attention parameters
        self.attention_weights = nn.Linear(hidden_dim, 1, bias=False)

    def forward(
        self, lstm_output: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply attention mechanism.

        Args:
            lstm_output (torch.Tensor): LSTM hidden states, shape (batch, seq_len, hidden_dim)
            mask (Optional[torch.Tensor], optional): Padding mask, shape (batch, seq_len).
                True for valid positions, False for padding. Defaults to None.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - context: Weighted sum of hidden states, shape (batch, hidden_dim)
                - attention_weights: Attention weights, shape (batch, seq_len)
        """
        # Compute attention scores
        # Shape: (batch, seq_len, 1)
        attention_scores = self.attention_weights(lstm_output)
        # Shape: (batch, seq_len)
        attention_scores = attention_scores.squeeze(-1)

        # Apply mask if provided (set padding positions to -inf)
        # Use -1e4 instead of -1e9 for float16 compatibility (mixed precision training)
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, -1e4)

        # Compute attention weights (softmax over sequence dimension)
        attention_weights = F.softmax(attention_scores, dim=1)

        # Compute weighted sum of hidden states
        # Shape: (batch, hidden_dim)
        context = torch.sum(
            lstm_output * attention_weights.unsqueeze(-1), dim=1
        )

        return context, attention_weights


class LSTMTextModel(nn.Module):
    """
    Bidirectional LSTM with attention for text classification.

    Architecture:
        Embedding → Bidirectional LSTM → Attention → FC → Dropout → Output

    Attributes:
        vocab_size (int): Size of vocabulary
        embedding_dim (int): Dimension of word embeddings
        hidden_dim (int): LSTM hidden dimension
        num_layers (int): Number of LSTM layers
        num_classes (int): Number of output classes
        use_attention (bool): Whether to use attention mechanism

    Example:
        >>> model = LSTMTextModel(
        ...     vocab_size=10000,
        ...     embedding_dim=300,
        ...     hidden_dim=256,
        ...     num_layers=2,
        ...     num_classes=18
        ... )
        >>> text = torch.randint(0, 10000, (32, 256))  # (batch, seq_len)
        >>> output = model(text)
        >>> output.shape
        torch.Size([32, 18])
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 300,
        hidden_dim: int = 256,
        num_layers: int = 2,
        num_classes: int = 18,
        dropout: float = 0.3,
        bidirectional: bool = True,
        use_attention: bool = True,
        pretrained_embeddings: Optional[torch.Tensor] = None,
    ):
        """
        Initialize LSTM text model.

        Args:
            vocab_size (int): Vocabulary size
            embedding_dim (int, optional): Embedding dimension. Defaults to 300.
            hidden_dim (int, optional): LSTM hidden dimension. Defaults to 256.
            num_layers (int, optional): Number of LSTM layers. Defaults to 2.
            num_classes (int, optional): Number of output classes. Defaults to 18.
            dropout (float, optional): Dropout probability. Defaults to 0.3.
            bidirectional (bool, optional): Use bidirectional LSTM. Defaults to True.
            use_attention (bool, optional): Use attention mechanism. Defaults to True.
            pretrained_embeddings (Optional[torch.Tensor], optional): Pretrained
                embedding weights (e.g., GloVe). Shape (vocab_size, embedding_dim).
                Defaults to None.
        """
        super(LSTMTextModel, self).__init__()

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.bidirectional = bidirectional
        self.use_attention = use_attention

        # Embedding layer
        self.embedding = nn.Embedding(
            vocab_size, embedding_dim, padding_idx=0
        )

        # Load pretrained embeddings if provided
        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(pretrained_embeddings)

        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
        )

        # Attention layer (if using attention)
        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        if use_attention:
            self.attention = AttentionLayer(lstm_output_dim)

        # Classification head
        self.fc = nn.Linear(lstm_output_dim, 512)
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(512, num_classes)

    def forward(
        self, input_ids: torch.Tensor, return_attention: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass.

        Args:
            input_ids (torch.Tensor): Token indices, shape (batch, seq_len)
            return_attention (bool, optional): Return attention weights.
                Defaults to False.

        Returns:
            Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
                - If return_attention=False: Logits, shape (batch, num_classes)
                - If return_attention=True: (logits, attention_weights)
        """
        # Embedding: (batch, seq_len) → (batch, seq_len, embedding_dim)
        embedded = self.embedding(input_ids)

        # LSTM: (batch, seq_len, embedding_dim) → (batch, seq_len, lstm_output_dim)
        lstm_output, (hidden, cell) = self.lstm(embedded)

        if self.use_attention:
            # Attention: (batch, seq_len, lstm_output_dim) → (batch, lstm_output_dim)
            # Create mask for padding (assuming padding_idx=0)
            mask = (input_ids != 0).long()  # Shape: (batch, seq_len)
            context, attention_weights = self.attention(lstm_output, mask)
        else:
            # Use last hidden state (concatenate forward and backward)
            if self.bidirectional:
                # hidden shape: (num_layers*2, batch, hidden_dim)
                # Get last layer, concatenate forward and backward
                context = torch.cat((hidden[-2], hidden[-1]), dim=1)
            else:
                # Get last layer
                context = hidden[-1]
            attention_weights = None

        # Classification head
        x = F.relu(self.fc(context))
        x = self.dropout(x)
        logits = self.output(x)

        if return_attention and self.use_attention:
            return logits, attention_weights
        else:
            return logits


class DistilBERTTextModel(nn.Module):
    """
    DistilBERT-based text classification model.

    Fine-tunes DistilBERT for multi-label genre classification.

    Architecture:
        DistilBERT → [CLS] token → Dropout → FC → Dropout → Output

    Attributes:
        bert (DistilBertModel): Pretrained DistilBERT encoder
        num_classes (int): Number of output classes
        fine_tune_all (bool): Whether to fine-tune all layers

    Example:
        >>> model = DistilBERTTextModel(num_classes=18)
        >>> input_ids = torch.randint(0, 30522, (32, 512))
        >>> attention_mask = torch.ones(32, 512)
        >>> output = model(input_ids, attention_mask)
        >>> output.shape
        torch.Size([32, 18])
    """

    def __init__(
        self,
        model_name: str = "distilbert-base-uncased",
        num_classes: int = 18,
        classifier_hidden_dim: int = 256,
        dropout: float = 0.3,
        fine_tune_all: bool = True,
    ):
        """
        Initialize DistilBERT text model.

        Args:
            model_name (str, optional): Hugging Face model name.
                Defaults to 'distilbert-base-uncased'.
            num_classes (int, optional): Number of output classes. Defaults to 18.
            classifier_hidden_dim (int, optional): Hidden dimension of classifier.
                Defaults to 256.
            dropout (float, optional): Dropout probability. Defaults to 0.3.
            fine_tune_all (bool, optional): Fine-tune all BERT layers (True) or
                freeze encoder and only train classifier (False). Defaults to True.
        """
        super(DistilBERTTextModel, self).__init__()

        self.num_classes = num_classes
        self.fine_tune_all = fine_tune_all

        # Load pretrained DistilBERT
        self.bert = DistilBertModel.from_pretrained(model_name)

        # Freeze BERT parameters if not fine-tuning all
        if not fine_tune_all:
            for param in self.bert.parameters():
                param.requires_grad = False

        # Classification head
        # DistilBERT hidden size is 768
        bert_hidden_size = self.bert.config.hidden_size  # 768

        self.dropout1 = nn.Dropout(dropout)
        self.fc = nn.Linear(bert_hidden_size, classifier_hidden_dim)
        self.dropout2 = nn.Dropout(dropout)
        self.output = nn.Linear(classifier_hidden_dim, num_classes)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            input_ids (torch.Tensor): Token indices from BERT tokenizer,
                shape (batch, seq_len)
            attention_mask (Optional[torch.Tensor], optional): Attention mask
                (1 for real tokens, 0 for padding), shape (batch, seq_len).
                Defaults to None.

        Returns:
            torch.Tensor: Logits, shape (batch, num_classes)
        """
        # DistilBERT forward pass
        # Returns last hidden state: (batch, seq_len, hidden_size)
        outputs = self.bert(
            input_ids=input_ids, attention_mask=attention_mask
        )

        # Get [CLS] token representation (first token)
        # Shape: (batch, hidden_size)
        cls_output = outputs.last_hidden_state[:, 0, :]

        # Classification head
        x = self.dropout1(cls_output)
        x = F.relu(self.fc(x))
        x = self.dropout2(x)
        logits = self.output(x)

        return logits

    def unfreeze_encoder(self):
        """Unfreeze BERT encoder for fine-tuning."""
        for param in self.bert.parameters():
            param.requires_grad = True
        self.fine_tune_all = True

    def freeze_encoder(self):
        """Freeze BERT encoder (only train classifier)."""
        for param in self.bert.parameters():
            param.requires_grad = False
        self.fine_tune_all = False


# ============================================================================
# Model Factory Function
# ============================================================================


def create_text_model(
    config: dict, vocab_size: Optional[int] = None
) -> nn.Module:
    """
    Factory function to create text models based on configuration.

    Args:
        config (dict): Model configuration dictionary
        vocab_size (Optional[int], optional): Vocabulary size (for LSTM).
            Required if model_type is 'lstm'. Defaults to None.

    Returns:
        nn.Module: Initialized text model

    Raises:
        ValueError: If model type is unknown or required parameters missing

    Example:
        >>> config = {
        ...     'type': 'lstm_text',
        ...     'embedding_dim': 300,
        ...     'hidden_dim': 256,
        ...     'num_layers': 2
        ... }
        >>> model = create_text_model(config, vocab_size=10000)
    """
    model_type = config.get("type", "").lower()

    if model_type == "lstm_text" or model_type == "lstm":
        if vocab_size is None:
            raise ValueError("vocab_size required for LSTM model")

        model = LSTMTextModel(
            vocab_size=vocab_size,
            embedding_dim=config.get("embedding_dim", 300),
            hidden_dim=config.get("hidden_dim", 256),
            num_layers=config.get("num_layers", 2),
            num_classes=config.get("num_classes", 18),
            dropout=config.get("dropout", 0.3),
            bidirectional=config.get("bidirectional", True),
            use_attention=config.get("attention", True),
        )

    elif model_type == "distilbert_text" or model_type == "distilbert":
        model = DistilBERTTextModel(
            model_name=config.get("model_name", "distilbert-base-uncased"),
            num_classes=config.get("num_classes", 18),
            classifier_hidden_dim=config.get("classifier_hidden_dim", 256),
            dropout=config.get("dropout", 0.3),
            fine_tune_all=config.get("fine_tune_all", True),
        )

    else:
        raise ValueError(f"Unknown text model type: {model_type}")

    return model


# Example usage and testing
if __name__ == "__main__":
    print("Testing Text Models...\n")

    # Test LSTM model
    print("1. LSTM Text Model")
    lstm_model = LSTMTextModel(
        vocab_size=10000,
        embedding_dim=300,
        hidden_dim=256,
        num_layers=2,
        num_classes=18,
        use_attention=True,
    )
    # Dummy input: batch_size=4, seq_len=100
    dummy_text = torch.randint(0, 10000, (4, 100))
    lstm_output, attention_weights = lstm_model(
        dummy_text, return_attention=True
    )
    print(f"   Input shape: {dummy_text.shape}")
    print(f"   Output shape: {lstm_output.shape}")
    print(f"   Attention weights shape: {attention_weights.shape}")
    print(
        f"   Parameters: {sum(p.numel() for p in lstm_model.parameters()):,}\n"
    )

    # Test DistilBERT model
    print("2. DistilBERT Text Model")
    bert_model = DistilBERTTextModel(num_classes=18, fine_tune_all=True)
    # Dummy input: batch_size=4, seq_len=512
    dummy_input_ids = torch.randint(0, 30522, (4, 512))
    dummy_attention_mask = torch.ones(4, 512)
    bert_output = bert_model(dummy_input_ids, dummy_attention_mask)
    print(f"   Input shape: {dummy_input_ids.shape}")
    print(f"   Output shape: {bert_output.shape}")
    print(
        f"   Parameters: {sum(p.numel() for p in bert_model.parameters()):,}\n"
    )

    print("All text models tested successfully!")
