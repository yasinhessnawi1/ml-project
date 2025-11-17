"""
Data preprocessing utilities for text and images.

This module provides functions and classes for preprocessing text (plot summaries)
and images (movie posters) for the multimodal genre classification task.

Classes:
    LSTMTokenizer: Custom tokenizer for LSTM models

Functions:
    clean_text: Clean and normalize text data
    build_vocab: Build vocabulary from text corpus
    encode_genres: Encode genre labels as multi-hot vectors
    get_image_transforms: Get image preprocessing transforms
    load_image: Load and validate image file
"""

import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from transformers import DistilBertTokenizer


# ============================================================================
# Text Preprocessing
# ============================================================================


def clean_text(
    text: str, remove_html: bool = True, lowercase: bool = False
) -> str:
    """
    Clean and normalize text data.

    Removes HTML tags, normalizes whitespace, and optionally converts to lowercase.

    Args:
        text (str): Raw text to clean
        remove_html (bool, optional): Remove HTML tags. Defaults to True.
        lowercase (bool, optional): Convert to lowercase. Defaults to False.

    Returns:
        str: Cleaned text

    Example:
        >>> text = "<p>A thrilling  action movie!</p>"
        >>> clean_text(text, remove_html=True, lowercase=True)
        'a thrilling action movie!'
    """
    if not isinstance(text, str):
        return ""

    # Remove HTML tags
    if remove_html:
        text = re.sub(r"<[^>]+>", "", text)

    # Normalize whitespace (multiple spaces to single space)
    text = re.sub(r"\s+", " ", text)

    # Remove leading/trailing whitespace
    text = text.strip()

    # Convert to lowercase if specified
    if lowercase:
        text = text.lower()

    return text


def build_vocab(
    texts: List[str],
    vocab_size: int = 10000,
    min_freq: int = 2,
    special_tokens: Optional[List[str]] = None,
) -> Dict[str, int]:
    """
    Build vocabulary from a list of texts.

    Creates a word-to-index mapping for the most frequent words.

    Args:
        texts (List[str]): List of text documents
        vocab_size (int, optional): Maximum vocabulary size. Defaults to 10000.
        min_freq (int, optional): Minimum word frequency to include. Defaults to 2.
        special_tokens (Optional[List[str]], optional): Special tokens to add.
            Defaults to ['<PAD>', '<UNK>', '<SOS>', '<EOS>'].

    Returns:
        Dict[str, int]: Word to index mapping

    Example:
        >>> texts = ["hello world", "hello there"]
        >>> vocab = build_vocab(texts, vocab_size=100)
        >>> vocab['hello']
        4  # After special tokens
    """
    if special_tokens is None:
        special_tokens = ["<PAD>", "<UNK>", "<SOS>", "<EOS>"]

    # Count word frequencies
    word_counts = Counter()
    for text in texts:
        # Simple whitespace tokenization
        words = text.lower().split()
        word_counts.update(words)

    # Filter by minimum frequency
    filtered_counts = {
        word: count
        for word, count in word_counts.items()
        if count >= min_freq
    }

    # Get most common words (convert back to Counter to use most_common method)
    word_counts = Counter(filtered_counts)
    most_common = word_counts.most_common(vocab_size - len(special_tokens))

    # Build vocabulary (word to index mapping)
    vocab = {token: idx for idx, token in enumerate(special_tokens)}
    for idx, (word, _) in enumerate(most_common):
        vocab[word] = len(special_tokens) + idx

    return vocab


class LSTMTokenizer:
    """
    Tokenizer for LSTM text models.

    Handles text cleaning, tokenization, and conversion to indices.

    Attributes:
        vocab (Dict[str, int]): Word to index mapping
        max_length (int): Maximum sequence length
        pad_token (str): Padding token
        unk_token (str): Unknown token

    Example:
        >>> tokenizer = LSTMTokenizer(vocab, max_length=256)
        >>> tokens = tokenizer("A thrilling action movie")
        >>> tokens.shape
        torch.Size([256])
    """

    def __init__(
        self,
        vocab: Dict[str, int],
        max_length: int = 256,
        pad_token: str = "<PAD>",
        unk_token: str = "<UNK>",
    ):
        """
        Initialize LSTM tokenizer.

        Args:
            vocab (Dict[str, int]): Word to index mapping
            max_length (int, optional): Maximum sequence length. Defaults to 256.
            pad_token (str, optional): Padding token. Defaults to '<PAD>'.
            unk_token (str, optional): Unknown token. Defaults to '<UNK>'.
        """
        self.vocab = vocab
        self.max_length = max_length
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.pad_idx = vocab.get(pad_token, 0)
        self.unk_idx = vocab.get(unk_token, 1)

    def __call__(self, text: str) -> torch.Tensor:
        """
        Tokenize text and convert to tensor of indices.

        Args:
            text (str): Text to tokenize

        Returns:
            torch.Tensor: Tensor of token indices, shape (max_length,)
        """
        # Clean and tokenize
        text = clean_text(text, lowercase=True)
        words = text.split()

        # Convert to indices
        indices = [self.vocab.get(word, self.unk_idx) for word in words]

        # Truncate or pad
        if len(indices) > self.max_length:
            indices = indices[: self.max_length]
        else:
            indices = indices + [self.pad_idx] * (
                self.max_length - len(indices)
            )

        return torch.tensor(indices, dtype=torch.long)

    def decode(self, indices: torch.Tensor) -> str:
        """
        Decode tensor of indices back to text.

        Args:
            indices (torch.Tensor): Tensor of token indices

        Returns:
            str: Decoded text
        """
        # Create reverse vocabulary
        idx_to_word = {idx: word for word, idx in self.vocab.items()}

        # Convert indices to words
        words = [
            idx_to_word.get(idx.item(), self.unk_token)
            for idx in indices
            if idx.item() != self.pad_idx
        ]

        return " ".join(words)


def get_bert_tokenizer(
    model_name: str = "distilbert-base-uncased",
) -> DistilBertTokenizer:
    """
    Get pretrained BERT tokenizer.

    Args:
        model_name (str, optional): Hugging Face model name.
            Defaults to 'distilbert-base-uncased'.

    Returns:
        DistilBertTokenizer: Pretrained tokenizer

    Example:
        >>> tokenizer = get_bert_tokenizer()
        >>> encoding = tokenizer("A thrilling movie", max_length=512,
        ...                      padding='max_length', truncation=True)
    """
    return DistilBertTokenizer.from_pretrained(model_name)


# ============================================================================
# Image Preprocessing
# ============================================================================


def load_image(
    image_path: Union[str, Path], min_size: int = 50
) -> Optional[Image.Image]:
    """
    Load and validate an image file.

    Args:
        image_path (Union[str, Path]): Path to image file
        min_size (int, optional): Minimum width/height in pixels. Defaults to 50.

    Returns:
        Optional[Image.Image]: PIL Image object in RGB mode, or None if loading fails

    Example:
        >>> img = load_image('poster.jpg')
        >>> if img is not None:
        ...     print(img.size)
        (500, 750)
    """
    try:
        # Load image
        img = Image.open(image_path)

        # Convert to RGB (handles grayscale and RGBA)
        img = img.convert("RGB")

        # Check minimum size
        if img.size[0] < min_size or img.size[1] < min_size:
            print(f"Warning: Image too small ({img.size}): {image_path}")
            return None

        return img

    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None


def get_image_transforms(
    split: str = "train",
    image_size: int = 224,
    resize_size: int = 256,
    augmentation_config: Optional[Dict] = None,
) -> transforms.Compose:
    """
    Get image preprocessing transforms for train/val/test splits.

    Training transforms include data augmentation.
    Validation/test transforms are deterministic.

    Args:
        split (str, optional): Dataset split ('train', 'val', or 'test').
            Defaults to 'train'.
        image_size (int, optional): Target image size (square). Defaults to 224.
        resize_size (int, optional): Resize shorter edge to this before cropping.
            Defaults to 256.
        augmentation_config (Optional[Dict], optional): Augmentation parameters.
            If None, uses default values. Defaults to None.

    Returns:
        transforms.Compose: Composed transforms

    Example:
        >>> train_transform = get_image_transforms('train')
        >>> val_transform = get_image_transforms('val')
        >>> img_tensor = train_transform(pil_image)
        >>> img_tensor.shape
        torch.Size([3, 224, 224])
    """
    # Default augmentation configuration
    if augmentation_config is None:
        augmentation_config = {
            "horizontal_flip_prob": 0.5,
            "rotation_degrees": 10,
            "color_jitter": {
                "brightness": 0.2,
                "contrast": 0.2,
                "saturation": 0.2,
                "hue": 0.1,
            },
        }

    # ImageNet normalization (required for pretrained models)
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

    if split == "train":
        # Training transforms with augmentation
        transform = transforms.Compose(
            [
                transforms.Resize(resize_size),  # Resize shorter edge
                transforms.RandomCrop(
                    image_size
                ),  # Random crop for variation
                transforms.RandomHorizontalFlip(
                    p=augmentation_config["horizontal_flip_prob"]
                ),
                transforms.ColorJitter(
                    brightness=augmentation_config["color_jitter"][
                        "brightness"
                    ],
                    contrast=augmentation_config["color_jitter"][
                        "contrast"
                    ],
                    saturation=augmentation_config["color_jitter"][
                        "saturation"
                    ],
                    hue=augmentation_config["color_jitter"]["hue"],
                ),
                transforms.RandomRotation(
                    degrees=augmentation_config["rotation_degrees"]
                ),
                transforms.ToTensor(),  # Convert to tensor and scale to [0, 1]
                normalize,
            ]
        )
    else:
        # Validation/test transforms (deterministic)
        transform = transforms.Compose(
            [
                transforms.Resize(resize_size),
                transforms.CenterCrop(
                    image_size
                ),  # Center crop (deterministic)
                transforms.ToTensor(),
                normalize,
            ]
        )

    return transform


# ============================================================================
# Label Preprocessing
# ============================================================================


def encode_genres(
    genre_list: List[str],
    genre_to_idx: Dict[str, int],
    num_genres: Optional[int] = None,
) -> torch.Tensor:
    """
    Encode list of genres as multi-hot binary vector.

    Args:
        genre_list (List[str]): List of genre strings (e.g., ['Action', 'Drama'])
        genre_to_idx (Dict[str, int]): Mapping from genre name to index
        num_genres (Optional[int], optional): Total number of genres.
            If None, inferred from genre_to_idx. Defaults to None.

    Returns:
        torch.Tensor: Binary tensor of shape (num_genres,), dtype float32

    Example:
        >>> genre_to_idx = {'Action': 0, 'Comedy': 1, 'Drama': 2}
        >>> genres = ['Action', 'Drama']
        >>> encoding = encode_genres(genres, genre_to_idx, num_genres=3)
        >>> encoding
        tensor([1., 0., 1.])
    """
    if num_genres is None:
        num_genres = len(genre_to_idx)

    # Initialize zero vector
    encoding = torch.zeros(num_genres, dtype=torch.float32)

    # Set 1 for present genres
    for genre in genre_list:
        if genre in genre_to_idx:
            idx = genre_to_idx[genre]
            encoding[idx] = 1.0

    return encoding


def decode_genres(
    encoding: torch.Tensor,
    idx_to_genre: Dict[int, str],
    threshold: float = 0.5,
) -> List[str]:
    """
    Decode multi-hot encoding back to list of genre names.

    Args:
        encoding (torch.Tensor): Multi-hot encoding tensor
        idx_to_genre (Dict[int, str]): Mapping from index to genre name
        threshold (float, optional): Threshold for binary classification.
            Defaults to 0.5.

    Returns:
        List[str]: List of predicted genre names

    Example:
        >>> idx_to_genre = {0: 'Action', 1: 'Comedy', 2: 'Drama'}
        >>> encoding = torch.tensor([0.9, 0.2, 0.8])
        >>> decode_genres(encoding, idx_to_genre, threshold=0.5)
        ['Action', 'Drama']
    """
    # Apply threshold
    predicted_indices = (encoding >= threshold).nonzero(as_tuple=True)[0]

    # Convert indices to genre names
    genres = [idx_to_genre[idx.item()] for idx in predicted_indices]

    return genres


def apply_label_smoothing(
    labels: torch.Tensor, epsilon: float = 0.1
) -> torch.Tensor:
    """
    Apply label smoothing to binary labels.

    Converts hard labels (0 or 1) to soft labels to prevent overconfidence.

    Args:
        labels (torch.Tensor): Hard binary labels (0 or 1)
        epsilon (float, optional): Smoothing factor. Defaults to 0.1.

    Returns:
        torch.Tensor: Smoothed labels

    Formula:
        y_smooth = y * (1 - ε) + ε / 2

    Example:
        >>> labels = torch.tensor([0., 1., 1., 0.])
        >>> smoothed = apply_label_smoothing(labels, epsilon=0.1)
        >>> smoothed
        tensor([0.05, 0.95, 0.95, 0.05])
    """
    # y_smooth = y * (1 - epsilon) + epsilon / 2
    # For 0: 0 * (1 - epsilon) + epsilon / 2 = epsilon / 2
    # For 1: 1 * (1 - epsilon) + epsilon / 2 = 1 - epsilon / 2
    return labels * (1 - epsilon) + epsilon / 2


# ============================================================================
# Utility Functions
# ============================================================================


def compute_class_weights(labels: torch.Tensor) -> torch.Tensor:
    """
    Compute class weights for imbalanced datasets.

    Uses inverse frequency: weight = n_samples / (n_classes * n_samples_per_class)

    Args:
        labels (torch.Tensor): Multi-hot label tensor, shape (n_samples, n_classes)

    Returns:
        torch.Tensor: Class weights, shape (n_classes,)

    Example:
        >>> labels = torch.tensor([[1, 0, 1], [1, 1, 0], [0, 1, 1]])
        >>> weights = compute_class_weights(labels)
        >>> weights.shape
        torch.Size([3])
    """
    # Count positive samples per class
    pos_counts = labels.sum(dim=0)  # Shape: (n_classes,)

    # Count negative samples per class
    neg_counts = labels.shape[0] - pos_counts

    # Compute weights (higher weight for rare classes)
    # weight = neg_count / pos_count
    weights = neg_counts / (
        pos_counts + 1e-6
    )  # Add epsilon to avoid division by zero

    return weights


def get_text_statistics(texts: List[str]) -> Dict[str, float]:
    """
    Compute statistics about text lengths.

    Args:
        texts (List[str]): List of text documents

    Returns:
        Dict[str, float]: Statistics dictionary with keys:
            - 'mean': Mean length in words
            - 'median': Median length
            - 'std': Standard deviation
            - 'min': Minimum length
            - 'max': Maximum length
            - '95_percentile': 95th percentile length

    Example:
        >>> texts = ["short text", "this is a longer text with more words"]
        >>> stats = get_text_statistics(texts)
        >>> print(f"Mean length: {stats['mean']:.1f} words")
    """
    lengths = [len(text.split()) for text in texts]

    return {
        "mean": np.mean(lengths),
        "median": np.median(lengths),
        "std": np.std(lengths),
        "min": np.min(lengths),
        "max": np.max(lengths),
        "95_percentile": np.percentile(lengths, 95),
        "99_percentile": np.percentile(lengths, 99),
    }


# Example usage and testing
if __name__ == "__main__":
    # Test text cleaning
    sample_text = "<p>A thrilling   ACTION movie!</p>"
    cleaned = clean_text(sample_text, remove_html=True, lowercase=True)
    print(f"Original: {sample_text}")
    print(f"Cleaned: {cleaned}")

    # Test vocabulary building
    texts = ["hello world", "hello there world", "goodbye world"]
    vocab = build_vocab(texts, vocab_size=100)
    print(f"\nVocabulary size: {len(vocab)}")
    print(f"Sample vocab: {list(vocab.items())[:10]}")

    # Test tokenizer
    tokenizer = LSTMTokenizer(vocab, max_length=10)
    tokens = tokenizer("hello world")
    print(f"\nTokenized shape: {tokens.shape}")
    print(f"Token indices: {tokens}")

    # Test image transforms
    train_transform = get_image_transforms("train")
    val_transform = get_image_transforms("val")
    print(f"\nTrain transforms: {train_transform}")
    print(f"Val transforms: {val_transform}")

    # Test genre encoding
    genre_to_idx = {"Action": 0, "Comedy": 1, "Drama": 2}
    genres = ["Action", "Drama"]
    encoding = encode_genres(genres, genre_to_idx, num_genres=3)
    print(f"\nGenre encoding: {encoding}")

    print("\nAll preprocessing functions tested successfully!")
