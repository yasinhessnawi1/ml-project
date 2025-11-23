"""
PyTorch Dataset classes for multimodal genre classification.

This module provides Dataset classes for loading and processing MM-IMDb data,
supporting both unimodal (text-only, image-only) and multimodal training.

Classes:
    MultimodalDataset: Main dataset class for multimodal learning
    TextOnlyDataset: Dataset for text-only models
    ImageOnlyDataset: Dataset for image-only models
"""

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
from torch.utils.data import Dataset
from PIL import Image

from src.data.preprocessing import (
    clean_text,
    encode_genres,
    load_image,
    get_image_transforms,
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MultimodalDataset(Dataset):
    """
    PyTorch Dataset for multimodal genre classification.

    Loads movie posters (images) and plot summaries (text) along with
    multi-label genre annotations.

    Attributes:
        data_dir (Path): Root data directory
        split (str): Dataset split ('train', 'val', or 'test')
        samples (List[Dict]): List of sample dictionaries
        text_tokenizer: Tokenizer for text preprocessing
        image_transform: Transforms for image preprocessing
        genre_to_idx (Dict[str, int]): Genre name to index mapping
        num_genres (int): Total number of genres

    Example:
        >>> dataset = MultimodalDataset(
        ...     data_dir='data/processed',
        ...     split='train',
        ...     text_tokenizer=tokenizer,
        ...     image_transform=transform,
        ...     genre_to_idx=genre_mapping
        ... )
        >>> sample = dataset[0]
        >>> print(sample['text'].shape, sample['image'].shape, sample['labels'].shape)
    """

    def __init__(
        self,
        data_dir: Union[str, Path],
        split: str,
        text_tokenizer,
        image_transform,
        genre_to_idx: Dict[str, int],
        min_text_length: int = 10,
        skip_missing: bool = True,
    ):
        """
        Initialize MultimodalDataset.

        Args:
            data_dir (Union[str, Path]): Root directory containing dataset
            split (str): Dataset split ('train', 'val', or 'test')
            text_tokenizer: Tokenizer for text (LSTM or BERT tokenizer)
            image_transform: Torchvision transforms for images
            genre_to_idx (Dict[str, int]): Mapping from genre name to index
            min_text_length (int, optional): Minimum text length in words.
                Defaults to 10.
            skip_missing (bool, optional): Skip samples with missing modalities.
                Defaults to True.
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.text_tokenizer = text_tokenizer
        self.image_transform = image_transform
        self.genre_to_idx = genre_to_idx
        self.num_genres = len(genre_to_idx)
        self.min_text_length = min_text_length
        self.skip_missing = skip_missing

        # Load metadata
        self.samples = self._load_metadata()

        # Statistics
        self.skipped_count = 0
        self.error_count = 0

        logger.info(f"Loaded {len(self.samples)} samples for {split} split")

    def _load_metadata(self) -> List[Dict]:
        """
        Load dataset metadata (file paths, genres, etc.).

        Returns:
            List[Dict]: List of sample dictionaries

        Note:
            This is a placeholder. Actual implementation depends on MM-IMDb format.
            Typically loads from a JSON/CSV file with movie IDs and metadata.
        """
        metadata_file = self.data_dir / f"{self.split}.json"

        # Convert to string to avoid Path circular reference in Python 3.12 Windows
        metadata_file_str = str(metadata_file)

        # Use os.path.exists instead of Path.exists() to avoid circular reference
        if not os.path.exists(metadata_file_str):
            raise FileNotFoundError(
                f"Metadata file not found: {metadata_file_str}"
            )

        with open(metadata_file_str, "r") as f:
            metadata = json.load(f)

        samples = []
        for item in metadata:
            # Expected format:
            # {
            #     'id': 'movie_id',
            #     'plot': 'path/to/plot.txt' or 'text content',
            #     'poster': 'path/to/poster.jpg',
            #     'genres': ['Action', 'Drama', ...]
            # }

            # Build file paths - store as strings to avoid Path circular reference in Python 3.12 Windows
            movie_id = item["id"]
            text_path_str = str(self.data_dir / item.get(
                "plot", f"{movie_id}/plot.txt"
            ))
            image_path_str = str(self.data_dir / item.get(
                "poster", f"{movie_id}/poster.jpg"
            ))

            samples.append(
                {
                    "id": movie_id,
                    "text_path": text_path_str,
                    "image_path": image_path_str,
                    "genres": item["genres"],
                }
            )

        return samples

    def _load_text(self, text_path: str) -> Optional[str]:
        """
        Load text from file.

        Args:
            text_path (str): Path to text file

        Returns:
            Optional[str]: Text content, or None if loading fails
        """
        try:
            # Path is already a string, no conversion needed
            if not os.path.exists(text_path):
                print(f"Warning: Text file not found: {text_path}")
                return None

            with open(
                text_path, "r", encoding="utf-8", errors="ignore"
            ) as f:
                text = f.read()

            # Clean text
            text = clean_text(text, remove_html=True, lowercase=False)

            # Check minimum length
            word_count = len(text.split())
            if word_count < self.min_text_length:
                print(f"Warning: Text too short ({word_count} words): {text_path}")
                return None

            return text

        except Exception as e:
            print(f"Error loading text {text_path}: {e}")
            self.error_count += 1
            return None

    def __len__(self) -> int:
        """Return number of samples in dataset."""
        return len(self.samples)

    def __getitem__(self, idx: int, _recursion_depth: int = 0) -> Optional[Dict[str, torch.Tensor]]:
        """
        Get a single sample.

        Args:
            idx (int): Sample index
            _recursion_depth (int): Internal recursion counter

        Returns:
            Optional[Dict[str, torch.Tensor]]: Dictionary containing:
                - 'text': Tokenized text tensor
                - 'image': Preprocessed image tensor
                - 'labels': Multi-hot genre encoding
                - 'movie_id': Movie identifier
            Returns None if sample cannot be loaded and skip_missing=True

        Raises:
            IndexError: If idx is out of bounds
        """
        # Prevent infinite recursion
        if _recursion_depth > len(self):
            raise RuntimeError(
                f"Could not load any valid samples after trying {_recursion_depth} attempts. "
                f"Check that text and image files exist in {self.data_dir}"
            )

        if idx >= len(self.samples):
            raise IndexError(
                f"Index {idx} out of range for dataset of size {len(self)}"
            )

        sample = self.samples[idx]

        # Load text
        text = self._load_text(sample["text_path"])
        if text is None:
            if self.skip_missing:
                self.skipped_count += 1
                # Return next sample (recursive call with bounds check)
                return self.__getitem__((idx + 1) % len(self), _recursion_depth + 1)
            else:
                # Use placeholder text
                text = "No summary available."

        # Tokenize text
        text_output = self.text_tokenizer(text)

        # Load image
        image = load_image(sample["image_path"])
        if image is None:
            if self.skip_missing:
                self.skipped_count += 1
                return self.__getitem__((idx + 1) % len(self), _recursion_depth + 1)
            else:
                # Use blank image
                image = Image.new("RGB", (224, 224), color=(0, 0, 0))

        # Transform image
        image_tensor = self.image_transform(image)

        # Encode genres
        labels = encode_genres(
            sample["genres"], self.genre_to_idx, self.num_genres
        )

        # Handle different tokenizer outputs
        # BERT tokenizers return dict with 'input_ids' and 'attention_mask'
        # LSTM tokenizers return a tensor directly
        if isinstance(text_output, dict):
            return {
                "text": text_output['input_ids'],
                "attention_mask": text_output['attention_mask'],
                "image": image_tensor,
                "labels": labels,
                "movie_id": sample["id"],
            }
        else:
            return {
                "text": text_output,
                "image": image_tensor,
                "labels": labels,
                "movie_id": sample["id"],
            }

    def get_statistics(self) -> Dict[str, int]:
        """
        Get dataset statistics.

        Returns:
            Dict[str, int]: Statistics including skipped and error counts
        """
        return {
            "total_samples": len(self.samples),
            "skipped_samples": self.skipped_count,
            "errors": self.error_count,
        }


class TextOnlyDataset(Dataset):
    """
    Dataset for text-only models.

    Only loads and preprocesses text data (plot summaries).

    Example:
        >>> dataset = TextOnlyDataset(
        ...     data_dir='data/processed',
        ...     split='train',
        ...     text_tokenizer=tokenizer,
        ...     genre_to_idx=genre_mapping
        ... )
        >>> sample = dataset[0]
        >>> print(sample['text'].shape, sample['labels'].shape)
    """

    def __init__(
        self,
        data_dir: Union[str, Path],
        split: str,
        text_tokenizer,
        genre_to_idx: Dict[str, int],
        min_text_length: int = 10,
    ):
        """
        Initialize TextOnlyDataset.

        Args:
            data_dir (Union[str, Path]): Root directory containing dataset
            split (str): Dataset split ('train', 'val', or 'test')
            text_tokenizer: Tokenizer for text (LSTM or BERT tokenizer)
            genre_to_idx (Dict[str, int]): Mapping from genre name to index
            min_text_length (int, optional): Minimum text length. Defaults to 10.
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.text_tokenizer = text_tokenizer
        self.genre_to_idx = genre_to_idx
        self.num_genres = len(genre_to_idx)
        self.min_text_length = min_text_length

        self.samples = self._load_metadata()
        logger.info(
            f"Loaded {len(self.samples)} text samples for {split} split"
        )

    def _load_metadata(self) -> List[Dict]:
        """Load metadata (same as MultimodalDataset)."""
        metadata_file = self.data_dir / f"{self.split}.json"

        # Convert to string to avoid Path circular reference in Python 3.12 Windows
        metadata_file_str = str(metadata_file)

        # Use os.path.exists instead of Path.exists() to avoid circular reference
        if not os.path.exists(metadata_file_str):
            raise FileNotFoundError(
                f"Metadata file not found: {metadata_file_str}"
            )

        with open(metadata_file_str, "r") as f:
            metadata = json.load(f)

        samples = []
        for item in metadata:
            movie_id = item["id"]
            # Store as string to avoid Path circular reference in Python 3.12 Windows
            text_path_str = str(self.data_dir / item.get(
                "plot", f"{movie_id}/plot.txt"
            ))

            samples.append(
                {
                    "id": movie_id,
                    "text_path": text_path_str,
                    "genres": item["genres"],
                }
            )

        return samples

    def _load_text(self, text_path: str) -> Optional[str]:
        """Load and validate text."""
        try:
            # Path is already a string, no conversion needed
            if not os.path.exists(text_path):
                return None

            with open(
                text_path, "r", encoding="utf-8", errors="ignore"
            ) as f:
                text = f.read()

            text = clean_text(text, remove_html=True, lowercase=False)

            if len(text.split()) < self.min_text_length:
                return None

            return text

        except Exception as e:
            # Avoid using logger to prevent recursion in error handling
            print(f"Error loading text {text_path}: {e}")
            return None

    def __len__(self) -> int:
        """Return number of samples."""
        return len(self.samples)

    def __getitem__(self, idx: int, _recursion_depth: int = 0) -> Dict[str, torch.Tensor]:
        """Get a single text sample."""
        # Prevent infinite recursion
        if _recursion_depth > len(self):
            raise RuntimeError(
                f"Could not load any valid samples after trying {_recursion_depth} attempts. "
                f"Check that text files exist in {self.data_dir}"
            )

        sample = self.samples[idx]

        # Load text
        text = self._load_text(sample["text_path"])
        if text is None:
            # Recursively get next valid sample
            return self.__getitem__((idx + 1) % len(self), _recursion_depth + 1)

        # Tokenize
        text_output = self.text_tokenizer(text)

        # Encode labels
        labels = encode_genres(
            sample["genres"], self.genre_to_idx, self.num_genres
        )

        # Handle BERT tokenizer (returns dict) vs LSTM tokenizer (returns tensor)
        if isinstance(text_output, dict):
            # BERT tokenizer - unpack input_ids and attention_mask
            return {
                "text": text_output['input_ids'],
                "attention_mask": text_output['attention_mask'],
                "labels": labels,
                "movie_id": sample["id"],
            }
        else:
            # LSTM tokenizer - returns tensor directly
            return {
                "text": text_output,
                "labels": labels,
                "movie_id": sample["id"],
            }


class ImageOnlyDataset(Dataset):
    """
    Dataset for image-only models.

    Only loads and preprocesses image data (movie posters).

    Example:
        >>> dataset = ImageOnlyDataset(
        ...     data_dir='data/processed',
        ...     split='train',
        ...     image_transform=transform,
        ...     genre_to_idx=genre_mapping
        ... )
        >>> sample = dataset[0]
        >>> print(sample['image'].shape, sample['labels'].shape)
    """

    def __init__(
        self,
        data_dir: Union[str, Path],
        split: str,
        image_transform,
        genre_to_idx: Dict[str, int],
    ):
        """
        Initialize ImageOnlyDataset.

        Args:
            data_dir (Union[str, Path]): Root directory containing dataset
            split (str): Dataset split ('train', 'val', or 'test')
            image_transform: Torchvision transforms for images
            genre_to_idx (Dict[str, int]): Mapping from genre name to index
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.image_transform = image_transform
        self.genre_to_idx = genre_to_idx
        self.num_genres = len(genre_to_idx)

        self.samples = self._load_metadata()
        logger.info(
            f"Loaded {len(self.samples)} image samples for {split} split"
        )

    def _load_metadata(self) -> List[Dict]:
        """Load metadata (same as MultimodalDataset)."""
        metadata_file = self.data_dir / f"{self.split}.json"

        # Convert to string to avoid Path circular reference in Python 3.12 Windows
        metadata_file_str = str(metadata_file)

        # Use os.path.exists instead of Path.exists() to avoid circular reference
        if not os.path.exists(metadata_file_str):
            raise FileNotFoundError(
                f"Metadata file not found: {metadata_file_str}"
            )

        with open(metadata_file_str, "r") as f:
            metadata = json.load(f)

        samples = []
        for item in metadata:
            movie_id = item["id"]
            # Store as string to avoid Path circular reference in Python 3.12 Windows
            image_path_str = str(self.data_dir / item.get(
                "poster", f"{movie_id}/poster.jpg"
            ))

            samples.append(
                {
                    "id": movie_id,
                    "image_path": image_path_str,
                    "genres": item["genres"],
                }
            )

        return samples

    def __len__(self) -> int:
        """Return number of samples."""
        return len(self.samples)

    def __getitem__(self, idx: int, _recursion_depth: int = 0) -> Dict[str, torch.Tensor]:
        """Get a single image sample."""
        # Prevent infinite recursion
        if _recursion_depth > len(self):
            raise RuntimeError(
                f"Could not load any valid samples after trying {_recursion_depth} attempts. "
                f"Check that image files exist in {self.data_dir}"
            )

        sample = self.samples[idx]

        # Load image
        image = load_image(sample["image_path"])
        if image is None:
            # Use blank image or get next sample
            return self.__getitem__((idx + 1) % len(self), _recursion_depth + 1)

        # Transform image
        image_tensor = self.image_transform(image)

        # Encode labels
        labels = encode_genres(
            sample["genres"], self.genre_to_idx, self.num_genres
        )

        return {
            "image": image_tensor,
            "labels": labels,
            "movie_id": sample["id"],
        }


# ============================================================================
# Utility Functions for DataLoader
# ============================================================================


def create_dataloaders(
    data_dir: Union[str, Path],
    text_tokenizer,
    genre_to_idx: Dict[str, int],
    config: Dict,
    dataset_type: str = "multimodal",
) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Create train, validation, and test dataloaders.

    Args:
        data_dir (Union[str, Path]): Root data directory
        text_tokenizer: Text tokenizer (LSTM or BERT)
        genre_to_idx (Dict[str, int]): Genre to index mapping
        config (Dict): Configuration dictionary
        dataset_type (str, optional): Type of dataset ('multimodal', 'text', 'image').
            Defaults to 'multimodal'.

    Returns:
        Tuple[Dataset, Dataset, Dataset]: Train, validation, and test datasets

    Example:
        >>> train_ds, val_ds, test_ds = create_dataloaders(
        ...     'data/processed', tokenizer, genre_mapping, config
        ... )
    """
    # Get image transforms
    train_img_transform = get_image_transforms(
        split="train",
        image_size=config["preprocessing"]["image"]["target_size"],
        resize_size=config["preprocessing"]["image"]["resize_size"],
        augmentation_config=config["preprocessing"]["augmentation"],
    )
    val_img_transform = get_image_transforms(
        split="val",
        image_size=config["preprocessing"]["image"]["target_size"],
        resize_size=config["preprocessing"]["image"]["resize_size"],
    )

    # Select dataset class
    if dataset_type == "multimodal":
        DatasetClass = MultimodalDataset
        dataset_kwargs = {
            "text_tokenizer": text_tokenizer,
            "image_transform": None,  # Will be set per split
            "genre_to_idx": genre_to_idx,
            "min_text_length": config["preprocessing"]["text"][
                "min_text_length"
            ],
        }
    elif dataset_type == "text":
        DatasetClass = TextOnlyDataset
        dataset_kwargs = {
            "text_tokenizer": text_tokenizer,
            "genre_to_idx": genre_to_idx,
            "min_text_length": config["preprocessing"]["text"][
                "min_text_length"
            ],
        }
    elif dataset_type == "image":
        DatasetClass = ImageOnlyDataset
        dataset_kwargs = {
            "image_transform": None,  # Will be set per split
            "genre_to_idx": genre_to_idx,
        }
    else:
        raise ValueError(f"Unknown dataset_type: {dataset_type}")

    # Create datasets
    if dataset_type == "multimodal":
        train_dataset = DatasetClass(
            data_dir=data_dir,
            split="train",
            **{**dataset_kwargs, "image_transform": train_img_transform},
        )
        val_dataset = DatasetClass(
            data_dir=data_dir,
            split="val",
            **{**dataset_kwargs, "image_transform": val_img_transform},
        )
        test_dataset = DatasetClass(
            data_dir=data_dir,
            split="test",
            **{**dataset_kwargs, "image_transform": val_img_transform},
        )
    elif dataset_type == "image":
        train_dataset = DatasetClass(
            data_dir=data_dir,
            split="train",
            **{**dataset_kwargs, "image_transform": train_img_transform},
        )
        val_dataset = DatasetClass(
            data_dir=data_dir,
            split="val",
            **{**dataset_kwargs, "image_transform": val_img_transform},
        )
        test_dataset = DatasetClass(
            data_dir=data_dir,
            split="test",
            **{**dataset_kwargs, "image_transform": val_img_transform},
        )
    else:  # text-only
        train_dataset = DatasetClass(
            data_dir=data_dir, split="train", **dataset_kwargs
        )
        val_dataset = DatasetClass(
            data_dir=data_dir, split="val", **dataset_kwargs
        )
        test_dataset = DatasetClass(
            data_dir=data_dir, split="test", **dataset_kwargs
        )

    return train_dataset, val_dataset, test_dataset


# Example usage
if __name__ == "__main__":
    print("Dataset module - use with actual data files")
    print("Example:")
    print("  from src.data.dataset import MultimodalDataset")
    print("  dataset = MultimodalDataset(...)")
