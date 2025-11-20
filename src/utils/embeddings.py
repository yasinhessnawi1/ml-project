"""
GloVe embeddings utilities for loading pretrained word embeddings.

This module handles downloading, caching, and loading GloVe embeddings.
"""

import os
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple
from tqdm import tqdm
import urllib.request
import zipfile


def download_glove(
    glove_dir: Path,
    glove_name: str = "glove.6B.300d"
) -> Path:
    """
    Download GloVe embeddings if not already present.

    Args:
        glove_dir: Directory to store GloVe files
        glove_name: GloVe variant (e.g., 'glove.6B.300d')

    Returns:
        Path to the GloVe text file
    """
    glove_dir.mkdir(parents=True, exist_ok=True)

    # Map glove names to download URLs
    glove_urls = {
        'glove.6B.50d': 'http://nlp.stanford.edu/data/glove.6B.zip',
        'glove.6B.100d': 'http://nlp.stanford.edu/data/glove.6B.zip',
        'glove.6B.200d': 'http://nlp.stanford.edu/data/glove.6B.zip',
        'glove.6B.300d': 'http://nlp.stanford.edu/data/glove.6B.zip',
    }

    glove_file = glove_dir / f"{glove_name}.txt"

    if glove_file.exists():
        print(f"GloVe embeddings already exist at {glove_file}")
        return glove_file

    # Download and extract
    if glove_name not in glove_urls:
        raise ValueError(f"Unknown GloVe variant: {glove_name}. "
                         f"Available: {list(glove_urls.keys())}")

    url = glove_urls[glove_name]
    zip_file = glove_dir / "glove.6B.zip"

    if not zip_file.exists():
        print(f"Downloading GloVe embeddings from {url}...")
        print("This is a 822MB download, it may take several minutes...")

        def reporthook(count, block_size, total_size):
            """Progress bar for download."""
            global start_time
            if count == 0:
                start_time = time.time()
                return

            duration = time.time() - start_time
            progress_size = int(count * block_size)
            speed = int(progress_size / (1024 * duration)) if duration > 0 else 0
            percent = int(count * block_size * 100 / total_size)

            print(f"\r{percent}% | {progress_size / (1024**2):.1f} MB / "
                  f"{total_size / (1024**2):.1f} MB | {speed} KB/s", end='')

        import time
        urllib.request.urlretrieve(url, zip_file, reporthook)
        print("\nDownload complete!")

    # Extract the specific file
    print(f"Extracting {glove_name}.txt...")
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extract(f"{glove_name}.txt", glove_dir)

    print(f"GloVe embeddings ready at {glove_file}")
    return glove_file


def load_glove_embeddings(
    glove_file: Path,
    vocab: Dict[str, int],
    embedding_dim: int
) -> torch.Tensor:
    """
    Load GloVe embeddings for a given vocabulary.

    Args:
        glove_file: Path to GloVe text file
        vocab: Vocabulary mapping word -> index
        embedding_dim: Embedding dimension (must match GloVe file)

    Returns:
        Embedding matrix of shape (vocab_size, embedding_dim)
    """
    print(f"\nLoading GloVe embeddings from {glove_file}...")
    print(f"Vocabulary size: {len(vocab)}")
    print(f"Embedding dimension: {embedding_dim}")

    # Initialize embedding matrix with random values
    vocab_size = len(vocab)
    embeddings = torch.randn(vocab_size, embedding_dim) * 0.01

    # Special tokens get zero embeddings
    embeddings[0] = torch.zeros(embedding_dim)  # <PAD>
    if '<UNK>' in vocab:
        embeddings[vocab['<UNK>']] = torch.zeros(embedding_dim)

    # Load GloVe vectors
    found = 0
    with open(glove_file, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Loading GloVe"):
            values = line.split()
            word = values[0]

            if word in vocab:
                idx = vocab[word]
                vector = np.array(values[1:], dtype=np.float32)

                if len(vector) == embedding_dim:
                    embeddings[idx] = torch.from_numpy(vector)
                    found += 1

    coverage = (found / vocab_size) * 100
    print(f"\nGloVe coverage: {found}/{vocab_size} words ({coverage:.2f}%)")
    print(f"Words not in GloVe: {vocab_size - found} (will use random init)")

    return embeddings


def create_embedding_matrix(
    vocab: Dict[str, int],
    embedding_dim: int = 300,
    glove_path: Optional[Path] = None,
    use_pretrained: bool = True
) -> Tuple[torch.Tensor, Dict[str, any]]:
    """
    Create embedding matrix, optionally using pretrained GloVe.

    Args:
        vocab: Vocabulary mapping word -> index
        embedding_dim: Embedding dimension
        glove_path: Path to GloVe directory (will download if needed)
        use_pretrained: Whether to use pretrained embeddings

    Returns:
        Tuple of (embedding_matrix, info_dict)
    """
    vocab_size = len(vocab)

    if not use_pretrained:
        print(f"Creating random embeddings ({vocab_size} x {embedding_dim})...")
        embeddings = torch.randn(vocab_size, embedding_dim) * 0.01
        embeddings[0] = torch.zeros(embedding_dim)  # <PAD>

        info = {
            'vocab_size': vocab_size,
            'embedding_dim': embedding_dim,
            'pretrained': False,
            'coverage': 0.0
        }
        return embeddings, info

    # Use GloVe
    if glove_path is None:
        glove_path = Path.home() / '.cache' / 'glove'

    glove_name = f"glove.6B.{embedding_dim}d"
    glove_file = glove_path / f"{glove_name}.txt"

    # Download if needed
    if not glove_file.exists():
        print(f"\nGloVe embeddings not found. Downloading...")
        download_glove(glove_path, glove_name)

    # Load embeddings
    embeddings = load_glove_embeddings(glove_file, vocab, embedding_dim)

    found = (embeddings.abs().sum(dim=1) > 0).sum().item()
    coverage = (found / vocab_size) * 100

    info = {
        'vocab_size': vocab_size,
        'embedding_dim': embedding_dim,
        'pretrained': True,
        'glove_file': str(glove_file),
        'coverage': coverage,
        'words_found': found,
        'words_missing': vocab_size - found
    }

    return embeddings, info


if __name__ == "__main__":
    # Example usage
    vocab = {
        '<PAD>': 0,
        '<UNK>': 1,
        'the': 2,
        'movie': 3,
        'great': 4
    }

    embeddings, info = create_embedding_matrix(
        vocab,
        embedding_dim=300,
        use_pretrained=True
    )

    print(f"\nEmbedding shape: {embeddings.shape}")
    print(f"Info: {info}")
