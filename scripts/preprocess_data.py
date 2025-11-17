#!/usr/bin/env python3
"""
MM-IMDb Dataset Preprocessing Script

This script processes the raw MM-IMDb dataset from HDF5 format into
organized train/val/test splits.

MM-IMDb Dataset Structure:
- Total samples: 25,959 movies
- Images: (25959, 3, 256, 160) - RGB images
- Sequences: (25959,) - Tokenized plot text (variable length)
- Genres: (25959, 23) - Multi-hot genre labels (23 genres)
- IMDb IDs: (25959,) - Movie identifiers

Output Structure:
data/processed/
├── train/
│   ├── 0000001/
│   │   ├── plot.npy (tokenized sequence)
│   │   ├── poster.jpg
│   │   └── metadata.json (genres, imdb_id)
│   └── ...
├── val/
├── test/
├── dataset_statistics.json
├── genre_mapping.json
└── vocab.json (from metadata.npy)

Usage:
    python scripts/preprocess_data.py
    python scripts/preprocess_data.py --train-ratio 0.8 --val-ratio 0.1
    python scripts/preprocess_data.py --seed 42 --max-samples 1000
"""

import argparse
import sys
import json
import numpy as np
import h5py
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import Counter
from PIL import Image
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


# MM-IMDb has 23 genres (based on dataset inspection)
# Genre indices will be determined from the dataset
MMIMDB_NUM_GENRES = 23


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Preprocess MM-IMDb dataset',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--input-dir',
        type=str,
        default='data/raw',
        help='Directory containing raw MM-IMDb data'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/processed',
        help='Directory to save processed data'
    )

    parser.add_argument(
        '--train-ratio',
        type=float,
        default=0.7,
        help='Training set ratio'
    )

    parser.add_argument(
        '--val-ratio',
        type=float,
        default=0.15,
        help='Validation set ratio'
    )

    parser.add_argument(
        '--test-ratio',
        type=float,
        default=0.15,
        help='Test set ratio'
    )

    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )

    parser.add_argument(
        '--max-samples',
        type=int,
        default=None,
        help='Maximum number of samples to process (for testing)'
    )

    parser.add_argument(
        '--min-sequence-length',
        type=int,
        default=10,
        help='Minimum text sequence length (in tokens)'
    )

    parser.add_argument(
        '--skip-images',
        action='store_true',
        help='Skip saving images (text-only preprocessing)'
    )

    return parser.parse_args()


def load_mmimdb_data(input_dir: Path) -> Tuple[h5py.File, Dict]:
    """
    Load MM-IMDb dataset from HDF5 and metadata files.

    Args:
        input_dir (Path): Directory containing raw data files

    Returns:
        Tuple[h5py.File, Dict]: (hdf5_file, metadata_dict)
    """
    print("\n" + "="*80)
    print("LOADING MM-IMDb DATASET")
    print("="*80)

    # Load HDF5 file
    hdf5_path = input_dir / 'multimodal_imdb.hdf5'
    if not hdf5_path.exists():
        raise FileNotFoundError(f"HDF5 file not found: {hdf5_path}")

    print(f"\nLoading HDF5 file: {hdf5_path}")
    print(f"File size: {hdf5_path.stat().st_size / 1e9:.2f} GB")

    hdf5_file = h5py.File(hdf5_path, 'r')

    # Load metadata (contains vocabulary)
    metadata_path = input_dir / 'metadata.npy'
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    print(f"Loading metadata: {metadata_path}")
    metadata = np.load(metadata_path, allow_pickle=True).item()

    print(f"\nDataset info:")
    print(f"  Total samples: {len(hdf5_file['imdb_ids'])}")
    print(f"  Images shape: {hdf5_file['images'].shape}")
    print(f"  Genres shape: {hdf5_file['genres'].shape}")
    print(f"  Sequences shape: {hdf5_file['sequences'].shape}")
    print(f"  Vocabulary size: {metadata.get('vocab_size', 'N/A')}")

    return hdf5_file, metadata


def extract_genre_names(genres_matrix: np.ndarray) -> List[str]:
    """
    Extract genre names from the dataset.

    Since we don't have explicit genre names, we'll create indexed names.
    In practice, these correspond to IMDb genres.

    Args:
        genres_matrix (np.ndarray): Genre matrix (n_samples, n_genres)

    Returns:
        List[str]: List of genre names
    """
    # Common IMDb genres (23 total based on dataset)
    # These are the most common genres in IMDb
    genre_names = [
        'Drama', 'Comedy', 'Romance', 'Thriller', 'Action',
        'Horror', 'Documentary', 'Crime', 'Adventure', 'Sci-Fi',
        'Mystery', 'Fantasy', 'Family', 'Biography', 'War',
        'History', 'Music', 'Animation', 'Musical', 'Western',
        'Sport', 'Film-Noir', 'Short'
    ]

    # Ensure we have the right number
    n_genres = genres_matrix.shape[1]
    if len(genre_names) != n_genres:
        genre_names = [f'Genre_{i}' for i in range(n_genres)]

    return genre_names


def process_dataset(
    hdf5_file: h5py.File,
    metadata: Dict,
    max_samples: Optional[int] = None,
    min_sequence_length: int = 10
) -> List[Dict]:
    """
    Process dataset and create sample dictionaries.

    Args:
        hdf5_file (h5py.File): HDF5 file object
        metadata (Dict): Metadata dictionary with vocabulary
        max_samples (Optional[int]): Maximum samples to process
        min_sequence_length (int): Minimum sequence length

    Returns:
        List[Dict]: List of valid samples
    """
    print("\n" + "="*80)
    print("PROCESSING DATASET")
    print("="*80)

    # Load all data
    imdb_ids = hdf5_file['imdb_ids'][:]
    images = hdf5_file['images']
    sequences = hdf5_file['sequences']
    genres = hdf5_file['genres'][:]

    # Extract genre names
    genre_names = extract_genre_names(genres)

    # Limit samples if needed
    n_samples = len(imdb_ids) if max_samples is None else min(max_samples, len(imdb_ids))

    print(f"\nProcessing {n_samples} samples...")

    valid_samples = []

    for i in tqdm(range(n_samples), desc="Extracting"):
        try:
            # Get IMDb ID
            imdb_id = imdb_ids[i].decode('utf-8') if isinstance(imdb_ids[i], bytes) else str(imdb_ids[i])

            # Get sequence (tokenized text)
            sequence = sequences[i]
            if isinstance(sequence, np.ndarray):
                sequence = sequence.tolist()

            # Filter by sequence length
            if len(sequence) < min_sequence_length:
                continue

            # Get image
            image = images[i]  # Shape: (3, 256, 160)
            # Convert from CHW to HWC format
            image = np.transpose(image, (1, 2, 0))  # Shape: (256, 160, 3)

            # Ensure valid image range
            if image.max() > 255:
                image = image % 256  # Handle overflow
            image = np.clip(image, 0, 255).astype(np.uint8)

            # Get genres (multi-hot vector)
            genre_vector = genres[i]
            active_genres = [genre_names[j] for j, val in enumerate(genre_vector) if val == 1]

            # Skip samples with no genres
            if len(active_genres) == 0:
                continue

            valid_samples.append({
                'imdb_id': imdb_id,
                'sequence': sequence,
                'image': image,
                'genres': active_genres,
                'genre_vector': genre_vector.tolist()
            })

        except Exception as e:
            print(f"Error processing sample {i}: {e}")
            continue

    print(f"\nProcessing complete:")
    print(f"  Total processed: {n_samples}")
    print(f"  Valid samples: {len(valid_samples)}")
    print(f"  Filtered out: {n_samples - len(valid_samples)}")

    return valid_samples, genre_names


def split_data(
    samples: List[Dict],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Split data into train/val/test sets.

    Args:
        samples (List[Dict]): All valid samples
        train_ratio (float): Training set ratio
        val_ratio (float): Validation set ratio
        test_ratio (float): Test set ratio
        seed (int): Random seed

    Returns:
        Tuple[List[Dict], List[Dict], List[Dict]]: (train, val, test) splits
    """
    print("\n" + "="*80)
    print("SPLITTING DATA")
    print("="*80)

    # Validate ratios
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Ratios must sum to 1.0"

    # Shuffle with seed
    np.random.seed(seed)
    indices = np.random.permutation(len(samples))

    # Calculate split points
    n_train = int(len(samples) * train_ratio)
    n_val = int(len(samples) * val_ratio)

    # Split indices
    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    # Create splits
    train_samples = [samples[i] for i in train_idx]
    val_samples = [samples[i] for i in val_idx]
    test_samples = [samples[i] for i in test_idx]

    print(f"\nSplit statistics:")
    print(f"  Train: {len(train_samples)} samples ({len(train_samples)/len(samples)*100:.1f}%)")
    print(f"  Val:   {len(val_samples)} samples ({len(val_samples)/len(samples)*100:.1f}%)")
    print(f"  Test:  {len(test_samples)} samples ({len(test_samples)/len(samples)*100:.1f}%)")

    return train_samples, val_samples, test_samples


def save_samples(
    samples: List[Dict],
    output_dir: Path,
    split_name: str,
    skip_images: bool = False
) -> None:
    """
    Save samples to disk.

    Args:
        samples (List[Dict]): Samples to save
        output_dir (Path): Output directory
        split_name (str): Split name (train/val/test)
        skip_images (bool): Skip saving images
    """
    split_dir = output_dir / split_name
    split_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nSaving {split_name} samples to {split_dir}...")

    for sample in tqdm(samples, desc=f"Saving {split_name}"):
        # Create sample directory
        sample_dir = split_dir / sample['imdb_id']
        sample_dir.mkdir(exist_ok=True)

        # Save tokenized sequence
        sequence_path = sample_dir / 'plot.npy'
        np.save(sequence_path, np.array(sample['sequence']))

        # Save poster image
        if not skip_images and sample['image'] is not None:
            poster_path = sample_dir / 'poster.jpg'
            try:
                img = Image.fromarray(sample['image'])
                img.save(poster_path, 'JPEG', quality=95)
            except Exception as e:
                print(f"Warning: Could not save poster for {sample['imdb_id']}: {e}")

        # Save metadata
        metadata = {
            'imdb_id': sample['imdb_id'],
            'genres': sample['genres'],
            'genre_vector': sample['genre_vector']
        }
        metadata_path = sample_dir / 'metadata.json'
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)


def compute_statistics(
    train_samples: List[Dict],
    val_samples: List[Dict],
    test_samples: List[Dict],
    genre_names: List[str],
    output_dir: Path
) -> Dict:
    """
    Compute and save dataset statistics.

    Args:
        train_samples (List[Dict]): Training samples
        val_samples (List[Dict]): Validation samples
        test_samples (List[Dict]): Test samples
        genre_names (List[str]): List of genre names
        output_dir (Path): Output directory

    Returns:
        Dict: Statistics dictionary
    """
    print("\n" + "="*80)
    print("COMPUTING STATISTICS")
    print("="*80)

    all_samples = train_samples + val_samples + test_samples

    # Genre distribution
    genre_counts = Counter()
    for sample in all_samples:
        genre_counts.update(sample['genres'])

    # Sequence statistics
    seq_lengths = [len(s['sequence']) for s in all_samples]

    # Image statistics (all have images in this dataset)
    has_images = len(all_samples)

    stats = {
        'total_samples': len(all_samples),
        'train_samples': len(train_samples),
        'val_samples': len(val_samples),
        'test_samples': len(test_samples),
        'genre_distribution': dict(genre_counts),
        'num_genres': len(genre_names),
        'genre_names': genre_names,
        'samples_with_images': has_images,
        'image_coverage': 1.0,  # All samples have images
        'sequence_statistics': {
            'mean_length': float(np.mean(seq_lengths)),
            'std_length': float(np.std(seq_lengths)),
            'min_length': int(np.min(seq_lengths)),
            'max_length': int(np.max(seq_lengths)),
            'median_length': float(np.median(seq_lengths))
        }
    }

    # Print statistics
    print(f"\nDataset Statistics:")
    print(f"  Total samples: {stats['total_samples']}")
    print(f"  Train/Val/Test: {stats['train_samples']}/{stats['val_samples']}/{stats['test_samples']}")
    print(f"  Unique genres: {stats['num_genres']}")
    print(f"  All samples have images: {stats['image_coverage'] == 1.0}")
    print(f"\nSequence Statistics:")
    print(f"  Mean length: {stats['sequence_statistics']['mean_length']:.0f} tokens")
    print(f"  Std length: {stats['sequence_statistics']['std_length']:.0f} tokens")
    print(f"  Min/Max length: {stats['sequence_statistics']['min_length']}/{stats['sequence_statistics']['max_length']} tokens")

    print(f"\nTop 10 Genres:")
    sorted_genres = sorted(genre_counts.items(), key=lambda x: x[1], reverse=True)
    for genre, count in sorted_genres[:10]:
        print(f"  {genre:15s}: {count:5d} samples ({count/len(all_samples)*100:.1f}%)")

    # Save statistics
    stats_path = output_dir / 'dataset_statistics.json'
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"\nStatistics saved to {stats_path}")

    # Save genre mapping
    genre_to_idx = {genre: idx for idx, genre in enumerate(genre_names)}
    mapping_path = output_dir / 'genre_mapping.json'
    with open(mapping_path, 'w') as f:
        json.dump(genre_to_idx, f, indent=2)
    print(f"Genre mapping saved to {mapping_path}")

    return stats


def main():
    """Main preprocessing function."""
    args = parse_args()

    print("\n" + "="*80)
    print("MM-IMDb DATASET PREPROCESSING")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Input directory: {args.input_dir}")
    print(f"  Output directory: {args.output_dir}")
    print(f"  Train/Val/Test ratio: {args.train_ratio}/{args.val_ratio}/{args.test_ratio}")
    print(f"  Random seed: {args.seed}")
    print(f"  Min sequence length: {args.min_sequence_length} tokens")
    print(f"  Max samples: {args.max_samples if args.max_samples else 'All'}")
    print(f"  Skip images: {args.skip_images}")

    # Setup paths
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    hdf5_file, metadata = load_mmimdb_data(input_dir)

    # Save vocabulary
    vocab_path = output_dir / 'vocab.json'
    vocab_data = {
        'vocab_size': int(metadata.get('vocab_size', 0)),
        'word_to_ix': {k: int(v) for k, v in metadata.get('word_to_ix', {}).items()} if 'word_to_ix' in metadata else {},
        'ix_to_word': {int(k): v for k, v in metadata.get('ix_to_word', {}).items()} if 'ix_to_word' in metadata else {}
    }
    with open(vocab_path, 'w') as f:
        json.dump(vocab_data, f, indent=2)
    print(f"Vocabulary saved to {vocab_path}")

    # Process dataset
    valid_samples, genre_names = process_dataset(
        hdf5_file,
        metadata,
        args.max_samples,
        args.min_sequence_length
    )

    if len(valid_samples) == 0:
        print("ERROR: No valid samples found!")
        hdf5_file.close()
        return

    # Split data
    train_samples, val_samples, test_samples = split_data(
        valid_samples,
        args.train_ratio,
        args.val_ratio,
        args.test_ratio,
        args.seed
    )

    # Save samples
    print("\n" + "="*80)
    print("SAVING PROCESSED DATA")
    print("="*80)

    save_samples(train_samples, output_dir, 'train', args.skip_images)
    save_samples(val_samples, output_dir, 'val', args.skip_images)
    save_samples(test_samples, output_dir, 'test', args.skip_images)

    # Compute statistics
    stats = compute_statistics(train_samples, val_samples, test_samples, genre_names, output_dir)

    # Close HDF5 file
    hdf5_file.close()

    print("\n" + "="*80)
    print("PREPROCESSING COMPLETE")
    print("="*80)
    print(f"\nProcessed dataset saved to: {output_dir}")
    print(f"Total samples: {stats['total_samples']}")
    print(f"  Train: {stats['train_samples']}")
    print(f"  Val: {stats['val_samples']}")
    print(f"  Test: {stats['test_samples']}")
    print(f"\nGenres: {stats['num_genres']}")
    print("\nNext steps:")
    print("  1. Update config.yaml if needed (num_genres = 23)")
    print("  2. Run training: python scripts/train.py --config config.yaml --model early_fusion")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
