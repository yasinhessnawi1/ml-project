"""
Compute class weights for handling class imbalance in MM-IMDb dataset.

This script computes pos_weight for BCEWithLogitsLoss based on genre frequencies.
"""

import json
import torch
from pathlib import Path
import numpy as np


def compute_pos_weights(stats_path, method='inverse_freq', smooth=1.0):
    """
    Compute positive class weights for multi-label classification.

    Args:
        stats_path: Path to dataset_statistics.json
        method: 'inverse_freq' or 'effective_samples'
        smooth: Smoothing factor (default 1.0)

    Returns:
        pos_weight: Tensor of shape (num_classes,)
    """
    # Load statistics
    with open(stats_path, 'r') as f:
        stats = json.load(f)

    genre_dist = stats['genre_distribution']
    total_samples = stats['total_samples']
    genre_names = stats['genre_names']

    print(f"\nDataset Statistics:")
    print(f"Total samples: {total_samples}")
    print(f"Number of genres: {len(genre_names)}")

    # Compute positive counts for each genre
    pos_counts = []
    for genre in genre_names:
        count = genre_dist[genre]
        pos_counts.append(count)

    pos_counts = np.array(pos_counts)
    neg_counts = total_samples - pos_counts

    print(f"\nGenre Distribution (sorted by frequency):")
    print(f"{'Genre':<15} {'Positive':<10} {'Negative':<10} {'Pos %':<10} {'Weight':<10}")
    print("=" * 70)

    if method == 'inverse_freq':
        # pos_weight = neg_count / pos_count
        # This makes the loss for rare classes higher
        pos_weights = neg_counts / (pos_counts + smooth)

    elif method == 'effective_samples':
        # Effective number of samples (from Class-Balanced Loss paper)
        beta = 0.9999
        effective_num = 1.0 - np.power(beta, pos_counts)
        pos_weights = (1.0 - beta) / (effective_num + 1e-8)
        # Normalize
        pos_weights = pos_weights / pos_weights.sum() * len(pos_weights)

    else:
        raise ValueError(f"Unknown method: {method}")

    # Print sorted by frequency
    indices = np.argsort(pos_counts)[::-1]
    for idx in indices:
        genre = genre_names[idx]
        pos = pos_counts[idx]
        neg = neg_counts[idx]
        pct = (pos / total_samples) * 100
        weight = pos_weights[idx]
        print(f"{genre:<15} {pos:<10} {neg:<10} {pct:<10.2f} {weight:<10.4f}")

    # Convert to tensor
    pos_weights_tensor = torch.FloatTensor(pos_weights)

    print(f"\nWeight Statistics:")
    print(f"Min weight: {pos_weights.min():.4f} (most common class)")
    print(f"Max weight: {pos_weights.max():.4f} (rarest class)")
    print(f"Mean weight: {pos_weights.mean():.4f}")
    print(f"Ratio (max/min): {pos_weights.max() / pos_weights.min():.2f}x")

    return pos_weights_tensor, genre_names


def save_weights(weights, genre_names, output_path):
    """Save weights to JSON file."""
    weights_dict = {
        'pos_weights': weights.tolist(),
        'genre_names': genre_names,
        'num_classes': len(genre_names)
    }

    with open(output_path, 'w') as f:
        json.dump(weights_dict, f, indent=2)

    print(f"\nWeights saved to: {output_path}")


def main():
    """Main function."""
    project_root = Path(__file__).parent.parent
    stats_path = project_root / 'data' / 'processed' / 'dataset_statistics.json'
    output_path = project_root / 'data' / 'processed' / 'class_weights.json'

    print("=" * 70)
    print("COMPUTING CLASS WEIGHTS FOR MM-IMDb DATASET")
    print("=" * 70)

    # Compute weights using inverse frequency
    print("\nMethod: Inverse Frequency (neg_count / pos_count)")
    weights, genre_names = compute_pos_weights(stats_path, method='inverse_freq', smooth=1.0)

    # Save weights
    save_weights(weights, genre_names, output_path)

    print("\n" + "=" * 70)
    print("HOW TO USE THESE WEIGHTS:")
    print("=" * 70)
    print("\nIn config.yaml, change:")
    print("  loss:")
    print("    type: 'weighted_bce'")
    print("    compute_weights: true  # Load from class_weights.json")
    print("\nOr use focal loss (automatically handles imbalance):")
    print("  loss:")
    print("    type: 'focal'")
    print("    alpha: 0.25")
    print("    gamma: 2.0")
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
