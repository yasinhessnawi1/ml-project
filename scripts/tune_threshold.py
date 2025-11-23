"""
Threshold tuning script for multi-label classification.

This script loads a trained model and finds the optimal threshold
that maximizes F1-macro score on the validation set.
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data.dataset import TextOnlyDataset, ImageOnlyDataset
from src.data.preprocessing import get_bert_tokenizer, get_image_transforms
from src.evaluation.metrics import compute_f1_scores
from src.models.text_models import DistilBERTTextModel
from src.models.vision_models import ResNetVisionModel, CustomCNNModel
from src.utils.config import load_config


def load_model_and_data(checkpoint_path, config_path, device):
    """Load trained model and validation data."""
    # Load config
    config = load_config(config_path)

    # Detect model type from checkpoint path
    checkpoint_path_obj = Path(checkpoint_path)
    model_type = checkpoint_path_obj.parent.name
    print(f"Detected model type: {model_type}")

    # Load genre mapping
    data_dir = Path(config['dataset']['data_dir'])
    genre_mapping_path = data_dir / 'genre_mapping.json'
    with open(genre_mapping_path, 'r') as f:
        genre_to_idx = json.load(f)  # Direct mapping: {"Drama": 0, "Comedy": 1, ...}
    num_genres = len(genre_to_idx)

    # Load checkpoint first to get model info
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Determine if this is a vision model or text model
    is_vision_model = 'resnet' in model_type or 'cnn' in model_type

    if is_vision_model:
        # ===== VISION MODELS =====
        print("Loading vision model...")

        # Get image transforms (validation - no augmentation)
        transform = get_image_transforms(
            split='val',
            image_size=config['preprocessing']['image']['target_size']
        )

        # Create appropriate vision model
        if 'resnet' in model_type:
            print("Creating ResNet vision model...")
            model_config = config['model_resnet']
            model = ResNetVisionModel(
                num_classes=num_genres,
                architecture=model_config.get('architecture', 'resnet18'),
                pretrained=model_config.get('pretrained', True),
                fine_tune_strategy=model_config.get('fine_tune_strategy', 'unfreeze_layer3_layer4'),
                classifier_hidden_dim=model_config.get('classifier_hidden_dim', 256),
                dropout=model_config.get('dropout', 0.5)
            )
        elif 'cnn' in model_type:
            print("Creating Custom CNN vision model...")
            model_config = config['model_custom_cnn']
            model = CustomCNNModel(
                num_classes=num_genres,
                channels=model_config.get('channels', [64, 128, 256, 512]),
                kernel_sizes=model_config.get('kernel_sizes', [7, 3, 3, 3]),
                dropout=model_config.get('dropout', 0.5)
            )
        else:
            raise ValueError(f"Unknown vision model type: {model_type}")

        # Create validation dataset (vision only)
        val_dataset = ImageOnlyDataset(
            data_dir=data_dir,
            split='val',
            image_transform=transform,
            genre_to_idx=genre_to_idx
        )

    else:
        # ===== TEXT MODELS =====
        print("Loading text model...")

        if 'bert' in model_type:
            # BERT model
            print("Creating BERT text model...")
            max_length = config['preprocessing']['text'].get('max_length_bert', 512)
            tokenizer = get_bert_tokenizer(
                model_name=config['model_distilbert'].get('model_name', 'distilbert-base-uncased'),
                max_length=max_length
            )

            model = DistilBERTTextModel(
                num_classes=num_genres,
                model_name=config['model_distilbert'].get('model_name', 'distilbert-base-uncased'),
                classifier_hidden_dim=config['model_distilbert'].get('classifier_hidden_dim', 256),
                dropout=config['model_distilbert'].get('dropout', 0.3),
                fine_tune_all=config['model_distilbert'].get('fine_tune_all', True)
            )
        else:
            # LSTM model
            print("Creating LSTM text model...")
            from src.data.preprocessing import LSTMTokenizer
            from src.models.text_models import LSTMTextModel

            # Get vocab size from checkpoint - infer from embedding weights if not stored
            if 'vocab_size' in checkpoint:
                vocab_size = checkpoint['vocab_size']
            elif 'embedding.weight' in checkpoint['model_state_dict']:
                vocab_size = checkpoint['model_state_dict']['embedding.weight'].shape[0]
            else:
                vocab_size = config['preprocessing']['text'].get('vocab_size', 70000)
            print(f"LSTM vocab size: {vocab_size}")

            # Load actual vocabulary from processed data
            vocab_path = data_dir / 'vocab.json'
            print(f"Loading vocabulary from: {vocab_path}")
            with open(vocab_path, 'r') as f:
                vocab_data = json.load(f)

            # Extract word_to_ix mapping (the actual vocabulary)
            vocab = vocab_data.get('word_to_ix', vocab_data)
            print(f"Loaded vocabulary with {len(vocab)} words")

            max_length = config['preprocessing']['text'].get('max_length_lstm', 128)
            tokenizer = LSTMTokenizer(
                vocab=vocab,
                max_length=max_length
            )

            # Create LSTM model
            model_config = config['model_lstm']
            model = LSTMTextModel(
                vocab_size=vocab_size,
                embedding_dim=model_config.get('embedding_dim', 300),
                hidden_dim=model_config.get('hidden_dim', 256),
                num_layers=model_config.get('num_layers', 2),
                num_classes=num_genres,
                dropout=model_config.get('dropout', 0.3),
                bidirectional=model_config.get('bidirectional', True),
                use_attention=model_config.get('attention', True)
            )

        # Create validation dataset (text only)
        val_dataset = TextOnlyDataset(
            data_dir=data_dir,
            split='val',
            text_tokenizer=tokenizer,
            genre_to_idx=genre_to_idx
        )

    # Create dataloader (same for both vision and text)
    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=0
    )

    # Load model weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    return model, val_loader, num_genres


@torch.no_grad()
def get_predictions(model, data_loader, device):
    """Get model predictions and ground truth labels.

    Works for both text and vision models.
    """
    all_predictions = []
    all_targets = []

    for batch in data_loader:
        labels = batch['labels'].to(device)

        # Check if this is a text model or vision model
        if 'text' in batch:
            # Text model
            text = batch['text'].to(device)
            attention_mask = batch.get('attention_mask')

            # Forward pass
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)
                outputs = model(text, attention_mask)
            else:
                outputs = model(text)

        elif 'image' in batch:
            # Vision model
            images = batch['image'].to(device)
            outputs = model(images)

        else:
            raise ValueError("Batch must contain either 'text' or 'image' key")

        # Apply sigmoid to get probabilities
        predictions = torch.sigmoid(outputs)

        all_predictions.append(predictions.cpu())
        all_targets.append(labels.cpu())

    all_predictions = torch.cat(all_predictions, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    return all_predictions, all_targets


def find_optimal_threshold(predictions, targets, thresholds=None):
    """Find optimal threshold that maximizes F1-macro."""
    if thresholds is None:
        thresholds = np.arange(0.1, 0.9, 0.05)

    best_threshold = 0.5
    best_f1 = 0.0
    results = []

    for threshold in thresholds:
        scores = compute_f1_scores(predictions, targets, threshold=threshold, average='macro')
        f1_macro = scores['f1_macro']

        results.append({
            'threshold': float(threshold),
            'f1_macro': float(f1_macro)
        })

        if f1_macro > best_f1:
            best_f1 = f1_macro
            best_threshold = threshold

    return best_threshold, best_f1, results


def plot_threshold_curve(results, save_path):
    """Plot F1 vs threshold curve."""
    thresholds = [r['threshold'] for r in results]
    f1_scores = [r['f1_macro'] for r in results]

    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, f1_scores, 'b-', linewidth=2, marker='o', markersize=4)
    plt.xlabel('Threshold', fontsize=12)
    plt.ylabel('F1-macro Score', fontsize=12)
    plt.title('Threshold Tuning: F1-macro vs Classification Threshold', fontsize=14)
    plt.grid(True, alpha=0.3)

    # Mark best threshold
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx]
    best_f1 = f1_scores[best_idx]
    plt.plot(best_threshold, best_f1, 'r*', markersize=15,
             label=f'Best: {best_threshold:.2f} (F1={best_f1:.4f})')

    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Threshold curve saved to: {save_path}")


def main():
    parser = argparse.ArgumentParser(description='Tune classification threshold')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to config file')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda/cpu)')
    parser.add_argument('--output', type=str, default='threshold_tuning_results.json',
                       help='Output file for results')

    args = parser.parse_args()

    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load model and data
    print(f"\nLoading model from: {args.checkpoint}")
    print(f"Loading config from: {args.config}")
    model, val_loader, num_genres = load_model_and_data(args.checkpoint, args.config, device)
    print(f"Model loaded successfully ({num_genres} genres)")

    # Get predictions
    print("\nComputing predictions on validation set...")
    predictions, targets = get_predictions(model, val_loader, device)
    print(f"Predictions shape: {predictions.shape}")
    print(f"Targets shape: {targets.shape}")

    # Find optimal threshold
    print("\nSearching for optimal threshold...")
    thresholds = np.arange(0.1, 0.9, 0.02)
    best_threshold, best_f1, results = find_optimal_threshold(predictions, targets, thresholds)

    print(f"\n{'='*80}")
    print("THRESHOLD TUNING RESULTS")
    print(f"{'='*80}")
    print(f"Optimal threshold: {best_threshold:.3f}")
    print(f"F1-macro at optimal threshold: {best_f1:.4f}")
    print(f"Default threshold (0.5): {[r['f1_macro'] for r in results if abs(r['threshold'] - 0.5) < 0.01][0]:.4f}")
    print(f"Improvement: {(best_f1 - [r['f1_macro'] for r in results if abs(r['threshold'] - 0.5) < 0.01][0]) * 100:.2f}%")
    print(f"{'='*80}\n")

    # Save results
    output_data = {
        'best_threshold': float(best_threshold),
        'best_f1_macro': float(best_f1),
        'default_f1_macro': float([r['f1_macro'] for r in results if abs(r['threshold'] - 0.5) < 0.01][0]),
        'all_results': results
    }

    output_path = Path(args.output)
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    print(f"Results saved to: {output_path}")

    # Plot threshold curve
    plot_path = output_path.parent / 'threshold_curve.png'
    plot_threshold_curve(results, plot_path)


if __name__ == '__main__':
    main()
