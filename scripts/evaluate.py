#!/usr/bin/env python3
"""
Evaluation script for trained models.

This script evaluates a trained model and generates comprehensive reports:
1. Load trained model from checkpoint
2. Evaluate on test set
3. Compute all metrics (F1, precision, recall, ROC-AUC, etc.)
4. Generate visualizations (confusion matrices, ROC curves, PR curves)
5. Save detailed evaluation report

Usage:
    python scripts/evaluate.py --checkpoint checkpoints/early_fusion/best.pth --config config.yaml
    python scripts/evaluate.py --checkpoint checkpoints/bert_text/best.pth --split test
    python scripts/evaluate.py --help

Arguments:
    --checkpoint: Path to model checkpoint
    --config: Path to configuration file
    --split: Dataset split to evaluate on (val/test, default: test)
    --output-dir: Directory to save evaluation results (default: evaluation/)
    --threshold: Classification threshold (default: 0.5)
    --device: Device to use (cuda/cpu/mps, default: auto-detect)
"""

import argparse
import sys
from pathlib import Path
import json
import torch
from torch.utils.data import DataLoader
from datetime import datetime
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.config import load_config, set_seed, setup_device
from src.data.dataset import MultimodalDataset, TextOnlyDataset, ImageOnlyDataset
from src.data.preprocessing import (
    LSTMTokenizer,
    get_bert_tokenizer,
    get_image_transforms
)
from src.models.text_models import create_text_model
from src.models.vision_models import create_vision_model
from src.models.fusion_models import (
    EarlyFusionModel,
    LateFusionModel,
    AttentionFusionModel
)
from src.evaluation.metrics import (
    compute_all_metrics,
    get_classification_report,
    compute_confusion_matrices
)
from src.utils.visualization import (
    plot_all_confusion_matrices,
    plot_roc_curves,
    plot_precision_recall_curves,
    plot_per_class_metrics,
    plot_prediction_comparison
)



def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Evaluate trained genre classification model',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to model checkpoint'
    )

    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to configuration file'
    )

    parser.add_argument(
        '--split',
        type=str,
        default='test',
        choices=['val', 'test'],
        help='Dataset split to evaluate on'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default='evaluation',
        help='Directory to save evaluation results'
    )

    parser.add_argument(
        '--threshold',
        type=float,
        default=0.5,
        help='Classification threshold'
    )

    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help='Device to use (cuda/cpu/mps). Auto-detect if not specified.'
    )

    parser.add_argument(
        '--batch-size',
        type=int,
        default=None,
        help='Batch size for evaluation. Uses config value if not specified.'
    )

    return parser.parse_args()


@torch.no_grad()
def evaluate_model(model, dataloader, device):
    """
    Evaluate model and collect predictions.

    Args:
        model (nn.Module): Model to evaluate
        dataloader (DataLoader): Data loader
        device (torch.device): Device

    Returns:
        tuple: (predictions, targets) as torch.Tensors
    """
    model.eval()
    all_predictions = []
    all_targets = []

    print(f"\nEvaluating on {len(dataloader)} batches...")

    for batch_idx, batch in enumerate(dataloader):
        # Move to device
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()}

        # Forward pass
        if 'text' in batch and 'image' in batch:
            # Multimodal
            if 'attention_mask' in batch:
                outputs = model(
                    text_input={'input_ids': batch['text'], 'attention_mask': batch['attention_mask']},
                    image_input=batch['image']
                )
            else:
                outputs = model(text_input=batch['text'], image_input=batch['image'])
        elif 'text' in batch:
            # Text-only
            if 'attention_mask' in batch:
                outputs = model(batch['text'], batch['attention_mask'])
            else:
                outputs = model(batch['text'])
        elif 'image' in batch:
            # Vision-only
            outputs = model(batch['image'])

        # Convert logits to probabilities
        predictions = torch.sigmoid(outputs)

        all_predictions.append(predictions.cpu())
        all_targets.append(batch['labels'].cpu())

        if (batch_idx + 1) % 10 == 0:
            print(f"  Processed {batch_idx + 1}/{len(dataloader)} batches")

    # Concatenate all batches
    all_predictions = torch.cat(all_predictions, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    return all_predictions, all_targets


def main():
    """Main evaluation function."""
    # Parse arguments
    args = parse_args()

    print("\n" + "="*80)
    print("MODEL EVALUATION")
    print("="*80)

    # Load configuration
    print(f"\nLoading configuration from {args.config}...")
    config = load_config(args.config)

    # Override batch size if provided
    if args.batch_size is not None:
        config['training']['batch_size'] = args.batch_size

    # Setup device
    device = setup_device(args.device)
    print(f"Using device: {device}")

    # Load checkpoint
    print(f"\nLoading checkpoint from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)

    # Infer model type from checkpoint path
    checkpoint_path = Path(args.checkpoint)
    model_type = checkpoint_path.parent.name

    print(f"Model type: {model_type}")
    print(f"Checkpoint epoch: {checkpoint.get('epoch', 'unknown')}")
    print(f"Best metric: {checkpoint.get('best_metric', 'unknown')}")

    # Use optimal threshold if not specified by user
    if args.threshold == 0.5:  # Default value, not user-specified
        optimal_thresholds = {
            'bert_text': 0.28,  # From threshold tuning (F1: 59.02%)
            'lstm_text': 0.34,  # From threshold tuning (F1: 45.53%)
        }
        if model_type in optimal_thresholds:
            args.threshold = optimal_thresholds[model_type]
            print(f"Using optimal threshold for {model_type}: {args.threshold}")
        else:
            print(f"Using default threshold: {args.threshold}")
    else:
        print(f"Using user-specified threshold: {args.threshold}")

    # Create output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path(args.output_dir) / model_type / f"{args.split}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Prepare data (similar to train.py but only for the specified split)
    print("\n" + "="*80)
    print("PREPARING DATA")
    print("="*80)

    data_dir = Path(config['dataset']['data_dir'])
    batch_size = config['training']['batch_size']

    # Determine modality
    is_multimodal = model_type in ['early_fusion', 'late_fusion', 'attention_fusion']
    is_text_only = model_type in ['lstm_text', 'bert_text']
    is_vision_only = model_type in ['resnet_vision', 'cnn_vision']

    # Prepare preprocessing
    vocab_or_tokenizer = None
    if is_text_only or is_multimodal:
        if 'bert' in model_type:
            max_length = config['preprocessing']['text'].get('max_length_bert', 512)
            tokenizer = get_bert_tokenizer(
                model_name=config['model_distilbert'].get('model_name', 'distilbert-base-uncased'),
                max_length=max_length
            )
            vocab_or_tokenizer = tokenizer
            text_tokenizer_fn = tokenizer
        else:
            # LSTM - load actual vocabulary from processed data
            vocab_path = data_dir / 'vocab.json'
            with open(vocab_path, 'r') as f:
                vocab_data = json.load(f)
            vocab = vocab_data.get('word_to_ix', vocab_data)
            print(f"Loaded vocabulary with {len(vocab)} words for LSTM tokenizer")

            tokenizer = LSTMTokenizer(
                vocab=vocab,
                max_length=config['preprocessing']['text']['max_length_lstm']
            )
            vocab_or_tokenizer = tokenizer
            text_tokenizer_fn = tokenizer

    if is_vision_only or is_multimodal:
        transform = get_image_transforms(
            split='val',  # Use validation transforms (no augmentation)
            image_size=config['preprocessing']['image']['target_size']
        )
    else:
        transform = None

    # Load genre mapping from processed data
    genre_mapping_path = data_dir / 'genre_mapping.json'
    with open(genre_mapping_path, 'r') as f:
        genre_to_idx = json.load(f)
    print(f"Loaded {len(genre_to_idx)} genres from genre_mapping.json")

    # Create dataset
    print(f"\nCreating {args.split} dataset...")

    if is_multimodal:
        dataset = MultimodalDataset(
            data_dir=data_dir,
            split=args.split,
            text_tokenizer=text_tokenizer_fn,
            image_transform=transform,
            genre_to_idx=genre_to_idx
        )
    elif is_text_only:
        dataset = TextOnlyDataset(
            data_dir=data_dir,
            split=args.split,
            text_tokenizer=text_tokenizer_fn,
            genre_to_idx=genre_to_idx
        )
    else:  # vision_only
        dataset = ImageOnlyDataset(
            data_dir=data_dir,
            split=args.split,
            image_transform=transform,
            genre_to_idx=genre_to_idx
        )

    print(f"Dataset size: {len(dataset)}")

    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config['training'].get('num_workers', 4),
        pin_memory=True if device.type == 'cuda' else False
    )

    # Recreate model (same logic as train.py)
    print("\n" + "="*80)
    print("CREATING MODEL")
    print("="*80)

    num_classes = config['dataset']['num_genres']

    # Create model based on type
    if model_type == 'lstm_text':
        model_config = config['model_lstm']
        model_config['type'] = 'lstm_text'
        model_config['num_classes'] = num_classes
        # Get vocab size from checkpoint - infer from embedding weights if not stored
        if 'vocab_size' in checkpoint:
            vocab_size = checkpoint['vocab_size']
        elif 'embedding.weight' in checkpoint['model_state_dict']:
            vocab_size = checkpoint['model_state_dict']['embedding.weight'].shape[0]
        else:
            vocab_size = config['preprocessing']['text'].get('vocab_size', 70000)
        print(f"LSTM vocab size: {vocab_size}")
        model = create_text_model(model_config, vocab_size=vocab_size)

    elif model_type == 'bert_text':
        model_config = config['model_distilbert']
        model_config['type'] = 'distilbert_text'
        model_config['num_classes'] = num_classes
        model = create_text_model(model_config)

    elif model_type == 'resnet_vision':
        model_config = config['model_resnet']
        model_config['type'] = 'resnet'
        model_config['num_classes'] = num_classes
        model = create_vision_model(model_config)

    elif model_type == 'cnn_vision':
        model_config = config['model_custom_cnn']
        model_config['type'] = 'custom_cnn'
        model_config['num_classes'] = num_classes
        model = create_vision_model(model_config)

    elif model_type in ['early_fusion', 'late_fusion', 'attention_fusion']:
        # Create component models
        text_config = config['model_distilbert']
        text_config['num_classes'] = num_classes
        text_model = create_text_model(text_config, vocab_size=10000)

        vision_config = config['model_resnet']
        vision_config['num_classes'] = num_classes
        vision_model = create_vision_model(vision_config)

        fusion_config = config['model_early_fusion']

        if model_type == 'early_fusion':
            model = EarlyFusionModel(
                text_model, vision_model, num_classes,
                **{k: v for k, v in fusion_config.items() if k != 'strategy'}
            )
        elif model_type == 'late_fusion':
            model = LateFusionModel(text_model, vision_model, num_classes,
                                   fusion_strategy=fusion_config.get('strategy', 'average'))
        else:
            model = AttentionFusionModel(
                text_model, vision_model, num_classes,
                **{k: v for k, v in fusion_config.items() if k != 'strategy'}
            )

    # Load model weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)

    print(f"Model loaded successfully")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Evaluate model
    print("\n" + "="*80)
    print("EVALUATION")
    print("="*80)

    predictions, targets = evaluate_model(model, dataloader, device)

    print(f"\nPredictions shape: {predictions.shape}")
    print(f"Targets shape: {targets.shape}")

    # Create class names list from genre_to_idx mapping
    genre_names = [name for name, _ in sorted(genre_to_idx.items(), key=lambda x: x[1])]

    # Compute all metrics
    print("\nComputing comprehensive metrics...")
    metrics = compute_all_metrics(
        predictions,
        targets,
        class_names=genre_names,
        threshold=args.threshold,
        verbose=True
    )

    # Generate classification report
    print("\n" + "="*80)
    print("CLASSIFICATION REPORT")
    print("="*80)
    report = get_classification_report(
        predictions,
        targets,
        class_names=genre_names,
        threshold=args.threshold
    )
    print(report)

    # Save metrics to JSON
    metrics_to_save = {
        'threshold': args.threshold,
        'split': args.split,
        'model_type': model_type,
        'checkpoint_path': str(args.checkpoint),
        'f1_macro': float(metrics['f1_macro']),
        'f1_micro': float(metrics['f1_micro']),
        'f1_weighted': float(metrics['f1_weighted']),
        'precision_macro': float(metrics['precision_macro']),
        'recall_macro': float(metrics['recall_macro']),
        'roc_auc_macro': float(metrics['roc_auc_macro']),
        'hamming_loss': float(metrics['hamming_loss']),
        'subset_accuracy': float(metrics['subset_accuracy']),
        'per_class_metrics': metrics['per_class_metrics']
    }

    metrics_path = output_dir / 'metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(metrics_to_save, f, indent=2)
    print(f"\nMetrics saved to {metrics_path}")

    # Save classification report
    report_path = output_dir / 'classification_report.txt'
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"Classification report saved to {report_path}")

    # Generate visualizations
    print("\n" + "="*80)
    print("GENERATING VISUALIZATIONS")
    print("="*80)

    # Confusion matrices
    print("\n1. Plotting confusion matrices...")
    plot_all_confusion_matrices(
        metrics['confusion_matrices'],
        genre_names,
        normalize=True,
        save_path=output_dir / 'confusion_matrices.png',
        show=False
    )

    # ROC curves
    print("2. Plotting ROC curves...")
    plot_roc_curves(
        predictions,
        targets,
        genre_names,
        save_path=output_dir / 'roc_curves.png',
        show=False
    )

    # Precision-Recall curves
    print("3. Plotting Precision-Recall curves...")
    plot_precision_recall_curves(
        predictions,
        targets,
        genre_names,
        save_path=output_dir / 'pr_curves.png',
        show=False
    )

    # Per-class F1 scores
    print("4. Plotting per-class F1 scores...")
    plot_per_class_metrics(
        metrics['per_class_metrics'],
        metric_name='f1',
        save_path=output_dir / 'per_class_f1.png',
        show=False
    )

    # Sample prediction comparisons
    print("5. Plotting sample prediction comparisons...")
    for i in range(min(5, len(predictions))):
        plot_prediction_comparison(
            predictions,
            targets,
            genre_names,
            sample_idx=i,
            threshold=args.threshold,
            save_path=output_dir / f'prediction_sample_{i}.png',
            show=False
        )

    print("\n" + "="*80)
    print("EVALUATION COMPLETED")
    print("="*80)
    print(f"F1-Score (Macro): {metrics['f1_macro']:.4f}")
    print(f"ROC-AUC (Macro): {metrics['roc_auc_macro']:.4f}")
    print(f"Results saved to: {output_dir}")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
