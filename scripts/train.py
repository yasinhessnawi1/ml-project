#!/usr/bin/env python3
"""
Main training script for multimodal genre classification.

This script handles end-to-end training:
1. Load configuration
2. Prepare data (datasets and dataloaders)
3. Initialize model
4. Setup training infrastructure (optimizer, scheduler, loss)
5. Train model
6. Save results and visualizations

Usage:
    python scripts/train.py --config config.yaml --model early_fusion
    python scripts/train.py --config config.yaml --model late_fusion --resume checkpoints/last.pth
    python scripts/train.py --help

Arguments:
    --config: Path to configuration file (default: config.yaml)
    --model: Model type (early_fusion, late_fusion, attention_fusion, lstm_text, bert_text,
             resnet_vision, cnn_vision)
    --resume: Path to checkpoint to resume from (optional)
    --output-dir: Directory to save results (default: results/)
    --checkpoint-dir: Directory to save checkpoints (default: checkpoints/)
    --device: Device to train on (cuda/cpu/mps, default: auto-detect)
    --seed: Random seed (default: from config or 42)
"""

import argparse
import sys
from pathlib import Path
import json
import torch
from torch.utils.data import DataLoader
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.config import load_config, set_seed, setup_device
from src.data.dataset import MultimodalDataset, TextOnlyDataset, ImageOnlyDataset
from src.data.preprocessing import (
    LSTMTokenizer,
    get_bert_tokenizer,
    get_image_transforms,
    build_vocab
)
from src.models.text_models import create_text_model
from src.models.vision_models import create_vision_model
from src.models.fusion_models import (
    EarlyFusionModel,
    LateFusionModel,
    AttentionFusionModel
)
from src.training.losses import get_loss_function, compute_class_weights
from src.training.trainer import Trainer, create_optimizer, create_scheduler
from src.evaluation.metrics import compute_all_metrics
from src.utils.visualization import (
    plot_training_history,
    plot_class_distribution,
    plot_per_class_metrics
)


# MM-IMDb genre names will be loaded from genre_mapping.json
# This ensures consistency with the preprocessing output


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train multimodal genre classification model',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to configuration file'
    )

    parser.add_argument(
        '--model',
        type=str,
        required=True,
        choices=[
            'early_fusion', 'late_fusion', 'attention_fusion',
            'lstm_text', 'bert_text',
            'resnet_vision', 'cnn_vision'
        ],
        help='Model type to train'
    )

    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Path to checkpoint to resume training from'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default='results',
        help='Directory to save results'
    )

    parser.add_argument(
        '--checkpoint-dir',
        type=str,
        default='checkpoints',
        help='Directory to save model checkpoints'
    )

    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help='Device to train on (cuda/cpu/mps). Auto-detect if not specified.'
    )

    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Random seed. Uses config value if not specified.'
    )

    parser.add_argument(
        '--num-epochs',
        type=int,
        default=None,
        help='Number of epochs to train. Uses config value if not specified.'
    )

    parser.add_argument(
        '--batch-size',
        type=int,
        default=None,
        help='Batch size. Uses config value if not specified.'
    )

    return parser.parse_args()


def prepare_data(config, model_type, device):
    """
    Prepare datasets and dataloaders.

    Args:
        config (dict): Configuration dictionary
        model_type (str): Model type
        device (torch.device): Device

    Returns:
        tuple: (train_loader, val_loader, test_loader, vocab_or_tokenizer)
    """
    print("\n" + "="*80)
    print("PREPARING DATA")
    print("="*80)

    data_dir = Path(config['dataset']['data_dir'])
    batch_size = config['training']['batch_size']

    # Determine if model uses text, vision, or both
    is_multimodal = model_type in ['early_fusion', 'late_fusion', 'attention_fusion']
    is_text_only = model_type in ['lstm_text', 'bert_text']
    is_vision_only = model_type in ['resnet_vision', 'cnn_vision']

    # Prepare text preprocessing
    vocab_or_tokenizer = None
    if is_text_only or is_multimodal:
        if 'bert' in model_type:
            # BERT tokenizer
            print("Loading BERT tokenizer...")
            max_length = config['preprocessing']['text'].get('max_length_bert', 512)
            tokenizer = get_bert_tokenizer(
                model_name=config['model_distilbert'].get('model_name', 'distilbert-base-uncased'),
                max_length=max_length
            )
            vocab_or_tokenizer = tokenizer
            text_tokenizer_fn = tokenizer
        else:
            # LSTM tokenizer - load vocabulary from preprocessed data
            print("Loading vocabulary from preprocessed data...")
            vocab_path = Path(data_dir) / 'vocab.json'
            with open(vocab_path, 'r') as f:
                vocab_data = json.load(f)

            # Extract word_to_ix mapping (the actual vocabulary)
            vocab = vocab_data.get('word_to_ix', vocab_data)
            print(f"Loaded vocabulary with {len(vocab)} words")

            max_length = config['preprocessing']['text']['max_length_lstm']
            tokenizer = LSTMTokenizer(
                vocab=vocab,
                max_length=max_length
            )
            vocab_or_tokenizer = tokenizer
            text_tokenizer_fn = tokenizer

    # Prepare image preprocessing
    if is_vision_only or is_multimodal:
        print("Preparing image transforms...")
        train_transform = get_image_transforms(
            split='train',
            image_size=config['preprocessing']['image']['target_size']
        )
        val_transform = get_image_transforms(
            split='val',
            image_size=config['preprocessing']['image']['target_size']
        )
    else:
        train_transform = None
        val_transform = None

    # Load genre to index mapping from preprocessed data
    print("Loading genre mapping...")
    genre_mapping_path = Path(data_dir) / 'genre_mapping.json'
    with open(genre_mapping_path, 'r') as f:
        genre_to_idx = json.load(f)
    print(f"Loaded {len(genre_to_idx)} genres from {genre_mapping_path}")

    # Create datasets
    print("\nCreating datasets...")

    if is_multimodal:
        # Multimodal datasets
        train_dataset = MultimodalDataset(
            data_dir=data_dir,
            split='train',
            text_tokenizer=text_tokenizer_fn,
            image_transform=train_transform,
            genre_to_idx=genre_to_idx
        )
        val_dataset = MultimodalDataset(
            data_dir=data_dir,
            split='val',
            text_tokenizer=text_tokenizer_fn,
            image_transform=val_transform,
            genre_to_idx=genre_to_idx
        )
        test_dataset = MultimodalDataset(
            data_dir=data_dir,
            split='test',
            text_tokenizer=text_tokenizer_fn,
            image_transform=val_transform,
            genre_to_idx=genre_to_idx
        )

    elif is_text_only:
        # Text-only datasets
        train_dataset = TextOnlyDataset(
            data_dir=data_dir,
            split='train',
            text_tokenizer=text_tokenizer_fn,
            genre_to_idx=genre_to_idx
        )
        val_dataset = TextOnlyDataset(
            data_dir=data_dir,
            split='val',
            text_tokenizer=text_tokenizer_fn,
            genre_to_idx=genre_to_idx
        )
        test_dataset = TextOnlyDataset(
            data_dir=data_dir,
            split='test',
            text_tokenizer=text_tokenizer_fn,
            genre_to_idx=genre_to_idx
        )

    else:  # is_vision_only
        # Vision-only datasets
        train_dataset = ImageOnlyDataset(
            data_dir=data_dir,
            split='train',
            image_transform=train_transform,
            genre_to_idx=genre_to_idx
        )
        val_dataset = ImageOnlyDataset(
            data_dir=data_dir,
            split='val',
            image_transform=val_transform,
            genre_to_idx=genre_to_idx
        )
        test_dataset = ImageOnlyDataset(
            data_dir=data_dir,
            split='test',
            image_transform=val_transform,
            genre_to_idx=genre_to_idx
        )

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Val dataset size: {len(val_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")

    # Create dataloaders
    print("\nCreating dataloaders...")
    # todo: add num_workers for windows
    # Note: Using num_workers=0 on Windows to avoid multiprocessing issues with Path objects
    num_workers = 0 if device.type == 'cuda' else config['training'].get('num_workers', 0)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if device.type == 'cuda' else False
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if device.type == 'cuda' else False
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if device.type == 'cuda' else False
    )

    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")

    return train_loader, val_loader, test_loader, vocab_or_tokenizer


def create_model(config, model_type, vocab_or_tokenizer, device):
    """
    Create and initialize model.

    Args:
        config (dict): Configuration dictionary
        model_type (str): Model type
        vocab_or_tokenizer: Vocabulary or tokenizer for text models
        device (torch.device): Device

    Returns:
        nn.Module: Initialized model
    """
    print("\n" + "="*80)
    print("CREATING MODEL")
    print("="*80)

    num_classes = config['dataset']['num_genres']

    if model_type == 'lstm_text':
        # LSTM text model
        print("Creating LSTM text model...")
        model_config = config['model_lstm'].copy()
        model_config['num_classes'] = num_classes

        vocab_size = len(vocab_or_tokenizer.vocab) if hasattr(vocab_or_tokenizer, 'vocab') else 10000

        # Load GloVe embeddings if specified
        pretrained_embeddings = None
        if model_config.get('pretrained_embeddings') and model_config['pretrained_embeddings'] != 'null':
            from src.utils.embeddings import create_embedding_matrix

            print(f"\nLoading pretrained embeddings: {model_config['pretrained_embeddings']}")
            embedding_dim = model_config.get('embedding_dim', 300)

            embeddings, emb_info = create_embedding_matrix(
                vocab=vocab_or_tokenizer.vocab,
                embedding_dim=embedding_dim,
                use_pretrained=True
            )

            pretrained_embeddings = embeddings
            print(f"Pretrained embeddings loaded: {emb_info['coverage']:.2f}% coverage")

        model_config['pretrained_embeddings'] = pretrained_embeddings
        model = create_text_model(model_config, vocab_size=vocab_size)

    elif model_type == 'bert_text':
        # DistilBERT text model
        print("Creating DistilBERT text model...")
        model_config = config['model_distilbert'].copy()
        model_config['num_classes'] = num_classes
        model = create_text_model(model_config)

    elif model_type == 'resnet_vision':
        # ResNet vision model
        print("Creating ResNet vision model...")
        model_config = config['model_resnet'].copy()
        model_config['num_classes'] = num_classes
        model = create_vision_model(model_config)

    elif model_type == 'cnn_vision':
        # Custom CNN vision model
        print("Creating Custom CNN vision model...")
        model_config = config['model_custom_cnn'].copy()
        model_config['num_classes'] = num_classes
        model = create_vision_model(model_config)

    elif model_type in ['early_fusion', 'late_fusion', 'attention_fusion']:
        # Multimodal fusion models
        print(f"Creating {model_type.replace('_', ' ').title()} model...")

        # Get fusion config
        fusion_config = config[f'model_{model_type}'].copy()

        # Determine which text and vision models to use
        text_model_type = fusion_config.get('text_model', 'distilbert')
        vision_model_type = fusion_config.get('vision_model', 'resnet18')

        # Create text model
        if 'bert' in text_model_type or 'distilbert' in text_model_type:
            text_config = config['model_distilbert'].copy()
            text_config['num_classes'] = num_classes
            text_model = create_text_model(text_config)
        else:  # lstm
            text_config = config['model_lstm'].copy()
            text_config['num_classes'] = num_classes
            vocab_size = len(vocab_or_tokenizer.vocab) if hasattr(vocab_or_tokenizer, 'vocab') else 10000

            # Load GloVe embeddings if specified
            pretrained_embeddings = None
            if text_config.get('pretrained_embeddings') and text_config['pretrained_embeddings'] != 'null':
                from src.utils.embeddings import create_embedding_matrix

                print(f"\nLoading pretrained embeddings for fusion model: {text_config['pretrained_embeddings']}")
                embedding_dim = text_config.get('embedding_dim', 300)

                embeddings, emb_info = create_embedding_matrix(
                    vocab=vocab_or_tokenizer.vocab,
                    embedding_dim=embedding_dim,
                    use_pretrained=True
                )

                pretrained_embeddings = embeddings
                print(f"Pretrained embeddings loaded: {emb_info['coverage']:.2f}% coverage")

            text_config['pretrained_embeddings'] = pretrained_embeddings
            text_model = create_text_model(text_config, vocab_size=vocab_size)

        # Create vision model
        if 'custom' in vision_model_type or 'cnn' in vision_model_type:
            vision_config = config['model_custom_cnn'].copy()
        else:  # resnet
            vision_config = config['model_resnet'].copy()
        vision_config['num_classes'] = num_classes
        vision_model = create_vision_model(vision_config)

        if model_type == 'early_fusion':
            model = EarlyFusionModel(
                text_model=text_model,
                vision_model=vision_model,
                num_classes=num_classes,
                text_projection_dim=fusion_config.get('text_projection_dim', 512),
                vision_projection_dim=fusion_config.get('vision_projection_dim', 512),
                fusion_hidden_dims=fusion_config.get('fusion_hidden_dims', [1024, 512, 256]),
                dropout=fusion_config.get('dropout', 0.3)
            )

        elif model_type == 'late_fusion':
            model = LateFusionModel(
                text_model=text_model,
                vision_model=vision_model,
                num_classes=num_classes,
                fusion_strategy=fusion_config.get('fusion_strategy', 'average')
            )

        else:  # attention_fusion
            model = AttentionFusionModel(
                text_model=text_model,
                vision_model=vision_model,
                num_classes=num_classes,
                text_projection_dim=fusion_config.get('text_projection_dim', 512),
                vision_projection_dim=fusion_config.get('vision_projection_dim', 512),
                fusion_hidden_dims=fusion_config.get('fusion_hidden_dims', [1024, 512, 256]),
                num_attention_heads=fusion_config.get('num_heads', 8),
                dropout=fusion_config.get('dropout', 0.3)
            )

    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Move model to device
    model = model.to(device)

    # Print model info
    num_params = sum(p.numel() for p in model.parameters())
    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel: {model_type}")
    print(f"Total parameters: {num_params:,}")
    print(f"Trainable parameters: {num_trainable:,}")

    return model


def main():
    """Main training function."""
    # Parse arguments
    args = parse_args()

    # Load configuration
    print(f"Loading configuration from {args.config}...")
    config = load_config(args.config)

    # Override config with command line arguments if provided
    if args.seed is not None:
        config['project']['seed'] = args.seed
    if args.num_epochs is not None:
        config['training']['num_epochs'] = args.num_epochs
    if args.batch_size is not None:
        config['training']['batch_size'] = args.batch_size

    # Set random seed
    seed = config['project']['seed']
    set_seed(seed)
    print(f"Random seed set to {seed}")

    # Setup device
    device = setup_device(args.device)
    print(f"Using device: {device}")

    # Create output directories
    output_dir = Path(args.output_dir) / args.model / datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_dir = Path(args.checkpoint_dir) / args.model
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    print(f"Output directory: {output_dir}")
    print(f"Checkpoint directory: {checkpoint_dir}")

    # Save configuration
    config_save_path = output_dir / 'config.json'
    with open(config_save_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"Configuration saved to {config_save_path}")

    # Prepare data
    train_loader, val_loader, test_loader, vocab_or_tokenizer = prepare_data(
        config, args.model, device
    )

    # Create model
    model = create_model(config, args.model, vocab_or_tokenizer, device)

    # Create loss function
    print("\n" + "="*80)
    print("SETTING UP TRAINING")
    print("="*80)

    loss_config = config['loss'].copy()

    # Load class weights if using weighted BCE
    if loss_config.get('type') == 'weighted_bce' and loss_config.get('compute_weights', False):
        weights_path = Path(config['dataset']['data_dir']) / 'class_weights.json'
        if weights_path.exists():
            print(f"Loading class weights from {weights_path}")
            with open(weights_path, 'r') as f:
                weights_data = json.load(f)
            pos_weights = torch.FloatTensor(weights_data['pos_weights'])
            loss_config['pos_weight'] = pos_weights
            print(f"Loaded {len(pos_weights)} class weights")
            print(f"Weight range: {pos_weights.min():.2f} - {pos_weights.max():.2f}")
        else:
            print(f"Warning: compute_weights=True but {weights_path} not found!")
            print("Run: python scripts/compute_class_weights.py")

    loss_fn = get_loss_function(loss_config, device=device)
    print(f"Loss function: {type(loss_fn).__name__}")

    # Create optimizer
    optimizer_config = config['training'].copy()

    # Select appropriate learning rate based on model type
    if 'bert' in args.model.lower():
        # Use BERT-specific learning rate
        lr = optimizer_config['learning_rate'].get('bert_fine_tune', 0.00002)
        print(f"Using BERT fine-tuning LR: {lr}")
    elif 'resnet' in args.model.lower() or 'vision' in args.model.lower():
        # Use fine-tuning LR for pretrained vision models
        lr = optimizer_config['learning_rate'].get('fine_tune', 0.0001)
        print(f"Using fine-tuning LR: {lr}")
    else:
        # Use from-scratch LR for LSTM and custom models
        lr = optimizer_config['learning_rate'].get('from_scratch', 0.0003)
        print(f"Using from-scratch LR: {lr}")

    optimizer_config['lr'] = lr
    optimizer = create_optimizer(model, optimizer_config)
    print(f"Optimizer: {type(optimizer).__name__}")

    # Create scheduler
    scheduler_config = config['training'].get('scheduler', {})
    scheduler = create_scheduler(optimizer, scheduler_config)
    if scheduler:
        print(f"Scheduler: {type(scheduler).__name__}")

    # Create trainer
    trainer_config = config['training'].copy()
    trainer_config['checkpoint_dir'] = checkpoint_dir
    trainer_config['log_dir'] = output_dir / 'logs'

    # Metric function for validation
    def metric_fn(predictions, targets):
        """Compute F1-macro as validation metric."""
        from src.evaluation.metrics import compute_f1_scores
        # Use standard 0.5 threshold for BCE loss
        # Note: Focal loss requires lower threshold (~0.2), but BCE works with 0.5
        scores = compute_f1_scores(predictions, targets, threshold=0.5, average='macro')
        return scores['f1_macro']

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device=device,
        config=trainer_config,
        scheduler=scheduler,
        metric_fn=metric_fn,
        checkpoint_dir=checkpoint_dir,
        resume_from=args.resume
    )

    # Train model
    print("\n" + "="*80)
    print("TRAINING")
    print("="*80)

    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=config['training']['num_epochs']
    )

    # Plot training history
    print("\nPlotting training history...")
    plot_training_history(
        history,
        save_path=output_dir / 'training_history.png',
        show=False
    )

    # Evaluate on test set
    print("\n" + "="*80)
    print("FINAL EVALUATION")
    print("="*80)

    print("Evaluating on test set...")
    test_metrics = trainer.validate(test_loader, epoch=trainer.current_epoch)

    print(f"\nTest Loss: {test_metrics['loss']:.4f}")
    print(f"Test Metric: {test_metrics['metric']:.4f}")

    # Save final results
    results = {
        'model_type': args.model,
        'best_val_metric': trainer.best_metric,
        'final_test_loss': test_metrics['loss'],
        'final_test_metric': test_metrics['metric'],
        'training_history': {k: [float(v) for v in vals] for k, vals in history.items()},
        'config': config
    }

    results_path = output_dir / 'results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    print("\n" + "="*80)
    print("TRAINING COMPLETED")
    print("="*80)
    print(f"Best validation metric: {trainer.best_metric:.4f}")
    print(f"Final test metric: {test_metrics['metric']:.4f}")
    print(f"Results saved to: {output_dir}")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
