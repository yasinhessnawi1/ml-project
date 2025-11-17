"""
Comprehensive Test Script for ML Project Implementation

This script tests:
1. Configuration loading
2. Data downloading (MM-IMDb dataset)
3. Data preprocessing
4. Train/val/test splitting
5. Text model (LSTM & DistilBERT)
6. Vision model (ResNet & Custom CNN)
7. DataLoader functionality

Run this script to verify the implementation before training.

Usage:
    python test_implementation.py [--config config.yaml] [--download-data]
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.utils.config import (
    load_config,
    set_seed,
    setup_device,
    validate_config,
)
from src.data.preprocessing import (
    clean_text,
    build_vocab,
    LSTMTokenizer,
    get_bert_tokenizer,
    get_image_transforms,
    encode_genres,
    compute_class_weights,
    get_text_statistics,
)
from src.data.dataset import (
    MultimodalDataset,
    TextOnlyDataset,
    ImageOnlyDataset,
)
from src.models.text_models import LSTMTextModel, DistilBERTTextModel
from src.models.vision_models import ResNetVisionModel, CustomCNNModel

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ============================================================================
# 1. Dataset Download and Preparation
# ============================================================================


def download_mmimdb_dataset(data_dir: Path) -> bool:
    """
    Download MM-IMDb dataset.

    Note: This is a placeholder. Actual implementation depends on dataset source.
    MM-IMDb is typically downloaded from:
    http://lisi1.unal.edu.co/mmimdb/

    Args:
        data_dir (Path): Directory to save dataset

    Returns:
        bool: True if successful, False otherwise
    """
    logger.info("=" * 80)
    logger.info("STEP 1: Dataset Download")
    logger.info("=" * 80)

    data_dir.mkdir(parents=True, exist_ok=True)

    logger.warning(
        "⚠️  MM-IMDb dataset download not implemented in this test script"
    )
    logger.info("Please download the dataset manually from:")
    logger.info("   http://lisi1.unal.edu.co/mmimdb/")
    logger.info(f"   Extract to: {data_dir}")
    logger.info("\nExpected structure:")
    logger.info("   data/raw/")
    logger.info("   ├── dataset/")
    logger.info("   │   ├── split.json")
    logger.info("   │   ├── [movie_id_1]/")
    logger.info("   │   │   ├── poster.jpg")
    logger.info("   │   │   └── plot.txt")
    logger.info("   │   └── ...")

    # For testing purposes, create dummy structure
    logger.info("\n📝 Creating dummy dataset structure for testing...")
    create_dummy_dataset(data_dir)

    return True


def create_dummy_dataset(data_dir: Path) -> None:
    """
    Create a small dummy dataset for testing purposes.

    Creates synthetic data that mimics MM-IMDb structure.

    Args:
        data_dir (Path): Directory to create dummy data
    """
    from PIL import Image, ImageDraw, ImageFont
    import random

    # Genre list
    genres = [
        "Action",
        "Adventure",
        "Animation",
        "Comedy",
        "Crime",
        "Drama",
        "Fantasy",
        "Horror",
        "Mystery",
        "Romance",
        "Sci-Fi",
        "Thriller",
    ]

    # Create dummy movies
    num_movies = 100  # Small dataset for testing
    movies = []

    logger.info(f"Creating {num_movies} dummy movie samples...")

    for i in tqdm(range(num_movies), desc="Creating dummy data"):
        movie_id = f"movie_{i:04d}"
        movie_dir = data_dir / movie_id
        movie_dir.mkdir(parents=True, exist_ok=True)

        # Create dummy plot text
        plot_templates = [
            f"A thrilling {random.choice(['action', 'adventure', 'mystery'])} story about {random.choice(['love', 'betrayal', 'revenge', 'discovery'])}.",
            f"An epic tale of {random.choice(['heroes', 'villains', 'explorers'])} facing {random.choice(['danger', 'challenges', 'mysteries'])}.",
            f"A {random.choice(['heartwarming', 'suspenseful', 'dramatic'])} journey through {random.choice(['time', 'space', 'unknown lands'])}.",
        ]
        plot = " ".join(
            random.choices(plot_templates, k=random.randint(2, 5))
        )

        with open(movie_dir / "plot.txt", "w") as f:
            f.write(plot)

        # Create dummy poster image (224x224 colored rectangle)
        img = Image.new(
            "RGB",
            (224, 224),
            color=(
                random.randint(50, 200),
                random.randint(50, 200),
                random.randint(50, 200),
            ),
        )

        # Add some text to make it look like a poster
        draw = ImageDraw.Draw(img)
        try:
            # Try to use a font, fall back to default if not available
            font = ImageFont.truetype(
                "/System/Library/Fonts/Helvetica.ttc", 20
            )
        except:
            font = ImageFont.load_default()

        draw.text((10, 100), f"Movie {i}", fill=(255, 255, 255), font=font)
        img.save(movie_dir / "poster.jpg")

        # Random genres (1-3 genres per movie)
        movie_genres = random.sample(genres, k=random.randint(1, 3))

        movies.append(
            {
                "id": movie_id,
                "plot": f"{movie_id}/plot.txt",
                "poster": f"{movie_id}/poster.jpg",
                "genres": movie_genres,
            }
        )

    # Create splits (70/15/15)
    random.shuffle(movies)
    train_size = int(0.7 * len(movies))
    val_size = int(0.15 * len(movies))

    splits = {
        "train": movies[:train_size],
        "val": movies[train_size : train_size + val_size],
        "test": movies[train_size + val_size :],
    }

    # Save split files
    for split_name, split_data in splits.items():
        split_file = data_dir / f"{split_name}.json"
        with open(split_file, "w") as f:
            json.dump(split_data, f, indent=2)

        logger.info(f"   {split_name}: {len(split_data)} samples")

    # Save genre mapping
    genre_to_idx = {genre: idx for idx, genre in enumerate(genres)}
    with open(data_dir / "genre_mapping.json", "w") as f:
        json.dump(genre_to_idx, f, indent=2)

    logger.info(f"✅ Dummy dataset created with {len(genres)} genres")


# ============================================================================
# 2. Data Preprocessing Tests
# ============================================================================


def test_preprocessing(config: Dict) -> Dict:
    """
    Test data preprocessing functions.

    Args:
        config (Dict): Configuration dictionary

    Returns:
        Dict: Test results
    """
    logger.info("\n" + "=" * 80)
    logger.info("STEP 2: Data Preprocessing Tests")
    logger.info("=" * 80)

    results = {}

    # Test text cleaning
    logger.info("\n📝 Testing text cleaning...")
    sample_text = "<p>A thrilling  ACTION movie!  </p>"
    cleaned = clean_text(sample_text, remove_html=True, lowercase=True)
    logger.info(f"   Original: '{sample_text}'")
    logger.info(f"   Cleaned:  '{cleaned}'")
    results["text_cleaning"] = (
        "PASSED" if cleaned == "a thrilling action movie!" else "FAILED"
    )

    # Test vocabulary building
    logger.info("\n📚 Testing vocabulary building...")
    sample_texts = [
        "action adventure movie",
        "romantic comedy film",
        "action thriller adventure",
    ]
    vocab = build_vocab(sample_texts, vocab_size=100)
    logger.info(f"   Vocabulary size: {len(vocab)}")
    logger.info(f"   Sample words: {list(vocab.keys())[:10]}")
    results["vocab_building"] = "PASSED" if len(vocab) > 0 else "FAILED"

    # Test LSTM tokenizer
    logger.info("\n🔤 Testing LSTM tokenizer...")
    tokenizer = LSTMTokenizer(vocab, max_length=20)
    tokens = tokenizer("action movie")
    logger.info(f"   Input: 'action movie'")
    logger.info(f"   Token IDs: {tokens[:10].tolist()}")
    logger.info(f"   Shape: {tokens.shape}")
    results["lstm_tokenizer"] = (
        "PASSED" if tokens.shape[0] == 20 else "FAILED"
    )

    # Test BERT tokenizer
    logger.info("\n🤖 Testing BERT tokenizer...")
    try:
        bert_tokenizer = get_bert_tokenizer()
        encoding = bert_tokenizer(
            "A thrilling action movie",
            max_length=512,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        logger.info(f"   Input IDs shape: {encoding['input_ids'].shape}")
        logger.info(
            f"   Attention mask shape: {encoding['attention_mask'].shape}"
        )
        results["bert_tokenizer"] = "PASSED"
    except Exception as e:
        logger.error(f"   BERT tokenizer failed: {e}")
        results["bert_tokenizer"] = "FAILED"

    # Test image transforms
    logger.info("\n🖼️  Testing image transforms...")
    train_transform = get_image_transforms("train", image_size=224)
    val_transform = get_image_transforms("val", image_size=224)
    logger.info(
        f"   Train transform: {len(train_transform.transforms)} operations"
    )
    logger.info(
        f"   Val transform: {len(val_transform.transforms)} operations"
    )
    results["image_transforms"] = "PASSED"

    # Test genre encoding
    logger.info("\n🏷️  Testing genre encoding...")
    genre_to_idx = {"Action": 0, "Comedy": 1, "Drama": 2}
    genres = ["Action", "Drama"]
    encoding = encode_genres(genres, genre_to_idx, num_genres=3)
    logger.info(f"   Input genres: {genres}")
    logger.info(f"   Encoding: {encoding.tolist()}")
    expected = [1.0, 0.0, 1.0]
    results["genre_encoding"] = (
        "PASSED" if encoding.tolist() == expected else "FAILED"
    )

    # Summary
    logger.info("\n" + "-" * 80)
    logger.info("Preprocessing Test Results:")
    for test_name, result in results.items():
        status = "✅" if result == "PASSED" else "❌"
        logger.info(f"   {status} {test_name}: {result}")

    return results


# ============================================================================
# 3. Dataset Tests
# ============================================================================


def test_datasets(config: Dict, data_dir: Path) -> Dict:
    """
    Test Dataset classes.

    Args:
        config (Dict): Configuration dictionary
        data_dir (Path): Data directory

    Returns:
        Dict: Test results
    """
    logger.info("\n" + "=" * 80)
    logger.info("STEP 3: Dataset Tests")
    logger.info("=" * 80)

    results = {}

    # Load genre mapping
    with open(data_dir / "genre_mapping.json", "r") as f:
        genre_to_idx = json.load(f)

    logger.info(f"\n📊 Dataset info:")
    logger.info(f"   Data directory: {data_dir}")
    logger.info(f"   Number of genres: {len(genre_to_idx)}")

    # Build vocabulary from training data
    logger.info("\n📚 Building vocabulary from training data...")
    train_texts = []
    with open(data_dir / "train.json", "r") as f:
        train_data = json.load(f)

    for item in train_data[:50]:  # Use subset for speed
        text_path = data_dir / item["plot"]
        if text_path.exists():
            with open(text_path, "r") as f:
                text = clean_text(f.read(), lowercase=True)
                train_texts.append(text)

    vocab = build_vocab(
        train_texts,
        vocab_size=config["preprocessing"]["text"]["vocab_size"],
    )
    logger.info(f"   Vocabulary size: {len(vocab)}")

    # Test TextOnlyDataset
    logger.info("\n📄 Testing TextOnlyDataset...")
    try:
        text_tokenizer = LSTMTokenizer(
            vocab,
            max_length=config["preprocessing"]["text"]["max_length_lstm"],
        )
        text_dataset = TextOnlyDataset(
            data_dir=data_dir,
            split="train",
            text_tokenizer=text_tokenizer,
            genre_to_idx=genre_to_idx,
            min_text_length=config["preprocessing"]["text"][
                "min_text_length"
            ],
        )
        logger.info(f"   Dataset size: {len(text_dataset)}")

        # Get a sample
        sample = text_dataset[0]
        logger.info(f"   Sample text shape: {sample['text'].shape}")
        logger.info(f"   Sample labels shape: {sample['labels'].shape}")
        logger.info(f"   Movie ID: {sample['movie_id']}")

        results["text_dataset"] = "PASSED"
    except Exception as e:
        logger.error(f"   ❌ TextOnlyDataset failed: {e}")
        results["text_dataset"] = "FAILED"

    # Test ImageOnlyDataset
    logger.info("\n🖼️  Testing ImageOnlyDataset...")
    try:
        image_transform = get_image_transforms("train", image_size=224)
        image_dataset = ImageOnlyDataset(
            data_dir=data_dir,
            split="train",
            image_transform=image_transform,
            genre_to_idx=genre_to_idx,
        )
        logger.info(f"   Dataset size: {len(image_dataset)}")

        # Get a sample
        sample = image_dataset[0]
        logger.info(f"   Sample image shape: {sample['image'].shape}")
        logger.info(f"   Sample labels shape: {sample['labels'].shape}")

        results["image_dataset"] = "PASSED"
    except Exception as e:
        logger.error(f"   ❌ ImageOnlyDataset failed: {e}")
        results["image_dataset"] = "FAILED"

    # Test MultimodalDataset
    logger.info("\n🎬 Testing MultimodalDataset...")
    try:
        multimodal_dataset = MultimodalDataset(
            data_dir=data_dir,
            split="train",
            text_tokenizer=text_tokenizer,
            image_transform=image_transform,
            genre_to_idx=genre_to_idx,
            min_text_length=config["preprocessing"]["text"][
                "min_text_length"
            ],
        )
        logger.info(f"   Dataset size: {len(multimodal_dataset)}")

        # Get a sample
        sample = multimodal_dataset[0]
        logger.info(f"   Sample text shape: {sample['text'].shape}")
        logger.info(f"   Sample image shape: {sample['image'].shape}")
        logger.info(f"   Sample labels shape: {sample['labels'].shape}")

        results["multimodal_dataset"] = "PASSED"
    except Exception as e:
        logger.error(f"   ❌ MultimodalDataset failed: {e}")
        results["multimodal_dataset"] = "FAILED"

    # Test DataLoader
    logger.info("\n🔄 Testing DataLoader...")
    try:
        dataloader = DataLoader(
            multimodal_dataset,
            batch_size=4,
            shuffle=True,
            num_workers=0,  # Use 0 for testing to avoid multiprocessing issues
        )

        batch = next(iter(dataloader))
        logger.info(f"   Batch text shape: {batch['text'].shape}")
        logger.info(f"   Batch image shape: {batch['image'].shape}")
        logger.info(f"   Batch labels shape: {batch['labels'].shape}")

        results["dataloader"] = "PASSED"
    except Exception as e:
        logger.error(f"   ❌ DataLoader failed: {e}")
        results["dataloader"] = "FAILED"

    # Summary
    logger.info("\n" + "-" * 80)
    logger.info("Dataset Test Results:")
    for test_name, result in results.items():
        status = "✅" if result == "PASSED" else "❌"
        logger.info(f"   {status} {test_name}: {result}")

    return results, vocab, genre_to_idx


# ============================================================================
# 4. Model Tests
# ============================================================================


def test_text_models(
    config: Dict, vocab: Dict, device: torch.device
) -> Dict:
    """
    Test text models.

    Args:
        config (Dict): Configuration dictionary
        vocab (Dict): Vocabulary dictionary
        device (torch.device): Compute device

    Returns:
        Dict: Test results
    """
    logger.info("\n" + "=" * 80)
    logger.info("STEP 4: Text Model Tests")
    logger.info("=" * 80)

    results = {}

    # Test LSTM model
    logger.info("\n📝 Testing LSTM Text Model...")
    try:
        lstm_config = config["model_lstm"]
        lstm_model = LSTMTextModel(
            vocab_size=len(vocab),
            embedding_dim=lstm_config["embedding_dim"],
            hidden_dim=lstm_config["hidden_dim"],
            num_layers=lstm_config["num_layers"],
            num_classes=config["dataset"]["num_genres"],
            dropout=lstm_config["dropout"],
            bidirectional=lstm_config["bidirectional"],
            use_attention=lstm_config["attention"],
        ).to(device)

        # Test forward pass
        batch_size = 4
        seq_len = config["preprocessing"]["text"]["max_length_lstm"]
        dummy_input = torch.randint(
            0, len(vocab), (batch_size, seq_len)
        ).to(device)

        output = lstm_model(dummy_input)
        logger.info(f"   Input shape: {dummy_input.shape}")
        logger.info(f"   Output shape: {output.shape}")
        logger.info(
            f"   Parameters: {sum(p.numel() for p in lstm_model.parameters()):,}"
        )

        # Test with attention
        output, attn_weights = lstm_model(
            dummy_input, return_attention=True
        )
        logger.info(f"   Attention weights shape: {attn_weights.shape}")

        results["lstm_model"] = "PASSED"
    except Exception as e:
        logger.error(f"   ❌ LSTM model failed: {e}")
        results["lstm_model"] = "FAILED"

    # Test DistilBERT model
    logger.info("\n🤖 Testing DistilBERT Text Model...")
    try:
        bert_config = config["model_distilbert"]
        bert_model = DistilBERTTextModel(
            model_name=bert_config["model_name"],
            num_classes=config["dataset"]["num_genres"],
            classifier_hidden_dim=bert_config["classifier_hidden_dim"],
            dropout=bert_config["dropout"],
            fine_tune_all=bert_config["fine_tune_all"],
        ).to(device)

        # Test forward pass
        batch_size = 4
        seq_len = config["preprocessing"]["text"]["max_length_bert"]
        dummy_input_ids = torch.randint(0, 30522, (batch_size, seq_len)).to(
            device
        )
        dummy_attention_mask = torch.ones(batch_size, seq_len).to(device)

        output = bert_model(dummy_input_ids, dummy_attention_mask)
        logger.info(f"   Input shape: {dummy_input_ids.shape}")
        logger.info(f"   Output shape: {output.shape}")
        logger.info(
            f"   Parameters: {sum(p.numel() for p in bert_model.parameters()):,}"
        )

        results["distilbert_model"] = "PASSED"
    except Exception as e:
        logger.error(f"   ❌ DistilBERT model failed: {e}")
        logger.warning(
            f"   Note: This may fail if transformers library is not installed or internet is unavailable"
        )
        results["distilbert_model"] = "FAILED"

    # Summary
    logger.info("\n" + "-" * 80)
    logger.info("Text Model Test Results:")
    for test_name, result in results.items():
        status = "✅" if result == "PASSED" else "❌"
        logger.info(f"   {status} {test_name}: {result}")

    return results


def test_vision_models(config: Dict, device: torch.device) -> Dict:
    """
    Test vision models.

    Args:
        config (Dict): Configuration dictionary
        device (torch.device): Compute device

    Returns:
        Dict: Test results
    """
    logger.info("\n" + "=" * 80)
    logger.info("STEP 5: Vision Model Tests")
    logger.info("=" * 80)

    results = {}

    # Test ResNet model
    logger.info("\n🖼️  Testing ResNet Vision Model...")
    try:
        resnet_config = config["model_resnet"]
        resnet_model = ResNetVisionModel(
            architecture=resnet_config["architecture"],
            num_classes=config["dataset"]["num_genres"],
            pretrained=False,  # Use False for faster testing
            fine_tune_strategy=resnet_config["fine_tune_strategy"],
            classifier_hidden_dim=resnet_config["classifier_hidden_dim"],
            dropout=resnet_config["dropout"],
        ).to(device)

        # Test forward pass
        batch_size = 4
        dummy_images = torch.randn(batch_size, 3, 224, 224).to(device)

        output = resnet_model(dummy_images)
        features = resnet_model.get_features(dummy_images)

        logger.info(f"   Input shape: {dummy_images.shape}")
        logger.info(f"   Output shape: {output.shape}")
        logger.info(f"   Feature shape: {features.shape}")
        logger.info(
            f"   Parameters: {sum(p.numel() for p in resnet_model.parameters()):,}"
        )

        results["resnet_model"] = "PASSED"
    except Exception as e:
        logger.error(f"   ❌ ResNet model failed: {e}")
        results["resnet_model"] = "FAILED"

    # Test Custom CNN model
    logger.info("\n🎨 Testing Custom CNN Model...")
    try:
        cnn_config = config["model_custom_cnn"]
        cnn_model = CustomCNNModel(
            channels=cnn_config["channels"],
            kernel_sizes=cnn_config["kernel_sizes"],
            num_classes=config["dataset"]["num_genres"],
            dropout=cnn_config["dropout"],
        ).to(device)

        # Test forward pass
        output = cnn_model(dummy_images)
        features = cnn_model.get_features(dummy_images)

        logger.info(f"   Input shape: {dummy_images.shape}")
        logger.info(f"   Output shape: {output.shape}")
        logger.info(f"   Feature shape: {features.shape}")
        logger.info(
            f"   Parameters: {sum(p.numel() for p in cnn_model.parameters()):,}"
        )

        results["custom_cnn_model"] = "PASSED"
    except Exception as e:
        logger.error(f"   ❌ Custom CNN model failed: {e}")
        results["custom_cnn_model"] = "FAILED"

    # Summary
    logger.info("\n" + "-" * 80)
    logger.info("Vision Model Test Results:")
    for test_name, result in results.items():
        status = "✅" if result == "PASSED" else "❌"
        logger.info(f"   {status} {test_name}: {result}")

    return results


# ============================================================================
# Main Test Function
# ============================================================================


def run_all_tests(
    config_path: str = "config.yaml", download_data: bool = False
):
    """
    Run all implementation tests.

    Args:
        config_path (str, optional): Path to config file. Defaults to 'config.yaml'.
        download_data (bool, optional): Whether to download dataset. Defaults to False.
    """
    logger.info("=" * 80)
    logger.info("ML PROJECT IMPLEMENTATION TEST")
    logger.info("=" * 80)

    try:
        # Load configuration
        logger.info("\n📋 Loading configuration...")
        config = load_config(config_path)
        validate_config(config)
        logger.info("✅ Configuration loaded and validated")

        # Set random seed
        logger.info(f"\n🌱 Setting random seed: {config['project']['seed']}")
        set_seed(config["project"]["seed"])

        # Setup device
        logger.info("\n💻 Setting up compute device...")
        device = setup_device(config["project"]["device"])

        # Download/prepare dataset
        data_dir = Path(config["paths"]["data_dir"])
        if download_data or not data_dir.exists():
            download_mmimdb_dataset(data_dir)

        # Run tests
        all_results = {}

        # Test preprocessing
        preprocessing_results = test_preprocessing(config)
        all_results.update(preprocessing_results)

        # Test datasets
        dataset_results, vocab, genre_to_idx = test_datasets(
            config, data_dir
        )
        all_results.update(dataset_results)

        # Test text models
        text_model_results = test_text_models(config, vocab, device)
        all_results.update(text_model_results)

        # Test vision models
        vision_model_results = test_vision_models(config, device)
        all_results.update(vision_model_results)

        # Final summary
        logger.info("\n" + "=" * 80)
        logger.info("FINAL TEST SUMMARY")
        logger.info("=" * 80)

        passed = sum(1 for r in all_results.values() if r == "PASSED")
        failed = sum(1 for r in all_results.values() if r == "FAILED")
        total = len(all_results)

        logger.info(
            f"\n📊 Results: {passed}/{total} tests passed ({failed} failed)"
        )
        logger.info("\nDetailed Results:")

        for test_name, result in all_results.items():
            status = "✅" if result == "PASSED" else "❌"
            logger.info(f"   {status} {test_name}: {result}")

        if failed == 0:
            logger.info(
                "\n🎉 All tests passed! Implementation is working correctly."
            )
            logger.info("\n🚀 Next steps:")
            logger.info("   1. Download the real MM-IMDb dataset")
            logger.info("   2. Update data paths in config.yaml")
            logger.info("   3. Run data exploration notebook")
            logger.info("   4. Start training models")
        else:
            logger.warning(
                f"\n⚠️  {failed} test(s) failed. Please review the errors above."
            )

    except Exception as e:
        logger.error(f"\n❌ Test failed with error: {e}")
        import traceback

        traceback.print_exc()
        return False

    return True


# ============================================================================
# Entry Point
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test ML project implementation"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--download-data",
        action="store_true",
        help="Download/create dataset",
    )

    args = parser.parse_args()

    success = run_all_tests(
        config_path=args.config, download_data=args.download_data
    )

    sys.exit(0 if success else 1)
