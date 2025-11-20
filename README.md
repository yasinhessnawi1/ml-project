# Multimodal Genre Classification - MM-IMDb

**Deep Learning Project for Multi-Label Movie Genre Prediction**

This project implements multimodal deep learning models for predicting movie genres from text (plot summaries) and images (movie posters) using the MM-IMDb dataset.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
  - [Windows](#windows-setup)
  - [macOS](#macos-setup)
  - [Linux](#linux-setup)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [Models](#models)
- [Documentation](#documentation)
- [Results](#results)
- [Contributing](#contributing)

---

## 🎯 Overview

This project explores various deep learning architectures for multimodal genre classification:

- **Text Models**: LSTM with Attention, DistilBERT
- **Vision Models**: ResNet-18/50, Custom CNN
- **Fusion Strategies**: Early Fusion, Late Fusion, Attention-Based Fusion
- **Task**: Multi-label classification (23 genres)
- **Dataset**: MM-IMDb (Movie Posters + Plot Summaries)

---

## ✨ Features

- ✅ **Multiple Model Architectures**: Text-only, Vision-only, and Multimodal fusion models
- ✅ **Flexible Configuration**: YAML-based configuration management
- ✅ **Preprocessing**: Automated text cleaning, tokenization, image augmentation
- ✅ **Training Infrastructure**: Modular trainer with checkpointing, early stopping, and logging
- ✅ **Evaluation Metrics**: F1-score (macro/micro/weighted), precision, recall, AUC-ROC
- ✅ **Experiment Tracking**: TensorBoard integration, detailed logging
- ✅ **Reproducibility**: Seed setting, deterministic training
---

## 📁 Project Structure

```
ml-project/
├── .docs/                          # Implementation documentation
│   ├── plan.md                     # Complete implementation plan
│   ├── 01_dataset_analysis.md     # Dataset analysis
│   ├── 02_architecture_decisions.md
│   ├── 03_preprocessing_strategy.md
│   ├── 04_training_strategy.md
│   ├── 05_experiments_log.md
│   └── 06_assumptions_considerations.md
├── data/                           # Data directory
│   ├── raw/                        # Original MM-IMDb dataset
│   ├── processed/                  # Preprocessed data
│   └── splits/                     # Train/val/test splits
├── models/                         # Saved models
│   ├── checkpoints/                # Training checkpoints
│   └── final/                      # Final trained models
├── notebooks/                      # Jupyter notebooks
│   ├── 01_data_exploration.ipynb
│   └── 02_results_analysis.ipynb
├── src/                            # Source code
│   ├── data/                       # Data processing
│   │   ├── dataset.py             # PyTorch Dataset classes
│   │   └── preprocessing.py       # Preprocessing utilities
│   ├── models/                     # Model architectures
│   │   ├── text_models.py         # LSTM, DistilBERT
│   │   ├── vision_models.py       # ResNet, Custom CNN
│   │   ├── fusion_models.py       # Multimodal fusion
│   │   └── baseline_models.py     # Classical ML baselines
│   ├── training/                   # Training infrastructure
│   │   ├── trainer.py             # Training loop
│   │   └── losses.py              # Loss functions
│   ├── evaluation/                 # Evaluation metrics
│   │   └── metrics.py
│   └── utils/                      # Utilities
│       ├── config.py              # Configuration management
│       └── visualization.py       # Plotting utilities
├── scripts/                        # Utility scripts
│   ├── download_data.sh           # Automated data download
│   ├── preprocess_data.py         # Data preprocessing
│   └── train.py                   # Training script
├── experiments/                    # Experiment configurations
│   └── configs/
├── results/                        # Results and outputs
│   ├── figures/                   # Plots and visualizations
│   └── tables/                    # Results tables
├── tests/                          # Unit tests
├── report/                         # LaTeX report
├── config.yaml                     # Main configuration file
├── requirements.txt                # Python dependencies
├── setup.py                        # Package installation
└── README.md                       # This file
```

---

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- (Optional) CUDA-capable GPU for faster training

### Windows Setup

#### 1. Create Virtual Environment

**Using venv (built-in):**
```bash
# Open Command Prompt or PowerShell
cd path\to\ml-project

# Create virtual environment
python -m venv venv

# Activate virtual environment
venv\Scripts\activate
```

**Using conda:**
```bash
# Open Anaconda Prompt
cd path\to\ml-project

# Create conda environment
conda create -n ml-project python=3.8 -y

# Activate environment
conda activate ml-project
```

#### 2. Install Dependencies

```bash
# Upgrade pip
python -m pip install --upgrade pip

# Install PyTorch (with CUDA support if available)
# For CUDA 11.8:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CPU only:
# pip install torch torchvision torchaudio

# Install other dependencies
pip install -r requirements.txt
```

#### 3. Install Project

```bash
pip install -e .
```

---

### macOS Setup

#### 1. Create Virtual Environment

**Using venv:**
```bash
# Open Terminal
cd /path/to/ml-project

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate
```

**Using conda:**
```bash
cd /path/to/ml-project

# Create conda environment
conda create -n ml-project python=3.8 -y

# Activate environment
conda activate ml-project
```

#### 2. Install Dependencies

```bash
# Upgrade pip
pip install --upgrade pip

# Install PyTorch
# For Apple Silicon (M1/M2):
pip install torch torchvision torchaudio

# For Intel Mac:
pip install torch torchvision torchaudio

# Install other dependencies
pip install -r requirements.txt
```

#### 3. Install Project

```bash
pip install -e .
```

---

### Linux Setup

#### 1. Create Virtual Environment

**Using venv:**
```bash
# Open Terminal
cd /path/to/ml-project

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate
```

**Using conda:**
```bash
cd /path/to/ml-project

# Create conda environment
conda create -n ml-project python=3.8 -y

# Activate environment
conda activate ml-project
```

#### 2. Install Dependencies

```bash
# Upgrade pip
pip install --upgrade pip

# Install PyTorch (with CUDA support)
# For CUDA 11.8:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1:
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# For CPU only:
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install other dependencies
pip install -r requirements.txt
```

#### 3. Install Project

```bash
pip install -e .
```

---

## ⚡ Quick Start

### Download Dataset

**Linux/macOS:**
```bash
./scripts/download_data.sh
```

**Windows (Command Prompt):**
```batch
scripts\download_data.bat
```

**Windows (PowerShell):**
```powershell
.\scripts\download_data.ps1
```

This will download and extract the MM-IMDb dataset (~3.7 GB download, ~15.6 GB extracted) to `data/raw/`.

**Note**: The dataset download may take several minutes depending on your internet connection.

### 2. Verify Installation

Run the test script:

```bash
python test_implementation.py --download-data
```

This will:
- Test all preprocessing functions
- Create dummy dataset (for testing)
- Test all model architectures
- Verify data loaders
- Check GPU availability

Expected output:
```
========================================
ML PROJECT IMPLEMENTATION TEST
========================================

✅ All tests passed! Implementation is working correctly.

📊 Results: 15/15 tests passed (0 failed)
```

### 3. Train Model

```bash
# Train text-only LSTM model
python scripts/train.py --model lstm --config config.yaml

# Train vision-only ResNet model
python scripts/train.py --model resnet --config config.yaml

# Train multimodal fusion model
python scripts/train.py --model early_fusion --config config.yaml
```

### 4. Evaluate Model

```bash
python scripts/evaluate.py --model path/to/model.pt --split test
```

---

## 📖 Usage

### Configuration

All hyperparameters and settings are in [config.yaml](config.yaml). Key sections:

```yaml
project:
  seed: 42              # Random seed for reproducibility
  device: cuda          # 'cuda', 'cpu', or 'mps' (Apple Silicon)

training:
  num_epochs: 50
  batch_size: 32
  learning_rate:
    from_scratch: 0.001
    fine_tune: 0.0001

preprocessing:
  text:
    max_length_lstm: 256
    max_length_bert: 512
  image:
    target_size: 224
    augmentation:
      enabled: true
```

### Training Models

#### Text-Only Models

```bash
# LSTM with Attention
python scripts/train.py --model lstm_text --num-epochs 50

# DistilBERT
python scripts/train.py --model distilbert --epochs 20
```

#### Vision-Only Models

```bash
# ResNet-18 (pretrained)
python scripts/train.py --model resnet18 --pretrained

# Custom CNN (from scratch)
python scripts/train.py --model custom_cnn
```

#### Multimodal Fusion Models

```bash
# Early Fusion
python scripts/train.py --model early_fusion

# Late Fusion
python scripts/train.py --model late_fusion

# Attention Fusion
python scripts/train.py --model attention_fusion
```

### Custom Training Script

```python
from src.utils.config import load_config, set_seed, setup_device
from src.models.fusion_models import EarlyFusionModel
from src.training.trainer import Trainer

# Load configuration
config = load_config('config.yaml')
set_seed(config['project']['seed'])
device = setup_device(config['project']['device'])

# Create model
model = EarlyFusionModel(...)

# Train
trainer = Trainer(model, config, device)
trainer.train(train_loader, val_loader)
```

### Jupyter Notebooks

Explore the data and results:

```bash
jupyter notebook notebooks/01_data_exploration.ipynb
```

---

## 🧠 Models

### Text Models

| Model | Architecture | Parameters | Description |
|-------|-------------|------------|-------------|
| **LSTM** | Bidirectional LSTM + Attention | ~10M | Custom LSTM with attention mechanism |
| **DistilBERT** | Pretrained Transformer | ~66M | Fine-tuned DistilBERT-base-uncased |

### Vision Models

| Model | Architecture | Parameters | Description |
|-------|-------------|------------|-------------|
| **ResNet-18** | Pretrained CNN | ~11M | ImageNet pretrained ResNet-18 |
| **ResNet-50** | Pretrained CNN | ~25M | ImageNet pretrained ResNet-50 |
| **Custom CNN** | 4-layer CNN | ~5M | Lightweight CNN trained from scratch |

### Fusion Models

| Model | Strategy | Description |
|-------|----------|-------------|
| **Early Fusion** | Concatenate embeddings | Combines text and image features before classification |
| **Late Fusion** | Weighted average of predictions | Trains separate models, combines predictions |
| **Attention Fusion** | Cross-attention | Text and image attend to each other |

---

## 📚 Documentation

Comprehensive documentation is available in the [`.docs/`](.docs/) directory:
(to be uploaded ...)
- **[Implementation Plan](.docs/plan.md)**: Complete 8-week implementation roadmap
- **[Dataset Analysis](.docs/01_dataset_analysis.md)**: Dataset statistics and EDA
- **[Architecture Decisions](.docs/02_architecture_decisions.md)**: Model design rationale
- **[Preprocessing Strategy](.docs/03_preprocessing_strategy.md)**: Data preprocessing pipeline
- **[Training Strategy](.docs/04_training_strategy.md)**: Training configuration and techniques
- **[Experiments Log](.docs/05_experiments_log.md)**: All experiments and results
- **[Assumptions & Considerations](.docs/06_assumptions_considerations.md)**: Edge cases and limitations

### API Documentation

All modules have comprehensive docstrings:

```python
from src.models.text_models import LSTMTextModel

help(LSTMTextModel)  # View full documentation
```

---

## 📊 Results

Results will be saved to the `results/` directory:

```
results/
├── figures/
│   ├── training_curves.png
│   ├── confusion_matrix.png
│   └── attention_visualization.png
└── tables/
    ├── model_comparison.csv
    └── per_genre_f1.csv
```

### Expected Performance (on MM-IMDb)

| Model | F1-Macro | F1-Micro | Accuracy |
|-------|----------|----------|----------|
| LSTM (text-only) | ~0.55 | ~0.60 | ~0.45 |
| DistilBERT (text-only) | ~0.62 | ~0.68 | ~0.52 |
| ResNet-18 (vision-only) | ~0.58 | ~0.64 | ~0.48 |
| **Early Fusion** | **~0.68** | **~0.74** | **~0.58** |
| Late Fusion | ~0.66 | ~0.72 | ~0.56 |
| Attention Fusion | ~0.67 | ~0.73 | ~0.57 |

*Note: These are estimated benchmarks. Actual results may vary.*

---

## 🧪 Testing

Run the test suite:

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_preprocessing.py

# With coverage
pytest --cov=src tests/
```

Run implementation tests:

```bash
# Test everything
python test_implementation.py --download-data

# Test without downloading data
python test_implementation.py
```

---

## 🔧 Troubleshooting

### Common Issues

#### 1. CUDA Out of Memory

```bash
# Reduce batch size in config.yaml
training:
  batch_size: 16  # or 8
```

#### 2. Module Not Found Errors

```bash
# Reinstall in editable mode
pip install -e .
```

#### 3. Download Script Fails

**Manual download**:
1. Download from: https://www.kaggle.com/datasets/johnarevalo/mmimdb
2. Extract to `data/raw/`

#### 4. Slow Training on CPU

```python
# In config.yaml, enable mixed precision
training:
  mixed_precision: true
```

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Code Style

- Follow PEP 8 guidelines
- Add docstrings to all functions/classes
- Include type hints
- Write unit tests for new features

---

## 📝 Citation

If you use this code in your research, please cite:

```bibtex
@misc{ml-project-2025,
  author = {Yasin Hessnawi, Anwar Debes},
  title = {Multimodal Genre Classification with Deep Learning},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/yasinhessnawi/ml-project}
}
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **MM-IMDb Dataset**: [Arevalo et al.](https://arxiv.org/abs/1701.06647)
- **PyTorch**: Deep learning framework
- **Hugging Face Transformers**: Pretrained models
- **IKT466 Course**: Introduction to Machine Learning

---

## 📧 Contact

**Yasin Hessnawi**
- GitHub: [@yasinhessnawi1](https://github.com/yasinhessnawi1)
- Email: yasinhessnawi@gmail.com

---

## 🗺️ Roadmap

- [x] Project structure and documentation
- [x] Data preprocessing pipeline
- [x] Text models (LSTM, DistilBERT)
- [x] Vision models (ResNet, Custom CNN)
- [x] Fusion models (Early, Late, Attention)
- [ ] Training infrastructure
- [ ] Evaluation metrics
- [ ] Baseline models (Optional)
- [ ] Experiment tracking
- [ ] Results analysis
- [ ] Report writing

---

**Last Updated**: November 2025
