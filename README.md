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
├── data/                           # Data directory
│   ├── raw/                        # Original MM-IMDb dataset
│   ├── processed/                  # Preprocessed data
│   └── splits/                     # Train/val/test splits
├── checkpoints/                         # Saved models
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
│   ├── train.py                   # Training script
|   ├──...
├── experiments/                    # Experiment configurations
│   └── configs/
├── results/                        # Results and outputs
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

### API Documentation

All modules have docstrings:

```python
from src.models.text_models import LSTMTextModel

help(LSTMTextModel)  # View full documentation
```

---

## 📊 Results

Results will be saved to the `results/` directory:

```
results/
├── training_curves.png
├── confusion_matrix.png
└── attention_visualization.png
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

## 📊 Results

**Project Status**: ✅ **COMPLETE** - All experiments finished (November 15, 2025)

### Final Model Performance (Test Set)

| Rank | Model | F1-Macro | F1-Micro | ROC-AUC | Subset Acc. | Architecture |
|------|-------|----------|----------|---------|-------------|--------------|
| 1 | **Attention Fusion** | **59.79%** | 65.90% | **90.61%** | 17.97% | BERT + ResNet-18 + Cross-Attention |
| 2 | **Late Fusion** | **59.43%** | 65.94% | 89.78% | **18.18%** | BERT + ResNet-18 + Learned Weighted Avg |
| 3 | **Early Fusion** | **58.47%** | 64.82% | 88.99% | 16.99% | BERT + ResNet-18 + Concatenation |
| 4 | **BERT Text** | **57.01%** | 64.74% | 88.38% | 18.46% | DistilBERT-base-uncased |
| 5 | **LSTM Text** | **43.05%** | 53.39% | 82.70% | 9.40% | BiLSTM + Attention |
| 6 | **ResNet Vision** | **29.73%** | 38.43% | 73.29% | 1.29% | ResNet-18 (pretrained) |
| 7 | **CNN Vision** | **24.17%** | 29.85% | 68.14% | 0.03% | 4-Layer CNN (from scratch) |

### Key Findings

#### 1. **Multimodal Fusion Works** ✅
- All three fusion strategies outperform unimodal approaches
- **Best improvement**: +2.78 percentage points over best text-only model (BERT)
- Consistent gains across all fusion strategies (Early, Late, Attention)

#### 2. **Attention Fusion Wins** 🏆
- **Best F1-Macro**: 59.79% (primary metric)
- **Best ROC-AUC**: 90.61% (excellent ranking ability)
- Cross-attention mechanism enables dynamic multimodal feature interaction
- Outperforms simple concatenation (Early) by +1.32%

#### 3. **Text Dominates, But Vision Adds Value** 📝
- **BERT text-only**: 57.01% F1 (very competitive)
- **ResNet vision-only**: 29.73% F1 (insufficient alone)
- **Ratio**: Text is **1.92x more informative** than vision
- Plot summaries contain definitive genre information; posters provide visual refinement

#### 4. **Transfer Learning is Essential** 🚀
- **BERT vs LSTM**: +13.96 percentage points (32.4% relative improvement)
- **ResNet vs CNN**: +5.56 percentage points (23.0% relative improvement)
- Pretrained models critical for both text and vision modalities

#### 5. **Multi-Label Classification is Challenging** 🎯
- **Best Subset Accuracy**: Only 18.18% (Late Fusion)
- Predicting exact genre combinations much harder than individual genres
- 23 genres → 2²³ = 8.4M possible combinations
- High F1-Macro (59.79%) but low subset accuracy indicates partial correctness

### Per-Genre Performance (Attention Fusion - Best Model)

**Top Performing Genres** (F1 > 70%):
- 🥇 **Adventure**: 78.90% F1 - Strong visual cues (landscapes), distinctive plots
- 🥈 **Drama**: 78.67% F1 - Largest class (2130 samples), well-represented
- 🥉 **Sport**: 73.64% F1 - Distinctive visuals (stadiums), specific vocabulary
- **Western**: 71.66% F1 - Iconic visual identity, era-specific language
- **Crime**: 71.02% F1 - Narrative structure markers
- **Mystery**: 70.37% F1 - Plot-driven, clear textual cues

**Challenging Genres** (F1 < 50%):
- **Short**: 33.00% F1 (50 samples) - Format label, not content-based genre
- **Musical**: 36.91% F1 (121 samples) - Visual cues not always on posters
- **Film-Noir**: 41.32% F1 (69 samples) - Small class, vintage aesthetic
- **History**: 42.25% F1 (147 samples) - Overlaps with War, Biography, Drama

### Model Comparisons

**Fusion Strategy Comparison**:
```
Attention Fusion:  59.79% F1 │████████████████████░│ Cross-attention, dynamic interaction
Late Fusion:       59.43% F1 │████████████████████░│ Learned weighting, modular
Early Fusion:      58.47% F1 │███████████████████░░│ Concatenation, simple
```

**Modality Contribution**:
```
Text (BERT):       57.01% F1 │███████████████████░░│ 70-80% of predictive power
Vision (ResNet):   29.73% F1 │██████████░░░░░░░░░░░│ 20-30% complementary refinement
```

### Reproducing Best Model

```bash
# Train attention fusion model (best performance)
python scripts/train.py --model attention_fusion --config config.yaml --num-epochs 50

# Evaluate on test set
python scripts/evaluate.py --checkpoint checkpoints/attention_fusion/best.pth --config config.yaml --split test

# Expected results:
# F1-Macro: 59.79%
# ROC-AUC: 90.61%
# Training time: ~4 hours (GPU)
```

### Research Contributions

1. **Fusion Comparison**: First systematic comparison of Early, Late, and Attention fusion on MM-IMDb with modern pretrained models (BERT, ResNet)

2. **Modality Imbalance Analysis**: Demonstrated that fusion improves performance even when one modality dominates (text 1.92x better than vision)

3. **Architecture Insights**: Showed that simple Late Fusion is surprisingly competitive (99.4% of Attention Fusion performance)

4. **Loss Function Guidance**: Documented architecture-dependent loss function effectiveness (Focal Loss for LSTM, Weighted BCE for BERT)

---

## 📝 Citation

If you want to use this code in your research, and wants to cite (no need):

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

This project is not a big deal but its licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

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

**Anwar Debes**
- Github: [@AnwarDebes](https://github.com/AnwarDebes)

---

## 🗺️ Roadmap followed

- [x] Project structure
- [x] Data preprocessing pipeline
- [x] Text models (LSTM, DistilBERT)
- [x] Vision models (ResNet, Custom CNN)
- [x] Fusion models (Early, Late, Attention)
- [x] Training infrastructure
- [x] Evaluation metrics
- [x] Experiment tracking (TensorBoard)
- [x] Results analysis and visualization
- [x] All experiments completed
- [x] Documentation for thesis writing
- [x] Final thesis report writing

