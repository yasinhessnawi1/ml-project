# Scripts Directory

This directory contains executable scripts for data preprocessing, training, and evaluation.

## Scripts Overview

### 1. Download Scripts
Downloads the MM-IMDb dataset from Kaggle.

**Linux/macOS (`download_data.sh`):**
```bash
./scripts/download_data.sh
```

**Windows Command Prompt (`download_data.bat`):**
```batch
scripts\download_data.bat
```

**Windows PowerShell (`download_data.ps1`):**
```powershell
.\scripts\download_data.ps1
```

**Output**: Downloads and extracts data to `data/raw/`
- `multimodal_imdb.hdf5` (15.6 GB) - Contains images and text sequences
- `metadata.npy` - Contains vocabulary and metadata

**Features**:
- Automatic download with progress indicator
- Auto-extraction of compressed files
- Verification of downloaded files
- Cross-platform compatibility
- Detailed error messages

### 2. `preprocess_data.py`
Processes raw HDF5 data into organized train/val/test splits.

```bash
# Basic usage
python scripts/preprocess_data.py

# Advanced usage
python scripts/preprocess_data.py \
    --input-dir data/raw \
    --output-dir data/processed \
    --train-ratio 0.7 \
    --val-ratio 0.15 \
    --test-ratio 0.15 \
    --seed 42 \
    --min-text-length 50

# Quick test with limited samples
python scripts/preprocess_data.py --max-samples 1000

# Text-only preprocessing (skip images to save space)
python scripts/preprocess_data.py --skip-images
```

**Output**: Organized dataset in `data/processed/`
```
data/processed/
├── train/
│   ├── tt0000001/
│   │   ├── plot.txt
│   │   ├── poster.jpg
│   │   └── metadata.json
│   └── ...
├── val/
├── test/
├── dataset_statistics.json
└── genre_mapping.json
```

**Arguments**:
- `--input-dir`: Directory with raw data (default: `data/raw`)
- `--output-dir`: Directory for processed data (default: `data/processed`)
- `--train-ratio`: Training set ratio (default: 0.7)
- `--val-ratio`: Validation set ratio (default: 0.15)
- `--test-ratio`: Test set ratio (default: 0.15)
- `--seed`: Random seed (default: 42)
- `--max-samples`: Limit processing for testing (default: None)
- `--min-text-length`: Minimum plot length in chars (default: 50)
- `--skip-images`: Skip saving images for text-only experiments

### 3. `train.py`
Main training script for all model architectures.

```bash
# Train early fusion model
python scripts/train.py --config config.yaml --model early_fusion

# Train BERT text-only model
python scripts/train.py --config config.yaml --model bert_text

# Train with custom settings
python scripts/train.py \
    --config config.yaml \
    --model attention_fusion \
    --num-epochs 30 \
    --batch-size 16 \
    --device cuda

# Resume from checkpoint
python scripts/train.py \
    --config config.yaml \
    --model early_fusion \
    --resume checkpoints/early_fusion/last.pth
```

**Model Types**:
- `early_fusion`: Early fusion multimodal model
- `late_fusion`: Late fusion multimodal model
- `attention_fusion`: Attention-based fusion model
- `lstm_text`: LSTM text-only model
- `bert_text`: DistilBERT text-only model
- `resnet_vision`: ResNet vision-only model
- `cnn_vision`: Custom CNN vision-only model

**Output**:
- Checkpoints saved to `checkpoints/{model_type}/`
- Results saved to `results/{model_type}/{timestamp}/`
- TensorBoard logs in results directory

### 4. `evaluate.py`
Comprehensive model evaluation script.

```bash
# Evaluate on test set
python scripts/evaluate.py \
    --checkpoint checkpoints/early_fusion/best.pth \
    --config config.yaml

# Evaluate on validation set
python scripts/evaluate.py \
    --checkpoint checkpoints/bert_text/best.pth \
    --split val

# Custom threshold
python scripts/evaluate.py \
    --checkpoint checkpoints/attention_fusion/best.pth \
    --threshold 0.6
```

**Output**:
- Comprehensive metrics (F1, precision, recall, ROC-AUC, etc.)
- Visualizations (confusion matrices, ROC curves, PR curves)
- Classification report
- Results saved to `evaluation/{model_type}/{split}_{timestamp}/`

## Complete Workflow

### Step 1: Download Data
```bash
./scripts/download_data.sh
```

### Step 2: Preprocess Data
```bash
# Process all data
python scripts/preprocess_data.py

# Or quick test with 1000 samples
python scripts/preprocess_data.py --max-samples 1000
```

### Step 3: Train Models
```bash
# Train multiple models in sequence
python scripts/train.py --config config.yaml --model bert_text
python scripts/train.py --config config.yaml --model resnet_vision
python scripts/train.py --config config.yaml --model early_fusion
```

### Step 4: Evaluate Models
```bash
# Evaluate each trained model
python scripts/evaluate.py --checkpoint checkpoints/bert_text/best.pth --config config.yaml
python scripts/evaluate.py --checkpoint checkpoints/resnet_vision/best.pth --config config.yaml
python scripts/evaluate.py --checkpoint checkpoints/early_fusion/best.pth --config config.yaml
```

## Quick Start Example

```bash
# Complete pipeline for early fusion model
./scripts/download_data.sh                                    # Download data
python scripts/preprocess_data.py --max-samples 5000         # Preprocess (5K samples for testing)
python scripts/train.py --config config.yaml --model early_fusion --num-epochs 10  # Train
python scripts/evaluate.py --checkpoint checkpoints/early_fusion/best.pth --config config.yaml  # Evaluate
```

## Troubleshooting

### Issue: Out of memory during preprocessing
**Solution**: Use `--max-samples` to limit dataset size or `--skip-images` for text-only

### Issue: CUDA out of memory during training
**Solution**: Reduce `--batch-size` or use CPU with `--device cpu`

### Issue: Missing dependencies
**Solution**: Install requirements: `pip install -r requirements.txt`

### Issue: Data not found
**Solution**: Ensure preprocessing completed successfully and check `config.yaml` paths

## Performance Tips

1. **Use GPU**: Training on GPU is 10-50x faster than CPU
2. **Batch Size**: Larger batches = faster training but more memory
3. **Mixed Precision**: Enable in config for 2x speedup on modern GPUs
4. **Data Workers**: Increase `num_workers` in config for faster data loading
5. **Start Small**: Use `--max-samples` to test pipeline before full training

## Monitoring Training

### TensorBoard
```bash
# Launch TensorBoard to monitor training
tensorboard --logdir results/
```

Then open http://localhost:6006 in your browser.

### Watch GPU Usage
```bash
# Monitor GPU memory and utilization
watch -n 1 nvidia-smi
```

## Next Steps

After running the scripts:
1. Review training curves in TensorBoard
2. Compare metrics across different models
3. Analyze per-class performance in evaluation results
4. Experiment with hyperparameters in `config.yaml`
