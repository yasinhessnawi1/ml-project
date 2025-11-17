#!/bin/bash
# Automated MM-IMDb Dataset Download Script
# This script downloads and extracts the MM-IMDb dataset

set -e  # Exit on error

echo "=========================================="
echo "MM-IMDb Dataset Download Script"
echo "=========================================="

# Configuration
DATA_DIR="data/raw"
DOWNLOAD_URL="https://www.kaggle.com/api/v1/datasets/download/johnarevalo/mmimdb"
ZIP_FILE="mmimdb.zip"

# Create data directory
echo "Creating data directory: $DATA_DIR"
mkdir -p "$DATA_DIR"

# Check if data already exists
if [ -d "$DATA_DIR/dataset" ]; then
    echo "⚠️  Data directory already exists: $DATA_DIR/dataset"
    read -p "Do you want to re-download and overwrite? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Skipping download. Exiting."
        exit 0
    fi
    echo "Removing existing data..."
    rm -rf "$DATA_DIR/dataset"
fi

# Download dataset
echo ""
echo "📥 Downloading MM-IMDb dataset..."
echo "This may take several minutes depending on your connection..."

if command -v curl &> /dev/null; then
    curl -L -o "$DATA_DIR/$ZIP_FILE" "$DOWNLOAD_URL"
elif command -v wget &> /dev/null; then
    wget -O "$DATA_DIR/$ZIP_FILE" "$DOWNLOAD_URL"
else
    echo "❌ Error: Neither curl nor wget found. Please install one of them."
    exit 1
fi

# Check if download was successful
if [ ! -f "$DATA_DIR/$ZIP_FILE" ]; then
    echo "❌ Error: Download failed. File not found: $DATA_DIR/$ZIP_FILE"
    exit 1
fi

echo "✅ Download complete!"

# Extract dataset
echo ""
echo "📦 Extracting dataset..."

if command -v unzip &> /dev/null; then
    unzip -q "$DATA_DIR/$ZIP_FILE" -d "$DATA_DIR"
else
    echo "❌ Error: unzip command not found. Please install unzip."
    exit 1
fi

echo "✅ Extraction complete!"

# Clean up zip file
echo ""
echo "🧹 Cleaning up..."
rm "$DATA_DIR/$ZIP_FILE"

# Verify extraction
if [ -d "$DATA_DIR/dataset" ]; then
    echo ""
    echo "✅ Dataset successfully downloaded and extracted!"
    echo "📊 Dataset location: $DATA_DIR/dataset"

    # Count files
    num_movies=$(find "$DATA_DIR/dataset" -maxdepth 1 -type d | wc -l)
    echo "📁 Number of movie directories: $((num_movies - 1))"

else
    echo "⚠️  Warning: Expected directory 'dataset' not found."
    echo "Contents of $DATA_DIR:"
    ls -la "$DATA_DIR"
fi

echo ""
echo "=========================================="
echo "Next steps:"
echo "1. Run data preprocessing: python scripts/preprocess_data.py"
echo "2. Or run the test script: python test_implementation.py --download-data"
echo "=========================================="
