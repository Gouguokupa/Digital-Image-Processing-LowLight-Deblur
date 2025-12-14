#!/bin/bash

# =============================================================================
# LOL Dataset Download and Setup Script
# =============================================================================

set -e  # Exit on error

echo "============================================================"
echo "LOL Dataset Download and Setup"
echo "============================================================"
echo ""

# Check if kaggle is installed
if ! command -v kaggle &> /dev/null; then
    echo "Kaggle CLI not found. Installing..."
    pip install kaggle
fi

# Check if kaggle credentials exist
if [ ! -f ~/.kaggle/kaggle.json ]; then
    echo ""
    echo "❌ Kaggle credentials not found!"
    echo ""
    echo "Please follow these steps:"
    echo "1. Go to https://www.kaggle.com/settings"
    echo "2. Scroll to 'API' section"
    echo "3. Click 'Create New API Token'"
    echo "4. Move the downloaded kaggle.json to ~/.kaggle/"
    echo ""
    echo "Run these commands:"
    echo "  mkdir -p ~/.kaggle"
    echo "  mv ~/Downloads/kaggle.json ~/.kaggle/"
    echo "  chmod 600 ~/.kaggle/kaggle.json"
    echo ""
    echo "Then run this script again."
    exit 1
fi

echo "✓ Kaggle CLI found"
echo "✓ Kaggle credentials found"
echo ""

# Navigate to project directory
cd "$(dirname "$0")"
PROJECT_DIR=$(pwd)

echo "Project directory: $PROJECT_DIR"
echo ""

# Download dataset
echo "============================================================"
echo "Step 1: Downloading LOL dataset from Kaggle..."
echo "============================================================"
echo ""

if [ -d "lol_raw" ]; then
    echo "⚠️  lol_raw directory already exists. Skipping download."
else
    kaggle datasets download -d soumikrakshit/lol-dataset
    
    echo ""
    echo "Extracting dataset..."
    unzip -q lol-dataset.zip -d lol_raw
    rm lol-dataset.zip
    
    echo "✓ Download complete"
fi

echo ""

# Organize dataset
echo "============================================================"
echo "Step 2: Organizing dataset..."
echo "============================================================"
echo ""

python organize_lol_dataset.py --input ./lol_raw --output ./dataset

echo ""

# Verify dataset
echo "============================================================"
echo "Step 3: Verifying dataset..."
echo "============================================================"
echo ""

python -c "
from dataset import create_dataloaders
try:
    train_loader, val_loader, test_loader = create_dataloaders('./dataset', batch_size=2, num_workers=0)
    print(f'✅ Dataset loaded successfully!')
    print(f'   - Training batches: {len(train_loader)}')
    print(f'   - Validation batches: {len(val_loader)}')
    print(f'   - Test batches: {len(test_loader)}')
    print('')
    print('🎉 Setup complete! You can now start training.')
except Exception as e:
    print(f'❌ Error loading dataset: {e}')
    exit(1)
"

echo ""
echo "============================================================"
echo "Next Steps:"
echo "============================================================"
echo ""
echo "1. Start training:"
echo "   python train.py --data_root ./dataset --epochs 100"
echo ""
echo "2. Monitor training:"
echo "   tensorboard --logdir ./logs"
echo ""
echo "3. Evaluate model:"
echo "   python evaluate.py --data_root ./dataset --checkpoint ./checkpoints/best_model.pth"
echo ""

