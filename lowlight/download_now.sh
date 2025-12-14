#!/bin/bash

# Quick download script for LOL dataset
# Run this after setting up kaggle.json

set -e  # Exit on error

echo "============================================================"
echo "Downloading LOL Dataset"
echo "============================================================"
echo ""

# Check if we're in the right directory
if [ ! -f "model.py" ]; then
    echo "❌ Error: Please run this from the project directory"
    echo "   cd /Users/guoguo/Desktop/253_low_light_project"
    exit 1
fi

# Check if kaggle credentials exist
if [ ! -f ~/.kaggle/kaggle.json ]; then
    echo "❌ Error: Kaggle credentials not found!"
    echo ""
    echo "Please create ~/.kaggle/kaggle.json with:"
    echo '{"username":"gouguokupa","key":"KGAT_54b4e1408a75f27c7bee1fefcd9d0ee2"}'
    echo ""
    echo "Run these commands:"
    echo "  mkdir -p ~/.kaggle"
    echo "  cat > ~/.kaggle/kaggle.json << 'EOF'"
    echo '  {"username":"gouguokupa","key":"KGAT_54b4e1408a75f27c7bee1fefcd9d0ee2"}'
    echo "  EOF"
    echo "  chmod 600 ~/.kaggle/kaggle.json"
    exit 1
fi

echo "✓ Kaggle credentials found"
echo ""

# Download dataset
echo "============================================================"
echo "Step 1: Downloading from Kaggle (this may take 2-5 minutes)..."
echo "============================================================"
echo ""

if [ -f "lol-dataset.zip" ]; then
    echo "⚠️  lol-dataset.zip already exists, skipping download"
else
    kaggle datasets download -d soumikrakshit/lol-dataset
    echo ""
    echo "✓ Download complete"
fi

echo ""

# Extract
echo "============================================================"
echo "Step 2: Extracting dataset..."
echo "============================================================"
echo ""

if [ -d "lol_raw" ]; then
    echo "⚠️  lol_raw directory already exists, skipping extraction"
else
    unzip -q lol-dataset.zip -d lol_raw
    echo "✓ Extraction complete"
fi

# Clean up zip
if [ -f "lol-dataset.zip" ]; then
    rm lol-dataset.zip
    echo "✓ Cleaned up zip file"
fi

echo ""

# Organize
echo "============================================================"
echo "Step 3: Organizing dataset into train/val/test..."
echo "============================================================"
echo ""

python organize_lol_dataset.py --input ./lol_raw --output ./dataset

echo ""

# Verify
echo "============================================================"
echo "Step 4: Verifying dataset..."
echo "============================================================"
echo ""

python -c "
from dataset import create_dataloaders
try:
    train_loader, val_loader, test_loader = create_dataloaders('./dataset', batch_size=2, num_workers=0)
    print('✅ Dataset loaded successfully!')
    print(f'')
    print(f'📊 Dataset Summary:')
    print(f'   Training:   {len(train_loader.dataset)} images')
    print(f'   Validation: {len(val_loader.dataset)} images')
    print(f'   Test:       {len(test_loader.dataset)} images')
    print(f'   Total:      {len(train_loader.dataset) + len(val_loader.dataset) + len(test_loader.dataset)} image pairs')
    print(f'')
    print('🎉 Setup complete! Ready to train.')
except Exception as e:
    print(f'❌ Error: {e}')
    exit(1)
"

echo ""
echo "============================================================"
echo "🎯 Next Steps:"
echo "============================================================"
echo ""
echo "Start training with:"
echo "  python train.py --data_root ./dataset --epochs 100 --batch_size 8"
echo ""
echo "Monitor training with:"
echo "  tensorboard --logdir ./logs"
echo ""
echo "============================================================"

