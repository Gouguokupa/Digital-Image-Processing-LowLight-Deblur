# How to Download and Setup LOL Dataset

## Method 1: Using Kaggle API (Recommended - Fastest)

### Step 1: Install Kaggle API

```bash
pip install kaggle
```

### Step 2: Setup Kaggle Credentials

1. Go to [Kaggle](https://www.kaggle.com) and sign in
2. Go to Account settings: https://www.kaggle.com/settings
3. Scroll down to "API" section
4. Click "Create New API Token"
5. This will download `kaggle.json` to your computer

6. Move the file to the correct location:

**On macOS/Linux:**
```bash
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

**On Windows:**
```cmd
mkdir %USERPROFILE%\.kaggle
move %USERPROFILE%\Downloads\kaggle.json %USERPROFILE%\.kaggle\
```

### Step 3: Download the Dataset

```bash
# Navigate to your project directory
cd /Users/guoguo/Desktop/253_low_light_project

# Download the LOL dataset
kaggle datasets download -d soumikrakshit/lol-dataset

# Unzip the dataset
unzip lol-dataset.zip -d lol_raw

# Clean up zip file
rm lol-dataset.zip
```

### Step 4: Organize the Dataset

```bash
# Run the organization script
python organize_lol_dataset.py --input ./lol_raw --output ./dataset
```

---

## Method 2: Manual Download (If Kaggle API doesn't work)

### Step 1: Manual Download

1. Go to: https://www.kaggle.com/datasets/soumikrakshit/lol-dataset/data
2. Click the "Download" button (you need to be logged in)
3. Save `lol-dataset.zip` to your computer
4. Move it to your project directory: `/Users/guoguo/Desktop/253_low_light_project/`

### Step 2: Unzip and Organize

```bash
# Navigate to your project directory
cd /Users/guoguo/Desktop/253_low_light_project

# Unzip the dataset
unzip lol-dataset.zip -d lol_raw

# Run the organization script
python organize_lol_dataset.py --input ./lol_raw --output ./dataset
```

---

## Expected Dataset Structure After Organization

```
/Users/guoguo/Desktop/253_low_light_project/dataset/
├── train/
│   ├── low/
│   │   ├── image_001.png
│   │   ├── image_002.png
│   │   └── ...
│   └── high/
│       ├── image_001.png
│       ├── image_002.png
│       └── ...
├── val/
│   ├── low/
│   └── high/
└── test/
    ├── low/
    └── high/
```

---

## Troubleshooting

### Issue: "kaggle: command not found"

**Solution:**
```bash
pip install --user kaggle
# Or if using conda:
conda install -c conda-forge kaggle
```

### Issue: "401 - Unauthorized"

**Solution:** Your Kaggle API credentials are not set up correctly. Repeat Step 2.

### Issue: "403 - Forbidden" or "404 - Not Found"

**Solution:** 
1. Make sure you have accepted the dataset's terms on the Kaggle website
2. Visit: https://www.kaggle.com/datasets/soumikrakshit/lol-dataset
3. Click "Download" at least once (even if it doesn't download)

### Issue: Dataset structure doesn't match

**Solution:** The organization script (`organize_lol_dataset.py`) will automatically detect and fix the structure.

---

## Quick Start Commands (All in One)

```bash
# Install Kaggle API
pip install kaggle

# Setup credentials (after downloading kaggle.json from Kaggle website)
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json

# Download and organize
cd /Users/guoguo/Desktop/253_low_light_project
kaggle datasets download -d soumikrakshit/lol-dataset
unzip lol-dataset.zip -d lol_raw
rm lol-dataset.zip
python organize_lol_dataset.py --input ./lol_raw --output ./dataset

# Verify
python -c "from dataset import create_dataloaders; train_loader, val_loader, test_loader = create_dataloaders('./dataset', batch_size=2); print(f'✓ Train batches: {len(train_loader)}'); print(f'✓ Val batches: {len(val_loader)}'); print(f'✓ Test batches: {len(test_loader)}')"
```

---

## Next Steps

After organizing the dataset, you can:

1. **Add custom images** (optional):
   ```bash
   python prepare_dataset.py \
       --lol_path ./dataset \
       --custom_path ./my_custom_images \
       --output_dir ./dataset_combined
   ```

2. **Start training**:
   ```bash
   python train.py --data_root ./dataset --epochs 100
   ```

3. **Verify dataset**:
   ```bash
   python dataset.py  # Test the dataset loader
   ```

