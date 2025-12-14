# 📥 LOL Dataset Download Guide

This guide explains how to download and setup the LOL dataset from Kaggle.

## 🚀 Quick Start (Automated - Recommended)

### Prerequisites
1. A Kaggle account (free): https://www.kaggle.com
2. Python and pip installed

### Step-by-Step Instructions

#### 1. Setup Kaggle API Credentials

**a) Get your API token:**
- Go to https://www.kaggle.com/settings
- Scroll down to the "API" section
- Click "**Create New API Token**"
- A file named `kaggle.json` will be downloaded

**b) Install the token:**
```bash
# Create Kaggle directory
mkdir -p ~/.kaggle

# Move the downloaded file
mv ~/Downloads/kaggle.json ~/.kaggle/

# Set correct permissions
chmod 600 ~/.kaggle/kaggle.json
```

#### 2. Run the Automated Setup

```bash
cd /Users/guoguo/Desktop/253_low_light_project
bash download_and_setup.sh
```

**That's it!** The script will:
- ✓ Download LOL dataset (~150MB)
- ✓ Extract and organize files
- ✓ Create train/val/test splits
- ✓ Verify everything is ready

---

## 🔧 Manual Download (Alternative)

If the automated method doesn't work, you can download manually:

### Step 1: Download from Kaggle Website

1. Go to: https://www.kaggle.com/datasets/soumikrakshit/lol-dataset/data
2. Click the **Download** button (requires login)
3. Save `lol-dataset.zip` to your Downloads folder

### Step 2: Extract and Organize

```bash
cd /Users/guoguo/Desktop/253_low_light_project

# Move the zip file here
mv ~/Downloads/lol-dataset.zip .

# Extract
unzip lol-dataset.zip -d lol_raw

# Clean up
rm lol-dataset.zip

# Organize the dataset
python organize_lol_dataset.py --input ./lol_raw --output ./dataset
```

---

## 📊 What You'll Get

After setup, you'll have this structure:

```
/Users/guoguo/Desktop/253_low_light_project/dataset/
├── train/          # 436 pairs (~90% of training data)
│   ├── low/
│   └── high/
├── val/            # 49 pairs (~10% of training data)
│   ├── low/
│   └── high/
└── test/           # 15 pairs (official test set)
    ├── low/
    └── high/
```

**Total: 500 image pairs**
- Original LOL dataset has 485 training + 15 test images
- We split the 485 training images into 436 train + 49 validation

---

## ✅ Verify Setup

Run this command to verify everything is working:

```bash
python -c "from dataset import create_dataloaders; train_loader, val_loader, test_loader = create_dataloaders('./dataset', batch_size=2); print(f'✓ Train batches: {len(train_loader)}'); print(f'✓ Val batches: {len(val_loader)}'); print(f'✓ Test batches: {len(test_loader)}'); print('✅ Dataset ready!')"
```

Expected output:
```
Loaded 436 image pairs from /Users/guoguo/Desktop/253_low_light_project/dataset/train/low
Loaded 49 image pairs from /Users/guoguo/Desktop/253_low_light_project/dataset/val/low
Loaded 15 image pairs from /Users/guoguo/Desktop/253_low_light_project/dataset/test/low
✓ Train batches: 218
✓ Val batches: 25
✓ Test batches: 15
✅ Dataset ready!
```

---

## 🐛 Troubleshooting

### Problem: "kaggle: command not found"

**Solution:**
```bash
pip install kaggle
```

### Problem: "401 - Unauthorized"

**Solution:** Your Kaggle credentials are incorrect or missing.
- Delete old credentials: `rm ~/.kaggle/kaggle.json`
- Follow Step 1 again to download new credentials

### Problem: "403 - Forbidden"

**Solution:** You need to accept the dataset terms.
1. Visit: https://www.kaggle.com/datasets/soumikrakshit/lol-dataset
2. Click "Download" (you may not actually download, just click it)
3. Try the script again

### Problem: "Could not find LOL dataset structure"

**Solution:** The dataset structure may have changed. Check the contents:
```bash
ls -R lol_raw/
```

If you see `our485`, `eval15`, `low`, and `high` folders, manually specify:
```bash
python organize_lol_dataset.py --input ./lol_raw/LOL --output ./dataset
```

### Problem: "ImportError: No module named 'dataset'"

**Solution:** Make sure you're in the project directory:
```bash
cd /Users/guoguo/Desktop/253_low_light_project
```

---

## 📝 Next Steps After Download

Once your dataset is ready:

### 1. Test the Dataset Loader
```bash
python dataset.py
```

### 2. Start Training
```bash
python train.py --data_root ./dataset --epochs 100
```

### 3. Monitor Training
```bash
tensorboard --logdir ./logs
```

Open http://localhost:6006 in your browser.

### 4. Add Your Own Images (Optional)

If you want to add custom images to improve the model:

```bash
# Organize your images in this structure:
# my_images/
#   ├── low/
#   └── high/

python prepare_dataset.py \
    --lol_path ./dataset \
    --custom_path ./my_images \
    --output_dir ./dataset_combined
```

---

## 💡 Tips

1. **Internet Speed**: The download is ~150MB and should take 1-5 minutes depending on your connection.

2. **Storage**: Make sure you have at least 500MB free space.

3. **Kaggle Rate Limits**: If download fails, wait 5 minutes and try again.

4. **Keep Raw Data**: Don't delete `lol_raw/` folder until you're sure the organized dataset works.

5. **Backup**: After organizing, you can backup just the `dataset/` folder (~300MB).

---

## 📚 Additional Resources

- **Full Documentation**: See `README.md`
- **Quick Start**: See `QUICKSTART.md`
- **Example Commands**: See `example_commands.sh`
- **Paper Details**: See your paper document

---

## 🎉 Ready to Train!

Once you see "✅ Dataset ready!", you can proceed with training:

```bash
python train.py --data_root ./dataset --epochs 100 --batch_size 8
```

Good luck with your experiments! 🚀

