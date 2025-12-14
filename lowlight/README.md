# Hybrid Low-Light Image Enhancement

This repository implements a hybrid approach for low-light image enhancement, combining **Retinex-based decomposition** (traditional method) with **curve-based brightness correction** (learning-based method). The implementation is based on our paper "Diving into Low-Light Image Enhancement and Deblurring".

## Overview

The proposed model features a **two-branch architecture**:
1. **Illumination Branch**: Uses Retinex theory to decompose images into illumination and reflectance components
2. **Curve Branch**: Learns pixel-wise nonlinear brightness adjustments
3. **Fusion Module**: Adaptively combines outputs from both branches

This hybrid approach maintains the physical interpretability of traditional methods while leveraging the flexibility of deep learning.

## Architecture

```
Input (Low-light Image)
    │
    ├─→ Shared Decoder (Feature Extraction)
    │       │
    │       ├─→ Illumination Branch (Retinex)
    │       │       └─→ x_retinex
    │       │
    │       └─→ Curve Branch (Learning-based)
    │               └─→ x_curve
    │
    └─→ Fusion Module
            └─→ Enhanced Image
```

## Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA (recommended for training)

Install dependencies:

```bash
pip install -r requirements.txt
```

## Dataset Preparation

### 1. LOL Dataset

Download the LOL (Low-Light) dataset from Kaggle: https://www.kaggle.com/datasets/soumikrakshit/lol-dataset/data

The dataset contains 500 pairs of low-light and normal-light images (485 for training, 15 for testing).

### 2. Self-Captured Image Pairs ⭐ **INCLUDED IN OUR TRAINING**

**Important**: Our model is trained on a **hybrid dataset** that combines:
- LOL public benchmark (437 training pairs)
- **Self-captured image pairs** (added to improve real-world generalization)

#### **Why Self-Captured Pairs:**
- ✅ Enhance model robustness to diverse lighting conditions
- ✅ Reduce dataset bias from single source
- ✅ Cover real-world scenarios not in LOL dataset
- ✅ Improve generalization performance

#### **How to Capture Your Own Pairs:**

**Method** (Same as described in our paper):
1. **Setup**: Place camera on tripod or stable surface
2. **Frame Scene**: Position camera to capture desired scene
3. **Low-Light Capture**: 
   - Turn off/dim lights to create low-light condition
   - Capture photo → save as `sceneX_low.png`
   - **DO NOT move camera!**
4. **Normal-Light Capture**:
   - Turn on all lights to brighten the scene
   - Capture photo → save as `sceneX_high.png`
5. **Verify**: Check both images are aligned with only lighting difference

**Equipment**:
- Any camera (smartphone or DSLR)
- Stable tripod or surface
- Indoor environment with controllable lighting

**Recommended Scenes**:
- Living rooms (lamp lighting variations)
- Bedrooms (curtains open/closed)
- Offices (desk lamp on/off)
- Kitchens (under-cabinet lighting)
- Hallways (different light levels)

### 3. Dataset Structure and Organization

Our final dataset structure integrates **both LOL and self-captured pairs**:

```
dataset/
├── train/
│   ├── low/      # Low-light images (LOL + Self-captured)
│   │   ├── 1.png              (LOL)
│   │   ├── 2.png              (LOL)
│   │   ├── ...                (LOL: 437 images)
│   │   ├── custom_001.png     (Self-captured)
│   │   ├── custom_002.png     (Self-captured)
│   │   └── ...                (Self-captured: X images)
│   │
│   └── high/     # Normal-light images (ground truth)
│       ├── 1.png              (LOL)
│       ├── 2.png              (LOL)
│       ├── ...
│       ├── custom_001.png     (Self-captured)
│       ├── custom_002.png     (Self-captured)
│       └── ...
│
├── val/
│   ├── low/      # Validation images (from LOL split)
│   └── high/
│
└── test/
    ├── low/      # Test images (LOL official test set: 15 images)
    └── high/
```

**Important Notes:**
- ✅ Image pairs **must have matching filenames** in `low/` and `high/` directories
- ✅ LOL images and self-captured images are **mixed together** in training
- ✅ DataLoader automatically loads and pairs images by filename
- ✅ Shuffling during training ensures even distribution
- ✅ Validation and test sets use LOL official splits for fair comparison

### 4. Download and Setup (Automated)

**Option A: One-Command Setup (Recommended)**

```bash
# Make sure you have set up Kaggle credentials first (see below)
bash download_and_setup.sh
```

This script will automatically:
- Download the LOL dataset from Kaggle
- Extract and organize it
- Create train/val/test splits
- Verify the dataset is ready

**Option B: Manual Setup**

1. **Setup Kaggle API credentials:**
```bash
# Install Kaggle CLI
pip install kaggle

# Get your API token from https://www.kaggle.com/settings
# Download kaggle.json and move it:
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

2. **Download and organize:**
```bash
# Download dataset
kaggle datasets download -d soumikrakshit/lol-dataset
unzip lol-dataset.zip -d lol_raw
rm lol-dataset.zip

# Organize into train/val/test
python organize_lol_dataset.py --input ./lol_raw --output ./dataset
```

**Option C: Manual Download (No Kaggle API)**

See `download_lol_dataset.md` for detailed instructions on manual download.

### 5. Add Custom Images (Optional)

If you want to add your own captured images:

```bash
python prepare_dataset.py \
    --lol_path ./dataset \
    --custom_path /path/to/custom/images \
    --output_dir ./dataset_combined \
    --train_ratio 0.75 \
    --test_ratio 0.15 \
    --val_ratio 0.10
```

## Training

### Basic Training

```bash
python train.py \
    --data_root ./dataset \
    --epochs 100 \
    --batch_size 8 \
    --lr 1e-4 \
    --image_size 256
```

### Advanced Training Options

```bash
python train.py \
    --data_root ./dataset \
    --epochs 100 \
    --batch_size 16 \
    --lr 1e-4 \
    --image_size 512 \
    --base_channels 32 \
    --lambda1 1.0 \
    --lambda2 1.0 \
    --lambda3 0.1 \
    --lambda4 0.5 \
    --checkpoint_dir ./checkpoints \
    --log_dir ./logs \
    --save_freq 5
```

### Training Parameters

- `--data_root`: Path to dataset directory
- `--epochs`: Number of training epochs (default: 100)
- `--batch_size`: Batch size (default: 8, adjust based on GPU memory)
- `--lr`: Learning rate (default: 1e-4)
- `--image_size`: Image size for training (default: 256)
- `--base_channels`: Base number of channels in model (default: 32)
- `--lambda1`: Weight for L1 loss (default: 1.0)
- `--lambda2`: Weight for SSIM loss (default: 1.0)
- `--lambda3`: Weight for Total Variation loss (default: 0.1)
- `--lambda4`: Weight for color consistency loss (default: 0.5)

### Resume Training

```bash
python train.py \
    --data_root ./dataset \
    --resume ./checkpoints/checkpoint_epoch_50.pth
```

### Monitor Training

Use TensorBoard to monitor training progress:

```bash
tensorboard --logdir ./logs
```

## Evaluation

### Evaluate Trained Model

```bash
python evaluate.py \
    --data_root ./dataset \
    --checkpoint ./checkpoints/best_model.pth \
    --save_results \
    --save_csv
```

This will compute:
- **PSNR** (Peak Signal-to-Noise Ratio)
- **SSIM** (Structural Similarity Index)
- **NIQE** (Natural Image Quality Evaluator)

Results will be saved to `./results/`.

### Evaluate Baseline (Histogram Equalization)

```bash
python baseline.py \
    --data_root ./dataset \
    --method clahe \
    --save_results \
    --save_csv
```

Available baseline methods:
- `histogram`: Traditional histogram equalization
- `clahe`: Contrast Limited Adaptive Histogram Equalization (recommended)

## Inference

Enhance your own low-light images:

```bash
python inference.py \
    --input ./input_images \
    --output ./enhanced_images \
    --checkpoint ./checkpoints/best_model.pth
```

## Results

### Quantitative Results

Example results on LOL test set:

| Method | PSNR (dB) | SSIM | NIQE |
|--------|-----------|------|------|
| Histogram Equalization | 15.2 | 0.65 | 5.8 |
| CLAHE | 16.5 | 0.71 | 5.2 |
| **Our Method** | **22.3** | **0.85** | **4.1** |

*(Note: These are example values. Actual results will vary based on your dataset and training)*

### Qualitative Results

Sample enhanced images will be saved during training in `./checkpoints/samples/`.

## Model Components

### 1. Shared Decoder
- Multi-layer CNN with skip connections
- Extracts low-level (edges, textures) and high-level (illumination) features
- Uses LeakyReLU activation and batch normalization

### 2. Illumination Branch (Retinex)
- Estimates illumination map: `L(x,y)`
- Computes reflectance: `R(x,y) = I_low(x,y) / (L(x,y) + ε)`
- Reconstructs enhanced image: `x_retinex = R(x,y) × L_enhanced(x,y)`

### 3. Curve Branch
- Three convolutional layers
- Predicts pixel-wise parameter α
- Applies curve transformation: `T(x) = x + α·x·(1-x)`

### 4. Fusion Module
- Learns adaptive weight map `w ∈ [0,1]`
- Final output: `x_enhanced = w·x_retinex + (1-w)·x_curve`

## Loss Function

The total loss combines multiple objectives:

```
L = λ1·||x_enhanced - x_gt||_1 
  + λ2·(1 - SSIM(x_enhanced, x_gt))
  + λ3·TV(I)
  + λ4·L_color
```

Where:
- **L1 loss**: Pixel-wise reconstruction accuracy
- **SSIM loss**: Structural similarity
- **TV loss**: Total Variation for illumination smoothness
- **Color loss**: Color consistency

## Project Structure

```
253_low_light_project/
├── README.md                  # This file
├── requirements.txt           # Python dependencies
├── model.py                   # Model architecture
├── losses.py                  # Loss functions
├── dataset.py                 # Dataset loader
├── train.py                   # Training script
├── evaluate.py                # Evaluation script
├── baseline.py                # Baseline methods
├── inference.py               # Inference script
├── metrics.py                 # Evaluation metrics
├── utils.py                   # Utility functions
├── prepare_dataset.py         # Dataset preparation script
├── dataset/                   # Dataset directory
│   ├── train/
│   ├── val/
│   └── test/
├── checkpoints/               # Model checkpoints
└── logs/                      # TensorBoard logs
```

## Citation

If you use this code in your research, please cite our paper:

```bibtex
@article{zheng2025lowlight,
  title={Diving into Low-Light Image Enhancement and Deblurring},
  author={Zheng, Hanxin and Shi, Tianai},
  year={2025}
}
```

## Tips for Best Results

1. **Dataset Quality**: Ensure your paired images are properly aligned
2. **Hyperparameters**: Tune λ1, λ2, λ3, λ4 based on your dataset
3. **Image Size**: Use 512×512 for better quality (requires more GPU memory)
4. **Batch Size**: Reduce if you encounter out-of-memory errors
5. **Training Time**: Expect ~2-4 hours on a single GPU for 100 epochs
6. **Data Augmentation**: Enabled by default (horizontal flip, rotation)

## Troubleshooting

### Out of Memory Error
- Reduce `--batch_size` (try 4 or 2)
- Reduce `--image_size` (try 128 or 256)
- Reduce `--base_channels` (try 16)

### Poor Results
- Check dataset quality and alignment
- Increase training epochs
- Tune loss weights (λ1, λ2, λ3, λ4)
- Try different learning rates

### Slow Training
- Enable mixed precision training (requires code modification)
- Use smaller image size
- Use fewer data augmentation techniques

## Future Work

- Incorporate unpaired datasets using unsupervised learning (EnlightenGAN, Zero-DCE)
- Add support for video enhancement
- Implement real-time inference optimization
- Add more baseline comparisons (RetinexNet, KinD, etc.)

## License

This project is for academic and research purposes.

## Acknowledgments

- LOL Dataset creators
- PyTorch community
- Research papers that inspired this work

## Contact

For questions or issues, please open an issue on the repository or contact the authors.

