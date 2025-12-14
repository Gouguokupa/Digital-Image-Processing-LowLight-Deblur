# Quick Start Guide

This guide will help you get started with the low-light enhancement project quickly.

## Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

## Step 2: Download and Prepare Dataset

### Option A: Use LOL Dataset Only

1. Download the LOL dataset from [official source]
2. Organize your LOL dataset:
```
LOL_dataset/
├── low/
│   ├── 1.png
│   ├── 2.png
│   └── ...
└── high/
    ├── 1.png
    ├── 2.png
    └── ...
```

3. Prepare the dataset:
```bash
python prepare_dataset.py \
    --lol_path /path/to/LOL_dataset \
    --output_dir ./dataset
```

### Option B: Use LOL + Custom Images

1. Prepare your custom images in the same structure:
```
my_images/
├── low/
│   ├── img1.png
│   └── ...
└── high/
    ├── img1.png
    └── ...
```

2. Prepare the combined dataset:
```bash
python prepare_dataset.py \
    --lol_path /path/to/LOL_dataset \
    --custom_path /path/to/my_images \
    --output_dir ./dataset
```

## Step 3: Train the Model

### Quick Training (Small Model, Fast)

```bash
python train.py \
    --data_root ./dataset \
    --epochs 50 \
    --batch_size 8 \
    --image_size 256 \
    --base_channels 16
```

**Training time**: ~1-2 hours on GPU

### Full Training (Best Quality)

```bash
python train.py \
    --data_root ./dataset \
    --epochs 100 \
    --batch_size 16 \
    --image_size 512 \
    --base_channels 32
```

**Training time**: ~3-5 hours on GPU

### Monitor Training

Open a new terminal and run:
```bash
tensorboard --logdir ./logs
```

Then open http://localhost:6006 in your browser.

## Step 4: Evaluate the Model

### Evaluate on Test Set

```bash
python evaluate.py \
    --data_root ./dataset \
    --checkpoint ./checkpoints/best_model.pth \
    --save_results \
    --save_csv
```

### Evaluate Baseline

```bash
python baseline.py \
    --data_root ./dataset \
    --method clahe \
    --save_results \
    --save_csv
```

## Step 5: Enhance Your Own Images

### Single Image

```bash
python inference.py \
    --input ./my_image.jpg \
    --output ./enhanced_images \
    --checkpoint ./checkpoints/best_model.pth
```

### Batch Processing

```bash
python inference.py \
    --input ./input_folder \
    --output ./enhanced_images \
    --checkpoint ./checkpoints/best_model.pth \
    --save_comparison
```

## Common Issues and Solutions

### Issue 1: Out of Memory Error

**Solution**: Reduce batch size and image size
```bash
python train.py \
    --data_root ./dataset \
    --batch_size 4 \
    --image_size 256
```

### Issue 2: No GPU Available

**Solution**: Training will automatically use CPU, but it will be slower. Consider using smaller model:
```bash
python train.py \
    --data_root ./dataset \
    --batch_size 2 \
    --base_channels 16 \
    --image_size 128
```

### Issue 3: Poor Results After Training

**Solutions**:
1. Train for more epochs (try 150-200)
2. Tune loss weights:
```bash
python train.py \
    --data_root ./dataset \
    --lambda1 1.0 \
    --lambda2 2.0 \
    --lambda3 0.05 \
    --lambda4 0.3
```
3. Check your dataset quality (images should be properly paired)

## Expected Results

After training, you should see:
- Training loss decreasing steadily
- Validation SSIM increasing to ~0.80-0.90
- PSNR improving to ~20-25 dB
- Enhanced images with better brightness and details

## Next Steps

1. Experiment with different hyperparameters
2. Collect more training data
3. Compare with baseline methods
4. Try different image sizes for inference

## Resources

- Full documentation: See README.md
- Model architecture: See model.py
- Loss functions: See losses.py

## Tips for Best Results

1. **Good Data Quality**: Ensure your image pairs are well-aligned
2. **Sufficient Training**: Don't stop too early, let the model converge
3. **Proper Evaluation**: Use the validation set to select the best model
4. **Hyperparameter Tuning**: Experiment with different loss weights
5. **Data Augmentation**: Already enabled by default, helps generalization

## Getting Help

If you encounter issues:
1. Check the error message carefully
2. Review this quick start guide
3. Check README.md for more details
4. Ensure all dependencies are installed correctly
5. Verify your dataset structure matches the expected format

Happy enhancing! 🌟

