#!/bin/bash

# Example Commands for Low-Light Enhancement Project
# Copy and paste these commands to get started quickly

# ============================================================
# 1. DATASET PREPARATION
# ============================================================

# Prepare dataset from LOL only
python prepare_dataset.py \
    --lol_path /path/to/LOL_dataset \
    --output_dir ./dataset \
    --train_ratio 0.75 \
    --test_ratio 0.15 \
    --val_ratio 0.10

# Prepare dataset from LOL + custom images
python prepare_dataset.py \
    --lol_path /path/to/LOL_dataset \
    --custom_path /path/to/custom_images \
    --output_dir ./dataset \
    --train_ratio 0.75 \
    --test_ratio 0.15 \
    --val_ratio 0.10

# ============================================================
# 2. TRAINING
# ============================================================

# Quick training (small model, fast)
python train.py \
    --data_root ./dataset \
    --epochs 50 \
    --batch_size 8 \
    --lr 1e-4 \
    --image_size 256 \
    --base_channels 16 \
    --checkpoint_dir ./checkpoints \
    --log_dir ./logs

# Full training (best quality)
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

# Training with custom loss weights
python train.py \
    --data_root ./dataset \
    --epochs 100 \
    --batch_size 8 \
    --lambda1 1.0 \
    --lambda2 2.0 \
    --lambda3 0.05 \
    --lambda4 0.3

# Resume training from checkpoint
python train.py \
    --data_root ./dataset \
    --resume ./checkpoints/checkpoint_epoch_50.pth

# ============================================================
# 3. EVALUATION
# ============================================================

# Evaluate trained model
python evaluate.py \
    --data_root ./dataset \
    --checkpoint ./checkpoints/best_model.pth \
    --base_channels 32 \
    --image_size 256 \
    --save_results \
    --save_csv \
    --output_dir ./results

# Evaluate without saving images (faster)
python evaluate.py \
    --data_root ./dataset \
    --checkpoint ./checkpoints/best_model.pth \
    --save_csv

# ============================================================
# 4. BASELINE COMPARISON
# ============================================================

# Evaluate CLAHE baseline
python baseline.py \
    --data_root ./dataset \
    --method clahe \
    --clip_limit 2.0 \
    --tile_size 8 \
    --save_results \
    --save_csv \
    --output_dir ./baseline_results

# Evaluate histogram equalization baseline
python baseline.py \
    --data_root ./dataset \
    --method histogram \
    --save_results \
    --save_csv

# ============================================================
# 5. INFERENCE (ENHANCE YOUR OWN IMAGES)
# ============================================================

# Enhance single image
python inference.py \
    --input ./my_low_light_image.jpg \
    --output ./enhanced_images \
    --checkpoint ./checkpoints/best_model.pth

# Enhance batch of images
python inference.py \
    --input ./input_folder \
    --output ./enhanced_images \
    --checkpoint ./checkpoints/best_model.pth

# Enhance with comparison images
python inference.py \
    --input ./input_folder \
    --output ./enhanced_images \
    --checkpoint ./checkpoints/best_model.pth \
    --save_comparison

# Enhance with specific image size
python inference.py \
    --input ./input_folder \
    --output ./enhanced_images \
    --checkpoint ./checkpoints/best_model.pth \
    --image_size 512

# ============================================================
# 6. MONITORING
# ============================================================

# Start TensorBoard to monitor training
tensorboard --logdir ./logs

# Start TensorBoard on specific port
tensorboard --logdir ./logs --port 6007

# ============================================================
# 7. TESTING MODEL COMPONENTS
# ============================================================

# Test dataset loading
python dataset.py

# Test model architecture
python model.py

# Test loss functions
python losses.py

# Test metrics
python metrics.py

# Test utilities
python utils.py

# ============================================================
# 8. GPU-SPECIFIC COMMANDS
# ============================================================

# Check GPU availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}')"

# Train on specific GPU
CUDA_VISIBLE_DEVICES=0 python train.py --data_root ./dataset

# Train on multiple GPUs (if available)
# Note: Current implementation doesn't support multi-GPU, would need DataParallel

# ============================================================
# 9. COMMON WORKFLOWS
# ============================================================

# Complete workflow from scratch
echo "Step 1: Prepare dataset"
python prepare_dataset.py --lol_path /path/to/LOL --output_dir ./dataset

echo "Step 2: Train model"
python train.py --data_root ./dataset --epochs 100

echo "Step 3: Evaluate model"
python evaluate.py --data_root ./dataset --checkpoint ./checkpoints/best_model.pth --save_results

echo "Step 4: Evaluate baseline"
python baseline.py --data_root ./dataset --method clahe --save_results

echo "Step 5: Enhance new images"
python inference.py --input ./new_images --output ./enhanced --checkpoint ./checkpoints/best_model.pth

# ============================================================
# 10. CLEANUP
# ============================================================

# Remove checkpoints except best model
# find ./checkpoints -name "checkpoint_epoch_*.pth" -type f -delete

# Remove old logs
# rm -rf ./logs/old_runs

# Remove temporary files
# rm -rf ./temp ./tmp

echo "Example commands loaded! Copy and paste to use."

