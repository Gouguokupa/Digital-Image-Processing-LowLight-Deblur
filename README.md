Image Deblurring with DeblurGAN-v2

This repository contains a PyTorch implementation of image deblurring using the DeblurGAN-v2 architecture. The project focuses on motion blur restoration and compares learning-based deblurring with a traditional Wiener filter baseline.

The main objective is to study the strengths and limitations of learning-based image deblurring on both public datasets and self-captured images.

1. Overview

In this project, DeblurGAN-v2 is used as the primary learning-based deblurring model. A Wiener filter is included as a classical baseline for comparison. Although a hybrid two-stage pipeline was explored, experiments showed that Wiener pre-processing does not consistently improve results and may introduce artifacts. Therefore, DeblurGAN-v2 is used as the final model.

2.Requirements

- Python 3.8+
- PyTorch 2.0+
- OpenCV
- numpy, scipy, scikit-image
- matplotlib, Pillow

3. Dataset

GoPro Dataset:
- Placed under Dataset/Gopro/ (To simplify, the dataset is not included in the repo)
- Downloaded from:
  https://www.kaggle.com/datasets/jishnuparayilshibu/a-curated-list-of-image-deblurring-datasets

Self-Captured Dataset:
Dataset/Self_captured/
  ├── Sharp/
  └── Blur/

4. Training

To speed up the overall training, this code is implemented on Colab A100GPU.

Model definitions and training loops are implemented in:
- DeblurGAN_Final.py
- DeblurGAN_Midpoint.py

The training could be conducted using the following code in Colab after uploading the file: 
!python DeblurGAN_Final.py \
  --train_blur gopro_full/train/blur \
  --train_sharp gopro_full/train/sharp \
  --val_blur gopro_full/val/blur \
  --val_sharp gopro_full/val/sharp \
  --epochs 100 \
  --batch_size 4 \
  --save_dir ./checkpoints_complete \
  --use_amp

Training settings and hyperparameters can be adjusted directly in these files. There is one file named best_model.pth, which is a saved best-performing checkpoint through training.

5. Evaluation

The Demo_Pick.py script visualizes the best deblurring results from the validation set.

Usage:
python Demo_Pick.py --ckpt ./checkpoints/best_model.pth \
  --blur_dir ./Dataset/Gopro/val/blur \
  --sharp_dir ./Dataset/Gopro/val/sharp \
  --output_dir ./demo_outputs

The output includes:
- Side-by-side comparison (Blur | Wiener | DeblurGAN | Ground Truth)
- PSNR and SSIM metrics

Testing on custom images:
python Test_Self_Captured.py --ckpt ./checkpoints/best_model.pth \
  --input Dataset/Self_captured/Blur --output results/

6. References

DeblurGAN-v2:
Kupyn et al., "DeblurGAN-v2: Deblurring (Orders-of-Magnitude) Faster and Better", ICCV 2019

GoPro Dataset:
Nah et al., "Deep Multi-scale CNN for Dynamic Scene Deblurring", CVPR 2017