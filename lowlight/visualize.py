"""
Visualization script for comparing results
Creates side-by-side comparisons and metric plots
"""

import os
import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torchvision import transforms

from model import HybridLLE
from utils import load_checkpoint


def create_comparison_grid(images_dict, titles, save_path=None):
    """
    Create a grid of images for comparison
    
    Args:
        images_dict: Dictionary of {title: image} where image is PIL Image or numpy array
        titles: List of titles for each image
        save_path: Path to save the comparison (optional)
    """
    n_images = len(images_dict)
    fig, axes = plt.subplots(1, n_images, figsize=(5 * n_images, 5))
    
    if n_images == 1:
        axes = [axes]
    
    for idx, (title, image) in enumerate(images_dict.items()):
        if isinstance(image, torch.Tensor):
            image = image.cpu().numpy().transpose(1, 2, 0)
        elif isinstance(image, Image.Image):
            image = np.array(image)
        
        image = np.clip(image, 0, 1)
        
        axes[idx].imshow(image)
        axes[idx].set_title(title, fontsize=12)
        axes[idx].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved comparison to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_metrics_comparison(csv_paths, labels, save_path=None):
    """
    Plot comparison of metrics from multiple CSV files
    
    Args:
        csv_paths: List of paths to CSV files with results
        labels: List of labels for each method
        save_path: Path to save the plot (optional)
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    metrics = ['psnr', 'ssim', 'niqe']
    metric_names = ['PSNR (dB)', 'SSIM', 'NIQE']
    
    for csv_path, label in zip(csv_paths, labels):
        if not os.path.exists(csv_path):
            print(f"Warning: {csv_path} not found")
            continue
        
        df = pd.read_csv(csv_path)
        
        for idx, (metric, metric_name) in enumerate(zip(metrics, metric_names)):
            if metric in df.columns:
                values = df[metric].dropna()
                axes[idx].hist(values, alpha=0.6, label=label, bins=20)
    
    for idx, metric_name in enumerate(metric_names):
        axes[idx].set_xlabel(metric_name)
        axes[idx].set_ylabel('Frequency')
        axes[idx].set_title(f'{metric_name} Distribution')
        axes[idx].legend()
        axes[idx].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved metrics comparison to {save_path}")
    else:
        plt.show()
    
    plt.close()


def visualize_model_outputs(model, image_path, device, save_dir=None):
    """
    Visualize all intermediate outputs of the model
    
    Args:
        model: Trained model
        image_path: Path to input image
        device: Device to run on
        save_dir: Directory to save visualizations (optional)
    """
    # Load and preprocess image
    img = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])
    img_tensor = transform(img).unsqueeze(0).to(device)
    
    # Get model outputs
    model.eval()
    with torch.no_grad():
        outputs = model(img_tensor)
    
    # Convert tensors to numpy
    def tensor_to_np(t):
        return t.squeeze(0).cpu().clamp(0, 1).numpy().transpose(1, 2, 0)
    
    low_img = tensor_to_np(img_tensor)
    enhanced = tensor_to_np(outputs['enhanced'])
    retinex = tensor_to_np(outputs['retinex'])
    curve = tensor_to_np(outputs['curve'])
    
    # Illumination map (grayscale)
    illum = outputs['illum_map'].squeeze().cpu().numpy()
    
    # Weight map (grayscale)
    weight = outputs['weight'].squeeze().cpu().numpy()
    
    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Row 1: Main results
    axes[0, 0].imshow(low_img)
    axes[0, 0].set_title('Input (Low-light)', fontsize=12)
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(retinex)
    axes[0, 1].set_title('Retinex Branch', fontsize=12)
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(curve)
    axes[0, 2].set_title('Curve Branch', fontsize=12)
    axes[0, 2].axis('off')
    
    # Row 2: Analysis
    axes[1, 0].imshow(enhanced)
    axes[1, 0].set_title('Final Enhanced', fontsize=12, fontweight='bold')
    axes[1, 0].axis('off')
    
    im1 = axes[1, 1].imshow(illum, cmap='gray')
    axes[1, 1].set_title('Illumination Map', fontsize=12)
    axes[1, 1].axis('off')
    plt.colorbar(im1, ax=axes[1, 1], fraction=0.046)
    
    im2 = axes[1, 2].imshow(weight, cmap='viridis')
    axes[1, 2].set_title('Fusion Weight', fontsize=12)
    axes[1, 2].axis('off')
    plt.colorbar(im2, ax=axes[1, 2], fraction=0.046)
    
    plt.tight_layout()
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, 'model_outputs.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_training_curves(log_dir, save_path=None):
    """
    Plot training curves from TensorBoard logs
    
    Args:
        log_dir: Directory containing TensorBoard logs
        save_path: Path to save the plot (optional)
    """
    try:
        from tensorboard.backend.event_processing import event_accumulator
        
        # Find event file
        event_files = []
        for root, dirs, files in os.walk(log_dir):
            for file in files:
                if 'events.out.tfevents' in file:
                    event_files.append(os.path.join(root, file))
        
        if not event_files:
            print(f"No TensorBoard event files found in {log_dir}")
            return
        
        # Load events
        ea = event_accumulator.EventAccumulator(event_files[0])
        ea.Reload()
        
        # Get scalars
        tags = ea.Tags()['scalars']
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Plot train/val loss
        if 'Loss/train' in tags and 'Loss/val' in tags:
            train_loss = [(s.step, s.value) for s in ea.Scalars('Loss/train')]
            val_loss = [(s.step, s.value) for s in ea.Scalars('Loss/val')]
            
            axes[0, 0].plot([s[0] for s in train_loss], [s[1] for s in train_loss], label='Train')
            axes[0, 0].plot([s[0] for s in val_loss], [s[1] for s in val_loss], label='Val')
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].set_title('Training and Validation Loss')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
        
        # Plot learning rate
        if 'Learning_rate' in tags:
            lr = [(s.step, s.value) for s in ea.Scalars('Learning_rate')]
            axes[0, 1].plot([s[0] for s in lr], [s[1] for s in lr])
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Learning Rate')
            axes[0, 1].set_title('Learning Rate Schedule')
            axes[0, 1].grid(True, alpha=0.3)
        
        # Plot SSIM
        if 'Val/ssim' in tags:
            ssim = [(s.step, s.value) for s in ea.Scalars('Val/ssim')]
            axes[1, 0].plot([s[0] for s in ssim], [s[1] for s in ssim])
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('SSIM Loss')
            axes[1, 0].set_title('Validation SSIM Loss')
            axes[1, 0].grid(True, alpha=0.3)
        
        # Plot L1 loss
        if 'Val/l1_enhanced' in tags:
            l1 = [(s.step, s.value) for s in ea.Scalars('Val/l1_enhanced')]
            axes[1, 1].plot([s[0] for s in l1], [s[1] for s in l1])
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('L1 Loss')
            axes[1, 1].set_title('Validation L1 Loss')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved training curves to {save_path}")
        else:
            plt.show()
        
        plt.close()
        
    except Exception as e:
        print(f"Error plotting training curves: {e}")


def main():
    parser = argparse.ArgumentParser(description='Visualize Results')
    
    parser.add_argument('--mode', type=str, required=True,
                        choices=['model_outputs', 'metrics', 'training'],
                        help='Visualization mode')
    
    # For model_outputs mode
    parser.add_argument('--checkpoint', type=str,
                        help='Path to model checkpoint')
    parser.add_argument('--image', type=str,
                        help='Path to input image')
    
    # For metrics mode
    parser.add_argument('--csv_paths', type=str, nargs='+',
                        help='Paths to CSV files with metrics')
    parser.add_argument('--labels', type=str, nargs='+',
                        help='Labels for each method')
    
    # For training mode
    parser.add_argument('--log_dir', type=str,
                        help='Path to TensorBoard logs')
    
    # General
    parser.add_argument('--output', type=str, default='./visualizations',
                        help='Output directory for visualizations')
    
    args = parser.parse_args()
    
    os.makedirs(args.output, exist_ok=True)
    
    if args.mode == 'model_outputs':
        if not args.checkpoint or not args.image:
            parser.error("--checkpoint and --image are required for model_outputs mode")
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = HybridLLE().to(device)
        load_checkpoint(args.checkpoint, model)
        
        visualize_model_outputs(
            model, args.image, device,
            save_dir=args.output
        )
    
    elif args.mode == 'metrics':
        if not args.csv_paths or not args.labels:
            parser.error("--csv_paths and --labels are required for metrics mode")
        
        if len(args.csv_paths) != len(args.labels):
            parser.error("Number of CSV paths must match number of labels")
        
        save_path = os.path.join(args.output, 'metrics_comparison.png')
        plot_metrics_comparison(args.csv_paths, args.labels, save_path)
    
    elif args.mode == 'training':
        if not args.log_dir:
            parser.error("--log_dir is required for training mode")
        
        save_path = os.path.join(args.output, 'training_curves.png')
        plot_training_curves(args.log_dir, save_path)


if __name__ == '__main__':
    main()

