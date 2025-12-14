"""
Visualize model results on a single image
Shows: Input, Baseline, Model Output, and Intermediate Results
"""

import os
import torch
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import cv2

from model import HybridLLE
from utils import load_checkpoint


def baseline_clahe(image_np):
    """Apply CLAHE baseline"""
    # Convert to uint8
    img_uint8 = (image_np * 255).astype(np.uint8)
    
    # Convert to YCrCb
    ycrcb = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2YCrCb)
    
    # Apply CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    ycrcb[:, :, 0] = clahe.apply(ycrcb[:, :, 0])
    
    # Convert back
    enhanced = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)
    
    return enhanced.astype(np.float32) / 255.0


def visualize_image(image_name, checkpoint_path, output_dir='./visualization'):
    """
    Visualize complete enhancement pipeline for one image
    
    Args:
        image_name: Name of image file (e.g., '1.png')
        checkpoint_path: Path to model checkpoint
        output_dir: Where to save visualization
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model
    device = torch.device('cpu')
    model = HybridLLE(base_channels=16).to(device)
    load_checkpoint(checkpoint_path, model)
    model.eval()
    
    # Paths
    low_path = f'./dataset/test/low/{image_name}'
    high_path = f'./dataset/test/high/{image_name}'
    
    print(f"\n{'='*60}")
    print(f"Visualizing: {image_name}")
    print(f"{'='*60}\n")
    
    # Load images
    low_img = Image.open(low_path).convert('RGB')
    high_img = Image.open(high_path).convert('RGB')
    
    print(f"✓ Loaded low-light image: {low_img.size}")
    print(f"✓ Loaded ground truth: {high_img.size}")
    
    # Resize to 256x256 for model
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])
    
    low_tensor = transform(low_img).unsqueeze(0).to(device)
    high_tensor = transform(high_img).unsqueeze(0).to(device)
    
    # Get model outputs
    print("\n✓ Running model inference...")
    with torch.no_grad():
        outputs = model(low_tensor)
    
    # Convert to numpy for visualization
    def to_np(tensor):
        return tensor.squeeze(0).cpu().numpy().transpose(1, 2, 0)
    
    low_np = to_np(low_tensor)
    high_np = to_np(high_tensor)
    enhanced_np = to_np(outputs['enhanced'])
    retinex_np = to_np(outputs['retinex'])
    curve_np = to_np(outputs['curve'])
    illum_np = outputs['illum_map'].squeeze().cpu().numpy()
    weight_np = outputs['weight'].squeeze().cpu().numpy()
    
    # Apply baseline
    print("✓ Applying baseline (CLAHE)...")
    baseline_np = baseline_clahe(low_np)
    
    # Calculate metrics
    from metrics import calculate_psnr, calculate_ssim
    
    baseline_psnr = calculate_psnr(baseline_np, high_np)
    baseline_ssim = calculate_ssim(baseline_np, high_np)
    
    model_psnr = calculate_psnr(enhanced_np, high_np)
    model_ssim = calculate_ssim(enhanced_np, high_np)
    
    print("\n" + "="*60)
    print("METRICS COMPARISON")
    print("="*60)
    print(f"{'Method':<20} {'PSNR (dB)':<15} {'SSIM'}")
    print("-"*60)
    print(f"{'Baseline (CLAHE)':<20} {baseline_psnr:<15.2f} {baseline_ssim:.4f}")
    print(f"{'Our Model':<20} {model_psnr:<15.2f} {model_ssim:.4f}")
    print(f"{'Improvement':<20} {(model_psnr-baseline_psnr):<15.2f} {(model_ssim-baseline_ssim):.4f}")
    print("="*60 + "\n")
    
    # Create comprehensive visualization
    fig = plt.figure(figsize=(20, 12))
    
    # Row 1: Main comparison
    ax1 = plt.subplot(3, 4, 1)
    ax1.imshow(np.clip(low_np, 0, 1))
    ax1.set_title('Input\n(Low-light)', fontsize=14, fontweight='bold')
    ax1.axis('off')
    
    ax2 = plt.subplot(3, 4, 2)
    ax2.imshow(np.clip(baseline_np, 0, 1))
    ax2.set_title(f'Baseline (CLAHE)\nPSNR: {baseline_psnr:.2f} dB', fontsize=14)
    ax2.axis('off')
    
    ax3 = plt.subplot(3, 4, 3)
    ax3.imshow(np.clip(enhanced_np, 0, 1))
    ax3.set_title(f'Our Model\nPSNR: {model_psnr:.2f} dB ↑', 
                  fontsize=14, fontweight='bold', color='green')
    ax3.axis('off')
    
    ax4 = plt.subplot(3, 4, 4)
    ax4.imshow(np.clip(high_np, 0, 1))
    ax4.set_title('Ground Truth\n(Reference)', fontsize=14, fontweight='bold')
    ax4.axis('off')
    
    # Row 2: Two-branch outputs
    ax5 = plt.subplot(3, 4, 5)
    ax5.imshow(np.clip(retinex_np, 0, 1))
    ax5.set_title('Retinex Branch\n(Physics-based)', fontsize=12)
    ax5.axis('off')
    
    ax6 = plt.subplot(3, 4, 6)
    ax6.imshow(np.clip(curve_np, 0, 1))
    ax6.set_title('Curve Branch\n(Learning-based)', fontsize=12)
    ax6.axis('off')
    
    ax7 = plt.subplot(3, 4, 7)
    im7 = ax7.imshow(illum_np, cmap='hot')
    ax7.set_title('Illumination Map\n(Estimated)', fontsize=12)
    ax7.axis('off')
    plt.colorbar(im7, ax=ax7, fraction=0.046)
    
    ax8 = plt.subplot(3, 4, 8)
    im8 = ax8.imshow(weight_np, cmap='viridis', vmin=0, vmax=1)
    ax8.set_title('Fusion Weight\n(Learned)', fontsize=12)
    ax8.axis('off')
    plt.colorbar(im8, ax=ax8, fraction=0.046)
    
    # Row 3: Detailed comparison
    ax9 = plt.subplot(3, 4, 9)
    diff_baseline = np.abs(baseline_np - high_np)
    ax9.imshow(diff_baseline)
    ax9.set_title(f'Baseline Error\n(MAE: {np.mean(diff_baseline):.4f})', fontsize=12)
    ax9.axis('off')
    
    ax10 = plt.subplot(3, 4, 10)
    diff_model = np.abs(enhanced_np - high_np)
    ax10.imshow(diff_model)
    ax10.set_title(f'Our Model Error\n(MAE: {np.mean(diff_model):.4f})', fontsize=12, color='green')
    ax10.axis('off')
    
    # Histogram comparison
    ax11 = plt.subplot(3, 4, 11)
    ax11.hist(low_np.ravel(), bins=50, alpha=0.5, label='Input', color='blue')
    ax11.hist(enhanced_np.ravel(), bins=50, alpha=0.5, label='Enhanced', color='green')
    ax11.set_title('Intensity Distribution', fontsize=12)
    ax11.legend()
    ax11.set_xlabel('Pixel Value')
    ax11.set_ylabel('Frequency')
    
    # SSIM comparison
    ax12 = plt.subplot(3, 4, 12)
    categories = ['Baseline', 'Our Model']
    ssim_values = [baseline_ssim, model_ssim]
    colors = ['orange', 'green']
    bars = ax12.bar(categories, ssim_values, color=colors, alpha=0.7)
    ax12.set_ylabel('SSIM', fontsize=12)
    ax12.set_title('SSIM Comparison', fontsize=12)
    ax12.set_ylim([0, 1])
    ax12.axhline(y=0.85, color='r', linestyle='--', label='Target (0.85)')
    ax12.legend()
    
    # Add values on bars
    for bar, val in zip(bars, ssim_values):
        height = bar.get_height()
        ax12.text(bar.get_x() + bar.get_width()/2., height,
                 f'{val:.4f}', ha='center', va='bottom', fontweight='bold')
    
    plt.suptitle(f'Low-Light Enhancement Results: {image_name}', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    
    # Save
    save_path = os.path.join(output_dir, f'{image_name.replace(".png", "")}_analysis.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved visualization to: {save_path}")
    
    # Also save individual images
    individual_dir = os.path.join(output_dir, 'individual')
    os.makedirs(individual_dir, exist_ok=True)
    
    Image.fromarray((np.clip(low_np, 0, 1) * 255).astype(np.uint8)).save(
        os.path.join(individual_dir, f'{image_name.replace(".png", "")}_input.png'))
    Image.fromarray((np.clip(baseline_np, 0, 1) * 255).astype(np.uint8)).save(
        os.path.join(individual_dir, f'{image_name.replace(".png", "")}_baseline.png'))
    Image.fromarray((np.clip(enhanced_np, 0, 1) * 255).astype(np.uint8)).save(
        os.path.join(individual_dir, f'{image_name.replace(".png", "")}_our_model.png'))
    Image.fromarray((np.clip(high_np, 0, 1) * 255).astype(np.uint8)).save(
        os.path.join(individual_dir, f'{image_name.replace(".png", "")}_ground_truth.png'))
    
    print(f"✓ Saved individual images to: {individual_dir}/")
    
    print(f"\n{'='*60}")
    print("✅ Visualization complete!")
    print(f"{'='*60}\n")
    
    plt.show()


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Visualize model results on specific image')
    parser.add_argument('--image', type=str, default='1.png',
                        help='Image filename (e.g., 1.png)')
    parser.add_argument('--checkpoint', type=str, 
                        default='./checkpoints/best_model.pth',
                        help='Path to model checkpoint')
    parser.add_argument('--output', type=str, default='./visualization',
                        help='Output directory')
    
    args = parser.parse_args()
    
    visualize_image(args.image, args.checkpoint, args.output)

