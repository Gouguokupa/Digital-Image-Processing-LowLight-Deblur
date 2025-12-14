"""
Baseline method: Histogram Equalization
Traditional method for low-light image enhancement
"""

import os
import argparse
import cv2
import numpy as np
from tqdm import tqdm
import pandas as pd

from metrics import calculate_psnr, calculate_ssim, calculate_niqe


def histogram_equalization_rgb(image):
    """
    Apply histogram equalization to RGB image
    
    Args:
        image: RGB image (numpy array [H, W, 3])
        
    Returns:
        Enhanced RGB image
    """
    # Convert to YCrCb color space
    ycrcb = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
    
    # Apply histogram equalization to Y channel only
    ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
    
    # Convert back to RGB
    enhanced = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)
    
    return enhanced


def clahe_enhancement(image, clip_limit=2.0, tile_grid_size=(8, 8)):
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    More sophisticated than regular histogram equalization
    
    Args:
        image: RGB image (numpy array [H, W, 3])
        clip_limit: Threshold for contrast limiting
        tile_grid_size: Size of grid for histogram equalization
        
    Returns:
        Enhanced RGB image
    """
    # Convert to YCrCb color space
    ycrcb = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
    
    # Create CLAHE object
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    
    # Apply CLAHE to Y channel only
    ycrcb[:, :, 0] = clahe.apply(ycrcb[:, :, 0])
    
    # Convert back to RGB
    enhanced = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)
    
    return enhanced


def evaluate_baseline(args):
    """Evaluate baseline histogram equalization method"""
    
    # Get list of test images
    low_dir = os.path.join(args.data_root, 'test', 'low')
    high_dir = os.path.join(args.data_root, 'test', 'high')
    
    if not os.path.exists(low_dir) or not os.path.exists(high_dir):
        raise ValueError(f"Test directories not found: {low_dir} or {high_dir}")
    
    low_files = sorted([f for f in os.listdir(low_dir) 
                       if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))])
    
    print(f"Found {len(low_files)} test images")
    
    # Metrics storage
    results = []
    psnr_list = []
    ssim_list = []
    niqe_list = []
    
    # Create output directory
    if args.save_results:
        os.makedirs(args.output_dir, exist_ok=True)
        print(f"Saving results to {args.output_dir}")
    
    # Process each image
    print(f"\nEvaluating baseline method: {args.method}")
    pbar = tqdm(low_files, desc='Processing')
    
    for filename in pbar:
        # Load images
        low_path = os.path.join(low_dir, filename)
        high_path = os.path.join(high_dir, filename)
        
        low_img = cv2.imread(low_path)
        high_img = cv2.imread(high_path)
        
        if low_img is None or high_img is None:
            print(f"Warning: Failed to load {filename}")
            continue
        
        # Convert BGR to RGB
        low_img = cv2.cvtColor(low_img, cv2.COLOR_BGR2RGB)
        high_img = cv2.cvtColor(high_img, cv2.COLOR_BGR2RGB)
        
        # Normalize to [0, 1]
        low_img = low_img.astype(np.float32) / 255.0
        high_img = high_img.astype(np.float32) / 255.0
        
        # Resize if needed
        if args.image_size > 0:
            low_img = cv2.resize(low_img, (args.image_size, args.image_size))
            high_img = cv2.resize(high_img, (args.image_size, args.image_size))
        
        # Convert back to uint8 for OpenCV processing
        low_img_uint8 = (low_img * 255).astype(np.uint8)
        
        # Apply enhancement
        if args.method == 'histogram':
            enhanced = histogram_equalization_rgb(low_img_uint8)
        elif args.method == 'clahe':
            enhanced = clahe_enhancement(low_img_uint8, 
                                        clip_limit=args.clip_limit,
                                        tile_grid_size=(args.tile_size, args.tile_size))
        else:
            raise ValueError(f"Unknown method: {args.method}")
        
        # Convert back to float [0, 1]
        enhanced = enhanced.astype(np.float32) / 255.0
        
        # Calculate metrics
        psnr = calculate_psnr(enhanced, high_img)
        ssim = calculate_ssim(enhanced, high_img)
        niqe = calculate_niqe(enhanced)
        
        psnr_list.append(psnr)
        ssim_list.append(ssim)
        if niqe > 0:
            niqe_list.append(niqe)
        
        results.append({
            'filename': filename,
            'psnr': psnr,
            'ssim': ssim,
            'niqe': niqe
        })
        
        # Update progress bar
        pbar.set_postfix({
            'PSNR': f'{psnr:.2f}',
            'SSIM': f'{ssim:.4f}',
            'NIQE': f'{niqe:.4f}'
        })
        
        # Save results if requested
        if args.save_results:
            # Save enhanced image
            save_path = os.path.join(args.output_dir, filename)
            enhanced_uint8 = (enhanced * 255).astype(np.uint8)
            enhanced_bgr = cv2.cvtColor(enhanced_uint8, cv2.COLOR_RGB2BGR)
            cv2.imwrite(save_path, enhanced_bgr)
    
    # Compute statistics
    psnr_mean = np.mean(psnr_list)
    psnr_std = np.std(psnr_list)
    ssim_mean = np.mean(ssim_list)
    ssim_std = np.std(ssim_list)
    niqe_mean = np.mean(niqe_list) if niqe_list else -1
    niqe_std = np.std(niqe_list) if niqe_list else -1
    
    # Print results
    print(f"\n{'='*50}")
    print(f"Baseline Evaluation Results ({args.method})")
    print(f"{'='*50}")
    print(f"Test images: {len(low_files)}")
    print(f"\nMetrics:")
    print(f"  PSNR: {psnr_mean:.2f} ± {psnr_std:.2f} dB")
    print(f"  SSIM: {ssim_mean:.4f} ± {ssim_std:.4f}")
    if niqe_mean > 0:
        print(f"  NIQE: {niqe_mean:.4f} ± {niqe_std:.4f}")
    print(f"{'='*50}\n")
    
    # Save results to CSV
    if args.save_csv:
        df = pd.DataFrame(results)
        csv_path = os.path.join(args.output_dir, f'baseline_{args.method}_results.csv')
        df.to_csv(csv_path, index=False)
        print(f"Results saved to {csv_path}")
        
        # Save summary statistics
        summary = {
            'method': [args.method],
            'psnr_mean': [psnr_mean],
            'psnr_std': [psnr_std],
            'ssim_mean': [ssim_mean],
            'ssim_std': [ssim_std],
            'niqe_mean': [niqe_mean],
            'niqe_std': [niqe_std]
        }
        summary_df = pd.DataFrame(summary)
        summary_path = os.path.join(args.output_dir, f'baseline_{args.method}_summary.csv')
        summary_df.to_csv(summary_path, index=False)
        print(f"Summary statistics saved to {summary_path}")


def main():
    parser = argparse.ArgumentParser(description='Baseline Low-Light Enhancement (Histogram Equalization)')
    
    # Dataset parameters
    parser.add_argument('--data_root', type=str, required=True,
                        help='Root directory of dataset (containing test folder)')
    parser.add_argument('--image_size', type=int, default=256,
                        help='Image size for evaluation (default: 256, set to -1 for original size)')
    
    # Method parameters
    parser.add_argument('--method', type=str, default='clahe', choices=['histogram', 'clahe'],
                        help='Enhancement method (default: clahe)')
    parser.add_argument('--clip_limit', type=float, default=2.0,
                        help='CLAHE clip limit (default: 2.0)')
    parser.add_argument('--tile_size', type=int, default=8,
                        help='CLAHE tile grid size (default: 8)')
    
    # Output parameters
    parser.add_argument('--output_dir', type=str, default='./baseline_results',
                        help='Directory to save results (default: ./baseline_results)')
    parser.add_argument('--save_results', action='store_true',
                        help='Save enhanced images')
    parser.add_argument('--save_csv', action='store_true',
                        help='Save results to CSV file')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Evaluate
    evaluate_baseline(args)


if __name__ == '__main__':
    main()

