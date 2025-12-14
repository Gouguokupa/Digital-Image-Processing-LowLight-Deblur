"""
Evaluation script for low-light enhancement model
Computes PSNR, SSIM, and NIQE metrics on test set
"""

import os
import argparse
import torch
from tqdm import tqdm
import numpy as np
import pandas as pd

from model import HybridLLE
from dataset import create_dataloaders
from metrics import calculate_psnr, calculate_ssim, calculate_niqe
from utils import load_checkpoint, tensor_to_numpy, save_images


def evaluate(args):
    """Main evaluation function"""
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create dataloader
    print(f"\nLoading test dataset from {args.data_root}")
    _, _, test_loader = create_dataloaders(
        data_root=args.data_root,
        batch_size=1,  # Evaluate one image at a time
        image_size=args.image_size,
        num_workers=args.num_workers
    )
    
    print(f"Test images: {len(test_loader)}")
    
    # Create model
    print(f"\nInitializing model")
    model = HybridLLE(base_channels=args.base_channels).to(device)
    
    # Load checkpoint
    if not os.path.isfile(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    
    print(f"Loading checkpoint from {args.checkpoint}")
    checkpoint = load_checkpoint(args.checkpoint, model)
    print(f"Checkpoint epoch: {checkpoint.get('epoch', 'unknown')}")
    
    # Set to evaluation mode
    model.eval()
    
    # Metrics storage
    results = []
    psnr_list = []
    ssim_list = []
    niqe_list = []
    
    # Create output directory for enhanced images
    if args.save_results:
        os.makedirs(args.output_dir, exist_ok=True)
        print(f"\nSaving results to {args.output_dir}")
    
    # Evaluate
    print(f"\nEvaluating on test set...")
    with torch.no_grad():
        pbar = tqdm(test_loader, desc='Evaluating')
        for idx, batch in enumerate(pbar):
            # Move to device
            low = batch['low'].to(device)
            high = batch['high'].to(device)
            filename = batch['filename'][0]
            
            # Forward pass
            outputs = model(low)
            enhanced = outputs['enhanced']
            
            # Convert to numpy for metrics
            enhanced_np = tensor_to_numpy(enhanced)
            high_np = tensor_to_numpy(high)
            
            # Calculate metrics
            psnr = calculate_psnr(enhanced_np, high_np)
            ssim = calculate_ssim(enhanced_np, high_np)
            niqe = calculate_niqe(enhanced_np)
            
            psnr_list.append(psnr)
            ssim_list.append(ssim)
            if niqe > 0:  # Valid NIQE score
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
                save_dir = os.path.join(args.output_dir, f'image_{idx:04d}')
                save_images(
                    low, high, enhanced,
                    outputs['retinex'], outputs['curve'],
                    save_dir
                )
    
    # Compute statistics
    psnr_mean = np.mean(psnr_list)
    psnr_std = np.std(psnr_list)
    ssim_mean = np.mean(ssim_list)
    ssim_std = np.std(ssim_list)
    niqe_mean = np.mean(niqe_list) if niqe_list else -1
    niqe_std = np.std(niqe_list) if niqe_list else -1
    
    # Print results
    print(f"\n{'='*50}")
    print(f"Evaluation Results")
    print(f"{'='*50}")
    print(f"Test images: {len(test_loader)}")
    print(f"\nMetrics:")
    print(f"  PSNR: {psnr_mean:.2f} ± {psnr_std:.2f} dB")
    print(f"  SSIM: {ssim_mean:.4f} ± {ssim_std:.4f}")
    if niqe_mean > 0:
        print(f"  NIQE: {niqe_mean:.4f} ± {niqe_std:.4f}")
    print(f"{'='*50}\n")
    
    # Save results to CSV
    if args.save_csv:
        df = pd.DataFrame(results)
        csv_path = os.path.join(args.output_dir, 'evaluation_results.csv')
        df.to_csv(csv_path, index=False)
        print(f"Results saved to {csv_path}")
        
        # Save summary statistics
        summary = {
            'metric': ['PSNR', 'SSIM', 'NIQE'],
            'mean': [psnr_mean, ssim_mean, niqe_mean],
            'std': [psnr_std, ssim_std, niqe_std]
        }
        summary_df = pd.DataFrame(summary)
        summary_path = os.path.join(args.output_dir, 'summary_statistics.csv')
        summary_df.to_csv(summary_path, index=False)
        print(f"Summary statistics saved to {summary_path}")
    
    return results, psnr_mean, ssim_mean, niqe_mean


def main():
    parser = argparse.ArgumentParser(description='Evaluate Hybrid Low-Light Enhancement Model')
    
    # Dataset parameters
    parser.add_argument('--data_root', type=str, required=True,
                        help='Root directory of dataset (containing test folder)')
    parser.add_argument('--image_size', type=int, default=256,
                        help='Image size for evaluation (default: 256)')
    
    # Model parameters
    parser.add_argument('--base_channels', type=int, default=32,
                        help='Base number of channels in model (default: 32)')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    
    # Evaluation parameters
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers (default: 4)')
    
    # Output parameters
    parser.add_argument('--output_dir', type=str, default='./results',
                        help='Directory to save results (default: ./results)')
    parser.add_argument('--save_results', action='store_true',
                        help='Save enhanced images')
    parser.add_argument('--save_csv', action='store_true',
                        help='Save results to CSV file')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Evaluate
    evaluate(args)


if __name__ == '__main__':
    main()

