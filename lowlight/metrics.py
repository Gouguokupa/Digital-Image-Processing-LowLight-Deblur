"""
Evaluation metrics for image quality assessment
Includes PSNR, SSIM, and NIQE
"""

import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
import cv2


def calculate_psnr(img1, img2, max_value=1.0):
    """
    Calculate Peak Signal-to-Noise Ratio (PSNR)
    
    Args:
        img1: First image (numpy array [H, W, C] or [H, W])
        img2: Second image (numpy array [H, W, C] or [H, W])
        max_value: Maximum possible pixel value (default: 1.0 for normalized images)
        
    Returns:
        PSNR value in dB
    """
    return compare_psnr(img1, img2, data_range=max_value)


def calculate_ssim(img1, img2, max_value=1.0):
    """
    Calculate Structural Similarity Index (SSIM)
    
    Args:
        img1: First image (numpy array [H, W, C] or [H, W])
        img2: Second image (numpy array [H, W, C] or [H, W])
        max_value: Maximum possible pixel value (default: 1.0 for normalized images)
        
    Returns:
        SSIM value (between -1 and 1, higher is better)
    """
    if len(img1.shape) == 3:
        # Multi-channel image
        return compare_ssim(img1, img2, data_range=max_value, channel_axis=2)
    else:
        # Single-channel image
        return compare_ssim(img1, img2, data_range=max_value)


def calculate_niqe(img):
    """
    Calculate Natural Image Quality Evaluator (NIQE)
    Lower NIQE score indicates better perceptual quality
    
    Note: This is a simplified implementation. For full NIQE, you may need
    the official MATLAB implementation or more sophisticated Python ports.
    
    Args:
        img: Image (numpy array [H, W, C] or [H, W])
        
    Returns:
        NIQE score (lower is better)
    """
    try:
        import scipy.ndimage
        from scipy.special import gamma
        
        # Convert to grayscale if needed
        if len(img.shape) == 3:
            gray = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY) / 255.0
        else:
            gray = img
        
        # Compute local mean
        mu = scipy.ndimage.gaussian_filter(gray, sigma=7/6)
        
        # Compute local variance
        sigma = np.sqrt(np.abs(scipy.ndimage.gaussian_filter(gray**2, sigma=7/6) - mu**2))
        
        # Normalized intensity
        normalized = (gray - mu) / (sigma + 1e-10)
        
        # Fit to Generalized Gaussian Distribution (simplified)
        # In full NIQE, this would be more sophisticated
        alpha = np.std(normalized)
        
        # Simple quality metric based on local statistics
        # Lower values indicate more natural statistics
        niqe_score = np.mean(np.abs(normalized)) + alpha
        
        return niqe_score
    
    except Exception as e:
        print(f"Warning: NIQE calculation failed: {e}")
        return -1.0


def calculate_mae(img1, img2):
    """
    Calculate Mean Absolute Error (MAE)
    
    Args:
        img1: First image (numpy array)
        img2: Second image (numpy array)
        
    Returns:
        MAE value
    """
    return np.mean(np.abs(img1 - img2))


def calculate_mse(img1, img2):
    """
    Calculate Mean Squared Error (MSE)
    
    Args:
        img1: First image (numpy array)
        img2: Second image (numpy array)
        
    Returns:
        MSE value
    """
    return np.mean((img1 - img2) ** 2)


def evaluate_image_pair(pred, target, metrics=['psnr', 'ssim', 'niqe']):
    """
    Evaluate a pair of images with multiple metrics
    
    Args:
        pred: Predicted/enhanced image (numpy array [H, W, C])
        target: Target/ground truth image (numpy array [H, W, C])
        metrics: List of metrics to compute
        
    Returns:
        Dictionary of metric values
    """
    results = {}
    
    if 'psnr' in metrics:
        results['psnr'] = calculate_psnr(pred, target)
    
    if 'ssim' in metrics:
        results['ssim'] = calculate_ssim(pred, target)
    
    if 'niqe' in metrics:
        results['niqe'] = calculate_niqe(pred)
    
    if 'mae' in metrics:
        results['mae'] = calculate_mae(pred, target)
    
    if 'mse' in metrics:
        results['mse'] = calculate_mse(pred, target)
    
    return results


def batch_evaluate(pred_batch, target_batch, metrics=['psnr', 'ssim', 'niqe']):
    """
    Evaluate a batch of images
    
    Args:
        pred_batch: Batch of predicted images (torch tensor [B, C, H, W])
        target_batch: Batch of target images (torch tensor [B, C, H, W])
        metrics: List of metrics to compute
        
    Returns:
        Dictionary of average metric values
    """
    from utils import tensor_to_numpy
    
    batch_size = pred_batch.shape[0]
    metric_sums = {metric: 0.0 for metric in metrics}
    
    for i in range(batch_size):
        # Convert to numpy
        pred = tensor_to_numpy(pred_batch[i:i+1])
        target = tensor_to_numpy(target_batch[i:i+1])
        
        # Compute metrics
        results = evaluate_image_pair(pred, target, metrics)
        
        for metric in metrics:
            if metric in results:
                metric_sums[metric] += results[metric]
    
    # Compute averages
    metric_avgs = {metric: metric_sums[metric] / batch_size for metric in metrics}
    
    return metric_avgs


if __name__ == '__main__':
    # Test metrics
    print("Testing evaluation metrics...")
    
    # Create dummy images
    img1 = np.random.rand(256, 256, 3)
    img2 = img1 + np.random.randn(256, 256, 3) * 0.1
    img2 = np.clip(img2, 0, 1)
    
    print("\nTesting PSNR:")
    psnr = calculate_psnr(img1, img2)
    print(f"  PSNR: {psnr:.2f} dB")
    
    print("\nTesting SSIM:")
    ssim = calculate_ssim(img1, img2)
    print(f"  SSIM: {ssim:.4f}")
    
    print("\nTesting NIQE:")
    niqe = calculate_niqe(img1)
    print(f"  NIQE: {niqe:.4f}")
    
    print("\nTesting batch evaluation:")
    pred_batch = torch.rand(4, 3, 256, 256)
    target_batch = torch.rand(4, 3, 256, 256)
    results = batch_evaluate(pred_batch, target_batch)
    print(f"  Results: {results}")

