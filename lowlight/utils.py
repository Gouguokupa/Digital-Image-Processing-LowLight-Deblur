"""
Utility functions for training and evaluation
"""

import os
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms


def save_checkpoint(filepath, model, optimizer, epoch, val_loss, best_val_loss, is_best):
    """
    Save model checkpoint
    
    Args:
        filepath: Path to save checkpoint
        model: Model to save
        optimizer: Optimizer state
        epoch: Current epoch
        val_loss: Current validation loss
        best_val_loss: Best validation loss so far
        is_best: Whether this is the best model
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss,
        'best_val_loss': best_val_loss,
        'is_best': is_best
    }
    torch.save(checkpoint, filepath)


def load_checkpoint(filepath, model, optimizer=None):
    """
    Load model checkpoint
    
    Args:
        filepath: Path to checkpoint
        model: Model to load weights into
        optimizer: Optimizer to load state into (optional)
        
    Returns:
        checkpoint dictionary
    """
    checkpoint = torch.load(filepath, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return checkpoint


def save_images(low, high, enhanced, retinex, curve, save_dir):
    """
    Save sample images for visualization
    
    Args:
        low: Low-light input images [B, 3, H, W]
        high: Ground truth images [B, 3, H, W]
        enhanced: Enhanced output images [B, 3, H, W]
        retinex: Retinex branch output [B, 3, H, W]
        curve: Curve branch output [B, 3, H, W]
        save_dir: Directory to save images
    """
    os.makedirs(save_dir, exist_ok=True)
    
    to_pil = transforms.ToPILImage()
    
    batch_size = low.shape[0]
    for i in range(batch_size):
        # Convert tensors to PIL images
        low_img = to_pil(low[i].cpu().clamp(0, 1))
        high_img = to_pil(high[i].cpu().clamp(0, 1))
        enhanced_img = to_pil(enhanced[i].cpu().clamp(0, 1))
        retinex_img = to_pil(retinex[i].cpu().clamp(0, 1))
        curve_img = to_pil(curve[i].cpu().clamp(0, 1))
        
        # Save individual images
        low_img.save(os.path.join(save_dir, f'sample_{i}_low.png'))
        high_img.save(os.path.join(save_dir, f'sample_{i}_gt.png'))
        enhanced_img.save(os.path.join(save_dir, f'sample_{i}_enhanced.png'))
        retinex_img.save(os.path.join(save_dir, f'sample_{i}_retinex.png'))
        curve_img.save(os.path.join(save_dir, f'sample_{i}_curve.png'))


def tensor_to_numpy(tensor):
    """
    Convert PyTorch tensor to numpy array for metric calculation
    
    Args:
        tensor: PyTorch tensor [B, C, H, W] or [C, H, W]
        
    Returns:
        numpy array [H, W, C]
    """
    if len(tensor.shape) == 4:
        # Batch of images, take first image
        tensor = tensor[0]
    
    # Convert to numpy and transpose to [H, W, C]
    img = tensor.cpu().detach().numpy()
    img = np.transpose(img, (1, 2, 0))
    
    # Clip to [0, 1]
    img = np.clip(img, 0, 1)
    
    return img


def denormalize(tensor, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    """
    Denormalize tensor if it was normalized
    
    Args:
        tensor: Normalized tensor
        mean: Mean used for normalization
        std: Std used for normalization
        
    Returns:
        Denormalized tensor
    """
    mean = torch.tensor(mean).view(-1, 1, 1).to(tensor.device)
    std = torch.tensor(std).view(-1, 1, 1).to(tensor.device)
    return tensor * std + mean


def create_grid(images, nrow=4):
    """
    Create a grid of images for visualization
    
    Args:
        images: List of image tensors
        nrow: Number of images per row
        
    Returns:
        Grid image tensor
    """
    import torchvision.utils as vutils
    return vutils.make_grid(images, nrow=nrow, normalize=True, scale_each=True)


class AverageMeter:
    """
    Computes and stores the average and current value
    """
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def print_network(model, name='Model'):
    """
    Print network architecture
    
    Args:
        model: PyTorch model
        name: Name of the model
    """
    num_params = 0
    for p in model.parameters():
        num_params += p.numel()
    
    print(f"\n{'='*50}")
    print(f"{name}")
    print(f"{'='*50}")
    print(model)
    print(f"{'='*50}")
    print(f"Total number of parameters: {num_params:,}")
    print(f"{'='*50}\n")


if __name__ == '__main__':
    # Test utilities
    print("Testing utility functions...")
    
    # Test tensor to numpy conversion
    tensor = torch.rand(1, 3, 256, 256)
    img = tensor_to_numpy(tensor)
    print(f"Tensor shape: {tensor.shape}")
    print(f"Numpy shape: {img.shape}")
    print(f"Numpy range: [{img.min():.3f}, {img.max():.3f}]")
    
    # Test average meter
    meter = AverageMeter()
    for i in range(10):
        meter.update(i * 0.1)
    print(f"\nAverage meter test: {meter.avg:.3f}")

