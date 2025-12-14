"""
Dataset loader for low-light image enhancement
Supports LOL dataset and custom paired images
"""

import os
import random
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import numpy as np


class LowLightDataset(Dataset):
    """
    Dataset for paired low-light and normal-light images
    Expected structure:
        dataset_root/
            low/
                image1.png
                image2.png
                ...
            high/
                image1.png
                image2.png
                ...
    """
    
    def __init__(self, low_dir, high_dir, image_size=256, augment=False):
        """
        Args:
            low_dir: Directory containing low-light images
            high_dir: Directory containing normal-light (ground truth) images
            image_size: Size to resize images to
            augment: Whether to apply data augmentation
        """
        self.low_dir = low_dir
        self.high_dir = high_dir
        self.image_size = image_size
        self.augment = augment
        
        # Get list of image files
        self.low_images = sorted([f for f in os.listdir(low_dir) 
                                 if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))])
        self.high_images = sorted([f for f in os.listdir(high_dir) 
                                  if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))])
        
        # Ensure paired images exist
        assert len(self.low_images) == len(self.high_images), \
            f"Number of low-light images ({len(self.low_images)}) must match normal-light images ({len(self.high_images)})"
        
        print(f"Loaded {len(self.low_images)} image pairs from {low_dir}")
        
        # Basic transforms
        self.to_tensor = transforms.ToTensor()
        self.resize = transforms.Resize((image_size, image_size))
        
    def __len__(self):
        return len(self.low_images)
    
    def __getitem__(self, idx):
        # Load images
        low_path = os.path.join(self.low_dir, self.low_images[idx])
        high_path = os.path.join(self.high_dir, self.high_images[idx])
        
        low_img = Image.open(low_path).convert('RGB')
        high_img = Image.open(high_path).convert('RGB')
        
        # Resize
        low_img = self.resize(low_img)
        high_img = self.resize(high_img)
        
        # Data augmentation
        if self.augment:
            # Random horizontal flip
            if random.random() > 0.5:
                low_img = transforms.functional.hflip(low_img)
                high_img = transforms.functional.hflip(high_img)
            
            # Random rotation (90, 180, 270 degrees)
            if random.random() > 0.5:
                angle = random.choice([90, 180, 270])
                low_img = transforms.functional.rotate(low_img, angle)
                high_img = transforms.functional.rotate(high_img, angle)
        
        # Convert to tensor and normalize to [0, 1]
        low_tensor = self.to_tensor(low_img)
        high_tensor = self.to_tensor(high_img)
        
        return {
            'low': low_tensor,
            'high': high_tensor,
            'filename': self.low_images[idx]
        }


def create_dataloaders(data_root, batch_size=8, image_size=256, num_workers=4):
    """
    Create train, validation, and test dataloaders
    
    Expected structure:
        data_root/
            train/
                low/
                high/
            val/
                low/
                high/
            test/
                low/
                high/
    
    Args:
        data_root: Root directory of the dataset
        batch_size: Batch size for training
        image_size: Size to resize images to
        num_workers: Number of workers for data loading
        
    Returns:
        train_loader, val_loader, test_loader
    """
    
    # Create datasets
    train_dataset = LowLightDataset(
        low_dir=os.path.join(data_root, 'train', 'low'),
        high_dir=os.path.join(data_root, 'train', 'high'),
        image_size=image_size,
        augment=True
    )
    
    val_dataset = LowLightDataset(
        low_dir=os.path.join(data_root, 'val', 'low'),
        high_dir=os.path.join(data_root, 'val', 'high'),
        image_size=image_size,
        augment=False
    )
    
    test_dataset = LowLightDataset(
        low_dir=os.path.join(data_root, 'test', 'low'),
        high_dir=os.path.join(data_root, 'test', 'high'),
        image_size=image_size,
        augment=False
    )
    
    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


if __name__ == '__main__':
    # Test dataset loading
    print("Testing dataset loader...")
    # Example usage:
    # train_loader, val_loader, test_loader = create_dataloaders('./dataset', batch_size=4)
    # for batch in train_loader:
    #     print(f"Low image shape: {batch['low'].shape}")
    #     print(f"High image shape: {batch['high'].shape}")
    #     break

