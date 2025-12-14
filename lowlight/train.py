"""
Training script for hybrid low-light enhancement model
"""

import os
import argparse
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
from datetime import datetime

from model import HybridLLE
from losses import HybridLoss
from dataset import create_dataloaders
from utils import save_checkpoint, load_checkpoint, save_images


def train_epoch(model, train_loader, criterion, optimizer, device, epoch):
    """Train for one epoch"""
    model.train()
    
    total_loss = 0
    loss_dict_sum = {
        'l1_enhanced': 0,
        'l1_retinex': 0,
        'ssim': 0,
        'tv': 0,
        'color': 0
    }
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
    for batch_idx, batch in enumerate(pbar):
        # Move to device
        low = batch['low'].to(device)
        high = batch['high'].to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(low)
        
        # Compute loss
        loss, loss_dict = criterion(outputs, high)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Accumulate losses
        total_loss += loss.item()
        for key in loss_dict_sum.keys():
            if key in loss_dict:
                loss_dict_sum[key] += loss_dict[key]
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'l1': f'{loss_dict["l1_enhanced"]:.4f}'
        })
    
    # Compute average losses
    avg_loss = total_loss / len(train_loader)
    avg_loss_dict = {k: v / len(train_loader) for k, v in loss_dict_sum.items()}
    
    return avg_loss, avg_loss_dict


def validate(model, val_loader, criterion, device):
    """Validate the model"""
    model.eval()
    
    total_loss = 0
    loss_dict_sum = {
        'l1_enhanced': 0,
        'l1_retinex': 0,
        'ssim': 0,
        'tv': 0,
        'color': 0
    }
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc='Validation')
        for batch in pbar:
            # Move to device
            low = batch['low'].to(device)
            high = batch['high'].to(device)
            
            # Forward pass
            outputs = model(low)
            
            # Compute loss
            loss, loss_dict = criterion(outputs, high)
            
            # Accumulate losses
            total_loss += loss.item()
            for key in loss_dict_sum.keys():
                if key in loss_dict:
                    loss_dict_sum[key] += loss_dict[key]
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    # Compute average losses
    avg_loss = total_loss / len(val_loader)
    avg_loss_dict = {k: v / len(val_loader) for k, v in loss_dict_sum.items()}
    
    return avg_loss, avg_loss_dict


def train(args):
    """Main training function"""
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create dataloaders
    print(f"\nLoading dataset from {args.data_root}")
    train_loader, val_loader, _ = create_dataloaders(
        data_root=args.data_root,
        batch_size=args.batch_size,
        image_size=args.image_size,
        num_workers=args.num_workers
    )
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    
    # Create model
    print(f"\nInitializing model with {args.base_channels} base channels")
    model = HybridLLE(base_channels=args.base_channels).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Create loss function
    criterion = HybridLoss(
        lambda1=args.lambda1,
        lambda2=args.lambda2,
        lambda3=args.lambda3,
        lambda4=args.lambda4
    )
    
    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    # Tensorboard writer
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = os.path.join(args.log_dir, f'run_{timestamp}')
    writer = SummaryWriter(log_dir)
    print(f"\nTensorboard logs: {log_dir}")
    
    # Load checkpoint if specified
    start_epoch = 0
    best_val_loss = float('inf')
    
    if args.resume:
        if os.path.isfile(args.resume):
            print(f"\nLoading checkpoint from {args.resume}")
            checkpoint = load_checkpoint(args.resume, model, optimizer)
            start_epoch = checkpoint['epoch']
            best_val_loss = checkpoint['best_val_loss']
            print(f"Resumed from epoch {start_epoch}")
        else:
            print(f"No checkpoint found at {args.resume}")
    
    # Create checkpoint directory
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # Training loop
    print(f"\n{'='*50}")
    print(f"Starting training for {args.epochs} epochs")
    print(f"{'='*50}\n")
    
    for epoch in range(start_epoch, args.epochs):
        # Train
        train_loss, train_loss_dict = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch + 1
        )
        
        # Validate
        val_loss, val_loss_dict = validate(model, val_loader, criterion, device)
        
        # Update learning rate
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Log to tensorboard
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Learning_rate', current_lr, epoch)
        
        for key in train_loss_dict.keys():
            writer.add_scalar(f'Train/{key}', train_loss_dict[key], epoch)
            writer.add_scalar(f'Val/{key}', val_loss_dict[key], epoch)
        
        # Print epoch summary
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss:   {val_loss:.4f}")
        print(f"  LR:         {current_lr:.6f}")
        
        # Save checkpoint
        is_best = val_loss < best_val_loss
        best_val_loss = min(val_loss, best_val_loss)
        
        checkpoint_path = os.path.join(args.checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pth')
        save_checkpoint(
            checkpoint_path,
            model,
            optimizer,
            epoch + 1,
            val_loss,
            best_val_loss,
            is_best
        )
        
        if is_best:
            best_path = os.path.join(args.checkpoint_dir, 'best_model.pth')
            save_checkpoint(
                best_path,
                model,
                optimizer,
                epoch + 1,
                val_loss,
                best_val_loss,
                is_best
            )
            print(f"  *** New best model saved! (Val Loss: {val_loss:.4f}) ***")
        
        # Save sample images every N epochs
        if (epoch + 1) % args.save_freq == 0:
            sample_dir = os.path.join(args.checkpoint_dir, 'samples', f'epoch_{epoch+1}')
            os.makedirs(sample_dir, exist_ok=True)
            
            model.eval()
            with torch.no_grad():
                batch = next(iter(val_loader))
                low = batch['low'].to(device)
                high = batch['high'].to(device)
                outputs = model(low)
                
                save_images(
                    low[:4],
                    high[:4],
                    outputs['enhanced'][:4],
                    outputs['retinex'][:4],
                    outputs['curve'][:4],
                    sample_dir
                )
            model.train()
    
    writer.close()
    print(f"\n{'='*50}")
    print(f"Training completed!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"{'='*50}\n")


def main():
    parser = argparse.ArgumentParser(description='Train Hybrid Low-Light Enhancement Model')
    
    # Dataset parameters
    parser.add_argument('--data_root', type=str, required=True,
                        help='Root directory of dataset (containing train/val/test folders)')
    parser.add_argument('--image_size', type=int, default=256,
                        help='Image size for training (default: 256)')
    
    # Model parameters
    parser.add_argument('--base_channels', type=int, default=32,
                        help='Base number of channels in model (default: 32)')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs to train (default: 100)')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for training (default: 8)')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate (default: 1e-4)')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers (default: 4)')
    
    # Loss weights
    parser.add_argument('--lambda1', type=float, default=1.0,
                        help='Weight for L1 loss (default: 1.0)')
    parser.add_argument('--lambda2', type=float, default=1.0,
                        help='Weight for SSIM loss (default: 1.0)')
    parser.add_argument('--lambda3', type=float, default=0.1,
                        help='Weight for TV loss (default: 0.1)')
    parser.add_argument('--lambda4', type=float, default=0.5,
                        help='Weight for color consistency loss (default: 0.5)')
    
    # Checkpoint and logging
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints',
                        help='Directory to save checkpoints (default: ./checkpoints)')
    parser.add_argument('--log_dir', type=str, default='./logs',
                        help='Directory for tensorboard logs (default: ./logs)')
    parser.add_argument('--resume', type=str, default='',
                        help='Path to checkpoint to resume from')
    parser.add_argument('--save_freq', type=int, default=5,
                        help='Save sample images every N epochs (default: 5)')
    
    args = parser.parse_args()
    
    # Train
    train(args)


if __name__ == '__main__':
    main()

