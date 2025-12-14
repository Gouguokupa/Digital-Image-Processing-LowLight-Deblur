"""
Dataset preparation script
Combines LOL dataset and custom images, then splits into train/val/test
"""

import os
import shutil
import argparse
import random
from pathlib import Path
from tqdm import tqdm


def copy_paired_images(low_dir, high_dir, output_low, output_high):
    """
    Copy paired images from source to destination
    
    Args:
        low_dir: Source directory for low-light images
        high_dir: Source directory for normal-light images
        output_low: Destination directory for low-light images
        output_high: Destination directory for normal-light images
        
    Returns:
        Number of pairs copied
    """
    os.makedirs(output_low, exist_ok=True)
    os.makedirs(output_high, exist_ok=True)
    
    # Get list of images
    low_files = sorted([f for f in os.listdir(low_dir) 
                       if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))])
    high_files = sorted([f for f in os.listdir(high_dir) 
                        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))])
    
    # Find matching pairs
    pairs = []
    for low_file in low_files:
        if low_file in high_files:
            pairs.append(low_file)
    
    print(f"Found {len(pairs)} matching pairs")
    
    # Copy pairs
    for filename in tqdm(pairs, desc="Copying images"):
        shutil.copy2(
            os.path.join(low_dir, filename),
            os.path.join(output_low, filename)
        )
        shutil.copy2(
            os.path.join(high_dir, filename),
            os.path.join(output_high, filename)
        )
    
    return len(pairs)


def split_dataset(source_low, source_high, output_dir, train_ratio, test_ratio, val_ratio):
    """
    Split dataset into train/val/test sets
    
    Args:
        source_low: Directory containing all low-light images
        source_high: Directory containing all normal-light images
        output_dir: Output directory for split dataset
        train_ratio: Ratio of training data
        test_ratio: Ratio of test data
        val_ratio: Ratio of validation data
    """
    # Verify ratios sum to 1
    assert abs(train_ratio + test_ratio + val_ratio - 1.0) < 1e-6, \
        f"Ratios must sum to 1.0, got {train_ratio + test_ratio + val_ratio}"
    
    # Get list of image pairs
    low_files = sorted([f for f in os.listdir(source_low) 
                       if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))])
    
    # Shuffle
    random.shuffle(low_files)
    
    total = len(low_files)
    train_count = int(total * train_ratio)
    test_count = int(total * test_ratio)
    
    train_files = low_files[:train_count]
    test_files = low_files[train_count:train_count + test_count]
    val_files = low_files[train_count + test_count:]
    
    print(f"\nSplitting {total} images:")
    print(f"  Train: {len(train_files)} ({len(train_files)/total*100:.1f}%)")
    print(f"  Test:  {len(test_files)} ({len(test_files)/total*100:.1f}%)")
    print(f"  Val:   {len(val_files)} ({len(val_files)/total*100:.1f}%)")
    
    # Create directories
    splits = {
        'train': train_files,
        'test': test_files,
        'val': val_files
    }
    
    for split_name, files in splits.items():
        split_low = os.path.join(output_dir, split_name, 'low')
        split_high = os.path.join(output_dir, split_name, 'high')
        os.makedirs(split_low, exist_ok=True)
        os.makedirs(split_high, exist_ok=True)
        
        print(f"\nCopying {split_name} set...")
        for filename in tqdm(files):
            shutil.copy2(
                os.path.join(source_low, filename),
                os.path.join(split_low, filename)
            )
            shutil.copy2(
                os.path.join(source_high, filename),
                os.path.join(split_high, filename)
            )


def prepare_dataset(args):
    """Main dataset preparation function"""
    
    print(f"{'='*50}")
    print("Dataset Preparation")
    print(f"{'='*50}\n")
    
    # Create temporary directory for combined dataset
    temp_dir = os.path.join(args.output_dir, 'temp')
    temp_low = os.path.join(temp_dir, 'low')
    temp_high = os.path.join(temp_dir, 'high')
    os.makedirs(temp_low, exist_ok=True)
    os.makedirs(temp_high, exist_ok=True)
    
    total_pairs = 0
    
    # Copy LOL dataset
    if args.lol_path and os.path.exists(args.lol_path):
        print("Processing LOL dataset...")
        lol_low = os.path.join(args.lol_path, 'low')
        lol_high = os.path.join(args.lol_path, 'high')
        
        if os.path.exists(lol_low) and os.path.exists(lol_high):
            count = copy_paired_images(lol_low, lol_high, temp_low, temp_high)
            total_pairs += count
            print(f"Added {count} pairs from LOL dataset\n")
        else:
            print(f"Warning: Could not find low/high directories in {args.lol_path}\n")
    
    # Copy custom dataset
    if args.custom_path and os.path.exists(args.custom_path):
        print("Processing custom dataset...")
        custom_low = os.path.join(args.custom_path, 'low')
        custom_high = os.path.join(args.custom_path, 'high')
        
        if os.path.exists(custom_low) and os.path.exists(custom_high):
            count = copy_paired_images(custom_low, custom_high, temp_low, temp_high)
            total_pairs += count
            print(f"Added {count} pairs from custom dataset\n")
        else:
            print(f"Warning: Could not find low/high directories in {args.custom_path}\n")
    
    if total_pairs == 0:
        raise ValueError("No image pairs found! Check your dataset paths.")
    
    print(f"Total image pairs: {total_pairs}\n")
    
    # Split dataset
    print("Splitting dataset into train/val/test...")
    split_dataset(
        temp_low, temp_high, args.output_dir,
        args.train_ratio, args.test_ratio, args.val_ratio
    )
    
    # Clean up temporary directory
    print("\nCleaning up...")
    shutil.rmtree(temp_dir)
    
    print(f"\n{'='*50}")
    print("Dataset preparation completed!")
    print(f"Dataset saved to: {args.output_dir}")
    print(f"{'='*50}\n")


def main():
    parser = argparse.ArgumentParser(description='Prepare Low-Light Enhancement Dataset')
    
    # Input paths
    parser.add_argument('--lol_path', type=str, default='',
                        help='Path to LOL dataset directory (containing low/ and high/ folders)')
    parser.add_argument('--custom_path', type=str, default='',
                        help='Path to custom dataset directory (containing low/ and high/ folders)')
    
    # Output
    parser.add_argument('--output_dir', type=str, default='./dataset',
                        help='Output directory for prepared dataset (default: ./dataset)')
    
    # Split ratios
    parser.add_argument('--train_ratio', type=float, default=0.75,
                        help='Training set ratio (default: 0.75)')
    parser.add_argument('--test_ratio', type=float, default=0.15,
                        help='Test set ratio (default: 0.15)')
    parser.add_argument('--val_ratio', type=float, default=0.10,
                        help='Validation set ratio (default: 0.10)')
    
    # Random seed
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility (default: 42)')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not args.lol_path and not args.custom_path:
        parser.error("At least one of --lol_path or --custom_path must be specified")
    
    # Set random seed
    random.seed(args.seed)
    
    # Prepare dataset
    prepare_dataset(args)


if __name__ == '__main__':
    main()

