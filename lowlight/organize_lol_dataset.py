"""
Script to organize the LOL dataset into the required structure
Handles different possible structures from Kaggle download
"""

import os
import shutil
import argparse
from pathlib import Path
from tqdm import tqdm


def find_lol_images(root_dir):
    """
    Find low and high light image directories in the downloaded structure
    
    Args:
        root_dir: Root directory of unzipped LOL dataset
        
    Returns:
        Dictionary with 'train_low', 'train_high', 'test_low', 'test_high' paths
    """
    paths = {}
    
    # Common possible structures
    possible_structures = [
        # Structure 1: Direct low/high folders
        {
            'train_low': 'our485/low',
            'train_high': 'our485/high',
            'test_low': 'eval15/low',
            'test_high': 'eval15/high'
        },
        # Structure 2: LOL prefix
        {
            'train_low': 'LOL/our485/low',
            'train_high': 'LOL/our485/high',
            'test_low': 'LOL/eval15/low',
            'test_high': 'LOL/eval15/high'
        },
        # Structure 3: lol-dataset prefix
        {
            'train_low': 'lol-dataset/our485/low',
            'train_high': 'lol-dataset/our485/high',
            'test_low': 'lol-dataset/eval15/low',
            'test_high': 'lol-dataset/eval15/high'
        },
        # Structure 4: Alternative naming
        {
            'train_low': 'train/low',
            'train_high': 'train/high',
            'test_low': 'test/low',
            'test_high': 'test/high'
        },
    ]
    
    # Try to find the correct structure
    for structure in possible_structures:
        all_exist = True
        temp_paths = {}
        
        for key, rel_path in structure.items():
            full_path = os.path.join(root_dir, rel_path)
            if os.path.exists(full_path):
                temp_paths[key] = full_path
            else:
                all_exist = False
                break
        
        if all_exist:
            print(f"✓ Found LOL dataset structure")
            return temp_paths
    
    # If no structure matched, try to find directories by searching
    print("Standard structure not found, searching for directories...")
    
    for root, dirs, files in os.walk(root_dir):
        for dir_name in dirs:
            dir_path = os.path.join(root, dir_name)
            
            if 'low' in dir_name.lower():
                if 'train' in root.lower() or 'our485' in root.lower():
                    paths['train_low'] = dir_path
                elif 'test' in root.lower() or 'eval' in root.lower():
                    paths['test_low'] = dir_path
            
            elif 'high' in dir_name.lower() or 'normal' in dir_name.lower():
                if 'train' in root.lower() or 'our485' in root.lower():
                    paths['train_high'] = dir_path
                elif 'test' in root.lower() or 'eval' in root.lower():
                    paths['test_high'] = dir_path
    
    if len(paths) == 4:
        print("✓ Found directories by searching")
        return paths
    
    raise ValueError(f"Could not find LOL dataset structure in {root_dir}. Found: {paths}")


def copy_images(src_dir, dst_dir, desc="Copying"):
    """
    Copy images from source to destination
    
    Args:
        src_dir: Source directory
        dst_dir: Destination directory
        desc: Description for progress bar
    """
    os.makedirs(dst_dir, exist_ok=True)
    
    image_files = [f for f in os.listdir(src_dir) 
                   if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    
    for filename in tqdm(image_files, desc=desc):
        src_path = os.path.join(src_dir, filename)
        dst_path = os.path.join(dst_dir, filename)
        shutil.copy2(src_path, dst_path)
    
    return len(image_files)


def organize_dataset(input_dir, output_dir, val_ratio=0.10):
    """
    Organize LOL dataset into train/val/test structure
    
    Args:
        input_dir: Directory containing raw LOL dataset
        output_dir: Output directory for organized dataset
        val_ratio: Ratio of training data to use for validation
    """
    print(f"\n{'='*60}")
    print(f"Organizing LOL Dataset")
    print(f"{'='*60}\n")
    
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    
    # Find LOL dataset directories
    lol_paths = find_lol_images(input_dir)
    
    print("\nFound directories:")
    for key, path in lol_paths.items():
        print(f"  {key}: {path}")
    
    # Create output directory structure
    output_train_low = os.path.join(output_dir, 'train', 'low')
    output_train_high = os.path.join(output_dir, 'train', 'high')
    output_val_low = os.path.join(output_dir, 'val', 'low')
    output_val_high = os.path.join(output_dir, 'val', 'high')
    output_test_low = os.path.join(output_dir, 'test', 'low')
    output_test_high = os.path.join(output_dir, 'test', 'high')
    
    # Copy test set (eval15)
    print("\n" + "="*60)
    print("Processing test set...")
    print("="*60)
    test_low_count = copy_images(lol_paths['test_low'], output_test_low, "Copying test low")
    test_high_count = copy_images(lol_paths['test_high'], output_test_high, "Copying test high")
    print(f"✓ Test set: {test_low_count} image pairs")
    
    # Split training set into train and validation
    print("\n" + "="*60)
    print("Processing training set...")
    print("="*60)
    
    train_low_files = sorted([f for f in os.listdir(lol_paths['train_low']) 
                             if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))])
    train_high_files = sorted([f for f in os.listdir(lol_paths['train_high']) 
                              if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))])
    
    # Find matching pairs
    train_pairs = []
    for filename in train_low_files:
        if filename in train_high_files:
            train_pairs.append(filename)
    
    print(f"Found {len(train_pairs)} training image pairs")
    
    # Shuffle and split
    import random
    random.seed(42)
    random.shuffle(train_pairs)
    
    val_count = int(len(train_pairs) * val_ratio)
    val_pairs = train_pairs[:val_count]
    train_pairs = train_pairs[val_count:]
    
    print(f"Split: {len(train_pairs)} train, {len(val_pairs)} validation")
    
    # Copy training images
    os.makedirs(output_train_low, exist_ok=True)
    os.makedirs(output_train_high, exist_ok=True)
    
    for filename in tqdm(train_pairs, desc="Copying train images"):
        shutil.copy2(
            os.path.join(lol_paths['train_low'], filename),
            os.path.join(output_train_low, filename)
        )
        shutil.copy2(
            os.path.join(lol_paths['train_high'], filename),
            os.path.join(output_train_high, filename)
        )
    
    # Copy validation images
    os.makedirs(output_val_low, exist_ok=True)
    os.makedirs(output_val_high, exist_ok=True)
    
    for filename in tqdm(val_pairs, desc="Copying validation images"):
        shutil.copy2(
            os.path.join(lol_paths['train_low'], filename),
            os.path.join(output_val_low, filename)
        )
        shutil.copy2(
            os.path.join(lol_paths['train_high'], filename),
            os.path.join(output_val_high, filename)
        )
    
    # Summary
    print(f"\n{'='*60}")
    print("Dataset Organization Complete!")
    print(f"{'='*60}")
    print(f"\nDataset summary:")
    print(f"  Training:   {len(train_pairs)} pairs ({len(train_pairs)/(len(train_pairs)+len(val_pairs)+test_low_count)*100:.1f}%)")
    print(f"  Validation: {len(val_pairs)} pairs ({len(val_pairs)/(len(train_pairs)+len(val_pairs)+test_low_count)*100:.1f}%)")
    print(f"  Test:       {test_low_count} pairs ({test_low_count/(len(train_pairs)+len(val_pairs)+test_low_count)*100:.1f}%)")
    print(f"  Total:      {len(train_pairs)+len(val_pairs)+test_low_count} pairs")
    print(f"\nOutput structure:")
    print(f"  {output_dir}/")
    print(f"    ├── train/")
    print(f"    │   ├── low/  ({len(train_pairs)} images)")
    print(f"    │   └── high/ ({len(train_pairs)} images)")
    print(f"    ├── val/")
    print(f"    │   ├── low/  ({len(val_pairs)} images)")
    print(f"    │   └── high/ ({len(val_pairs)} images)")
    print(f"    └── test/")
    print(f"        ├── low/  ({test_low_count} images)")
    print(f"        └── high/ ({test_low_count} images)")
    print(f"\n✅ Ready for training!")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description='Organize LOL Dataset')
    
    parser.add_argument('--input', type=str, required=True,
                        help='Path to raw LOL dataset directory')
    parser.add_argument('--output', type=str, default='./dataset',
                        help='Output directory for organized dataset (default: ./dataset)')
    parser.add_argument('--val_ratio', type=float, default=0.10,
                        help='Ratio of training data for validation (default: 0.10)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        raise ValueError(f"Input directory not found: {args.input}")
    
    organize_dataset(args.input, args.output, args.val_ratio)
    
    # Verify the dataset can be loaded
    print("\nVerifying dataset...")
    try:
        from dataset import create_dataloaders
        train_loader, val_loader, test_loader = create_dataloaders(
            args.output, batch_size=2, num_workers=0
        )
        print(f"✅ Dataset verification successful!")
        print(f"   - Train batches: {len(train_loader)}")
        print(f"   - Val batches: {len(val_loader)}")
        print(f"   - Test batches: {len(test_loader)}")
    except Exception as e:
        print(f"⚠️  Warning: Could not verify dataset: {e}")


if __name__ == '__main__':
    main()

