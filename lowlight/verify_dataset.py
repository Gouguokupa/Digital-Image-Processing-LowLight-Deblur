"""
Verify dataset composition and identify self-captured vs LOL images
"""

import os


def analyze_dataset(dataset_root='./dataset'):
    """
    Analyze dataset composition and identify sources
    
    Args:
        dataset_root: Root directory of dataset
    """
    print("\n" + "="*70)
    print("DATASET COMPOSITION ANALYSIS")
    print("="*70 + "\n")
    
    # Count images in each split
    splits = ['train', 'val', 'test']
    
    for split in splits:
        low_dir = os.path.join(dataset_root, split, 'low')
        high_dir = os.path.join(dataset_root, split, 'high')
        
        if not os.path.exists(low_dir):
            print(f"⚠️  {split}/ directory not found, skipping...")
            continue
        
        low_files = sorted([f for f in os.listdir(low_dir) 
                           if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))])
        high_files = sorted([f for f in os.listdir(high_dir) 
                            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))])
        
        # Identify LOL vs custom images
        lol_images = []
        custom_images = []
        
        for filename in low_files:
            # LOL images typically have numeric names or specific patterns
            # Custom images might have different naming
            if filename.startswith('custom') or filename.startswith('self'):
                custom_images.append(filename)
            else:
                lol_images.append(filename)
        
        # Count total
        total = len(low_files)
        lol_count = len(lol_images)
        custom_count = len(custom_images)
        
        print(f"📁 {split.upper()} SET:")
        print(f"   Total pairs: {total}")
        print(f"   ├─ LOL dataset: {lol_count} pairs")
        print(f"   └─ Self-captured: {custom_count} pairs")
        
        if custom_count > 0:
            print(f"   ✅ Custom pairs detected:")
            for img in custom_images[:5]:  # Show first 5
                print(f"      - {img}")
            if custom_count > 5:
                print(f"      ... and {custom_count - 5} more")
        
        # Check for matching pairs
        unmatched = []
        for low_file in low_files:
            if low_file not in high_files:
                unmatched.append(low_file)
        
        if unmatched:
            print(f"   ⚠️  WARNING: {len(unmatched)} unmatched images found:")
            for img in unmatched[:3]:
                print(f"      - {img}")
        else:
            print(f"   ✅ All pairs properly matched")
        
        print()
    
    # Summary
    train_low_dir = os.path.join(dataset_root, 'train', 'low')
    if os.path.exists(train_low_dir):
        total_train = len([f for f in os.listdir(train_low_dir) 
                          if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))])
        
        print("="*70)
        print("SUMMARY")
        print("="*70)
        print(f"\n📊 Dataset Statistics:")
        print(f"   Total training pairs: {total_train}")
        print(f"   Expected LOL training: 437 pairs")
        
        if total_train > 437:
            custom_count_estimate = total_train - 437
            print(f"   Estimated self-captured: ~{custom_count_estimate} pairs ✅")
            print(f"\n✅ SUCCESS: Self-captured pairs detected in training set!")
            print(f"   Your model is trained on HYBRID dataset (LOL + Custom)")
        elif total_train == 437:
            print(f"   Self-captured: 0 pairs")
            print(f"\n⚠️  NOTE: Only LOL dataset detected")
            print(f"   To add custom pairs, place them in:")
            print(f"   - ./dataset/train/low/custom_XXX.png")
            print(f"   - ./dataset/train/high/custom_XXX.png")
        else:
            print(f"\n⚠️  WARNING: Fewer than expected LOL images")
        
        print("\n" + "="*70 + "\n")


def check_data_loader():
    """Verify DataLoader can load the dataset"""
    try:
        from dataset import create_dataloaders
        
        print("="*70)
        print("DATALOADER VERIFICATION")
        print("="*70 + "\n")
        
        train_loader, val_loader, test_loader = create_dataloaders(
            './dataset', batch_size=2, num_workers=0
        )
        
        print(f"✅ DataLoader successfully created!")
        print(f"   Training batches: {len(train_loader)}")
        print(f"   Training images: {len(train_loader.dataset)}")
        print(f"   Validation batches: {len(val_loader)}")
        print(f"   Test batches: {len(test_loader)}")
        
        # Get one batch to verify
        batch = next(iter(train_loader))
        print(f"\n✅ Sample batch loaded:")
        print(f"   Low image shape: {batch['low'].shape}")
        print(f"   High image shape: {batch['high'].shape}")
        print(f"   Filenames: {batch['filename']}")
        
        print("\n" + "="*70 + "\n")
        
    except Exception as e:
        print(f"\n❌ Error loading dataset: {e}\n")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Verify Dataset Composition')
    parser.add_argument('--data_root', type=str, default='./dataset',
                        help='Root directory of dataset')
    
    args = parser.parse_args()
    
    # Analyze dataset
    analyze_dataset(args.data_root)
    
    # Verify DataLoader
    check_data_loader()
    
    print("\n💡 TIP: Your training uses ALL images in train/low/ and train/high/")
    print("   This includes both LOL dataset and any self-captured pairs!")
    print("   The model automatically benefits from the hybrid dataset.\n")

