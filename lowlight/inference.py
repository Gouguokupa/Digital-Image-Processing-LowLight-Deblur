"""
Inference script for enhancing low-light images
"""

import os
import argparse
import torch
from PIL import Image
import torchvision.transforms as transforms
from tqdm import tqdm

from model import HybridLLE
from utils import load_checkpoint


def enhance_image(model, image_path, device, image_size=None):
    """
    Enhance a single low-light image
    
    Args:
        model: Trained model
        image_path: Path to input image
        device: Device to run inference on
        image_size: Size to resize image to (optional)
        
    Returns:
        enhanced_image: Enhanced PIL Image
    """
    # Load image
    img = Image.open(image_path).convert('RGB')
    original_size = img.size
    
    # Prepare transforms
    transform_list = []
    if image_size is not None:
        transform_list.append(transforms.Resize((image_size, image_size)))
    transform_list.append(transforms.ToTensor())
    transform = transforms.Compose(transform_list)
    
    # Transform and add batch dimension
    img_tensor = transform(img).unsqueeze(0).to(device)
    
    # Inference
    with torch.no_grad():
        outputs = model(img_tensor)
        enhanced_tensor = outputs['enhanced']
    
    # Convert back to PIL Image
    enhanced_tensor = enhanced_tensor.squeeze(0).cpu().clamp(0, 1)
    to_pil = transforms.ToPILImage()
    enhanced_img = to_pil(enhanced_tensor)
    
    # Resize back to original size if needed
    if image_size is not None:
        enhanced_img = enhanced_img.resize(original_size, Image.LANCZOS)
    
    return enhanced_img


def batch_enhance(args):
    """Enhance a batch of images"""
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
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
    
    # Get list of input images
    if os.path.isfile(args.input):
        # Single image
        image_files = [args.input]
        input_dir = os.path.dirname(args.input)
    else:
        # Directory of images
        input_dir = args.input
        image_files = [
            os.path.join(input_dir, f) for f in os.listdir(input_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))
        ]
    
    if len(image_files) == 0:
        raise ValueError(f"No images found in {args.input}")
    
    print(f"\nFound {len(image_files)} images to enhance")
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    print(f"Saving results to {args.output}")
    
    # Process images
    print("\nEnhancing images...")
    pbar = tqdm(image_files, desc='Processing')
    
    for image_path in pbar:
        try:
            # Get filename
            filename = os.path.basename(image_path)
            basename, ext = os.path.splitext(filename)
            
            # Enhance image
            enhanced_img = enhance_image(
                model, image_path, device, 
                image_size=args.image_size if args.image_size > 0 else None
            )
            
            # Save enhanced image
            output_path = os.path.join(args.output, f"{basename}_enhanced{ext}")
            enhanced_img.save(output_path)
            
            # Optionally save comparison
            if args.save_comparison:
                original_img = Image.open(image_path).convert('RGB')
                comparison = Image.new('RGB', (original_img.width * 2, original_img.height))
                comparison.paste(original_img, (0, 0))
                comparison.paste(enhanced_img, (original_img.width, 0))
                comparison_path = os.path.join(args.output, f"{basename}_comparison{ext}")
                comparison.save(comparison_path)
            
            pbar.set_postfix({'current': filename})
            
        except Exception as e:
            print(f"\nError processing {image_path}: {e}")
            continue
    
    print(f"\n{'='*50}")
    print(f"Enhancement completed!")
    print(f"Enhanced images saved to: {args.output}")
    print(f"{'='*50}\n")


def main():
    parser = argparse.ArgumentParser(description='Enhance Low-Light Images')
    
    # Input/Output
    parser.add_argument('--input', type=str, required=True,
                        help='Path to input image or directory of images')
    parser.add_argument('--output', type=str, required=True,
                        help='Path to output directory')
    
    # Model
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--base_channels', type=int, default=32,
                        help='Base number of channels in model (default: 32)')
    
    # Processing
    parser.add_argument('--image_size', type=int, default=-1,
                        help='Resize images to this size (default: -1, keep original size)')
    parser.add_argument('--save_comparison', action='store_true',
                        help='Save side-by-side comparison images')
    
    args = parser.parse_args()
    
    # Run inference
    batch_enhance(args)


if __name__ == '__main__':
    main()

