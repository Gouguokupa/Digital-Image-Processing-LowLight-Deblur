import os
import argparse
import numpy as np
from PIL import Image
import torch
import matplotlib.pyplot as plt
import sys
import importlib.util
import cv2

try:
    from skimage.metrics import peak_signal_noise_ratio as psnr
    from skimage.metrics import structural_similarity as ssim
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False
    def psnr(img1, img2, data_range=255):
        mse = np.mean((img1.astype(np.float32) - img2.astype(np.float32)) ** 2)
        if mse == 0:
            return float('inf')
        return 20 * np.log10(data_range / np.sqrt(mse))

def load_model_and_checkpoint(checkpoint_path):
    model_path = 'DeblurGAN_COMPLETE.py'
    if not os.path.exists(model_path):
        exit(1)
    
    spec = importlib.util.spec_from_file_location("deblurgan_complete", model_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules['deblurgan_complete'] = module
    spec.loader.exec_module(module)
    FPNGenerator = module.FPNGenerator
    DeblurDataset = module.DeblurDataset
    
    if not os.path.exists(checkpoint_path):
        exit(1)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = FPNGenerator().to(device)
    
    state_dict = checkpoint.get('G') or checkpoint.get('generator_state_dict') or checkpoint
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    
    return model, device, DeblurDataset

def apply_tta(model, blur_tensor, device):
    outputs = []
    
    def get_output(tensor):
        with torch.no_grad():
            out = model(tensor)
            return out['scale1'] if isinstance(out, dict) else out
    
    outputs.append(get_output(blur_tensor))
    outputs.append(torch.flip(get_output(torch.flip(blur_tensor, [3])), [3]))
    outputs.append(torch.flip(get_output(torch.flip(blur_tensor, [2])), [2]))
    outputs.append(torch.flip(get_output(torch.flip(blur_tensor, [2, 3])), [2, 3]))
    outputs.append(torch.rot90(get_output(torch.rot90(blur_tensor, 1, [2, 3])), -1, [2, 3]))
    outputs.append(torch.rot90(get_output(torch.rot90(blur_tensor, 2, [2, 3])), -2, [2, 3]))
    outputs.append(torch.rot90(get_output(torch.rot90(blur_tensor, 3, [2, 3])), -3, [2, 3]))
    outputs.append(get_output(blur_tensor.transpose(2, 3)).transpose(2, 3))
    
    return torch.stack(outputs).mean(0)

def create_motion_blur_kernel(length=15, angle=0):
    kernel = np.zeros((length, length), dtype=np.float32)
    angle_rad = np.radians(angle)
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
    center = length // 2
    
    for i in range(length):
        x = int(center + (i - center) * cos_a)
        y = int(center + (i - center) * sin_a)
        if 0 <= x < length and 0 <= y < length:
            kernel[y, x] = 1.0
    
    return kernel / np.sum(kernel)

def wiener_filter_single_channel(image, kernel, noise_var=0.01):
    h, w = image.shape
    kh, kw = kernel.shape
    img_float = image.astype(np.float32)
    
    kernel_padded = np.zeros((h, w), dtype=np.float32)
    ky_center, kx_center = h // 2, w // 2
    ky_start = max(0, ky_center - kh // 2)
    kx_start = max(0, kx_center - kw // 2)
    ky_end = min(h, ky_start + kh)
    kx_end = min(w, kx_start + kw)
    
    k_start_y = max(0, -ky_start)
    k_start_x = max(0, -kx_start)
    k_end_y = k_start_y + (ky_end - ky_start)
    k_end_x = k_start_x + (kx_end - kx_start)
    
    kernel_padded[ky_start:ky_end, kx_start:kx_end] = kernel[k_start_y:k_end_y, k_start_x:k_end_x]
    kernel_padded = kernel_padded / (np.sum(kernel_padded) + 1e-10)
    kernel_shifted = np.fft.ifftshift(kernel_padded)
    
    img_fft = np.fft.fft2(img_float)
    kernel_fft = np.fft.fft2(kernel_shifted)
    kernel_conj = np.conj(kernel_fft)
    kernel_mag_sq = np.abs(kernel_fft) ** 2
    wiener_fft = kernel_conj / (kernel_mag_sq + noise_var + 1e-10)
    
    result = np.real(np.fft.ifft2(img_fft * wiener_fft))
    
    result_var = np.var(result)
    original_var = np.var(img_float)
    if result_var < original_var * 0.1 or np.sum(result < -10) > result.size * 0.01:
        result_shifted = np.fft.fftshift(result)
        if np.var(result_shifted) > result_var and np.var(result_shifted) > original_var * 0.5:
            result = result_shifted
    
    return np.clip(result, 0, 255).astype(np.uint8)

def wiener_filter(image, kernel, noise_var=0.01):
    if len(image.shape) == 2:
        return wiener_filter_single_channel(image, kernel, noise_var)
    result = np.zeros_like(image)
    for c in range(image.shape[2]):
        result[:, :, c] = wiener_filter_single_channel(image[:, :, c], kernel, noise_var)
    return result

def optimized_wiener_filter(image, sharp_image=None):
    best_result = image.copy()
    best_score = -np.inf
    
    for length in [5, 9, 15]:
        for angle in [0, 45, 90, 135]:
            for noise_var in [0.005, 0.01, 0.02]:
                try:
                    kernel = create_motion_blur_kernel(length=length, angle=angle)
                    result = wiener_filter(image, kernel, noise_var=noise_var)
                    
                    if sharp_image is not None:
                        score = psnr(sharp_image, result, data_range=255)
                    else:
                        gray = cv2.cvtColor(result, cv2.COLOR_RGB2GRAY) if len(result.shape) == 3 else result
                        score = np.var(gray)
                    
                    if score > best_score:
                        best_score = score
                        best_result = result
                except Exception:
                    continue
    
    return best_result

def unsharp_mask(img, amount=0.5, sigma=1.0, threshold=0):
    from scipy.ndimage import gaussian_filter
    blurred = gaussian_filter(img.astype(np.float32), sigma=sigma)
    sharpened = img.astype(np.float32) + amount * (img.astype(np.float32) - blurred)
    return np.clip(sharpened, 0, 255).astype(np.uint8)

def parse_zoom_arg(zoom_str, img_h, img_w, zoom_size=200):
    if zoom_str is None:
        return None
    
    if ',' not in zoom_str:
        raise ValueError(f"Invalid zoom format: {zoom_str}. Use 'x,y,w,h' or 'center,scale'")
    
    parts = [p.strip() for p in zoom_str.split(',')]
    
    if parts[0].lower() == 'center':
        if len(parts) < 2:
            raise ValueError("center format requires scale: 'center,scale' or 'center,scale,x,y'")
        
        scale = float(parts[1])
        w, h = int(img_w / scale), int(img_h / scale)
        
        if len(parts) >= 4:
            center_x, center_y = int(float(parts[2])), int(float(parts[3]))
        else:
            center_x, center_y = img_w // 2, img_h // 2
        
        x = max(0, min(center_x - w // 2, img_w - w))
        y = max(0, min(center_y - h // 2, img_h - h))
        w = min(w, img_w - x)
        h = min(h, img_h - y)
        
        return (x, y, w, h)
    else:
        if len(parts) != 4:
            raise ValueError("Pixel format requires 4 values: 'x,y,w,h'")
        
        x = max(0, min(int(float(parts[0])), img_w - 1))
        y = max(0, min(int(float(parts[1])), img_h - 1))
        w = min(int(float(parts[2])), img_w - x)
        h = min(int(float(parts[3])), img_h - y)
        
        return (x, y, w, h)

def crop_image(img, crop_coords):
    if crop_coords is None:
        return img
    x, y, w, h = crop_coords
    return img[y:y+h, x:x+w]


def tensor_to_uint8(tensor):
    img = tensor.detach().cpu().numpy() if torch.is_tensor(tensor) else tensor
    if img.ndim == 4:
        img = img[0]
    if img.ndim == 3 and img.shape[0] == 3:
        img = img.transpose(1, 2, 0)
    if img.min() < 0:
        img = ((img + 1) * 127.5).clip(0, 255)
    else:
        img = (img * 255).clip(0, 255)
    return img.astype(np.uint8)

def find_checkpoint(auto_search=True):
    if not auto_search:
        return None
    
    candidates = [
        './checkpoints_complete_stable/best_model.pth',
        './checkpoints_complete/best_model.pth',
        './checkpoints_fixed/best_model.pth',
        './checkpoints_optimized/best_model.pth',
        './checkpoints/best_model.pth',
        '/content/checkpoints_complete_stable/best_model.pth',
        '/content/checkpoints_complete/best_model.pth',
        '/content/drive/MyDrive/deblurgan_checkpoints/best_model.pth',
        '/content/drive/MyDrive/deblurgan_checkpoints/checkpoints_complete_stable/best_model.pth',
        '/content/drive/MyDrive/checkpoints_complete_stable/best_model.pth',
        '/content/drive/MyDrive/checkpoints/best_model.pth',
        '/content/drive/MyDrive/ECE253project/checkpoints_complete_stable/best_model.pth',
        '/content/drive/MyDrive/ECE253project/checkpoints/best_model.pth',
        '/content/drive/MyDrive/best_model.pth',
    ]
    
    for ckpt_path in candidates:
        if os.path.exists(ckpt_path):
            return ckpt_path
    
    if os.path.exists('/content/drive/MyDrive'):
        try:
            import subprocess
            result = subprocess.run(
                ['find', '/content/drive/MyDrive', '-name', 'best_model.pth', '-type', 'f', '-maxdepth', '4'],
                capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0 and result.stdout.strip():
                found_paths = [p.strip() for p in result.stdout.strip().split('\n') if p.strip()]
                if found_paths:
                    return found_paths[0]
        except Exception:
            pass
    
    return None

def main(args_list=None):
    parser = argparse.ArgumentParser(description='Demo: Pick best deblurred results for slides')
    parser.add_argument('--ckpt', default=None, help='Path to checkpoint .pth file')
    parser.add_argument('--blur_dir', default=None, help='Validation blur directory')
    parser.add_argument('--sharp_dir', default=None, help='Validation sharp directory')
    parser.add_argument('--output_dir', default='demo_outputs', help='Output directory')
    parser.add_argument('--tta', action='store_true', help='Use 8x test-time augmentation')
    parser.add_argument('--unsharp', action='store_true', help='Apply unsharp mask')
    parser.add_argument('--single', default=None, help='Process only one image by basename')
    parser.add_argument('--image_size', type=int, default=256, help='Image size')
    parser.add_argument('--zoom', default=None, help='Zoom region: "x,y,w,h" or "center,scale"')
    parser.add_argument('--zoom_size', type=int, default=200, help='Size of zoomed region')
    
    if args_list is None:
        args, unknown = parser.parse_known_args()
        if unknown:
            filtered_unknown = [a for a in unknown if not a.startswith('-f') and 'kernel-' not in a]
            if filtered_unknown:
                print(f"Ignoring unknown arguments: {filtered_unknown}")
    else:
        args = parser.parse_args(args_list)
    
    if args.ckpt is None:
        args.ckpt = find_checkpoint()
    
    if args.ckpt is None or not os.path.exists(args.ckpt):
        print("Checkpoint not found! Please provide --ckpt <path>")
        return
    
    print(f"Using checkpoint: {args.ckpt}")
    
    model, device, DeblurDataset = load_model_and_checkpoint(args.ckpt)
    
    blur_dir = args.blur_dir
    sharp_dir = args.sharp_dir
    
    if blur_dir is None or sharp_dir is None:
        candidates = [
            ('gopro_full/val/blur', 'gopro_full/val/sharp'),
            ('./gopro_full/val/blur', './gopro_full/val/sharp'),
            ('/content/gopro_full/val/blur', '/content/gopro_full/val/sharp'),
            ('DBlur/Gopro/val/blur', 'DBlur/Gopro/val/sharp'),
            ('gopro_subset/val/blur', 'gopro_subset/val/sharp'),
            ('gopro/val/blur', 'gopro/val/sharp'),
            ('/content/drive/MyDrive/gopro_full/val/blur', '/content/drive/MyDrive/gopro_full/val/sharp'),
            ('/content/drive/MyDrive/DBlur/Gopro/val/blur', '/content/drive/MyDrive/DBlur/Gopro/val/sharp'),
            ('/content/drive/MyDrive/gopro_subset/val/blur', '/content/drive/MyDrive/gopro_subset/val/sharp'),
            ('/content/drive/MyDrive/gopro/val/blur', '/content/drive/MyDrive/gopro/val/sharp'),
            ('/content/drive/MyDrive/ECE253project/gopro_full/val/blur', '/content/drive/MyDrive/ECE253project/gopro_full/val/sharp'),
        ]
        for bd, sd in candidates:
            if os.path.exists(bd) and os.path.exists(sd):
                blur_dir, sharp_dir = bd, sd
                print(f"Found validation set: {blur_dir}")
                break
        
        if blur_dir is None:
            print("Validation set not found! Please specify --blur_dir and --sharp_dir")
            return
    
    dataset = DeblurDataset(blur_dir, sharp_dir, image_size=args.image_size, augment=False)
    print(f"Dataset: {len(dataset)} pairs")
    
    if len(dataset) == 0:
        print("Dataset is empty!")
        exit(1)
    
    if args.single:
        filtered_indices = [idx for idx, (bp, sp) in enumerate(dataset.pairs) 
                          if os.path.basename(bp) == args.single or os.path.basename(sp) == args.single]
        if not filtered_indices:
            print(f"Image '{args.single}' not found in dataset")
            exit(1)
        process_indices = filtered_indices[:1]
    else:
        process_indices = list(range(len(dataset)))
    
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"\nProcessing {len(process_indices)} pair(s)...")
    
    results = []
    
    for idx in process_indices:
        if (idx + 1) % 50 == 0:
            print(f"  Processed {idx+1}/{len(dataset)}...")
        
        blur_tensor, sharp_tensor = dataset[idx]
        blur_path, sharp_path = dataset.pairs[idx]
        basename = os.path.basename(blur_path)
        
        blur_batch = blur_tensor.unsqueeze(0).to(device)
        
        with torch.no_grad():
            if args.tta:
                output_tensor = apply_tta(model, blur_batch, device)
            else:
                output = model(blur_batch)
                output_tensor = output['scale1'] if isinstance(output, dict) else output
        
        blur_img = tensor_to_uint8(blur_tensor)
        output_img = tensor_to_uint8(output_tensor.squeeze(0))
        sharp_img = tensor_to_uint8(sharp_tensor)
        
        wiener_img = optimized_wiener_filter(blur_img, sharp_image=sharp_img)
        
        if args.unsharp:
            output_img = unsharp_mask(output_img, amount=0.3, sigma=1.0)
            output_label = " (Unsharp)"
        else:
            output_label = ""
        
        p = psnr(sharp_img, output_img, data_range=255)
        wiener_psnr = psnr(sharp_img, wiener_img, data_range=255)
        input_psnr = psnr(sharp_img, blur_img, data_range=255)
        improvement = p - input_psnr
        wiener_improvement = wiener_psnr - input_psnr
        
        result = {
            'idx': idx, 'basename': basename, 'blur_path': blur_path, 'sharp_path': sharp_path,
            'blur_img': blur_img, 'wiener_img': wiener_img, 'output_img': output_img, 'sharp_img': sharp_img,
            'psnr': p, 'wiener_psnr': wiener_psnr, 'input_psnr': input_psnr,
            'improvement': improvement, 'wiener_improvement': wiener_improvement,
            'output_label': output_label
        }
        
        if HAS_SKIMAGE:
            s = ssim(sharp_img, output_img, channel_axis=2, data_range=255)
            wiener_ssim = ssim(sharp_img, wiener_img, channel_axis=2, data_range=255)
            input_ssim = ssim(sharp_img, blur_img, channel_axis=2, data_range=255)
            result.update({
                'ssim': s, 'wiener_ssim': wiener_ssim, 'input_ssim': input_ssim,
                'ssim_improvement': s - input_ssim,
                'wiener_ssim_improvement': wiener_ssim - input_ssim
            })
        else:
            result.update({'ssim': None, 'wiener_ssim': None, 'input_ssim': None,
                          'ssim_improvement': None, 'wiener_ssim_improvement': None})
        
        results.append(result)
    
    results.sort(key=lambda x: x.get('improvement', 0), reverse=True)
    
    print(f"\nSaving top 5 results to {args.output_dir}/")
    top_n = min(5, len(results))
    
    for rank, result in enumerate(results[:top_n], 1):
        basename = result['basename']
        psnr_val = result['psnr']
        ssim_val = result.get('ssim')
        input_psnr_val = result.get('input_psnr', 0)
        wiener_psnr_val = result.get('wiener_psnr', 0)
        wiener_improvement_val = result.get('wiener_improvement', 0)
        improvement_val = result.get('improvement', 0)
        
        crop_coords = None
        if args.zoom:
            img_h, img_w = result['blur_img'].shape[:2]
            try:
                crop_coords = parse_zoom_arg(args.zoom, img_h, img_w, args.zoom_size)
            except Exception as e:
                print(f"Invalid zoom argument '{args.zoom}': {e}")
                crop_coords = None
        
        if crop_coords is not None:
            from matplotlib.gridspec import GridSpec
            fig = plt.figure(figsize=(24, 12))
            gs = GridSpec(2, 4, figure=fig, hspace=0.3, wspace=0.1)
            axes = [fig.add_subplot(gs[0, i]) for i in range(4)]
        else:
            fig, axes = plt.subplots(1, 4, figsize=(24, 6))
        
        axes[0].imshow(result['blur_img'])
        axes[0].set_title(f'Blurred Input\nPSNR: {input_psnr_val:.2f} dB', fontsize=12, fontweight='bold')
        axes[0].axis('off')
        
        axes[1].imshow(result['wiener_img'])
        title_wiener = f'Wiener Filter\nPSNR: {wiener_psnr_val:.2f} dB ({wiener_improvement_val:+.2f})'
        if result.get('wiener_ssim') is not None:
            title_wiener += f'\nSSIM: {result.get("wiener_ssim", 0):.3f} ({result.get("wiener_ssim_improvement", 0):+.3f})'
        axes[1].set_title(title_wiener, fontsize=12, fontweight='bold', 
                         color='green' if wiener_improvement_val > 0 else 'orange')
        axes[1].axis('off')
        
        axes[2].imshow(result['output_img'])
        title = f'DeblurGAN Output{result["output_label"]}\nPSNR: {psnr_val:.2f} dB ({improvement_val:+.2f})'
        if ssim_val is not None:
            title += f'\nSSIM: {ssim_val:.3f} ({result.get("ssim_improvement", 0):+.3f})'
        axes[2].set_title(title, fontsize=12, fontweight='bold', 
                         color='green' if improvement_val > 0 else 'red')
        axes[2].axis('off')
        
        axes[3].imshow(result['sharp_img'])
        axes[3].set_title('Ground Truth', fontsize=12, fontweight='bold', color='blue')
        axes[3].axis('off')
        
        if crop_coords is not None:
            images = ['blur_img', 'wiener_img', 'output_img', 'sharp_img']
            titles_zoom = ['Blurred (Zoom)', 'Wiener (Zoom)', 'DeblurGAN (Zoom)', 'Ground Truth (Zoom)']
            axes_zoom = [fig.add_subplot(gs[1, i]) for i in range(4)]
            
            for idx, (img_key, title_zoom, ax_zoom) in enumerate(zip(images, titles_zoom, axes_zoom)):
                cropped = crop_image(result[img_key], crop_coords)
                ax_zoom.imshow(cropped)
                ax_zoom.set_title(title_zoom, fontsize=10, fontweight='bold')
                ax_zoom.axis('off')
                
                x, y, w, h = crop_coords
                rect = plt.Rectangle((x, y), w, h, linewidth=2, edgecolor='red', facecolor='none')
                axes[idx].add_patch(rect)
        
        filename = f"rank{rank:02d}_improvement{improvement_val:+.2f}dB_{basename}"
        filepath = os.path.join(args.output_dir, filename)
        
        plt.suptitle(f'Rank #{rank}: {basename} (Improvement: {improvement_val:+.2f} dB, Input: {input_psnr_val:.2f} dB → Output: {psnr_val:.2f} dB)', 
                     fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        
        output_filepath = os.path.join(args.output_dir, f"{os.path.splitext(basename)[0]}_pred.png")
        Image.fromarray(result['output_img']).save(output_filepath)
        
        print(f"  [{rank}] {basename}: Input={input_psnr_val:.2f} dB | Wiener={wiener_psnr_val:.2f} dB ({wiener_improvement_val:+.2f}) | DeblurGAN={psnr_val:.2f} dB ({improvement_val:+.2f})", end="")
        if ssim_val is not None:
            print(f" | SSIM: Wiener {result.get('wiener_ssim_improvement', 0):+.3f}, DeblurGAN {result.get('ssim_improvement', 0):+.3f}")
        else:
            print()
    
    avg_improvement = np.mean([r.get('improvement', 0) for r in results[:top_n]])
    
    if avg_improvement < 0:
        print(f"\nWARNING: Model is degrading quality! Average loss: {avg_improvement:.2f} dB")
    elif avg_improvement < 1:
        print(f"\nModel shows minimal improvement ({avg_improvement:.2f} dB)")
    elif avg_improvement < 3:
        print(f"\nModel shows moderate improvement: {avg_improvement:.2f} dB gain")
    else:
        print(f"\nModel shows excellent improvement: {avg_improvement:.2f} dB gain!")
    
    if HAS_SKIMAGE:
        avg_input_ssim = np.mean([r.get('input_ssim', 0) for r in results[:top_n] if r.get('input_ssim')])
        avg_wiener_ssim = np.mean([r.get('wiener_ssim', 0) for r in results[:top_n] if r.get('wiener_ssim')])
        avg_output_ssim = np.mean([r['ssim'] for r in results[:top_n] if r['ssim']])
        avg_wiener_ssim_improvement = np.mean([r.get('wiener_ssim_improvement', 0) for r in results[:top_n] if r.get('wiener_ssim_improvement')])
        avg_ssim_improvement = np.mean([r.get('ssim_improvement', 0) for r in results[:top_n] if r.get('ssim_improvement')])
        print(f"\nInput SSIM: {avg_input_ssim:.3f}")
        print(f"Wiener SSIM: {avg_wiener_ssim:.3f} (improvement: {avg_wiener_ssim_improvement:+.3f})")
        print(f"DeblurGAN SSIM: {avg_output_ssim:.3f} (improvement: {avg_ssim_improvement:+.3f})")

if __name__ == '__main__':
    main()

