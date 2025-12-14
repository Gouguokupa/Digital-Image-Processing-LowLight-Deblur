import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from google.colab import files
from scipy.ndimage import convolve
import os

uploaded = files.upload()
image_path = list(uploaded.keys())[0]

original_img = np.array(Image.open(image_path).convert('RGB'))
original_img = np.array(Image.fromarray(original_img).resize((256, 256)))

def create_realistic_camera_shake_blur(length=20, intensity=1.0):
    kernel_size = max(int(length * 2.5), 51)
    kernel = np.zeros((kernel_size, kernel_size))
    center = kernel_size // 2
    
    num_points = int(length * 2)
    path_points = []
    
    angle = np.random.uniform(0, 2 * np.pi)
    x, y = center, center
    path_points.append((int(x), int(y)))
    
    for i in range(num_points):
        angle += np.random.uniform(-0.3, 0.3)
        
        t = i / num_points
        speed = np.sin(t * np.pi) * intensity
        
        x += speed * np.cos(angle) * (length / num_points)
        y += speed * np.sin(angle) * (length / num_points)
        
        if i % 3 == 0:
            angle += np.random.uniform(-0.1, 0.1)
        
        path_points.append((int(x), int(y)))
    
    for i, (px, py) in enumerate(path_points):
        if 0 <= px < kernel_size and 0 <= py < kernel_size:
            intensity_factor = np.sin(i / len(path_points) * np.pi)
            kernel[py, px] += intensity_factor
    
    from scipy.ndimage import gaussian_filter
    kernel = gaussian_filter(kernel, sigma=1.0)
    kernel = kernel / np.sum(kernel) if np.sum(kernel) > 0 else kernel
    return kernel

def add_camera_noise(image, noise_level=0.005):
    noise = np.random.normal(0, noise_level * 255, image.shape)
    noisy = image.astype(np.float32) + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)

blur_length = 22
blur_intensity = 1.2

motion_kernel = create_realistic_camera_shake_blur(blur_length, blur_intensity)

blurred_img = np.zeros_like(original_img)
for c in range(3):
    blurred_img[:, :, c] = convolve(original_img[:, :, c], motion_kernel, mode='reflect')

blurred_img = blurred_img.astype(np.uint8)

blurred_img = add_camera_noise(blurred_img, noise_level=0.005)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

checkpoint_paths = [
    './checkpoints_complete_stable/best_model.pth',
    './checkpoints_complete/best_model.pth',
    '/content/drive/MyDrive/deblurgan_checkpoints/checkpoints_complete_stable_best_model.pth',
    '/content/drive/MyDrive/deblur_checkpoints_complete/best_model.pth',
    './checkpoints_optimized/best_model.pth',
    './checkpoints_fixed/best_model.pth'
]

checkpoint_path = None
for path in checkpoint_paths:
    if os.path.exists(path):
        checkpoint_path = path
        break

if checkpoint_path is None:
    raise FileNotFoundError("No checkpoint found!")

import DeblurGAN_COMPLETE
FPNGenerator = DeblurGAN_COMPLETE.FPNGenerator

checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
state_dict = checkpoint.get('G') or checkpoint.get('generator_state_dict')
if state_dict is None:
    raise KeyError(f"Checkpoint must contain 'G' or 'generator_state_dict'. Found keys: {list(checkpoint.keys())}")

generator = FPNGenerator().to(device)
generator.load_state_dict(state_dict, strict=False)
generator.eval()

transform = torch.nn.Sequential(
    torch.nn.AdaptiveAvgPool2d((256, 256))
)

blur_tensor = torch.from_numpy(blurred_img).permute(2, 0, 1).float() / 255.0
blur_tensor = (blur_tensor - 0.5) / 0.5
blur_tensor = blur_tensor.unsqueeze(0).to(device)

with torch.no_grad():
    outputs = generator(blur_tensor)
    if isinstance(outputs, dict):
        deblurred_tensor = outputs['scale1']
    else:
        deblurred_tensor = outputs

deblurred_img = deblurred_tensor.squeeze(0).cpu().numpy().transpose(1, 2, 0)
deblurred_img = ((deblurred_img + 1) * 127.5).clip(0, 255).astype(np.uint8)

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

axes[0].imshow(blurred_img)
axes[0].set_title('Blurred', fontsize=14, fontweight='bold', color='red')
axes[0].axis('off')

axes[1].imshow(deblurred_img)
axes[1].set_title('DeblurGAN Output', fontsize=14, fontweight='bold', color='green')
axes[1].axis('off')

axes[2].imshow(original_img)
axes[2].set_title('Original (Sharp)', fontsize=14, fontweight='bold', color='blue')
axes[2].axis('off')

plt.suptitle('DeblurGAN Test Results', fontsize=16, fontweight='bold', y=0.98)
plt.tight_layout()
plt.show()

from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

blurred_psnr = psnr(original_img, blurred_img, data_range=255)
deblurred_psnr = psnr(original_img, deblurred_img, data_range=255)
blurred_ssim = ssim(original_img, blurred_img, channel_axis=2, data_range=255)
deblurred_ssim = ssim(original_img, deblurred_img, channel_axis=2, data_range=255)

print(f"Blurred:    PSNR={blurred_psnr:.2f} dB, SSIM={blurred_ssim:.4f}")
print(f"Deblurred:  PSNR={deblurred_psnr:.2f} dB, SSIM={deblurred_ssim:.4f}")
print(f"Gain:       PSNR={deblurred_psnr - blurred_psnr:+.2f} dB, SSIM={deblurred_ssim - blurred_ssim:+.4f}")
