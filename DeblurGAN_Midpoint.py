import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import vgg19
import numpy as np
from PIL import Image
import cv2
import os
import sys
import glob
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

class ResidualBlock(nn.Module):
    def __init__(self, num_filters):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(num_filters, num_filters, 3, padding=1, padding_mode='reflect'),
            nn.InstanceNorm2d(num_filters),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_filters, num_filters, 3, padding=1, padding_mode='reflect'),
            nn.InstanceNorm2d(num_filters)
        )
    
    def forward(self, x):
        return x + self.block(x)


class FPNHead(nn.Module):
    def __init__(self, num_in_channels, num_out_channels):
        super(FPNHead, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(num_in_channels, num_out_channels, 3, padding=1),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.conv(x)


class CompleteFPNGenerator(nn.Module):
    def __init__(self, input_channels=3, output_channels=3, num_filters=64):
        super(CompleteFPNGenerator, self).__init__()
        
        self.pyramid_channels = num_filters * 2  # 128
        
        self.enc1 = nn.Sequential(
            nn.Conv2d(input_channels, num_filters, 7, padding=3, padding_mode='reflect'),
            nn.InstanceNorm2d(num_filters),
            nn.ReLU(inplace=True)
        )
        
        self.enc2 = nn.Sequential(
            nn.Conv2d(num_filters, num_filters * 2, 3, stride=2, padding=1),
            nn.InstanceNorm2d(num_filters * 2),
            nn.ReLU(inplace=True)
        )
        
        self.enc3 = nn.Sequential(
            nn.Conv2d(num_filters * 2, num_filters * 4, 3, stride=2, padding=1),
            nn.InstanceNorm2d(num_filters * 4),
            nn.ReLU(inplace=True)
        )
        
        self.res_blocks = nn.Sequential(*[
            ResidualBlock(num_filters * 4) for _ in range(9)
        ])
        
        self.lateral3 = nn.Conv2d(num_filters * 4, self.pyramid_channels, 1)
        self.lateral2 = nn.Conv2d(num_filters * 2, self.pyramid_channels, 1)
        self.lateral1 = nn.Conv2d(num_filters, self.pyramid_channels, 1)
        
        self.head3 = FPNHead(self.pyramid_channels, self.pyramid_channels)
        self.head2 = FPNHead(self.pyramid_channels, self.pyramid_channels)
        self.head1 = FPNHead(self.pyramid_channels, self.pyramid_channels)
        
        self.output3 = nn.Sequential(
            nn.Conv2d(self.pyramid_channels, output_channels, 7, padding=3, padding_mode='reflect'),
            nn.Tanh()
        )
        
        self.output2 = nn.Sequential(
            nn.Conv2d(self.pyramid_channels, output_channels, 7, padding=3, padding_mode='reflect'),
            nn.Tanh()
        )
        
        self.output1 = nn.Sequential(
            nn.Conv2d(self.pyramid_channels, output_channels, 7, padding=3, padding_mode='reflect'),
            nn.Tanh()
        )
    
    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        r = self.res_blocks(e3)
        p3 = self.lateral3(r)
        p3_up = F.interpolate(p3, scale_factor=2, mode='bilinear', align_corners=False)
        p2 = self.lateral2(e2) + p3_up
        p2_up = F.interpolate(p2, scale_factor=2, mode='bilinear', align_corners=False)
        p1 = self.lateral1(e1) + p2_up
        return {
            'scale1': self.output1(self.head1(p1)),
            'scale2': self.output2(self.head2(p2)),
            'scale3': self.output3(self.head3(p3))
        }

class PatchDiscriminator(nn.Module):
    def __init__(self, input_channels=3, num_filters=64, use_spectral_norm=False):
        super(PatchDiscriminator, self).__init__()
        
        def maybe_spectral_norm(layer):
            return nn.utils.spectral_norm(layer) if use_spectral_norm else layer
        
        self.model = nn.Sequential(
            maybe_spectral_norm(nn.Conv2d(input_channels, num_filters, 4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
            
            maybe_spectral_norm(nn.Conv2d(num_filters, num_filters * 2, 4, stride=2, padding=1)),
            nn.InstanceNorm2d(num_filters * 2),
            nn.LeakyReLU(0.2, inplace=True),
            
            maybe_spectral_norm(nn.Conv2d(num_filters * 2, num_filters * 4, 4, stride=2, padding=1)),
            nn.InstanceNorm2d(num_filters * 4),
            nn.LeakyReLU(0.2, inplace=True),
            
            maybe_spectral_norm(nn.Conv2d(num_filters * 4, num_filters * 8, 4, stride=1, padding=1)),
            nn.InstanceNorm2d(num_filters * 8),
            nn.LeakyReLU(0.2, inplace=True),
            
            maybe_spectral_norm(nn.Conv2d(num_filters * 8, 1, 4, stride=1, padding=1))
        )
    
    def forward(self, x):
        return self.model(x)


class DoubleScaleDiscriminator(nn.Module):
    def __init__(self, input_channels=3, num_filters=64, use_spectral_norm=False):
        super(DoubleScaleDiscriminator, self).__init__()
        self.discriminator_full = PatchDiscriminator(input_channels, num_filters, use_spectral_norm)
        self.discriminator_half = PatchDiscriminator(input_channels, num_filters, use_spectral_norm)
    
    def forward(self, x):
        out_full = self.discriminator_full(x)
        x_half = F.interpolate(x, scale_factor=0.5, mode='bilinear', align_corners=False)
        out_half = self.discriminator_half(x_half)
        return out_full, out_half

class MultiScalePerceptualLoss(nn.Module):
    def __init__(self):
        super(MultiScalePerceptualLoss, self).__init__()
        vgg = vgg19(pretrained=True).features.eval()
        
        self.slice1 = nn.Sequential(*list(vgg.children())[:4])
        self.slice2 = nn.Sequential(*list(vgg.children())[4:9])
        self.slice3 = nn.Sequential(*list(vgg.children())[9:18])
        self.slice4 = nn.Sequential(*list(vgg.children())[18:27])
        
        for param in self.parameters():
            param.requires_grad = False
    
    def forward(self, pred, target):
        pred_01 = (pred + 1.0) / 2.0
        target_01 = (target + 1.0) / 2.0
        
        pred_f1 = self.slice1(pred_01)
        pred_f2 = self.slice2(pred_f1)
        pred_f3 = self.slice3(pred_f2)
        pred_f4 = self.slice4(pred_f3)
        
        target_f1 = self.slice1(target_01)
        target_f2 = self.slice2(target_f1)
        target_f3 = self.slice3(target_f2)
        target_f4 = self.slice4(target_f3)
        
        loss = (F.mse_loss(pred_f1, target_f1) +
                F.mse_loss(pred_f2, target_f2) +
                F.mse_loss(pred_f3, target_f3) +
                F.mse_loss(pred_f4, target_f4))
        
        return loss

def compute_gradient_penalty(d_full, d_half, real, fake, device):
    batch_size = real.size(0)
    
    real = real.detach().requires_grad_(False)
    fake = fake.detach().requires_grad_(False)
    
    real = torch.clamp(real, -1.0, 1.0)
    fake = torch.clamp(fake, -1.0, 1.0)
    
    alpha = torch.rand(batch_size, 1, 1, 1, device=device)
    alpha = torch.clamp(alpha, 0.1, 0.9)
    
    interpolates_full = (alpha * real + (1 - alpha) * fake).requires_grad_(True)
    interpolates_half = F.interpolate(interpolates_full, scale_factor=0.5, mode='bilinear', align_corners=False)
    
    d_inter_full = d_full(interpolates_full)
    d_inter_half = d_half(interpolates_half)
    
    d_inter_full = torch.clamp(d_inter_full, -10.0, 10.0)
    d_inter_half = torch.clamp(d_inter_half, -10.0, 10.0)
    
    d_inter = torch.cat([d_inter_full.view(batch_size, -1),
                         d_inter_half.view(batch_size, -1)], dim=1)
    
    grad_outputs = torch.ones(batch_size, d_inter.size(1), device=device)
    gradients = torch.autograd.grad(
        outputs=d_inter, 
        inputs=interpolates_full, 
        grad_outputs=grad_outputs,
        create_graph=True, 
        retain_graph=True, 
        only_inputs=True
    )[0]
    
    if gradients is None:
        return torch.tensor(0.0, device=device, requires_grad=False)
    
    gradients = gradients.view(batch_size, -1)
    
    grad_norms = gradients.norm(2, dim=1)
    grad_norms = torch.clamp(grad_norms, 0.0, 10.0)
    
    gp = ((grad_norms - 1.0) ** 2).mean()
    
    gp = torch.clamp(gp, 0.0, 10.0)
    
    return gp

class CompleteDeblurTrainer:
    def __init__(self, generator, discriminator, device,
                 lambda_adv=0.01, lambda_l1=1.0, lambda_perc=0.1, lambda_ms=0.5,
                 lr_g=1e-4, lr_d=4e-4, use_amp=False):
        self.generator = generator.to(device)
        self.discriminator = discriminator.to(device)
        self.device = device
        
        self.lambda_adv = lambda_adv
        self.lambda_l1 = lambda_l1
        self.lambda_perc = lambda_perc
        self.lambda_ms = lambda_ms
        
        self.optimizer_g = optim.Adam(generator.parameters(), lr=lr_g, betas=(0.5, 0.999), eps=1e-8)
        stable_lr_d = min(lr_d, 2e-4)
        self.optimizer_d = optim.Adam(discriminator.parameters(), lr=stable_lr_d, betas=(0.5, 0.9), eps=1e-8)
        
        self.l1_loss = nn.L1Loss()
        self.perceptual_loss = MultiScalePerceptualLoss().to(device)
        
        self.use_amp = use_amp and torch.cuda.is_available()
        if self.use_amp:
            try:
                self.scaler_g = torch.amp.GradScaler('cuda')
                self.scaler_d = torch.amp.GradScaler('cuda')
            except:
                self.scaler_g = torch.cuda.amp.GradScaler()
                self.scaler_d = torch.cuda.amp.GradScaler()
    
    def train_step(self, blur_imgs, sharp_imgs):
        blur_imgs = blur_imgs.to(self.device)
        sharp_imgs = sharp_imgs.to(self.device)
        
        self.optimizer_d.zero_grad()
        
        if self.use_amp:
            try:
                with torch.amp.autocast('cuda'):
                    fake_outputs = self.generator(blur_imgs)
                    fake_imgs = fake_outputs['scale1']
                    
                    real_full, real_half = self.discriminator(sharp_imgs)
                    fake_full, fake_half = self.discriminator(fake_imgs.detach())
                    real_full = torch.clamp(real_full, -10.0, 10.0)
                    real_half = torch.clamp(real_half, -10.0, 10.0)
                    fake_full = torch.clamp(fake_full, -10.0, 10.0)
                    fake_half = torch.clamp(fake_half, -10.0, 10.0)
                    d_loss = (-torch.mean(real_full) + torch.mean(fake_full) +
                              -torch.mean(real_half) + torch.mean(fake_half))
                    gp = compute_gradient_penalty(
                        self.discriminator.discriminator_full,
                        self.discriminator.discriminator_half,
                        sharp_imgs, fake_imgs, self.device
                    )
                    d_loss_total = d_loss + 3 * gp
                    
                    if torch.isnan(d_loss_total) or torch.isinf(d_loss_total) or \
                       torch.isnan(d_loss) or torch.isinf(d_loss) or \
                       torch.isnan(gp) or torch.isinf(gp):
                        d_loss_total = None
            except:
                with torch.cuda.amp.autocast():
                    fake_outputs = self.generator(blur_imgs)
                    fake_imgs = fake_outputs['scale1']
                    
                    real_full, real_half = self.discriminator(sharp_imgs)
                    fake_full, fake_half = self.discriminator(fake_imgs.detach())
                    
                    real_full = torch.clamp(real_full, -10.0, 10.0)
                    real_half = torch.clamp(real_half, -10.0, 10.0)
                    fake_full = torch.clamp(fake_full, -10.0, 10.0)
                    fake_half = torch.clamp(fake_half, -10.0, 10.0)
                    
                    d_loss = (-torch.mean(real_full) + torch.mean(fake_full) +
                              -torch.mean(real_half) + torch.mean(fake_half))
                    
                    gp = compute_gradient_penalty(
                        self.discriminator.discriminator_full,
                        self.discriminator.discriminator_half,
                        sharp_imgs, fake_imgs, self.device
                    )
                    
                    d_loss_total = d_loss + 3 * gp
                    
                    if torch.isnan(d_loss_total) or torch.isinf(d_loss_total) or \
                       torch.isnan(d_loss) or torch.isinf(d_loss) or \
                       torch.isnan(gp) or torch.isinf(gp):
                        d_loss_total = None
        else:
            fake_outputs = self.generator(blur_imgs)
            fake_imgs = fake_outputs['scale1']
            
            real_full, real_half = self.discriminator(sharp_imgs)
            fake_full, fake_half = self.discriminator(fake_imgs.detach())
            
            real_full = torch.clamp(real_full, -10.0, 10.0)
            real_half = torch.clamp(real_half, -10.0, 10.0)
            fake_full = torch.clamp(fake_full, -10.0, 10.0)
            fake_half = torch.clamp(fake_half, -10.0, 10.0)
            
            d_loss = (-torch.mean(real_full) + torch.mean(fake_full) +
                      -torch.mean(real_half) + torch.mean(fake_half))
            
            gp = compute_gradient_penalty(
                self.discriminator.discriminator_full,
                self.discriminator.discriminator_half,
                sharp_imgs, fake_imgs, self.device
            )
            
            d_loss_total = d_loss + 3 * gp
            
            if torch.isnan(d_loss_total) or torch.isinf(d_loss_total) or \
               torch.isnan(d_loss) or torch.isinf(d_loss) or \
               torch.isnan(gp) or torch.isinf(gp):
                d_loss_total = None
        
        if d_loss_total is not None:
            if self.use_amp:
                self.scaler_d.scale(d_loss_total).backward()
                self.scaler_d.unscale_(self.optimizer_d)
            torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), max_norm=0.5)
                self.scaler_d.step(self.optimizer_d)
                self.scaler_d.update()
            else:
                d_loss_total.backward()
            torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), max_norm=0.5)
                self.optimizer_d.step()
        else:
            if self.use_amp:
                self.scaler_d.update()
            d_loss_total = torch.tensor(0.0, device=self.device)
        
        # Train Generator
        self.optimizer_g.zero_grad()
        
        if self.use_amp:
            try:
                autocast_context = torch.amp.autocast('cuda')
            except:
                autocast_context = torch.cuda.amp.autocast()
            
            with autocast_context:
                fake_outputs = self.generator(blur_imgs)
                fake_scale1 = fake_outputs['scale1']
                fake_scale2 = fake_outputs['scale2']
                fake_scale3 = fake_outputs['scale3']
                
                fake_full, fake_half = self.discriminator(fake_scale1)
                fake_full = torch.clamp(fake_full, -10.0, 10.0)
                fake_half = torch.clamp(fake_half, -10.0, 10.0)
                adv_loss = -(torch.mean(fake_full) + torch.mean(fake_half)) / 2.0
                l1_loss = self.l1_loss(fake_scale1, sharp_imgs)
                perc_loss = self.perceptual_loss(fake_scale1, sharp_imgs)
                
                target_size = (sharp_imgs.shape[2], sharp_imgs.shape[3])
                fake_scale2_up = F.interpolate(fake_scale2, size=target_size, mode='bilinear', align_corners=False)
                fake_scale3_up = F.interpolate(fake_scale3, size=target_size, mode='bilinear', align_corners=False)
                
                ms_loss = (self.l1_loss(fake_scale2_up, sharp_imgs) +
                          self.l1_loss(fake_scale3_up, sharp_imgs))
                
                if torch.isnan(adv_loss) or torch.isinf(adv_loss):
                    adv_loss = torch.tensor(0.0, device=self.device)
                if torch.isnan(l1_loss) or torch.isinf(l1_loss):
                    l1_loss = torch.tensor(0.0, device=self.device)
                if torch.isnan(perc_loss) or torch.isinf(perc_loss):
                    perc_loss = torch.tensor(0.0, device=self.device)
                if torch.isnan(ms_loss) or torch.isinf(ms_loss):
                    ms_loss = torch.tensor(0.0, device=self.device)
                
                g_loss = (self.lambda_adv * adv_loss +
                         self.lambda_l1 * l1_loss +
                         self.lambda_perc * perc_loss +
                         self.lambda_ms * ms_loss)
                
                if torch.isnan(g_loss) or torch.isinf(g_loss):
                    g_loss = None
        else:
            fake_outputs = self.generator(blur_imgs)
            fake_scale1 = fake_outputs['scale1']
            fake_scale2 = fake_outputs['scale2']
            fake_scale3 = fake_outputs['scale3']
            
            fake_full, fake_half = self.discriminator(fake_scale1)
            fake_full = torch.clamp(fake_full, -10.0, 10.0)
            fake_half = torch.clamp(fake_half, -10.0, 10.0)
            adv_loss = -(torch.mean(fake_full) + torch.mean(fake_half)) / 2.0
            l1_loss = self.l1_loss(fake_scale1, sharp_imgs)
            perc_loss = self.perceptual_loss(fake_scale1, sharp_imgs)
            
            target_size = (sharp_imgs.shape[2], sharp_imgs.shape[3])
            fake_scale2_up = F.interpolate(fake_scale2, size=target_size, mode='bilinear', align_corners=False)
            fake_scale3_up = F.interpolate(fake_scale3, size=target_size, mode='bilinear', align_corners=False)
            
            ms_loss = (self.l1_loss(fake_scale2_up, sharp_imgs) +
                      self.l1_loss(fake_scale3_up, sharp_imgs))
            
            if torch.isnan(adv_loss) or torch.isinf(adv_loss):
                adv_loss = torch.tensor(0.0, device=self.device)
            if torch.isnan(l1_loss) or torch.isinf(l1_loss):
                l1_loss = torch.tensor(0.0, device=self.device)
            if torch.isnan(perc_loss) or torch.isinf(perc_loss):
                perc_loss = torch.tensor(0.0, device=self.device)
            if torch.isnan(ms_loss) or torch.isinf(ms_loss):
                ms_loss = torch.tensor(0.0, device=self.device)
            
            g_loss = (self.lambda_adv * adv_loss +
                     self.lambda_l1 * l1_loss +
                     self.lambda_perc * perc_loss +
                     self.lambda_ms * ms_loss)
            
            if torch.isnan(g_loss) or torch.isinf(g_loss):
                g_loss = None
        
        if self.use_amp:
            self.scaler_g.scale(g_loss).backward()
            self.scaler_g.unscale_(self.optimizer_g)
            torch.nn.utils.clip_grad_norm_(self.generator.parameters(), max_norm=0.5)
            self.scaler_g.step(self.optimizer_g)
            self.scaler_g.update()
        else:
            g_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.generator.parameters(), max_norm=0.5)
            self.optimizer_g.step()
        
        def safe_item(tensor):
            if tensor is None:
                return 0.0
            if torch.is_tensor(tensor):
                val = tensor.item()
                return 0.0 if (val != val or abs(val) == float('inf')) else val
            return float(tensor) if not (tensor != tensor or abs(tensor) == float('inf')) else 0.0
        
        return {
            'd_loss': safe_item(d_loss_total) if d_loss_total is not None else 0.0,
            'g_loss': safe_item(g_loss) if g_loss is not None else 0.0,
            'adv_loss': safe_item(adv_loss),
            'l1_loss': safe_item(l1_loss),
            'perc_loss': safe_item(perc_loss),
            'multi_scale_loss': safe_item(ms_loss)
        }
    
    def evaluate(self, dataloader):
        self.generator.eval()
        total_psnr = 0
        total_ssim = 0
        count = 0
        
        with torch.no_grad():
            for blur_imgs, sharp_imgs in dataloader:
                blur_imgs = blur_imgs.to(self.device)
                sharp_imgs = sharp_imgs.to(self.device)
                
                fake_outputs = self.generator(blur_imgs)
                fake_imgs = fake_outputs['scale1']
                
                for i in range(blur_imgs.size(0)):
                    pred = fake_imgs[i].cpu().numpy().transpose(1, 2, 0)
                    target = sharp_imgs[i].cpu().numpy().transpose(1, 2, 0)
                    
                    pred = ((pred + 1) * 127.5).clip(0, 255).astype(np.uint8)
                    target = ((target + 1) * 127.5).clip(0, 255).astype(np.uint8)
                    
                    total_psnr += psnr(target, pred, data_range=255)
                    total_ssim += ssim(target, pred, channel_axis=2, data_range=255)
                    count += 1
        
        self.generator.train()
        return {'psnr': total_psnr / count, 'ssim': total_ssim / count}

class DeblurDataset(Dataset):
    def __init__(self, blur_dir, sharp_dir, transform=None, image_size=256):
        self.blur_dir = blur_dir
        self.sharp_dir = sharp_dir
        self.transform = transform
        self.image_size = image_size
        
        blur_files = sorted(glob.glob(os.path.join(blur_dir, '*.png')) + 
                           glob.glob(os.path.join(blur_dir, '*.jpg')))
        sharp_files = sorted(glob.glob(os.path.join(sharp_dir, '*.png')) + 
                            glob.glob(os.path.join(sharp_dir, '*.jpg')))
        
        self.pairs = []
        sharp_dict = {os.path.splitext(os.path.basename(f))[0]: f for f in sharp_files}
        
        for blur_file in blur_files:
            base_name = os.path.splitext(os.path.basename(blur_file))[0]
            if base_name in sharp_dict:
                self.pairs.append((blur_file, sharp_dict[base_name]))
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        blur_path, sharp_path = self.pairs[idx]
        
        blur_img = Image.open(blur_path).convert('RGB').resize((self.image_size, self.image_size))
        sharp_img = Image.open(sharp_path).convert('RGB').resize((self.image_size, self.image_size))
        
        if self.transform:
            blur_img = self.transform(blur_img)
            sharp_img = self.transform(sharp_img)
        
        return blur_img, sharp_img

def train_complete_deblurgan(train_blur_dir, train_sharp_dir,
                             val_blur_dir, val_sharp_dir,
                             epochs=50, batch_size=4, image_size=256,
                             save_dir='./checkpoints_complete',
                             use_amp=False, use_spectral_norm=True,
                             backup_to_drive=False, drive_backup_dir=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(save_dir, exist_ok=True)
    
    if backup_to_drive and drive_backup_dir:
        try:
            os.makedirs(drive_backup_dir, exist_ok=True)
        except:
            backup_to_drive = False
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    train_dataset = DeblurDataset(train_blur_dir, train_sharp_dir, transform, image_size)
    val_dataset = DeblurDataset(val_blur_dir, val_sharp_dir, transform, image_size)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    generator = CompleteFPNGenerator(input_channels=3, output_channels=3, num_filters=64)
    discriminator = DoubleScaleDiscriminator(input_channels=3, num_filters=64, 
                                            use_spectral_norm=use_spectral_norm)
    
    def init_weights(m):
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.xavier_uniform_(m.weight, gain=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d)):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
    
    generator.apply(init_weights)
    discriminator.apply(init_weights)
    
    trainer = CompleteDeblurTrainer(
        generator, discriminator, device,
        lambda_adv=0.005, lambda_l1=1.0, lambda_perc=0.1, lambda_ms=0.5,
        lr_g=1e-4, lr_d=2e-4, use_amp=use_amp
    )
    
    best_psnr = 0
    
    for epoch in range(epochs):
        trainer.generator.train()
        epoch_losses = {'d_loss': 0, 'g_loss': 0, 'adv_loss': 0, 'l1_loss': 0, 
                       'perc_loss': 0, 'multi_scale_loss': 0}
        
        pbar = tqdm(train_loader, desc="Training")
        for blur_imgs, sharp_imgs in pbar:
            losses = trainer.train_step(blur_imgs, sharp_imgs)
            
            for key in epoch_losses:
                epoch_losses[key] += losses[key]
            
            pbar.set_postfix({
                'G': f"{losses['g_loss']:.4f}",
                'D': f"{losses['d_loss']:.4f}",
                'L1': f"{losses['l1_loss']:.4f}",
                'MS': f"{losses['multi_scale_loss']:.4f}"
            })
        
        for key in epoch_losses:
            epoch_losses[key] /= len(train_loader)
        
        if (epoch + 1) % 5 == 0:
            metrics = trainer.evaluate(val_loader)
            if metrics['psnr'] > best_psnr:
                best_psnr = metrics['psnr']
                best_model_path = os.path.join(save_dir, 'best_model.pth')
                torch.save({
                    'epoch': epoch,
                    'generator_state_dict': generator.state_dict(),
                    'discriminator_state_dict': discriminator.state_dict(),
                    'psnr': metrics['psnr'],
                    'ssim': metrics['ssim']
                }, best_model_path)
                
                if backup_to_drive and drive_backup_dir:
                    try:
                        import shutil
                        drive_best = os.path.join(drive_backup_dir, 'best_model.pth')
                        shutil.copy(best_model_path, drive_best)
                    except Exception:
                        pass
        
        if (epoch + 1) % 20 == 0:
            checkpoint_path = os.path.join(save_dir, f'checkpoint_epoch_{epoch+1}.pth')
            torch.save({
                'epoch': epoch,
                'generator_state_dict': generator.state_dict(),
                'discriminator_state_dict': discriminator.state_dict(),
            }, checkpoint_path)
            
            import glob
            checkpoints = sorted(glob.glob(os.path.join(save_dir, 'checkpoint_epoch_*.pth')))
            if len(checkpoints) > 2:
                for old_ckpt in checkpoints[:-2]:
                    try:
                        os.remove(old_ckpt)
                    except Exception:
                        pass
            
            if backup_to_drive and drive_backup_dir:
                try:
                    import shutil
                    drive_checkpoint = os.path.join(drive_backup_dir, f'checkpoint_epoch_{epoch+1}.pth')
                    shutil.copy(checkpoint_path, drive_checkpoint)
                except Exception:
                    pass
    return generator, discriminator


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_blur', required=True)
    parser.add_argument('--train_sharp', required=True)
    parser.add_argument('--val_blur', required=True)
    parser.add_argument('--val_sharp', required=True)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--image_size', type=int, default=256)
    parser.add_argument('--save_dir', default='./checkpoints_complete')
    parser.add_argument('--use_amp', action='store_true')
    parser.add_argument('--use_spectral_norm', action='store_true')
    parser.add_argument('--backup_to_drive', action='store_true', 
                       help='Automatically backup checkpoints to Google Drive')
    parser.add_argument('--drive_backup_dir', default='/content/drive/MyDrive/deblur_checkpoints_complete',
                       help='Drive directory for backups')
    
    args = parser.parse_args()
    
    backup_enabled = False
    drive_dir = None
    if args.backup_to_drive:
        try:
            import sys
            is_colab = 'google.colab' in sys.modules
            
            if is_colab:
                from google.colab import drive
                try:
                    drive.mount('/content/drive', force_remount=False)
                except Exception as e:
                    if 'already mounted' in str(e).lower():
                        print("Drive already mounted")
                    else:
                        raise
            
            if os.path.exists('/content/drive'):
                backup_enabled = True
                drive_dir = args.drive_backup_dir
                os.makedirs(drive_dir, exist_ok=True)
        except Exception:
            backup_enabled = False
    
    train_complete_deblurgan(
        args.train_blur, args.train_sharp,
        args.val_blur, args.val_sharp,
        epochs=args.epochs,
        batch_size=args.batch_size,
        image_size=args.image_size,
        save_dir=args.save_dir,
        use_amp=args.use_amp,
        use_spectral_norm=args.use_spectral_norm,
        backup_to_drive=backup_enabled,
        drive_backup_dir=drive_dir
    )