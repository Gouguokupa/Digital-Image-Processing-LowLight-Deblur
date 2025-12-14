"""
Loss functions for low-light image enhancement
Includes L1, SSIM, Total Variation, and Color Consistency losses
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SSIMLoss(nn.Module):
    """
    Structural Similarity Index (SSIM) loss
    Measures structural similarity between images
    """
    
    def __init__(self, window_size=11, size_average=True):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 3
        self.window = self.create_window(window_size, self.channel)
        
    def create_window(self, window_size, channel):
        """Create Gaussian window for SSIM"""
        def gaussian(window_size, sigma):
            gauss = torch.Tensor([
                torch.exp(torch.tensor(-(x - window_size//2)**2 / float(2*sigma**2))) 
                for x in range(window_size)
            ])
            return gauss / gauss.sum()
        
        _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window
        
    def ssim(self, img1, img2):
        """Compute SSIM between two images"""
        (_, channel, _, _) = img1.size()
        
        if self.window.device != img1.device:
            self.window = self.window.to(img1.device)
            self.window = self.window.type_as(img1)
        
        window = self.window
        
        mu1 = F.conv2d(img1, window, padding=self.window_size//2, groups=channel)
        mu2 = F.conv2d(img2, window, padding=self.window_size//2, groups=channel)
        
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = F.conv2d(img1 * img1, window, padding=self.window_size//2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=self.window_size//2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=self.window_size//2, groups=channel) - mu1_mu2
        
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        
        if self.size_average:
            return ssim_map.mean()
        else:
            return ssim_map.mean(1).mean(1).mean(1)
    
    def forward(self, img1, img2):
        """Return SSIM loss (1 - SSIM)"""
        return 1 - self.ssim(img1, img2)


class TVLoss(nn.Module):
    """
    Total Variation Loss
    Encourages spatial smoothness in illumination map
    """
    
    def __init__(self):
        super(TVLoss, self).__init__()
        
    def forward(self, x):
        """
        Args:
            x: Input tensor [B, C, H, W]
        Returns:
            TV loss value
        """
        batch_size = x.size(0)
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :-1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :-1]), 2).sum()
        return (h_tv + w_tv) / batch_size


class ColorConstancyLoss(nn.Module):
    """
    Color Consistency Loss
    Maintains natural color balance in enhanced images
    Penalizes deviations from gray-world assumption
    """
    
    def __init__(self):
        super(ColorConstancyLoss, self).__init__()
        
    def forward(self, x):
        """
        Args:
            x: Enhanced image [B, 3, H, W]
        Returns:
            Color consistency loss
        """
        # Compute mean for each channel
        mean_rgb = torch.mean(x, dim=(2, 3), keepdim=True)  # [B, 3, 1, 1]
        
        # Compute differences between channels
        mr, mg, mb = mean_rgb[:, 0], mean_rgb[:, 1], mean_rgb[:, 2]
        
        # Gray-world assumption: R, G, B means should be similar
        d_rg = torch.pow(mr - mg, 2)
        d_rb = torch.pow(mr - mb, 2)
        d_gb = torch.pow(mg - mb, 2)
        
        return torch.sqrt(d_rg + d_rb + d_gb + 1e-8).mean()


class PerceptualLoss(nn.Module):
    """
    Perceptual Loss using VGG features (optional, for better quality)
    """
    
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        try:
            import torchvision.models as models
            vgg = models.vgg16(pretrained=True).features
            self.vgg_layers = nn.Sequential(*list(vgg.children())[:16]).eval()
            for param in self.vgg_layers.parameters():
                param.requires_grad = False
        except:
            print("Warning: VGG not available, perceptual loss will not work")
            self.vgg_layers = None
    
    def forward(self, x, y):
        if self.vgg_layers is None:
            return torch.tensor(0.0, device=x.device)
        
        x_vgg = self.vgg_layers(x)
        y_vgg = self.vgg_layers(y)
        return F.l1_loss(x_vgg, y_vgg)


class HybridLoss(nn.Module):
    """
    Combined loss function for hybrid low-light enhancement
    Equation (1) from paper:
    L = λ1*||x_retinex - x_gt||_1 + λ2*(1 - SSIM(x_retinex, x_gt)) 
        + λ3*TV(I) + λ4*L_color
    """
    
    def __init__(self, lambda1=1.0, lambda2=1.0, lambda3=0.1, lambda4=0.5):
        super(HybridLoss, self).__init__()
        
        self.lambda1 = lambda1  # L1 weight
        self.lambda2 = lambda2  # SSIM weight
        self.lambda3 = lambda3  # TV weight
        self.lambda4 = lambda4  # Color consistency weight
        
        self.l1_loss = nn.L1Loss()
        self.ssim_loss = SSIMLoss()
        self.tv_loss = TVLoss()
        self.color_loss = ColorConstancyLoss()
        
        print(f"Hybrid Loss initialized with weights:")
        print(f"  λ1 (L1): {self.lambda1}")
        print(f"  λ2 (SSIM): {self.lambda2}")
        print(f"  λ3 (TV): {self.lambda3}")
        print(f"  λ4 (Color): {self.lambda4}")
        
    def forward(self, outputs, target):
        """
        Args:
            outputs: Dictionary from model containing:
                - enhanced: Final enhanced image
                - retinex: Retinex branch output
                - illum_map: Illumination map
            target: Ground truth normal-light image
            
        Returns:
            total_loss: Combined loss
            loss_dict: Dictionary of individual losses
        """
        enhanced = outputs['enhanced']
        retinex = outputs['retinex']
        illum_map = outputs['illum_map']
        
        # L1 reconstruction loss on final output
        l1_enhanced = self.l1_loss(enhanced, target)
        
        # L1 loss on retinex branch (for supervision)
        l1_retinex = self.l1_loss(retinex, target)
        
        # SSIM loss on enhanced output
        ssim = self.ssim_loss(enhanced, target)
        
        # Total Variation loss on illumination map (smoothness)
        tv = self.tv_loss(illum_map)
        
        # Color consistency loss on enhanced output
        color = self.color_loss(enhanced)
        
        # Total loss
        total_loss = (
            self.lambda1 * (l1_enhanced + l1_retinex) +
            self.lambda2 * ssim +
            self.lambda3 * tv +
            self.lambda4 * color
        )
        
        # Return individual losses for logging
        loss_dict = {
            'total': total_loss.item(),
            'l1_enhanced': l1_enhanced.item(),
            'l1_retinex': l1_retinex.item(),
            'ssim': ssim.item(),
            'tv': tv.item(),
            'color': color.item()
        }
        
        return total_loss, loss_dict


if __name__ == '__main__':
    # Test losses
    print("Testing loss functions...")
    
    # Create dummy data
    pred = torch.rand(2, 3, 256, 256)
    target = torch.rand(2, 3, 256, 256)
    illum = torch.rand(2, 1, 256, 256)
    
    outputs = {
        'enhanced': pred,
        'retinex': pred,
        'illum_map': illum
    }
    
    # Test individual losses
    print("\nTesting individual losses:")
    l1 = nn.L1Loss()(pred, target)
    print(f"L1 Loss: {l1.item():.4f}")
    
    ssim = SSIMLoss()(pred, target)
    print(f"SSIM Loss: {ssim.item():.4f}")
    
    tv = TVLoss()(illum)
    print(f"TV Loss: {tv.item():.4f}")
    
    color = ColorConstancyLoss()(pred)
    print(f"Color Loss: {color.item():.4f}")
    
    # Test combined loss
    print("\nTesting combined loss:")
    criterion = HybridLoss()
    total_loss, loss_dict = criterion(outputs, target)
    print(f"Total Loss: {total_loss.item():.4f}")
    print(f"Loss breakdown: {loss_dict}")

