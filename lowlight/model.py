"""
Hybrid Low-Light Enhancement Model
Combines Retinex-based decomposition with curve-based correction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SharedDecoder(nn.Module):
    """
    Shared feature decoder with skip connections
    Extracts multi-dimensional features including edges, textures, and illumination
    """
    
    def __init__(self, in_channels=3, base_channels=32):
        super(SharedDecoder, self).__init__()
        
        # Encoder layers
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        self.enc2 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels * 2, 3, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        self.enc3 = nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels * 4, 3, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 4),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        self.enc4 = nn.Sequential(
            nn.Conv2d(base_channels * 4, base_channels * 8, 3, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 8),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(base_channels * 8, base_channels * 8, 3, padding=1),
            nn.BatchNorm2d(base_channels * 8),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Decoder layers with skip connections
        self.dec4 = nn.Sequential(
            nn.Conv2d(base_channels * 8, base_channels * 4, 3, padding=1),
            nn.BatchNorm2d(base_channels * 4),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        self.dec3 = nn.Sequential(
            nn.Conv2d(base_channels * 8, base_channels * 2, 3, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        self.dec2 = nn.Sequential(
            nn.Conv2d(base_channels * 4, base_channels, 3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        self.dec1 = nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels, 3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Global illumination aggregation
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.global_fc = nn.Sequential(
            nn.Linear(base_channels * 8, base_channels * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(base_channels * 4, base_channels)
        )
        
    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        
        # Bottleneck
        b = self.bottleneck(e4)
        
        # Global illumination context
        global_feat = self.global_pool(b).squeeze(-1).squeeze(-1)
        global_context = self.global_fc(global_feat)
        
        # Decoder with skip connections
        d4 = F.interpolate(self.dec4(b), scale_factor=2, mode='bilinear', align_corners=False)
        d4 = torch.cat([d4, e3], dim=1)
        
        d3 = F.interpolate(self.dec3(d4), scale_factor=2, mode='bilinear', align_corners=False)
        d3 = torch.cat([d3, e2], dim=1)
        
        d2 = F.interpolate(self.dec2(d3), scale_factor=2, mode='bilinear', align_corners=False)
        d2 = torch.cat([d2, e1], dim=1)
        
        features = self.dec1(d2)
        
        return features, global_context


class IlluminationBranch(nn.Module):
    """
    Retinex-based illumination decomposition branch
    Estimates illumination map and computes reflectance
    """
    
    def __init__(self, feature_channels=32):
        super(IlluminationBranch, self).__init__()
        
        # Illumination estimation network
        self.illum_net = nn.Sequential(
            nn.Conv2d(feature_channels, 32, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 16, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(16, 1, 3, padding=1),
            nn.Sigmoid()  # Illumination map in [0, 1]
        )
        
        # Reflectance refinement (optional)
        self.refine_net = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(16, 3, 3, padding=1),
            nn.Sigmoid()
        )
        
    def forward(self, features, x_low):
        """
        Args:
            features: Decoded features from shared decoder
            x_low: Input low-light image
            
        Returns:
            x_retinex: Enhanced image using Retinex model
            illum_map: Estimated illumination map
        """
        # Estimate illumination map
        illum_map = self.illum_net(features)
        
        # Compute reflectance (Retinex decomposition)
        # R(x,y) = I_low(x,y) / (L(x,y) + epsilon)
        epsilon = 1e-4
        reflectance = x_low / (illum_map + epsilon)
        
        # Enhance illumination (simple gamma correction)
        illum_enhanced = torch.pow(illum_map, 0.5)  # Brighten illumination
        
        # Reconstruct enhanced image
        # x_retinex = R(x,y) * L_enhanced(x,y)
        x_retinex = reflectance * illum_enhanced
        
        # Optional: Refine reflectance to reduce artifacts
        x_retinex = self.refine_net(x_retinex)
        
        # Clip to [0, 1]
        x_retinex = torch.clamp(x_retinex, 0, 1)
        
        return x_retinex, illum_map


class CurveBranch(nn.Module):
    """
    Curve-based brightness correction branch
    Learns pixel-wise nonlinear mapping for illumination adjustment
    """
    
    def __init__(self, feature_channels=32):
        super(CurveBranch, self).__init__()
        
        # Three convolutional layers as specified in paper
        self.conv1 = nn.Sequential(
            nn.Conv2d(feature_channels, 32, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 16, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(16, 3, 3, padding=1),  # Output 3 channels for RGB
            nn.Sigmoid()  # α in [0, 1]
        )
        
    def forward(self, features, x_low):
        """
        Args:
            features: Decoded features from shared decoder
            x_low: Input low-light image
            
        Returns:
            x_curve: Enhanced image using curve adjustment
        """
        # Predict pixel-wise parameter α
        alpha = self.conv1(features)
        alpha = self.conv2(alpha)
        alpha = self.conv3(alpha)
        
        # Apply curve transformation: T(x) = x + α * x * (1 - x)
        x_curve = x_low + alpha * x_low * (1 - x_low)
        
        # Clip to [0, 1]
        x_curve = torch.clamp(x_curve, 0, 1)
        
        return x_curve


class FusionModule(nn.Module):
    """
    Learnable fusion module that combines Retinex and Curve outputs
    Predicts adaptive weight map for combining both branches
    """
    
    def __init__(self):
        super(FusionModule, self).__init__()
        
        # Weight prediction network
        self.weight_net = nn.Sequential(
            nn.Conv2d(6, 32, 3, padding=1),  # Input: concatenation of both outputs
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 16, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(16, 1, 3, padding=1),
            nn.Sigmoid()  # Weight in [0, 1]
        )
        
    def forward(self, x_retinex, x_curve):
        """
        Args:
            x_retinex: Output from illumination branch
            x_curve: Output from curve branch
            
        Returns:
            x_fused: Final fused output
            weight: Fusion weight map
        """
        # Concatenate both outputs
        concat = torch.cat([x_retinex, x_curve], dim=1)
        
        # Predict weight map
        weight = self.weight_net(concat)
        
        # Fuse: x_enhanced = w * x_retinex + (1 - w) * x_curve
        x_fused = weight * x_retinex + (1 - weight) * x_curve
        
        return x_fused, weight


class HybridLLE(nn.Module):
    """
    Complete Hybrid Low-Light Enhancement Model
    Combines Retinex-based decomposition with curve-based correction
    """
    
    def __init__(self, base_channels=32):
        super(HybridLLE, self).__init__()
        
        self.decoder = SharedDecoder(in_channels=3, base_channels=base_channels)
        self.illumination = IlluminationBranch(feature_channels=base_channels)
        self.curve = CurveBranch(feature_channels=base_channels)
        self.fusion = FusionModule()
        
    def forward(self, x_low):
        """
        Args:
            x_low: Input low-light image [B, 3, H, W]
            
        Returns:
            outputs: Dictionary containing:
                - enhanced: Final enhanced image
                - retinex: Retinex branch output
                - curve: Curve branch output
                - illum_map: Illumination map
                - weight: Fusion weight map
        """
        # Extract features
        features, global_context = self.decoder(x_low)
        
        # Two parallel branches
        x_retinex, illum_map = self.illumination(features, x_low)
        x_curve = self.curve(features, x_low)
        
        # Fusion
        x_enhanced, weight = self.fusion(x_retinex, x_curve)
        
        return {
            'enhanced': x_enhanced,
            'retinex': x_retinex,
            'curve': x_curve,
            'illum_map': illum_map,
            'weight': weight
        }


if __name__ == '__main__':
    # Test model
    print("Testing model architecture...")
    model = HybridLLE(base_channels=32)
    
    # Test forward pass
    x = torch.randn(2, 3, 256, 256)
    outputs = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Enhanced output shape: {outputs['enhanced'].shape}")
    print(f"Retinex output shape: {outputs['retinex'].shape}")
    print(f"Curve output shape: {outputs['curve'].shape}")
    print(f"Illumination map shape: {outputs['illum_map'].shape}")
    print(f"Weight map shape: {outputs['weight'].shape}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {total_params:,}")

