"""
Metric Depth Estimation Model
==============================
Encoder-decoder that predicts METRIC depth (meters) from a single RGB image.

Key architectural choices:
    1. Pretrained encoder (EfficientNet/ResNet/MobileNet) for feature extraction
    2. Camera intrinsics are INJECTED into the decoder so the model learns
       to adapt its predictions based on the camera being used
    3. Sigmoid output scaled to [min_depth, max_depth] ensures valid metric range
    4. Skip connections preserve fine-grained spatial detail

Why intrinsics matter:
    Without intrinsics, "depth = 0.7" is ambiguous — 0.7 of what?
    With intrinsics (fx, fy), the model knows the camera's field of view
    and can predict actual metric distances.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class ConvBnRelu(nn.Module):
    """Basic building block: Conv → BatchNorm → ReLU (×2)."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.block(x)


class UpBlock(nn.Module):
    """Upsample → concatenate skip → ConvBnRelu."""
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
        self.conv = ConvBnRelu(in_ch // 2 + skip_ch, out_ch)
    
    def forward(self, x, skip=None):
        x = self.up(x)
        if skip is not None:
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, skip.shape[2:], mode='bilinear', align_corners=True)
            x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class CameraEncoder(nn.Module):
    """
    Encodes normalized intrinsics [fx/W, fy/H, cx/W, cy/H] into a
    spatial feature map that is concatenated with decoder features.
    
    This is what lets the model adapt to DIFFERENT CAMERAS.
    """
    def __init__(self, out_ch=64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(4, 64), nn.ReLU(inplace=True),
            nn.Linear(64, 128), nn.ReLU(inplace=True),
            nn.Linear(128, out_ch), nn.ReLU(inplace=True),
        )
    
    def forward(self, intrinsics, spatial_size):
        """intrinsics: (B, 4), spatial_size: (H, W)."""
        feat = self.mlp(intrinsics)                       # (B, C)
        feat = feat[:, :, None, None]                     # (B, C, 1, 1)
        return feat.expand(-1, -1, *spatial_size)         # (B, C, H, W)


class MetricDepthModel(nn.Module):
    """
    Main model. Predicts depth in meters from RGB + camera intrinsics.
    
    Architecture:
        RGB image → [Pretrained Encoder] → multi-scale features
        features  → [Decoder with skip connections] → low-res depth features
        features + camera_intrinsics → [Depth Head] → metric depth map
    """
    
    def __init__(self, encoder_name="efficientnet_b4", pretrained=True,
                 decoder_channels=(256, 128, 64, 32, 16),
                 use_camera_intrinsics=True,
                 max_depth=10.0, min_depth=0.1):
        super().__init__()
        
        self.max_depth = max_depth
        self.min_depth = min_depth
        self.use_camera = use_camera_intrinsics
        
        # --- Encoder: pretrained backbone ---
        self.encoder = timm.create_model(
            encoder_name, pretrained=pretrained,
            features_only=True, out_indices=[1, 2, 3, 4]
        )
        
        # Determine encoder output channels
        with torch.no_grad():
            dummy = torch.randn(1, 3, 480, 640)
            feats = self.encoder(dummy)
        enc_ch = [f.shape[1] for f in feats]
        print(f"  Encoder ({encoder_name}) channels: {enc_ch}")
        
        # --- Decoder ---
        dc = list(decoder_channels)
        self.bottleneck = ConvBnRelu(enc_ch[-1], dc[0])
        
        self.ups = nn.ModuleList()
        in_c = dc[0]
        for i, out_c in enumerate(dc[1:]):
            skip_c = enc_ch[-(i+2)] if i < len(enc_ch) - 1 else 0
            self.ups.append(UpBlock(in_c, skip_c, out_c))
            in_c = out_c
        
        self.final_up = nn.Sequential(
            nn.ConvTranspose2d(dc[-1], dc[-1], 2, stride=2),
            nn.ReLU(inplace=True)
        )
        
        # --- Camera intrinsics encoder ---
        cam_ch = 0
        if use_camera_intrinsics:
            cam_ch = 64
            self.cam_enc = CameraEncoder(cam_ch)
        
        # --- Depth prediction head ---
        self.head = nn.Sequential(
            nn.Conv2d(dc[-1] + cam_ch, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 1),
            nn.Sigmoid()  # output in [0,1], then scale to [min_depth, max_depth]
        )
        
        n_params = sum(p.numel() for p in self.parameters())
        n_train  = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"  Parameters: {n_params:,} total, {n_train:,} trainable")
    
    def forward(self, rgb, intrinsics=None):
        """
        rgb:        (B, 3, H, W)
        intrinsics: (B, 4) normalized [fx/W, fy/H, cx/W, cy/H]
        returns:    (B, 1, H, W) depth in meters
        """
        B, _, H, W = rgb.shape
        
        feats = self.encoder(rgb)       # 4 multi-scale feature maps
        x = self.bottleneck(feats[-1])  # deepest features
        
        for i, up in enumerate(self.ups):
            skip = feats[-(i+2)] if i < len(feats) - 1 else None
            x = up(x, skip)
        
        x = self.final_up(x)
        if x.shape[2:] != (H, W):
            x = F.interpolate(x, (H, W), mode='bilinear', align_corners=True)
        
        # Inject camera info
        if self.use_camera and intrinsics is not None:
            cam = self.cam_enc(intrinsics, (H, W))
            x = torch.cat([x, cam], dim=1)
        
        d = self.head(x)  # [0, 1]
        depth = self.min_depth + d * (self.max_depth - self.min_depth)
        return depth


class LightweightDepthModel(nn.Module):
    """
    Smaller model using MobileNetV3 — for machines with limited GPU memory.
    Same idea, fewer parameters.
    """
    def __init__(self, max_depth=10.0, min_depth=0.1, use_camera_intrinsics=True):
        super().__init__()
        self.max_depth = max_depth
        self.min_depth = min_depth
        self.use_camera = use_camera_intrinsics
        
        self.encoder = timm.create_model(
            "mobilenetv3_small_100", pretrained=True,
            features_only=True, out_indices=[1, 2, 3, 4]
        )
        with torch.no_grad():
            ec = [f.shape[1] for f in self.encoder(torch.randn(1, 3, 480, 640))]
        
        self.up1 = UpBlock(ec[3], ec[2], 128)
        self.up2 = UpBlock(128, ec[1], 64)
        self.up3 = UpBlock(64, ec[0], 32)
        self.up4 = nn.Sequential(nn.ConvTranspose2d(32, 16, 2, stride=2), nn.ReLU(True))
        
        head_ch = 16
        if use_camera_intrinsics:
            self.cam_enc = CameraEncoder(32)
            head_ch += 32
        
        self.head = nn.Sequential(
            nn.Conv2d(head_ch, 16, 3, padding=1), nn.ReLU(True),
            nn.Conv2d(16, 1, 1), nn.Sigmoid()
        )
        print(f"  Lightweight model: {sum(p.numel() for p in self.parameters()):,} params")
    
    def forward(self, rgb, intrinsics=None):
        B, _, H, W = rgb.shape
        f = self.encoder(rgb)
        x = self.up1(f[3], f[2])
        x = self.up2(x, f[1])
        x = self.up3(x, f[0])
        x = self.up4(x)
        if x.shape[2:] != (H, W):
            x = F.interpolate(x, (H, W), mode='bilinear', align_corners=True)
        if self.use_camera and intrinsics is not None:
            x = torch.cat([x, self.cam_enc(intrinsics, (H, W))], 1)
        d = self.head(x)
        return self.min_depth + d * (self.max_depth - self.min_depth)


def create_model(config):
    """Factory function: build model from config dict."""
    m = config.get("model", {})
    d = config.get("dataset", {})
    return MetricDepthModel(
        encoder_name=m.get("encoder", "efficientnet_b4"),
        pretrained=m.get("pretrained_encoder", True),
        decoder_channels=m.get("decoder_channels", [256, 128, 64, 32, 16]),
        use_camera_intrinsics=m.get("use_camera_intrinsics", True),
        max_depth=d.get("max_depth", 10.0),
        min_depth=d.get("min_depth", 0.1),
    )
