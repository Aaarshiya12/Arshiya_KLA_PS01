"""
model.py — Noise-conditioned restoration network for SEMICON India Hackathon 2026, Track 1 (KLA).

Architecture: U-Net style encoder-decoder with skip connections, taking
[degraded image, noise-level map] as 2-channel input, ending in a PixelShuffle
upsampling head (2x super-resolution) with two output branches:
  1. Restoration head -> the restored image
  2. Confidence head  -> per-pixel confidence map (self-supervised, Tier 2 novelty)

Two size variants controlled by `base_channels`:
  - small: base_channels=24  (~1-2M params, speed-optimized)
  - large: base_channels=48  (~8-10M params, quality-optimized)
This gives you the speed-quality Pareto pair for benchmarking.
"""

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.GroupNorm(min(8, out_ch), out_ch),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.GroupNorm(min(8, out_ch), out_ch),
            nn.GELU(),
        )

    def forward(self, x):
        return self.block(x)


class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch, 3, padding=1)
        self.norm1 = nn.GroupNorm(min(8, ch), ch)
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(min(8, ch), ch)
        self.act = nn.GELU()

    def forward(self, x):
        residual = x
        x = self.act(self.norm1(self.conv1(x)))
        x = self.norm2(self.conv2(x))
        return self.act(x + residual)


class RestorationNet(nn.Module):
    """
    in_channels: 2 (degraded grayscale image + noise-level map)
    base_channels: controls model size (24 = small, 48 = large)
    scale: super-resolution factor (2 for 256->512, matches problem statement)
    """

    def __init__(self, in_channels: int = 2, base_channels: int = 24, scale: int = 2):
        super().__init__()
        c = base_channels

        # Encoder: 4 downsampling stages
        self.enc1 = ConvBlock(in_channels, c)
        self.enc2 = ConvBlock(c, c * 2)
        self.enc3 = ConvBlock(c * 2, c * 4)
        self.enc4 = ConvBlock(c * 4, c * 8)
        self.pool = nn.AvgPool2d(2)

        # Bottleneck: 2 residual blocks
        self.bottleneck = nn.Sequential(ResBlock(c * 8), ResBlock(c * 8))

        # Decoder: 4 upsampling stages with skip connections
        # (up-conv keeps channel count so concat with same-level encoder skip is exact)
        self.up4 = nn.ConvTranspose2d(c * 8, c * 8, 2, stride=2)
        self.dec4 = ConvBlock(c * 16, c * 4)
        self.up3 = nn.ConvTranspose2d(c * 4, c * 4, 2, stride=2)
        self.dec3 = ConvBlock(c * 8, c * 2)
        self.up2 = nn.ConvTranspose2d(c * 2, c * 2, 2, stride=2)
        self.dec2 = ConvBlock(c * 4, c)
        self.up1 = nn.ConvTranspose2d(c, c, 2, stride=2)
        self.dec1 = ConvBlock(c * 2, c)

        # PixelShuffle super-resolution head (avoids checkerboard artifacts
        # that transposed-conv upsampling causes -- explicitly penalized by
        # the rubric's "no ringing/artifacts" requirement)
        self.pre_shuffle = nn.Conv2d(c, c * (scale ** 2), 3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(scale)

        # Restoration head
        self.restoration_head = nn.Conv2d(c, 1, 3, padding=1)

        # Confidence head (self-supervised, trained against error-derived target)
        self.confidence_head = nn.Sequential(
            nn.Conv2d(c, 1, 3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, image, noise_map):
        x = torch.cat([image, noise_map], dim=1)  # noise-conditioning input

        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        b = self.bottleneck(self.pool(e4))

        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        upsampled = self.pixel_shuffle(self.pre_shuffle(d1))

        restored = self.restoration_head(upsampled)
        confidence = self.confidence_head(upsampled)

        return restored, confidence


def build_model(size: str = "small") -> RestorationNet:
    base_channels = {"small": 24, "large": 48}[size]
    return RestorationNet(in_channels=2, base_channels=base_channels, scale=2)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    for size in ["small", "large"]:
        model = build_model(size)
        n_params = count_params(model)
        dummy_img = torch.randn(1, 1, 256, 256)
        dummy_noise = torch.randn(1, 1, 256, 256)
        restored, conf = model(dummy_img, dummy_noise)
        print(f"[{size}] params: {n_params:,} | restored shape: {tuple(restored.shape)} | confidence shape: {tuple(conf.shape)}")
