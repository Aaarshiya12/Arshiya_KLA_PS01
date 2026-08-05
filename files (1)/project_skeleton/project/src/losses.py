"""
losses.py — Physics-informed loss for the KLA restoration task.

L_total = w1*Charbonnier + w2*SSIM + w3*Frequency + w4*ClipConsistency + w5*Confidence

Why not plain L1/L2:
  - Speckle is multiplicative and pushes pixels out of range -> plain L2 over-
    penalizes outliers and encourages the network to over-smooth (blur), which
    directly conflicts with the brief's "do not blur to remove noise" requirement.
  - Charbonnier (robust L1) handles those outliers more gracefully.
  - SSIM adds structural/perceptual awareness.
  - A frequency-domain term explicitly pushes the recovery of high-frequency
    detail that plain pixel losses tend to average away.
  - Clip-consistency down-weights loss on pixels where the *input* was clipped/
    out-of-range, since ground truth there is inherently ambiguous -- forces
    the model to lean on neighboring context instead of overfitting to noise
    it fundamentally cannot resolve.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def charbonnier_loss(pred, target, eps=1e-3):
    diff = pred - target
    return torch.mean(torch.sqrt(diff * diff + eps * eps))


def gaussian_window(window_size=11, sigma=1.5, device="cpu"):
    coords = torch.arange(window_size, dtype=torch.float32, device=device) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    window_2d = g.unsqueeze(0) * g.unsqueeze(1)
    return window_2d.unsqueeze(0).unsqueeze(0)


def ssim_loss(pred, target, window_size=11):
    """Returns 1 - SSIM (so it can be minimized like a normal loss)."""
    window = gaussian_window(window_size, device=pred.device)
    c1, c2 = 0.01 ** 2, 0.03 ** 2

    mu_p = F.conv2d(pred, window, padding=window_size // 2)
    mu_t = F.conv2d(target, window, padding=window_size // 2)

    mu_p_sq, mu_t_sq, mu_pt = mu_p ** 2, mu_t ** 2, mu_p * mu_t

    sigma_p_sq = F.conv2d(pred * pred, window, padding=window_size // 2) - mu_p_sq
    sigma_t_sq = F.conv2d(target * target, window, padding=window_size // 2) - mu_t_sq
    sigma_pt = F.conv2d(pred * target, window, padding=window_size // 2) - mu_pt

    ssim_map = ((2 * mu_pt + c1) * (2 * sigma_pt + c2)) / (
        (mu_p_sq + mu_t_sq + c1) * (sigma_p_sq + sigma_t_sq + c2)
    )
    return 1.0 - ssim_map.mean()


def frequency_loss(pred, target):
    """L1 distance between log-magnitude FFT spectra -- pushes recovery of
    high-frequency detail lost during downsampling."""
    pred_fft = torch.fft.fft2(pred)
    target_fft = torch.fft.fft2(target)
    pred_mag = torch.log(torch.abs(pred_fft) + 1e-8)
    target_mag = torch.log(torch.abs(target_fft) + 1e-8)
    return F.l1_loss(pred_mag, target_mag)


def clip_consistency_loss(pred, input_upsampled, target, clip_low=0.0, clip_high=255.0, tolerance=2.0):
    """
    Down-weight loss on pixels where the (upsampled) input was at/near the
    clipping boundary -- ground truth there is inherently ambiguous since
    speckle could have pushed the true value in either direction.
    """
    is_clipped = ((input_upsampled <= clip_low + tolerance) | (input_upsampled >= clip_high - tolerance)).float()
    weight = 1.0 - 0.7 * is_clipped  # reduce (not zero) weight on clipped regions
    diff = torch.abs(pred - target)
    return (diff * weight).mean()


def confidence_loss(confidence_pred, pred, target, tau=10.0):
    """
    Self-supervised confidence target: high confidence where restoration
    error is low. Gradient is stopped on the error term so this doesn't
    distort the main restoration objective.
    """
    with torch.no_grad():
        error = torch.abs(pred - target)
        target_confidence = torch.exp(-error / tau)
    return F.mse_loss(confidence_pred, target_confidence)


class CombinedLoss(nn.Module):
    def __init__(self, w_charbonnier=1.0, w_ssim=0.3, w_freq=0.1, w_clip=0.2, w_conf=0.1):
        super().__init__()
        self.w_charbonnier = w_charbonnier
        self.w_ssim = w_ssim
        self.w_freq = w_freq
        self.w_clip = w_clip
        self.w_conf = w_conf

    def forward(self, pred, target, input_upsampled, confidence_pred):
        l_char = charbonnier_loss(pred, target)
        l_ssim = ssim_loss(pred, target)
        l_freq = frequency_loss(pred, target)
        l_clip = clip_consistency_loss(pred, input_upsampled, target)
        l_conf = confidence_loss(confidence_pred, pred, target)

        total = (
            self.w_charbonnier * l_char
            + self.w_ssim * l_ssim
            + self.w_freq * l_freq
            + self.w_clip * l_clip
            + self.w_conf * l_conf
        )

        breakdown = {
            "charbonnier": l_char.item(),
            "ssim": l_ssim.item(),
            "freq": l_freq.item(),
            "clip": l_clip.item(),
            "conf": l_conf.item(),
            "total": total.item(),
        }
        return total, breakdown


if __name__ == "__main__":
    pred = torch.rand(1, 1, 64, 64, requires_grad=True)
    target = torch.rand(1, 1, 64, 64)
    input_upsampled = torch.rand(1, 1, 64, 64)
    confidence_pred = torch.rand(1, 1, 64, 64)

    loss_fn = CombinedLoss()
    total, breakdown = loss_fn(pred, target, input_upsampled, confidence_pred)
    total.backward()
    print("Loss breakdown:", breakdown)
    print("Backward pass OK, grad exists:", pred.grad is not None)
