"""
degrade.py — Synthetic degradation simulator for SEMICON India Hackathon 2026, Track 1 (KLA)

Simulates the three degradations described in the problem statement:
  1. Speckle noise (multiplicative, can push pixels outside true range)
  2. Gaussian noise / blur (softens edges)
  3. Downsampling (2x super-resolution target)

Use this to generate extra hard training pairs, and to build proof-of-concept
before/after demo images for the registration slides.
"""

import numpy as np
import cv2


def add_speckle_noise(image: np.ndarray, sigma: float = 0.15) -> np.ndarray:
    """
    Multiplicative speckle noise: I_noisy = I * (1 + n), n ~ N(0, sigma^2)
    This can push pixel values outside the original [0, 255] range, which is
    why we do NOT clip immediately -- the network needs to learn from this.
    image: float32 array, range [0, 255]
    """
    noise = np.random.normal(loc=0.0, scale=sigma, size=image.shape).astype(np.float32)
    noisy = image * (1.0 + noise)
    return noisy


def add_gaussian_noise(image: np.ndarray, sigma: float = 10.0) -> np.ndarray:
    """
    Additive Gaussian noise -- simulates sensor read noise that softens
    apparent edge sharpness.
    """
    noise = np.random.normal(loc=0.0, scale=sigma, size=image.shape).astype(np.float32)
    return image + noise


def gaussian_blur(image: np.ndarray, ksize: int = 3, sigma: float = 0.8) -> np.ndarray:
    """Mild blur to simulate softened edges before noise is applied."""
    return cv2.GaussianBlur(image, (ksize, ksize), sigmaX=sigma)


def downsample(image: np.ndarray, scale: int = 2, method: str = "area") -> np.ndarray:
    """
    Downsample by `scale` (e.g. 512->256 for scale=2). This is the information
    that must be recovered by super-resolution -- it's genuinely gone, not
    just blurred.
    """
    h, w = image.shape[:2]
    new_h, new_w = h // scale, w // scale
    interp = cv2.INTER_AREA if method == "area" else cv2.INTER_CUBIC
    return cv2.resize(image, (new_w, new_h), interpolation=interp)


def degrade_pipeline(
    clean_image: np.ndarray,
    scale: int = 2,
    speckle_sigma: float = 0.15,
    gaussian_sigma: float = 10.0,
    blur_sigma: float = 0.8,
    clip_output: bool = False,
) -> np.ndarray:
    """
    Full degradation pipeline matching the KLA problem statement:
    clean (e.g. 512x512) -> blur -> speckle -> gaussian noise -> downsample -> degraded (e.g. 256x256)

    Order matters: blur/noise are applied at full resolution to mimic real
    sensor-level degradation, THEN downsampled -- this is closer to how a real
    inspection camera would produce a noisy low-res capture, versus noising
    an already-downsampled image.

    clip_output: if False (default), pixel values are allowed to exceed
    [0, 255] after speckle, matching the brief's note that speckle can push
    values outside the true range. Set True only for visualization/saving.
    """
    img = clean_image.astype(np.float32)

    img = gaussian_blur(img, ksize=3, sigma=blur_sigma)
    img = add_speckle_noise(img, sigma=speckle_sigma)
    img = add_gaussian_noise(img, sigma=gaussian_sigma)
    img = downsample(img, scale=scale, method="area")

    if clip_output:
        img = np.clip(img, 0, 255)

    return img


def make_synthetic_test_pattern(size: int = 512, seed: int = 0) -> np.ndarray:
    """
    Generates a synthetic grayscale test pattern resembling a semiconductor
    inspection image (periodic grid lines + circular contact-like dots +
    varying intensity regions) -- useful for prototyping before real KLA data
    is available. NOT a substitute for real data once it drops.
    """
    rng = np.random.default_rng(seed)
    img = np.full((size, size), 60, dtype=np.float32)  # dark background

    # periodic grid lines (like a DRAM/FinFET-style repeating structure)
    spacing = size // 16
    for i in range(0, size, spacing):
        img[i:i + 2, :] = 200
        img[:, i:i + 2] = 200

    # contact-dot-like circles at grid intersections
    for i in range(spacing, size, spacing):
        for j in range(spacing, size, spacing):
            if rng.random() > 0.3:
                cv2.circle(img, (i, j), radius=spacing // 6, color=230, thickness=-1)

    # a few irregular "defect" blobs for realism
    for _ in range(5):
        cx, cy = rng.integers(0, size, size=2)
        r = rng.integers(5, 20)
        cv2.circle(img, (int(cx), int(cy)), int(r), color=int(rng.integers(80, 180)), thickness=-1)

    return img


if __name__ == "__main__":
    clean = make_synthetic_test_pattern(size=512)
    degraded = degrade_pipeline(clean, scale=2, clip_output=True)
    cv2.imwrite("data/synthetic_hard/clean_demo.png", clean.astype(np.uint8))
    cv2.imwrite("data/synthetic_hard/degraded_demo.png", degraded.astype(np.uint8))
    print("Saved clean_demo.png and degraded_demo.png to data/synthetic_hard/")
    print(f"Clean shape: {clean.shape}, Degraded shape: {degraded.shape}")
