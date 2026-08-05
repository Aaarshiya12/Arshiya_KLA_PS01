"""
noise_estimator.py — cheap, non-learned per-image noise level estimation.

Used to feed a noise-level map as an extra input channel to the restoration
network, so the model can adapt its denoising strength per-image instead of
assuming one fixed noise level (this is the Tier-1 novelty: noise-conditioning).

Method: split the image into small patches, take the flattest (lowest local
variance) patches -- these are least likely to contain real structure -- and
estimate noise std from those. This is a standard trick from classical
denoising literature (e.g. used in BM3D-style noise estimation).
"""

import numpy as np


def estimate_noise_map(image: np.ndarray, patch_size: int = 8, flat_fraction: float = 0.2) -> np.ndarray:
    """
    image: 2D float array (grayscale)
    Returns: a low-res noise level map, one value per patch, upsampled back
    to the original image size (so it can be concatenated as a channel).
    """
    h, w = image.shape
    ph, pw = h // patch_size, w // patch_size
    variances = np.zeros((ph, pw), dtype=np.float32)

    for i in range(ph):
        for j in range(pw):
            patch = image[i * patch_size:(i + 1) * patch_size, j * patch_size:(j + 1) * patch_size]
            variances[i, j] = patch.var()

    # take the flattest patches globally to estimate the noise floor
    flat_thresh = np.quantile(variances, flat_fraction)
    flat_patch_vars = variances[variances <= flat_thresh]
    noise_std = float(np.sqrt(flat_patch_vars.mean())) if len(flat_patch_vars) > 0 else 0.0

    noise_map = np.full((h, w), noise_std, dtype=np.float32)
    return noise_map


def estimate_scalar_noise_level(image: np.ndarray, patch_size: int = 8, flat_fraction: float = 0.2) -> float:
    """Single scalar version, if you'd rather condition on one number instead of a map."""
    noise_map = estimate_noise_map(image, patch_size, flat_fraction)
    return float(noise_map[0, 0])


if __name__ == "__main__":
    import cv2
    degraded = cv2.imread("data/synthetic_hard/degraded_demo.png", cv2.IMREAD_GRAYSCALE).astype(np.float32)
    level = estimate_scalar_noise_level(degraded)
    print(f"Estimated noise std for degraded_demo.png: {level:.3f}")
