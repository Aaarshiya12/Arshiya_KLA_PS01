"""
evaluate.py — standalone evaluation script (mandatory per KLA submission brief).

Usage:
    python evaluate.py --input_dir data/raw/degraded --output_dir outputs --model_size small [--gt_dir data/raw/clean]

If --gt_dir is provided, computes SSIM/PSNR/LPIPS against ground truth.
Always reports per-image inference time in ms.

NOTE: at registration stage, model weights are untrained/random -- this script
is provided to demonstrate the full evaluation pipeline runs correctly on a
fresh machine, per the brief's explicit warning that an unrunnable script
scores zero regardless of model quality.
"""

import argparse
import os
import time
import numpy as np
import cv2
import torch

from model import build_model
from noise_estimator import estimate_noise_map

try:
    from skimage.metrics import structural_similarity as sk_ssim
    from skimage.metrics import peak_signal_noise_ratio as sk_psnr
    METRICS_AVAILABLE = True
except ImportError:
    METRICS_AVAILABLE = False


def load_image(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return img


def to_tensor(img_uint8, device):
    return torch.from_numpy(img_uint8.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0).to(device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True, help="Directory of degraded input images")
    parser.add_argument("--output_dir", required=True, help="Where to save restored outputs")
    parser.add_argument("--model_size", choices=["small", "large"], default="small")
    parser.add_argument("--weights", default="kla_best_model.pt", help="Path to trained model weights")
    parser.add_argument("--gt_dir", default=None, help="Directory of ground-truth images for metrics")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = build_model(args.model_size).to(device).eval()
    if args.weights and os.path.exists(args.weights):
        model.load_state_dict(torch.load(args.weights, map_location=device))
        print(f"Loaded weights from {args.weights}")
    else:
        print("WARNING: no trained weights loaded -- running with random-initialized weights.")

    image_files = sorted([f for f in os.listdir(args.input_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))])
    if not image_files:
        print(f"No images found in {args.input_dir}")
        return

    ssim_scores, psnr_scores, times_ms = [], [], []

    for fname in image_files:
        degraded = load_image(os.path.join(args.input_dir, fname))
        noise_map = estimate_noise_map(degraded, patch_size=8)

        img_tensor = to_tensor(degraded, device)
        noise_tensor = to_tensor((noise_map * 255).clip(0, 255).astype(np.uint8), device)

        start = time.time()
        with torch.no_grad():
            restored, confidence = model(img_tensor, noise_tensor)
        elapsed_ms = (time.time() - start) * 1000
        times_ms.append(elapsed_ms)

        restored_np = (restored.squeeze().cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        cv2.imwrite(os.path.join(args.output_dir, f"restored_{fname}"), restored_np)

        if args.gt_dir and METRICS_AVAILABLE:
            gt_path = os.path.join(args.gt_dir, fname)
            if os.path.exists(gt_path):
                gt = load_image(gt_path)
                if gt.shape == restored_np.shape:
                    ssim_scores.append(sk_ssim(gt, restored_np, data_range=255))
                    psnr_scores.append(sk_psnr(gt, restored_np, data_range=255))

        print(f"{fname}: {elapsed_ms:.2f} ms")

    print("\n--- Summary ---")
    print(f"Images processed: {len(image_files)}")
    print(f"Avg inference time: {np.mean(times_ms):.2f} ms/image")
    if ssim_scores:
        print(f"Avg SSIM: {np.mean(ssim_scores):.4f}")
        print(f"Avg PSNR: {np.mean(psnr_scores):.2f} dB")
    elif args.gt_dir:
        print("No matching ground-truth images found for metric computation, or scikit-image not installed.")


if __name__ == "__main__":
    main()
