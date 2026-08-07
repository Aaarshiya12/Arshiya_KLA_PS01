"""
evaluate.py — standalone evaluation script.

Usage:
python src/evaluate.py --input_dir <test_images_directory> --output_dir outputs

Optional:
python src/evaluate.py --input_dir <test_images_directory> \
    --output_dir outputs \
    --gt_dir <ground_truth_directory>

If --gt_dir is provided, computes SSIM and PSNR against ground truth.
Reports per-image and average inference time.

The script loads the trained model weights and runs the complete
restoration pipeline on all supported images in the input directory.
"""

import argparse
import os
import time

import cv2
import numpy as np
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
    """Load an image as grayscale."""
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")

    return img


def to_tensor(img_uint8, device):
    """Convert a uint8 grayscale image to a normalized PyTorch tensor."""
    return (
        torch.from_numpy(img_uint8.astype(np.float32) / 255.0)
        .unsqueeze(0)
        .unsqueeze(0)
        .to(device)
    )


def main():

    parser = argparse.ArgumentParser(
        description="NanoSight semiconductor image restoration evaluation script"
    )

    parser.add_argument(
        "--input_dir",
        required=True,
        help="Directory containing degraded input images",
    )

    parser.add_argument(
        "--output_dir",
        required=True,
        help="Directory where restored images will be saved",
    )

    parser.add_argument(
        "--model_size",
        choices=["small", "large"],
        default="small",
        help="Model architecture size",
    )

    parser.add_argument(
        "--weights",
        default="kla_best_model.pt",
        help="Path to trained model weights",
    )

    parser.add_argument(
        "--gt_dir",
        default=None,
        help="Optional directory containing ground-truth images",
    )

    args = parser.parse_args()

    # ---------------------------------------------------------
    # Setup
    # ---------------------------------------------------------

    os.makedirs(args.output_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Using device: {device}")

    # ---------------------------------------------------------
    # Load model
    # ---------------------------------------------------------

    model = build_model(args.model_size).to(device)

    if not os.path.exists(args.weights):
        raise FileNotFoundError(
            f"Trained model weights not found: {args.weights}"
        )

    state_dict = torch.load(
        args.weights,
        map_location=device,
    )

    model.load_state_dict(state_dict)
    model.eval()

    print(f"Loaded trained weights from: {args.weights}")

    # ---------------------------------------------------------
    # Find input images
    # ---------------------------------------------------------

    if not os.path.isdir(args.input_dir):
        raise NotADirectoryError(
            f"Input directory not found: {args.input_dir}"
        )

    image_files = sorted(
        [
            f
            for f in os.listdir(args.input_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]
    )

    if not image_files:
        raise RuntimeError(
            f"No supported images found in: {args.input_dir}"
        )

    print(f"Found {len(image_files)} input images.")

    # ---------------------------------------------------------
    # Evaluation
    # ---------------------------------------------------------

    ssim_scores = []
    psnr_scores = []
    times_ms = []

    for fname in image_files:

        input_path = os.path.join(args.input_dir, fname)

        degraded = load_image(input_path)

        # Estimate degradation/noise map
        noise_map = estimate_noise_map(
            degraded,
            patch_size=8,
        )

        img_tensor = to_tensor(
            degraded,
            device,
        )

        noise_uint8 = (
            (noise_map * 255)
            .clip(0, 255)
            .astype(np.uint8)
        )

        noise_tensor = to_tensor(
            noise_uint8,
            device,
        )

        # -----------------------------------------------------
        # Inference
        # -----------------------------------------------------

        if device == "cuda":
            torch.cuda.synchronize()

        start = time.perf_counter()

        with torch.no_grad():
            restored, confidence = model(
                img_tensor,
                noise_tensor,
            )

        if device == "cuda":
            torch.cuda.synchronize()

        elapsed_ms = (
            time.perf_counter() - start
        ) * 1000.0

        times_ms.append(elapsed_ms)

        # -----------------------------------------------------
        # Convert output to image
        # -----------------------------------------------------

        restored_np = (
            restored.squeeze()
            .detach()
            .cpu()
            .numpy()
            * 255.0
        )

        restored_np = (
            restored_np
            .clip(0, 255)
            .astype(np.uint8)
        )

        # -----------------------------------------------------
        # Save restored output
        # -----------------------------------------------------

        output_path = os.path.join(
            args.output_dir,
            f"restored_{fname}",
        )

        success = cv2.imwrite(
            output_path,
            restored_np,
        )

        if not success:
            raise RuntimeError(
                f"Failed to save output image: {output_path}"
            )

        # -----------------------------------------------------
        # Optional ground-truth metrics
        # -----------------------------------------------------

        if args.gt_dir and METRICS_AVAILABLE:

            gt_path = os.path.join(
                args.gt_dir,
                fname,
            )

            if os.path.exists(gt_path):

                gt = load_image(gt_path)

                if gt.shape == restored_np.shape:

                    ssim_scores.append(
                        sk_ssim(
                            gt,
                            restored_np,
                            data_range=255,
                        )
                    )

                    psnr_scores.append(
                        sk_psnr(
                            gt,
                            restored_np,
                            data_range=255,
                        )
                    )

        print(
            f"{fname}: {elapsed_ms:.2f} ms "
            f"-> {output_path}"
        )

    # ---------------------------------------------------------
    # Summary
    # ---------------------------------------------------------

    print("\n========== Evaluation Summary ==========")

    print(
        f"Images processed: {len(image_files)}"
    )

    print(
        f"Average inference time: "
        f"{np.mean(times_ms):.2f} ms/image"
    )

    if ssim_scores:

        print(
            f"Average SSIM: "
            f"{np.mean(ssim_scores):.4f}"
        )

        print(
            f"Average PSNR: "
            f"{np.mean(psnr_scores):.2f} dB"
        )

    elif args.gt_dir:

        if not METRICS_AVAILABLE:

            print(
                "Ground-truth metrics unavailable because "
                "scikit-image is not installed."
            )

        else:

            print(
                "No matching ground-truth images were found "
                "for metric computation."
            )

    print(
        f"Restored outputs saved to: "
        f"{args.output_dir}"
    )

    print("========================================")


if __name__ == "__main__":
    main()
