"""
make_demo_figure.py — generates the proof-of-concept demo figure for the
registration slides (clean vs. degraded vs. untrained-model-pass), plus
measures inference time for small vs. large model variants (Slide 7 Pareto data).

Note: the model here is UNTRAINED (random weights) since registration stage
happens before real training. This demo proves the pipeline runs end-to-end,
not restoration quality -- label it honestly on the slide.
"""

import time
import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt

from degrade import make_synthetic_test_pattern, degrade_pipeline
from noise_estimator import estimate_noise_map
from model import build_model, count_params


def normalize_for_model(img_uint8):
    return torch.from_numpy(img_uint8.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0)


def run_demo():
    clean = make_synthetic_test_pattern(size=512, seed=1)
    degraded = degrade_pipeline(clean, scale=2, clip_output=True)

    noise_map = estimate_noise_map(degraded, patch_size=8)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on device: {device}")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(clean, cmap="gray", vmin=0, vmax=255)
    axes[0].set_title("Clean ground truth (512x512)")
    axes[0].axis("off")

    axes[1].imshow(degraded, cmap="gray", vmin=0, vmax=255)
    axes[1].set_title("Degraded input (256x256)\nspeckle + gaussian + downsample")
    axes[1].axis("off")

    results = {}
    for size in ["small", "large"]:
        model = build_model(size).to(device).eval()
        n_params = count_params(model)

        img_tensor = normalize_for_model(degraded).to(device)
        noise_tensor = normalize_for_model((noise_map * 255).clip(0, 255).astype(np.uint8)).to(device)

        # warmup
        with torch.no_grad():
            _ = model(img_tensor, noise_tensor)

        n_runs = 20
        start = time.time()
        with torch.no_grad():
            for _ in range(n_runs):
                restored, confidence = model(img_tensor, noise_tensor)
        elapsed_ms = (time.time() - start) / n_runs * 1000

        results[size] = {"params": n_params, "ms_per_image": elapsed_ms}
        print(f"[{size}] params={n_params:,}  avg inference time={elapsed_ms:.2f} ms/image (device={device})")

        if size == "large":
            restored_np = (restored.squeeze().cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
            axes[2].imshow(restored_np, cmap="gray", vmin=0, vmax=255)
            axes[2].set_title("Model output (UNTRAINED weights)\npipeline runs end-to-end")
            axes[2].axis("off")

    plt.tight_layout()
    plt.savefig("outputs/demo_before_after.png", dpi=150, bbox_inches="tight")
    print("\nSaved outputs/demo_before_after.png")

    # Pareto benchmark table
    print("\n--- Speed/size benchmark (for Slide 7) ---")
    print(f"{'Model':<8}{'Params':<15}{'ms/image':<12}")
    for size, r in results.items():
        print(f"{size:<8}{r['params']:<15,}{r['ms_per_image']:<12.2f}")

    return results


if __name__ == "__main__":
    run_demo()
