# Arshiya_KLA_PS01 — AI-Based Restoration of Degraded Semiconductor Inspection Images

SEMICON India Hackathon 2026 — Track 1 (KLA)

## Problem

Reconstruct a clean, full-resolution semiconductor inspection image from a degraded input that has been simultaneously downsampled (2x), speckle-noised (multiplicative), and Gaussian-noised (edge softening). Evaluated on out-of-distribution structure types and on inference speed.

## Approach

A single-stage, noise-conditioned U-Net with residual bottleneck and PixelShuffle super-resolution head, ending in two branches:
- **Restoration head** — the restored image
- **Confidence head** — a self-supervised per-pixel confidence map, trained against an error-derived target

The model takes the degraded image *and* a cheaply-estimated per-image noise-level map as input, so denoising strength adapts per-image instead of assuming a fixed noise level. See `src/model.py`.

**Loss**: Charbonnier + SSIM + frequency-domain + clip-consistency + confidence terms. See `src/losses.py` for the full formulation and reasoning.

Two model sizes are provided for a speed/quality Pareto comparison:

| Model | Params | CPU inference (untrained, measured) |
|---|---|---|
| small | 2.77M | ~350 ms/image |
| large | 11.06M | ~790 ms/image |

(GPU numbers will be substantially lower; these are prototyping-stage CPU benchmarks.)

## Repo structure

```
src/
  degrade.py           # speckle + gaussian + downsample degradation simulator
  noise_estimator.py   # cheap per-image noise-level estimation (conditioning input)
  model.py             # U-Net backbone, noise conditioning, PixelShuffle, dual heads
  losses.py            # Charbonnier + SSIM + frequency + clip-consistency + confidence loss
  make_demo_figure.py  # generates before/after demo + speed benchmark
  evaluate.py           # standalone evaluation script (tested on fresh environment)
data/
  raw/                 # KLA-provided pairs go here once released
  synthetic_hard/      # extra synthetic hard-case pairs for generalization
weights/                # trained model weights (added after training phase)
outputs/                # restored images + demo figures
```

## Status (registration stage)

This repo currently contains the full pipeline skeleton and architecture,
verified to run end-to-end on synthetic data. Model weights are untrained —
training begins in the next phase once real KLA data is released. All
scripts below have been tested and run successfully:

- `python src/degrade.py` — generates a synthetic degraded/clean pair
- `python src/model.py` — builds both model sizes, verifies forward pass and param counts
- `python src/losses.py` — verifies the combined loss computes and backpropagates
- `python src/make_demo_figure.py` — produces `outputs/demo_before_after.png` and a speed benchmark
- `python src/evaluate.py --input_dir data/raw/degraded --output_dir outputs --model_size small --gt_dir data/raw/clean` — runs the full evaluation pipeline

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Planned novelties

1. **Noise-level conditioning** — per-image noise estimate fed as an extra input channel
2. **Physics-informed loss** — Charbonnier + SSIM + frequency + clip-consistency, tuned for multiplicative speckle rather than assuming additive noise
3. **Self-supervised confidence maps** — flags per-pixel restoration reliability
4. **Speed-quality Pareto reporting** — small vs. large variant, explicit tradeoff
5. **Synthetic hard-case augmentation** — training beyond the given noise range for out-of-distribution generalization

## Planned evaluation metrics (post-training)

| Model | SSIM | PSNR | LPIPS | ms/image (GPU) |
|---|---|---|---|---|
| small | TBD | TBD | TBD | TBD |
| large | TBD | TBD | TBD | TBD |

## References

1. Charbonnier et al., "Two deterministic half-quadratic regularization algorithms for computed imaging," ICIP 1994.
2. Shi et al., "Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network" (PixelShuffle / ESPCN), CVPR 2016.
3. Wang et al., "Image Quality Assessment: From Error Visibility to Structural Similarity" (SSIM), IEEE TIP 2004.
4. Ronneberger et al., "U-Net: Convolutional Networks for Biomedical Image Segmentation," MICCAI 2015.
