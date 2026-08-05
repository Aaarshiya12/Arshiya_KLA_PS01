import argparse
from pathlib import Path
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split

from model import build_model
from losses import CombinedLoss
from noise_estimator import estimate_noise_map


class KLAPairedDataset(Dataset):
    def __init__(self, data_dir):
        self.root = Path(data_dir)
        self.gt_dir = self.root / "GT"
        self.lr_dir = self.root / "NoisyLR"

        if not self.gt_dir.exists() or not self.lr_dir.exists():
            raise FileNotFoundError(
                f"Expected {self.root}/GT and {self.root}/NoisyLR"
            )

        gt_names = {p.name for p in self.gt_dir.glob("*.npy")}
        lr_names = {p.name for p in self.lr_dir.glob("*.npy")}
        self.names = sorted(gt_names & lr_names)

        if not self.names:
            raise RuntimeError("No matching .npy pairs found.")

        print(f"Found {len(self.names)} paired samples.")

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        name = self.names[idx]
        lr = np.load(self.lr_dir / name).astype(np.float32).squeeze()
        gt = np.load(self.gt_dir / name).astype(np.float32).squeeze()

        # KLA arrays are expected to be grayscale 2-D arrays.
        if lr.ndim != 2 or gt.ndim != 2:
            raise ValueError(f"{name}: expected 2-D arrays, got LR={lr.shape}, GT={gt.shape}")

        # Preserve KLA's out-of-range noisy values. Only scale if data is clearly 0..255.
        scale = 255.0 if max(float(np.nanmax(gt)), float(np.nanmax(lr))) > 4.0 else 1.0
        lr = lr / scale
        gt = gt / scale

        noise = estimate_noise_map(lr, patch_size=8).astype(np.float32)

        lr_t = torch.from_numpy(lr).unsqueeze(0)
        gt_t = torch.from_numpy(gt).unsqueeze(0)
        noise_t = torch.from_numpy(noise).unsqueeze(0)
        return lr_t, noise_t, gt_t, name


def psnr(pred, target):
    mse = F.mse_loss(pred, target).item()
    if mse <= 1e-12:
        return 99.0
    return 10.0 * np.log10(1.0 / mse)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", required=True, help="Folder containing GT/ and NoisyLR/")
    p.add_argument("--model_size", choices=["small", "large"], default="small")
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--val_fraction", type=float, default=0.1)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--output", default="weights/best_model.pt")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    ds = KLAPairedDataset(args.data_dir)
    n_val = max(1, int(len(ds) * args.val_fraction))
    n_train = len(ds) - n_val
    train_ds, val_ds = random_split(
        ds, [n_train, n_val],
        generator=torch.Generator().manual_seed(args.seed)
    )

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda")
    )
    val_loader = DataLoader(
        val_ds, batch_size=1, shuffle=False,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda")
    )

    model = build_model(args.model_size).to(device)
    loss_fn = CombinedLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses = []

        for step, (lr_img, noise, gt, _) in enumerate(train_loader, 1):
            lr_img = lr_img.to(device)
            noise = noise.to(device)
            gt = gt.to(device)

            opt.zero_grad(set_to_none=True)
            pred, conf = model(lr_img, noise)

            if pred.shape[-2:] != gt.shape[-2:]:
                raise RuntimeError(
                    f"Resolution mismatch: model output {tuple(pred.shape)} vs GT {tuple(gt.shape)}"
                )

            lr_up = F.interpolate(lr_img, size=gt.shape[-2:], mode="bilinear", align_corners=False)

            # Existing losses.py uses clip thresholds/tau designed for 0..255 values.
            # Feed 0..255 copies to that loss while the network itself works in ~0..1 scale.
            total, parts = loss_fn(pred * 255.0, gt * 255.0, lr_up * 255.0, conf)
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            train_losses.append(total.item())
            if step % 25 == 0 or step == len(train_loader):
                print(f"Epoch {epoch}/{args.epochs} step {step}/{len(train_loader)} loss={np.mean(train_losses[-25:]):.4f}")

        model.eval()
        val_losses, val_psnr = [], []
        with torch.no_grad():
            for lr_img, noise, gt, _ in val_loader:
                lr_img = lr_img.to(device)
                noise = noise.to(device)
                gt = gt.to(device)

                pred, conf = model(lr_img, noise)
                lr_up = F.interpolate(lr_img, size=gt.shape[-2:], mode="bilinear", align_corners=False)
                total, _ = loss_fn(pred * 255.0, gt * 255.0, lr_up * 255.0, conf)
                val_losses.append(total.item())
                val_psnr.append(psnr(pred.clamp(0, 1), gt.clamp(0, 1)))

        mean_train = float(np.mean(train_losses))
        mean_val = float(np.mean(val_losses))
        mean_psnr = float(np.mean(val_psnr))
        print(f"\nEpoch {epoch}: train_loss={mean_train:.4f} val_loss={mean_val:.4f} val_PSNR={mean_psnr:.2f} dB\n")

        if mean_val < best_val:
            best_val = mean_val
            torch.save(model.state_dict(), out)
            print(f"Saved best weights -> {out}")

    print("Training complete.")
    print(f"Best validation loss: {best_val:.4f}")
    print(f"Weights: {out}")


if __name__ == "__main__":
    main()
