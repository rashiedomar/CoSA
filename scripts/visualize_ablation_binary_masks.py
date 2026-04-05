#!/usr/bin/env python3
"""
Qualitative ablation visualizations (binary masks only).

Goal (paper-style):
- 6 columns per row: GT + 5 model predictions.
- Black background, binary masks.
- CoSA v3 panel highlights in GREEN where it fixed errors that other variants made.

Models compared (5):
- baseline               (B0)
- cosa                   (B2 v2: single-scale CoSA)
- cosa_multiscale_only   (ablation)
- cosa_learnable_gate_only (ablation)
- cosa_v3                (B2 v3: full CoSA, multi-scale + learnable gate)

Selection:
- Automatically scan the dataset and pick N samples where CoSA v3 F1
  is clearly better than the other variants (per-image).
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np
import torch
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(REPO_ROOT))

from models.siamese_unet import (
    SiameseUNet,
    SiameseUNetCoSA,
)
from scripts.data.levir_dataset_fixed import LEVIRCDDatasetFixed


VARIANTS: List[str] = [
    "baseline",
    "cosa",
    "cosa_multiscale_only",
    "cosa_learnable_gate_only",
    "cosa_v3",
]

PANEL_TITLES = {
    "gt": "GT",
    "baseline": "Baseline",
    "cosa": "CoSA (single-scale)",
    "cosa_multiscale_only": "Multi-scale only",
    "cosa_learnable_gate_only": "Gate only",
    "cosa_v3": "CoSA v3 (ours)",
}


def load_model(checkpoint_path: Path, variant: str, device: str) -> torch.nn.Module:
    if variant == "baseline":
        model = SiameseUNet(in_channels=3, n_classes=1, base_channels=64, fusion="diff")
    elif variant == "cosa":
        # Original single-scale CoSA (no multiscale, no learnable gate)
        model = SiameseUNetCoSA(
            in_channels=3,
            n_classes=1,
            base_channels=64,
            fusion="diff",
            topk=32,
            use_multiscale=False,
            use_learnable_gate=False,
        )
    elif variant == "cosa_multiscale_only":
        model = SiameseUNetCoSA(
            in_channels=3,
            n_classes=1,
            base_channels=64,
            fusion="diff",
            topk=32,
            use_multiscale=True,
            use_learnable_gate=False,
        )
    elif variant == "cosa_learnable_gate_only":
        model = SiameseUNetCoSA(
            in_channels=3,
            n_classes=1,
            base_channels=64,
            fusion="diff",
            topk=32,
            use_multiscale=False,
            use_learnable_gate=True,
        )
    elif variant == "cosa_v3":
        model = SiameseUNetCoSA(
            in_channels=3,
            n_classes=1,
            base_channels=64,
            fusion="diff",
            topk=32,
            use_multiscale=True,
            use_learnable_gate=True,
        )
    else:
        raise ValueError(f"Unknown variant: {variant}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model = model.to(device)
    model.eval()
    return model


def predict(model: torch.nn.Module, img1: torch.Tensor, img2: torch.Tensor, variant: str, device: str) -> np.ndarray:
    with torch.no_grad():
        if img1.dim() == 3:
            img1 = img1.unsqueeze(0).to(device)
            img2 = img2.unsqueeze(0).to(device)
        else:
            img1 = img1.to(device)
            img2 = img2.to(device)

        if isinstance(model, SiameseUNetCoSA):
            output, _ = model(img1, img2)
        else:
            output = model(img1, img2)

        pred = torch.sigmoid(output).squeeze().cpu().numpy()
        return (pred > 0.5).astype(np.float32)


def compute_metrics(pred: np.ndarray, gt: np.ndarray) -> Dict[str, float]:
    pred = (pred > 0.5).astype(np.uint8)
    gt = (gt > 0.5).astype(np.uint8)
    tp = (pred & gt).sum()
    fp = (pred & (1 - gt)).sum()
    fn = ((1 - pred) & gt).sum()

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    iou = tp / (tp + fp + fn + 1e-8)

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "iou": float(iou),
    }


def get_sample(dataset: LEVIRCDDatasetFixed, sample_idx: int):
    img1, img2, label, name = dataset[sample_idx]
    return img1, img2, label, name


def denormalize_image(img_tensor: torch.Tensor) -> np.ndarray:
    """Denormalize ImageNet-normalized image for visualization (same as training)."""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(img_tensor.device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(img_tensor.device)
    img = img_tensor * std + mean
    img = torch.clamp(img, 0, 1)
    return img.permute(1, 2, 0).cpu().numpy()


def create_six_column_grid(
    img2_tensor: torch.Tensor,
    gt: np.ndarray,
    preds: Dict[str, np.ndarray],
    output_path: Path,
    sample_name: str,
    bw_mode: bool = False,
) -> None:
    """
    Create 1x6 grid:
    [GT | baseline | cosa | multi-scale-only | gate-only | cosa_v3 (+green fixes)]
    All as binary masks on black background.
    """
    gt_bin = (gt > 0.5).astype(np.float32)
    h, w = gt_bin.shape
    gt_mask = gt_bin.astype(bool)

    # Background RGB (T2 image)
    if img2_tensor.dim() == 4:
        img2_tensor = img2_tensor.squeeze(0)
    img_bg_color = denormalize_image(img2_tensor)

    # Where CoSA v3 correctly detects change (TP) that others miss / get wrong
    cosa_v3_pred = (preds["cosa_v3"] > 0.5)
    cosa_v3_tp = cosa_v3_pred & gt_mask  # only true positives (detections)
    others_wrong = np.zeros_like(gt_mask, dtype=bool)
    for v in ["baseline", "cosa", "cosa_multiscale_only", "cosa_learnable_gate_only"]:
        p = (preds[v] > 0.5)
        # others either miss the change (FN) or make a false alarm (FP)
        others_wrong |= (p != gt_mask)
    improvement = cosa_v3_tp & others_wrong

    fig, axes = plt.subplots(1, 6, figsize=(18, 3))

    panels = [
        ("gt", gt_bin, None),
        ("baseline", preds["baseline"], None),
        ("cosa", preds["cosa"], None),
        ("cosa_multiscale_only", preds["cosa_multiscale_only"], None),
        ("cosa_learnable_gate_only", preds["cosa_learnable_gate_only"], None),
        ("cosa_v3", preds["cosa_v3"], improvement.astype(np.float32)),
    ]

    for ax, (key, mask, hl) in zip(axes, panels):
        if key == "gt":
            # GT: pure binary mask, black background and white foreground
            ax.imshow(gt_bin, cmap="gray", vmin=0.0, vmax=1.0)
        else:
            # Background: either original T2 or pure black (B/W mode)
            if bw_mode:
                ax.imshow(np.zeros_like(img_bg_color))
            else:
                ax.imshow(img_bg_color)

            pred_mask = (mask > 0.5).astype(bool)

            # Correct detections (TP: pred=1, GT=1): green on detection pixels only
            tp = pred_mask & gt_mask
            if tp.any():
                corr = np.zeros((h, w, 4), dtype=np.float32)
                corr[tp, 1] = 1.0  # green
                corr[tp, 3] = 0.7
                ax.imshow(corr)

            # Wrong detections (FP: pred=1, GT=0): red on detection pixels only
            fp = pred_mask & (~gt_mask)
            if fp.any():
                err = np.zeros((h, w, 4), dtype=np.float32)
                err[fp, 0] = 1.0  # red
                err[fp, 3] = 0.7
                ax.imshow(err)

            # False negatives (FN: pred=0, GT=1): highlight missed changes (yellow)
            fn = (~pred_mask) & gt_mask
            if fn.any():
                fn_ov = np.zeros((h, w, 4), dtype=np.float32)
                fn_ov[fn, 0] = 1.0  # red
                fn_ov[fn, 1] = 1.0  # green -> yellow
                fn_ov[fn, 3] = 0.7
                ax.imshow(fn_ov)

            # CoSA v3: additionally highlight pixels it uniquely fixes in blue
            if key == "cosa_v3" and hl is not None:
                hl_mask = hl > 0.5
                if hl_mask.any():
                    blue = np.zeros((h, w, 4), dtype=np.float32)
                    blue[hl_mask, 2] = 1.0  # blue channel
                    blue[hl_mask, 3] = 0.9
                    ax.imshow(blue)

        ax.set_title(PANEL_TITLES[key], fontsize=10, fontweight="bold")
        ax.axis("off")

    fig.suptitle(f"{sample_name}", fontsize=12, fontweight="bold", y=0.98)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def auto_select_samples(
    dataset: LEVIRCDDatasetFixed,
    models: Dict[str, torch.nn.Module],
    device: str,
    num_select: int,
    min_change_pixels: int = 50,
    stride: int = 1,
) -> List[int]:
    """
    Scan dataset and pick indices where CoSA v3 has the largest positive F1 gain
    over the best of the other variants.
    """
    candidates: List[Tuple[float, int]] = []
    n = len(dataset)

    for idx in range(0, n, stride):
        img1, img2, label, _ = dataset[idx]
        gt_np = label.squeeze().cpu().numpy()
        if (gt_np > 0.5).sum() < min_change_pixels:
            continue

        preds: Dict[str, np.ndarray] = {}
        metrics: Dict[str, Dict[str, float]] = {}

        for variant in VARIANTS:
            pred = predict(models[variant], img1, img2, variant, device)
            preds[variant] = pred
            metrics[variant] = compute_metrics(pred, gt_np)

        # CoSA v3 improvement over the best of others
        best_other_f1 = max(
            metrics["baseline"]["f1"],
            metrics["cosa"]["f1"],
            metrics["cosa_multiscale_only"]["f1"],
            metrics["cosa_learnable_gate_only"]["f1"],
        )
        score = metrics["cosa_v3"]["f1"] - best_other_f1
        candidates.append((score, idx))

    candidates.sort(key=lambda x: x[0], reverse=True)
    selected = [idx for _, idx in candidates[:num_select]]
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description="Qualitative ablation binary-mask visualizations")
    parser.add_argument("--dataset_dir", type=str, required=True)
    parser.add_argument("--ablation_dir", type=str, default="results/ablation_study")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/ablation_study/qualitative_binary_masks",
    )
    parser.add_argument(
        "--auto_select",
        type=int,
        default=6,
        help="Number of samples to auto-select (by CoSA v3 improvement).",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test"],
    )
    parser.add_argument(
        "--min_change_pixels",
        type=int,
        default=30000,
        help="Minimum number of positive pixels in GT to consider a sample.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Stride when scanning dataset (e.g., 2 = every second sample).",
    )
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--bw",
        action="store_true",
        help="If set, use black background instead of T2 image (detection colors only).",
    )
    args = parser.parse_args()

    ablation_dir = REPO_ROOT / args.ablation_dir
    output_dir = REPO_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    device = args.device

    # Load dataset
    dataset = LEVIRCDDatasetFixed(
        root_dir=args.dataset_dir,
        split=args.split,
        base_size=512,
        augment=False,
        eval_full_res=True,
    )

    # Load models
    models: Dict[str, torch.nn.Module] = {}
    for variant in VARIANTS:
        ckpt = ablation_dir / variant / "checkpoint_best.pth"
        if not ckpt.exists():
            raise FileNotFoundError(f"Missing checkpoint for {variant}: {ckpt}")
        models[variant] = load_model(ckpt, variant, device)

    # Auto-select indices
    selected_indices = auto_select_samples(
        dataset,
        models,
        device,
        num_select=args.auto_select,
        min_change_pixels=args.min_change_pixels,
        stride=args.stride,
    )

    # Generate visualizations for selected samples
    for rank, idx in enumerate(selected_indices, start=1):
        img1, img2, label, name = get_sample(dataset, idx)
        gt_np = label.squeeze().cpu().numpy()

        preds: Dict[str, np.ndarray] = {}
        for variant in VARIANTS:
            preds[variant] = predict(models[variant], img1, img2, variant, device)

        # Clean name for filename
        if isinstance(name, (tuple, list)):
            name_str = name[0] if len(name) > 0 else f"sample_{idx}"
        else:
            name_str = str(name)
        name_str = str(name_str).replace("'", "").replace("(", "").replace(")", "").replace(",", "")

        out_path = output_dir / f"ablation_binary_{rank:02d}_{name_str}.png"
        create_six_column_grid(img2, gt_np, preds, out_path, sample_name=name_str, bw_mode=args.bw)

    print(f"✅ Saved {len(selected_indices)} qualitative binary-mask figures to {output_dir}")


if __name__ == "__main__":
    main()

