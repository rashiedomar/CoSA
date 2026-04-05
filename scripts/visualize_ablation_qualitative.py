#!/usr/bin/env python3
"""
Generate qualitative ablation comparisons for selected samples.

Layout (3x3):
Row 1: T1, T2, GT
Row 2: Baseline, Attention-only, Aligned
Row 3: Gate-only, Multi-scale-only, CoSA v3
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from models.siamese_unet import (
    SiameseUNet,
    SiameseUNetAttention,
    SiameseUNetCoSA,
    SiameseUNetAligned,
)
from scripts.data.levir_dataset_fixed import LEVIRCDDatasetFixed


VARIANT_ORDER = [
    "baseline",
    "attention",
    "aligned",
    "cosa_learnable_gate_only",
    "cosa_multiscale_only",
    "cosa_v3",
]

PANEL_TITLES = {
    "baseline": "Baseline",
    "attention": "Attention-only",
    "aligned": "Alignment-first",
    "cosa_learnable_gate_only": "Gate-only",
    "cosa_multiscale_only": "Multi-scale-only",
    "cosa_v3": "CoSA v3",
}


def load_model(checkpoint_path: Path, variant: str, device: str) -> torch.nn.Module:
    if variant == "baseline":
        model = SiameseUNet(in_channels=3, n_classes=1, base_channels=64, fusion="diff")
    elif variant == "attention":
        model = SiameseUNetAttention(in_channels=3, n_classes=1, base_channels=64, fusion="diff")
    elif variant == "aligned":
        model = SiameseUNetAligned(in_channels=3, n_classes=1, base_channels=64, fusion="diff")
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

        if variant in ["aligned", "cosa_multiscale_only", "cosa_learnable_gate_only", "cosa_v3"]:
            output, _ = model(img1, img2)
        else:
            output = model(img1, img2)

        pred = torch.sigmoid(output).squeeze().cpu().numpy()
        return (pred > 0.5).astype(np.float32)


def denormalize_image(img_tensor: torch.Tensor) -> np.ndarray:
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img = img_tensor * std + mean
    img = torch.clamp(img, 0, 1)
    return img.permute(1, 2, 0).cpu().numpy()


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


def get_sample(datasets: Dict[str, LEVIRCDDatasetFixed], preferred_split: str, sample_id: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    splits = [preferred_split] + [s for s in datasets.keys() if s != preferred_split]
    for split in splits:
        dataset = datasets[split]
        for i, pair in enumerate(dataset.image_pairs):
            if pair["id"] == sample_id:
                img1, img2, label, _ = dataset[i]
                return img1, img2, label
    raise KeyError(f"Sample id not found in any split: {sample_id}")


def create_ablation_grid(img1, img2, gt, preds: Dict[str, np.ndarray], metrics: Dict[str, Dict[str, float]], output_path: Path, sample_id: str) -> None:
    img1_np = denormalize_image(img1)
    img2_np = denormalize_image(img2)
    gt_np = (gt.squeeze().cpu().numpy() > 0.5).astype(np.float32)

    fig, axes = plt.subplots(3, 3, figsize=(14, 12))
    axes = axes.reshape(3, 3)

    # Row 1
    axes[0, 0].imshow(img1_np)
    axes[0, 0].set_title("T1", fontsize=11, fontweight="bold")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(img2_np)
    axes[0, 1].set_title("T2", fontsize=11, fontweight="bold")
    axes[0, 1].axis("off")

    axes[0, 2].imshow(img1_np)
    axes[0, 2].imshow(gt_np, alpha=0.5, cmap="Reds")
    axes[0, 2].set_title("GT", fontsize=11, fontweight="bold")
    axes[0, 2].axis("off")

    # Row 2
    for col, variant in enumerate(["baseline", "attention", "aligned"]):
        pred = preds[variant]
        axes[1, col].imshow(img1_np)
        axes[1, col].imshow(pred, alpha=0.5, cmap="Reds")
        title = PANEL_TITLES[variant]
        title += f"\nF1 {metrics[variant]['f1']:.3f}"
        axes[1, col].set_title(title, fontsize=10, fontweight="bold")
        axes[1, col].axis("off")

    # Row 3
    for col, variant in enumerate(["cosa_learnable_gate_only", "cosa_multiscale_only", "cosa_v3"]):
        pred = preds[variant]
        axes[2, col].imshow(img1_np)
        axes[2, col].imshow(pred, alpha=0.5, cmap="Reds")
        title = PANEL_TITLES[variant]
        title += f"\nF1 {metrics[variant]['f1']:.3f}"
        axes[2, col].set_title(title, fontsize=10, fontweight="bold")
        axes[2, col].axis("off")

    fig.suptitle(f"Ablation Qualitative: {sample_id}", fontsize=14, fontweight="bold", y=0.99)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Qualitative ablation grid generator")
    parser.add_argument("--dataset_dir", type=str, required=True)
    parser.add_argument("--ablation_dir", type=str, default="results/ablation_study")
    parser.add_argument("--output_dir", type=str, default="results/baseline_cosa_visualizations/hero_comparisons")
    parser.add_argument("--samples", type=str, nargs="+",
                        help="Sample IDs like train_332, train_385, val_57")
    parser.add_argument("--auto_select", type=int, default=0,
                        help="If >0, auto-select top-N samples by CoSA v3 improvement")
    parser.add_argument("--auto_split", type=str, default="test", choices=["train", "val", "test"],
                        help="Split to scan when auto-selecting samples")
    parser.add_argument("--min_change_pixels", type=int, default=50,
                        help="Skip samples with fewer GT change pixels than this threshold")
    parser.add_argument("--auto_stride", type=int, default=1,
                        help="Stride when scanning samples for auto-select (e.g., 2 = every other sample)")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    ablation_dir = Path(args.ablation_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load datasets per split lazily
    datasets: Dict[str, LEVIRCDDatasetFixed] = {}

    def get_dataset(split: str) -> LEVIRCDDatasetFixed:
        if split not in datasets:
            datasets[split] = LEVIRCDDatasetFixed(
                root_dir=args.dataset_dir,
                split=split,
                base_size=512,
                augment=False,
                eval_full_res=True,
            )
        return datasets[split]

    # Preload datasets for all splits (split files may include mixed prefixes)
    for split in ["train", "val", "test"]:
        get_dataset(split)

    # Load models
    models = {}
    for variant in VARIANT_ORDER:
        ckpt = ablation_dir / variant / "checkpoint_best.pth"
        if not ckpt.exists():
            raise FileNotFoundError(f"Missing checkpoint: {ckpt}")
        models[variant] = load_model(ckpt, variant, args.device)

    summary_lines = []

    if args.auto_select > 0:
        dataset = get_dataset(args.auto_split)
        candidates = []
        selection_variants = [
            "baseline",
            "cosa_learnable_gate_only",
            "cosa_multiscale_only",
            "cosa_v3",
        ]
        for idx, pair in enumerate(dataset.image_pairs):
            if args.auto_stride > 1 and (idx % args.auto_stride) != 0:
                continue
            sample_id = pair["id"]
            img1, img2, gt = get_sample(datasets, args.auto_split, sample_id)
            gt_np = gt.squeeze().cpu().numpy()
            if (gt_np > 0.5).sum() < args.min_change_pixels:
                continue

            preds: Dict[str, np.ndarray] = {}
            metrics: Dict[str, Dict[str, float]] = {}
            for variant in selection_variants:
                pred = predict(models[variant], img1, img2, variant, args.device)
                preds[variant] = pred
                metrics[variant] = compute_metrics(pred, gt_np)

            score = metrics["cosa_v3"]["f1"] - max(
                metrics["baseline"]["f1"],
                metrics["cosa_learnable_gate_only"]["f1"],
                metrics["cosa_multiscale_only"]["f1"],
            )
            candidates.append((score, sample_id, metrics))

        candidates.sort(key=lambda x: x[0], reverse=True)
        args.samples = [c[1] for c in candidates[: args.auto_select]]

    if not args.samples:
        raise ValueError("No samples provided or selected. Use --samples or --auto_select.")

    for sample_id in args.samples:
        if "_" not in sample_id:
            raise ValueError(f"Sample ID must include split prefix (e.g., train_332): {sample_id}")

        split = sample_id.split("_")[0]
        dataset = get_dataset(split)
        img1, img2, gt = get_sample(datasets, split, sample_id)

        preds: Dict[str, np.ndarray] = {}
        metrics: Dict[str, Dict[str, float]] = {}

        for variant in VARIANT_ORDER:
            pred = predict(models[variant], img1, img2, variant, args.device)
            preds[variant] = pred
            metrics[variant] = compute_metrics(pred, gt.squeeze().cpu().numpy())

        out_path = output_dir / f"ablation_qual_{sample_id}.png"
        create_ablation_grid(img1, img2, gt, preds, metrics, out_path, sample_id)

        # Collect summary
        summary_lines.append(f"## {sample_id}\n")
        summary_lines.append("| Variant | Precision | Recall | F1 | IoU |\n")
        summary_lines.append("|---|---:|---:|---:|---:|\n")
        for variant in VARIANT_ORDER:
            m = metrics[variant]
            summary_lines.append(
                f"| {PANEL_TITLES[variant]} | {m['precision']:.3f} | {m['recall']:.3f} | {m['f1']:.3f} | {m['iou']:.3f} |\n"
            )
        summary_lines.append("\n")

    summary_path = output_dir / "ablation_qualitative_summary.md"
    summary_path.write_text("".join(summary_lines))
    print(f"✅ Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
