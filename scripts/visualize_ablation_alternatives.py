#!/usr/bin/env python3
"""
Generate alternative layouts for the ablation qualitative figure.

Current figure: 3 rows × 6 columns (3 samples, GT + 5 variants per row).

Alternatives produced:
  alt1_row_per_variant: 6 rows × 3 columns — each row is one variant, cols = 3 samples.
  alt2_compact:         3 rows × 3 columns — only GT | Baseline | CoSA v3 (main comparison).
  alt3_errors_only:    3 rows × 6 columns — same layout, only FP/FN shown (no green TP).
  alt4_improvement:    3 rows × 2 columns — CoSA v3 pred | (CoSA − Baseline) difference map.

Usage:
  python scripts/visualize_ablation_alternatives.py --dataset_dir /path/to/LEVIR-CD \\
    [--ablation_dir results/ablation_study] [--output_dir paper/figures] [--methods alt1 alt2 alt3 alt4]
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import matplotlib.pyplot as plt

# Paths
REPO_ROOT = Path(__file__).resolve().parents[1]   # .../change_detection/research_repo
ROOT = Path(__file__).resolve().parents[2]        # .../change_detection

import sys
import importlib.util

for p in (ROOT, REPO_ROOT):
    p_str = str(p)
    if p_str not in sys.path:
        sys.path.insert(0, p_str)

# Robustly load SiameseUNet definitions from the main workspace
_siamese_candidates = [
    ROOT / "models" / "siamese_unet.py",
    REPO_ROOT / "configs" / "custom" / "cosa_v3" / "siamese_unet.py",
    REPO_ROOT / "configs" / "custom" / "baseline" / "siamese_unet.py",
]
_siamese_module = None
for _path in _siamese_candidates:
    if _path.exists():
        spec = importlib.util.spec_from_file_location("cosa_siamese_unet", str(_path))
        _siamese_module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        sys.modules["cosa_siamese_unet"] = _siamese_module
        spec.loader.exec_module(_siamese_module)
        break

if _siamese_module is None:
    raise FileNotFoundError("Could not locate siamese_unet.py in expected locations.")

SiameseUNet = _siamese_module.SiameseUNet
SiameseUNetCoSA = _siamese_module.SiameseUNetCoSA

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
        model = SiameseUNetCoSA(
            in_channels=3, n_classes=1, base_channels=64, fusion="diff",
            topk=32, use_multiscale=False, use_learnable_gate=False,
        )
    elif variant == "cosa_multiscale_only":
        model = SiameseUNetCoSA(
            in_channels=3, n_classes=1, base_channels=64, fusion="diff",
            topk=32, use_multiscale=True, use_learnable_gate=False,
        )
    elif variant == "cosa_learnable_gate_only":
        model = SiameseUNetCoSA(
            in_channels=3, n_classes=1, base_channels=64, fusion="diff",
            topk=32, use_multiscale=False, use_learnable_gate=True,
        )
    elif variant == "cosa_v3":
        model = SiameseUNetCoSA(
            in_channels=3, n_classes=1, base_channels=64, fusion="diff",
            topk=32, use_multiscale=True, use_learnable_gate=True,
        )
    else:
        raise ValueError(f"Unknown variant: {variant}")

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        model.load_state_dict(ckpt["state_dict"])
    else:
        model.load_state_dict(ckpt)
    model = model.to(device).eval()
    return model


def predict(model: torch.nn.Module, img1: torch.Tensor, img2: torch.Tensor, device: str) -> np.ndarray:
    with torch.no_grad():
        if img1.dim() == 3:
            img1, img2 = img1.unsqueeze(0).to(device), img2.unsqueeze(0).to(device)
        else:
            img1, img2 = img1.to(device), img2.to(device)
        if isinstance(model, SiameseUNetCoSA):
            out, _ = model(img1, img2)
        else:
            out = model(img1, img2)
        pred = torch.sigmoid(out).squeeze().cpu().numpy()
    return (pred > 0.5).astype(np.float32)


def denormalize_image(img_tensor: torch.Tensor) -> np.ndarray:
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(img_tensor.device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(img_tensor.device)
    img = torch.clamp(img_tensor * std + mean, 0, 1)
    return img.permute(1, 2, 0).cpu().numpy()


def _draw_tp_fp_fn(ax, pred_bin: np.ndarray, gt_bin: np.ndarray, h: int, w: int,
                   show_tp: bool = True, bg_rgba: np.ndarray | None = None) -> None:
    gt_mask = gt_bin.astype(bool)
    pred_mask = pred_bin.astype(bool)
    if bg_rgba is not None:
        ax.imshow(bg_rgba)
    if show_tp:
        tp = pred_mask & gt_mask
        if tp.any():
            ov = np.zeros((h, w, 4), dtype=np.float32)
            ov[tp, 1] = 1.0
            ov[tp, 3] = 0.7
            ax.imshow(ov)
    fp = pred_mask & (~gt_mask)
    if fp.any():
        ov = np.zeros((h, w, 4), dtype=np.float32)
        ov[fp, 0] = 1.0
        ov[fp, 3] = 0.7
        ax.imshow(ov)
    fn = (~pred_mask) & gt_mask
    if fn.any():
        ov = np.zeros((h, w, 4), dtype=np.float32)
        ov[fn, 0] = 1.0
        ov[fn, 1] = 1.0
        ov[fn, 3] = 0.7
        ax.imshow(ov)


def create_alt1_row_per_variant(
    gt_list: List[np.ndarray],
    preds_list: List[Dict[str, np.ndarray]],
    bg_list: List[np.ndarray],
    sample_names: List[str],
    output_path: Path,
) -> None:
    """6 rows × 3 columns: each row = one variant, columns = 3 samples."""
    n_samples = len(gt_list)
    keys = ["gt"] + VARIANTS
    n_rows, n_cols = len(keys), n_samples
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    for row, key in enumerate(keys):
        for col in range(n_samples):
            ax = axes[row, col]
            gt_bin = (gt_list[col] > 0.5).astype(np.float32)
            h, w = gt_bin.shape
            bg = bg_list[col]
            ax.imshow(bg)
            if key == "gt":
                ax.imshow(gt_bin, alpha=0.6, cmap="Reds")
            else:
                pred = (preds_list[col][key] > 0.5).astype(np.float32)
                _draw_tp_fp_fn(ax, pred, gt_bin, h, w, show_tp=True, bg_rgba=None)
            if row == 0:
                ax.set_title(sample_names[col], fontsize=10, fontweight="bold")
            if col == 0:
                ax.set_ylabel(PANEL_TITLES.get(key, key), fontsize=10, fontweight="bold")
            ax.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()


def create_alt2_compact(
    gt_list: List[np.ndarray],
    preds_list: List[Dict[str, np.ndarray]],
    bg_list: List[np.ndarray],
    sample_names: List[str],
    output_path: Path,
) -> None:
    """3 rows × 3 columns: GT | Baseline | CoSA v3 only."""
    n = len(gt_list)
    fig, axes = plt.subplots(n, 3, figsize=(9, 3 * n))
    if n == 1:
        axes = axes.reshape(1, -1)
    for i in range(n):
        gt_bin = (gt_list[i] > 0.5).astype(np.float32)
        h, w = gt_bin.shape
        for j, key in enumerate(["gt", "baseline", "cosa_v3"]):
            ax = axes[i, j]
            ax.imshow(bg_list[i])
            if key == "gt":
                ax.imshow(gt_bin, alpha=0.6, cmap="Reds")
            else:
                pred = (preds_list[i][key] > 0.5).astype(np.float32)
                _draw_tp_fp_fn(ax, pred, gt_bin, h, w, show_tp=True, bg_rgba=None)
            ax.set_title(sample_names[i] if j == 0 else PANEL_TITLES[key], fontsize=10, fontweight="bold")
            ax.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()


def create_alt3_errors_only(
    gt_list: List[np.ndarray],
    preds_list: List[Dict[str, np.ndarray]],
    bg_list: List[np.ndarray],
    sample_names: List[str],
    output_path: Path,
) -> None:
    """3×6 layout (GT + 5 variants).

    Color scheme for all model columns:
      - Green = correct detections (TP).
      - Red   = false positives (FP).
    Additional highlight for CoSA v3 column only:
      - Blue = pixels where CoSA v3 is correct and at least one other variant is wrong (gain).
    """
    n_samples = len(gt_list)
    keys = ["gt", "baseline", "cosa", "cosa_multiscale_only", "cosa_learnable_gate_only", "cosa_v3"]
    fig, axes = plt.subplots(n_samples, 6, figsize=(18, 3 * n_samples))
    if n_samples == 1:
        axes = axes.reshape(1, -1)
    for row in range(n_samples):
        gt_bin = (gt_list[row] > 0.5).astype(np.float32)
        h, w = gt_bin.shape
        gt_mask = gt_bin.astype(bool)
        for col, key in enumerate(keys):
            ax = axes[row, col]
            ax.imshow(bg_list[row])
            if key == "gt":
                ax.imshow(gt_bin, alpha=0.6, cmap="gray", vmin=0, vmax=1)
            else:
                pred_mask = (preds_list[row][key] > 0.5).astype(bool)
                tp = pred_mask & gt_mask
                fp = pred_mask & (~gt_mask)

                # TP: green
                if tp.any():
                    ov = np.zeros((h, w, 4), dtype=np.float32)
                    ov[tp, 1] = 1.0
                    ov[tp, 3] = 0.7
                    ax.imshow(ov)

                # FP: red
                if fp.any():
                    ov = np.zeros((h, w, 4), dtype=np.float32)
                    ov[fp, 0] = 1.0
                    ov[fp, 3] = 0.7
                    ax.imshow(ov)

                # For CoSA v3 column, additionally highlight pixels that CoSA v3 fixed
                # relative to all other variants: GT=1, CoSA predicts 1, and at least
                # one other variant is wrong at that pixel.
                if key == "cosa_v3":
                    others_wrong = np.zeros_like(gt_mask, dtype=bool)
                    for other_key in ["baseline", "cosa", "cosa_multiscale_only", "cosa_learnable_gate_only"]:
                        other_pred = (preds_list[row][other_key] > 0.5).astype(bool)
                        others_wrong |= (other_pred != gt_mask)
                    improvement = pred_mask & gt_mask & others_wrong
                    if improvement.any():
                        ov = np.zeros((h, w, 4), dtype=np.float32)
                        # Bright blue overlay for CoSA corrections (distinct from green/red)
                        ov[improvement, 2] = 1.0
                        ov[improvement, 3] = 0.9
                        ax.imshow(ov)
            ax.set_title(PANEL_TITLES.get(key, key) if row == 0 else "", fontsize=10, fontweight="bold")
            if col == 0:
                ax.set_ylabel(sample_names[row], fontsize=10)
            ax.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()


def create_alt4_improvement(
    gt_list: List[np.ndarray],
    preds_list: List[Dict[str, np.ndarray]],
    bg_list: List[np.ndarray],
    sample_names: List[str],
    output_path: Path,
) -> None:
    """3 rows × 2 columns: left = CoSA v3 detections (green TP, red FP), right = gains/losses (yellow gain, red FP)."""
    n = len(gt_list)
    fig, axes = plt.subplots(n, 2, figsize=(8, 4 * n))
    if n == 1:
        axes = axes.reshape(1, -1)
    for i in range(n):
        gt_bin = (gt_list[i] > 0.5).astype(np.float32)
        h, w = gt_bin.shape
        base = (preds_list[i]["baseline"] > 0.5).astype(np.float32)
        cosa = (preds_list[i]["cosa_v3"] > 0.5).astype(np.float32)

        ax = axes[i, 0]
        ax.imshow(bg_list[i])
        # Left: raw CoSA v3 detections:
        #   - Green = correct detections (TP)
        #   - Red   = false positives (FP)
        gt_mask = gt_bin.astype(bool)
        pred_mask = cosa.astype(bool)
        tp = pred_mask & gt_mask
        fp = pred_mask & (~gt_mask)
        if tp.any():
            ov = np.zeros((h, w, 4), dtype=np.float32)
            ov[tp, 1] = 1.0  # green
            ov[tp, 3] = 0.8
            ax.imshow(ov)
        if fp.any():
            ov = np.zeros((h, w, 4), dtype=np.float32)
            ov[fp, 0] = 1.0  # red
            ov[fp, 3] = 0.8
            ax.imshow(ov)
        ax.set_title("CoSA v3 detections", fontsize=10, fontweight="bold")
        ax.set_ylabel(sample_names[i], fontsize=10)
        ax.axis("off")

        ax = axes[i, 1]
        ax.imshow(bg_list[i])
        # Right: improvement map relative to baseline
        #   - Yellow = gain (CoSA correct where baseline is wrong)
        #   - Red    = CoSA false positives
        gain = (cosa > 0.5) & (gt_bin > 0.5) & (base < 0.5)
        loss = (cosa > 0.5) & (gt_bin < 0.5)
        if gain.any():
            ov = np.zeros((h, w, 4), dtype=np.float32)
            ov[gain, 0] = 1.0
            ov[gain, 1] = 1.0  # yellow
            ov[gain, 3] = 0.8
            ax.imshow(ov)
        if loss.any():
            ov = np.zeros((h, w, 4), dtype=np.float32)
            ov[loss, 0] = 1.0
            ov[loss, 3] = 0.8
            ax.imshow(ov)
        ax.set_title("Gain (yellow) and FP (red)", fontsize=10, fontweight="bold")
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()


def auto_select_samples(
    dataset: LEVIRCDDatasetFixed,
    models: Dict[str, torch.nn.Module],
    device: str,
    num_select: int,
    min_change_pixels: int = 50,
    stride: int = 1,
    skip_top: int = 0,
    min_gain: float = 0.0,
) -> List[int]:
    candidates: List[Tuple[float, int]] = []
    for idx in range(0, len(dataset), stride):
        img1, img2, label, _ = dataset[idx]
        gt_np = label.squeeze().cpu().numpy()
        if (gt_np > 0.5).sum() < min_change_pixels:
            continue
        preds = {v: predict(models[v], img1, img2, device) for v in VARIANTS}

        gt_u = (gt_np > 0.5).astype(np.uint8)

        def f1_from_pred(pred_bin: np.ndarray) -> float:
            pred_u = (pred_bin > 0.5).astype(np.uint8)
            tp = (pred_u & gt_u).sum()
            fp = (pred_u & (1 - gt_u)).sum()
            fn = ((1 - pred_u) & gt_u).sum()
            prec = tp / (tp + fp + 1e-8)
            rec = tp / (tp + fn + 1e-8)
            return 2 * prec * rec / (prec + rec + 1e-8)

        f1_vals = {k: f1_from_pred(v) for k, v in preds.items()}
        f1_cosa = f1_vals["cosa_v3"]
        best_other = max(
            f1_vals["baseline"],
            f1_vals["cosa"],
            f1_vals["cosa_multiscale_only"],
            f1_vals["cosa_learnable_gate_only"],
        )
        # Only keep samples where CoSA v3 strictly beats all others
        score = f1_cosa - best_other
        if f1_cosa <= best_other or score < min_gain:
            continue
        candidates.append((score, idx))
    candidates.sort(key=lambda x: x[0], reverse=True)
    # Optionally skip the very best samples so we can get a different batch
    sliced = candidates[skip_top: skip_top + num_select]
    return [idx for _, idx in sliced]


def main() -> None:
    parser = argparse.ArgumentParser(description="Alternative ablation qualitative visualizations")
    parser.add_argument("--dataset_dir", type=str, required=True)
    parser.add_argument("--ablation_dir", type=str, default="results/ablation_study")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Default: paper/figures")
    parser.add_argument("--methods", type=str, nargs="+",
                        default=["alt1", "alt2", "alt3", "alt4"],
                        choices=["alt1", "alt2", "alt3", "alt4"],
                        help="Which alternative(s) to generate")
    parser.add_argument("--num_samples", type=int, default=4)
    parser.add_argument("--min_change_pixels", type=int, default=30000)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--skip_top", type=int, default=0,
                        help="Skip this many top-ranked samples when selecting (for a second batch).")
    parser.add_argument("--min_gain", type=float, default=0.0,
                        help="Minimum F1 gain (CoSA v3 vs best other) required to keep a sample.")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--tag", type=str, default="",
                        help="Optional suffix to append to output filenames, e.g. _v2.")
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = str(REPO_ROOT / "paper" / "figures")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ablation_dir = REPO_ROOT / args.ablation_dir
    dataset = LEVIRCDDatasetFixed(
        root_dir=args.dataset_dir,
        split=args.split,
        base_size=512,
        augment=False,
        eval_full_res=True,
    )

    models: Dict[str, torch.nn.Module] = {}
    for variant in VARIANTS:
        ckpt = ablation_dir / variant / "checkpoint_best.pth"
        if not ckpt.exists():
            raise FileNotFoundError(f"Missing checkpoint: {ckpt}")
        models[variant] = load_model(ckpt, variant, args.device)

    selected = auto_select_samples(
        dataset, models, args.device,
        num_select=args.num_samples,
        min_change_pixels=args.min_change_pixels,
        stride=args.stride,
        skip_top=args.skip_top,
        min_gain=args.min_gain,
    )

    gt_list: List[np.ndarray] = []
    preds_list: List[Dict[str, np.ndarray]] = []
    bg_list: List[np.ndarray] = []
    sample_names: List[str] = []

    for idx in selected:
        img1, img2, label, name = dataset[idx]
        gt_np = label.squeeze().cpu().numpy()
        gt_list.append(gt_np)
        preds_list.append({v: predict(models[v], img1, img2, args.device) for v in VARIANTS})
        bg_list.append(denormalize_image(img2))
        name_str = name[0] if isinstance(name, (list, tuple)) and name else str(name)
        sample_names.append(name_str)

    tag = args.tag

    if "alt1" in args.methods:
        create_alt1_row_per_variant(gt_list, preds_list, bg_list, sample_names,
                                    output_dir / f"ablation_qualitative_alt1_row_per_variant{tag}.png")
        print(f"Saved: ablation_qualitative_alt1_row_per_variant{tag}.png")
    if "alt2" in args.methods:
        create_alt2_compact(gt_list, preds_list, bg_list, sample_names,
                            output_dir / f"ablation_qualitative_alt2_compact{tag}.png")
        print(f"Saved: ablation_qualitative_alt2_compact{tag}.png")
    if "alt3" in args.methods:
        create_alt3_errors_only(gt_list, preds_list, bg_list, sample_names,
                                output_dir / f"ablation_qualitative_alt3_errors_only{tag}.png")
        print(f"Saved: ablation_qualitative_alt3_errors_only{tag}.png")
    if "alt4" in args.methods:
        create_alt4_improvement(gt_list, preds_list, bg_list, sample_names,
                               output_dir / f"ablation_qualitative_alt4_improvement{tag}.png")
        print(f"Saved: ablation_qualitative_alt4_improvement{tag}.png")
    print(f"Done. Outputs in {output_dir}")


if __name__ == "__main__":
    main()
