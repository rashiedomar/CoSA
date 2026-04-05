#!/usr/bin/env python3
"""
Generate 3 High-Impact Quantitative Tables for:
  - FC-Siam-diff vs FC-Siam-diff+CoSA (baseline_bs8 vs cosa_v3)
  - STANet vs STANet+CoSA
  - BIT vs BIT+CoSA

Use --pair fc_siam | stanet | bit to run one, or --pair all to run all three
and write the report (full tables for BIT+CoSA as "Star Example", 1–2 sentence
summaries for STANet and FC-Siam).
"""

from pathlib import Path
import sys
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader

# Repo root (parent of change_detection) for "from change_detection. ..." imports
REPO_ROOT = Path(__file__).resolve().parents[2]
# Change-detection project root for paths (dataset_dir, output_dir, checkpoints)
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from change_detection.scripts.quantitative_tables_common import (
    generate_table_1,
    generate_table_2,
    generate_table_3,
    one_line_summary,
)


def _metrics_from_binary(pred, gt):
    """pred, gt: numpy [H,W] or [1,H,W], values 0 or 1."""
    p = np.asarray(pred).flatten().astype(np.uint8)
    g = np.asarray(gt).flatten().astype(np.uint8)
    tp = int(((p == 1) & (g == 1)).sum())
    fp = int(((p == 1) & (g == 0)).sum())
    fn = int(((p == 0) & (g == 1)).sum())
    tn = int(((p == 0) & (g == 0)).sum())
    
    # Handle empty ground truth (no-change images)
    gt_sum = g.sum()
    is_no_change = (gt_sum == 0)
    
    if is_no_change:
        # For no-change images: F1 is undefined, use accuracy-based metric
        # Perfect prediction (no FPs) = 1.0, any FP = 0.0
        accuracy = 1.0 if fp == 0 else 0.0
        # Use accuracy as proxy for F1 (1.0 if perfect, 0.0 if any false positive)
        f1 = accuracy
        precision = 1.0 if (tp + fp) == 0 else tp / (tp + fp + 1e-8)
        recall = 1.0  # No missed changes if GT is empty
        iou = 1.0 if fp == 0 else 0.0
    else:
        # Normal case: GT has changes, F1 is well-defined
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        iou = tp / (tp + fp + fn + 1e-8)
    
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "f1": float(f1), "precision": float(precision), "recall": float(recall), "iou": float(iou),
        "is_no_change": bool(is_no_change),  # Flag for filtering
    }


# -----------------------------------------------------------------------------
# FC-Siam-diff vs FC-Siam-diff+CoSA
# ------------------------------------------------------------------------------

def run_fc_siam(device, dataset_dir, baseline_ckpt, cosa_ckpt):
    from change_detection.models.siamese_unet import SiameseUNet, SiameseUNetCoSA
    from change_detection.scripts.data.levir_dataset_fixed import LEVIRCDDatasetFixed
    from change_detection.scripts.metrics import f1_change_class, iou_change_class

    def load_model(ckpt_path, variant):
        if variant == "baseline":
            m = SiameseUNet(in_channels=3, n_classes=1, base_channels=64, fusion="diff")
        else:
            m = SiameseUNetCoSA(
                in_channels=3, n_classes=1, base_channels=64, fusion="diff",
                topk=32, use_multiscale=True, use_learnable_gate=True,
            )
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        sd = ckpt.get("model_state_dict") or ckpt.get("state_dict") or ckpt
        m.load_state_dict(sd)
        return m.to(device).eval()

    def compute_metrics_from_logits(logits, gt, threshold=0.5):
        if isinstance(logits, np.ndarray):
            logits = torch.from_numpy(logits).float()
        if isinstance(gt, np.ndarray):
            gt = torch.from_numpy(gt).float()
        if logits.dim() == 2:
            logits = logits.unsqueeze(0).unsqueeze(0)
        elif logits.dim() == 3:
            logits = logits.unsqueeze(0)
        if gt.dim() == 2:
            gt = gt.unsqueeze(0).unsqueeze(0)
        elif gt.dim() == 3:
            gt = gt.unsqueeze(0)
        f1, precision, recall = f1_change_class(logits, gt, threshold=threshold)
        iou = iou_change_class(logits, gt, threshold=threshold)
        pred = (torch.sigmoid(logits) > threshold).to(torch.uint8)
        gt_binary = (gt > 0).to(torch.uint8)
        tp = ((pred == 1) & (gt_binary == 1)).sum().item()
        fp = ((pred == 1) & (gt_binary == 0)).sum().item()
        fn = ((pred == 0) & (gt_binary == 1)).sum().item()
        tn = ((pred == 0) & (gt_binary == 0)).sum().item()
        return {"f1": f1, "precision": precision, "recall": recall, "iou": iou, "tp": tp, "fp": fp, "fn": fn, "tn": tn}

    ds = LEVIRCDDatasetFixed(
        root_dir=str(dataset_dir),
        split="test",
        base_size=512,
        augment=False,
        eval_full_res=True,
    )
    dl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)
    baseline = load_model(baseline_ckpt, "baseline")
    cosa = load_model(cosa_ckpt, "cosa_v3")
    all_samples = []
    for idx, (img1, img2, label, name) in enumerate(dl):
        img1 = img1.squeeze(0).to(device)
        img2 = img2.squeeze(0).to(device)
        lab = label.squeeze().numpy()
        lab_bin = (lab > 0.5).astype(np.float32)
        lab_t = torch.from_numpy(lab_bin).float().unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            lb = baseline(img1.unsqueeze(0), img2.unsqueeze(0))
            lc, _ = cosa.forward_with_attention(img1.unsqueeze(0), img2.unsqueeze(0))
        m_b = compute_metrics_from_logits(lb.cpu(), lab_t, threshold=0.5)
        m_c = compute_metrics_from_logits(lc.cpu(), lab_t, threshold=0.5)
        imp = m_c["f1"] - m_b["f1"]
        # Check if no-change image (empty GT)
        gt_sum = lab_bin.sum()
        is_no_change = (gt_sum == 0)
        all_samples.append({
            "name": name[0] if isinstance(name, (list, tuple)) else name,
            "baseline_f1": m_b["f1"],
            "cosa_f1": m_c["f1"],
            "improvement": imp,
            "baseline_metrics": m_b,
            "cosa_metrics": m_c,
            "is_no_change": is_no_change,
        })
        if (idx + 1) % 50 == 0:
            print(f"  FC-Siam: {idx+1}/{len(ds)}")
    return all_samples


# -----------------------------------------------------------------------------
# STANet vs STANet+CoSA
# ------------------------------------------------------------------------------

def run_stanet(device, dataroot_256, stanet_ckpt_dir, cosa_ckpt_path):
    import copy
    import os

    stanet_dir = ROOT / "checkpoints_official" / "STANet_LEVIR" / "STANet-master"
    if str(stanet_dir) not in sys.path:
        sys.path.insert(0, str(stanet_dir))

    # Avoid polluting argparse; use minimal argv for TrainOptions
    argv_save = sys.argv
    sys.argv = [
        "",
        "--model", "CDFA",
        "--dataroot", str(dataroot_256),
        "--phase", "val",
        "--dataset_mode", "changedetection",
        "--batch_size", "1",
        "--num_threads", "0",
        "--serial_batches",
        "--preprocess", "none",
        "--angle", "0",
        "--no_flip", "True",
        "--load_size", "256",
        "--crop_size", "256",
        "--max_dataset_size", "1000000",
    ]
    try:
        from options.train_options import TrainOptions
        from data import create_dataset
        from models import create_model
        from models.cosa_block import CoSABlock
        import torch.nn as nn
        import torch.nn.functional as F

        opt = TrainOptions().parse()
        opt.model = "CDFA"
        opt.isTrain = False
        opt.istest = False
    finally:
        sys.argv = argv_save

    # Build STANet baseline model (CDFA has netF + netA)
    opt_net = copy.deepcopy(opt)
    opt_net.gpu_ids = "0" if torch.cuda.is_available() else ""
    opt_net.checkpoints_dir = str(stanet_dir / "checkpoints")
    opt_net.name = "stanet_official_paper"
    opt_net.continue_train = False
    opt_net.epoch = "90_F1_1_0.80165"
    stanet_model = create_model(opt_net)
    stanet_model.save_dir = str(Path(stanet_ckpt_dir))
    stanet_model.setup(opt_net)
    try:
        stanet_model.load_networks("90_F1_1_0.80165")
    except Exception:
        stanet_model.load_networks("latest")
    stanet_model.eval()

    # STANet+CoSA wrapper (mirrors test_cosa_model.py)
    class STANetWithCoSA(nn.Module):
        def __init__(self, stanet, scales=(8, 16)):
            super().__init__()
            self.stanet = stanet
            for p in self.stanet.netF.parameters():
                p.requires_grad = False
            for p in self.stanet.netA.parameters():
                p.requires_grad = False
            self.stanet.netF.eval()
            self.stanet.netA.eval()
            self.cosa = CoSABlock(in_channels=64, scales=list(scales))
            self.pred_conv = nn.Sequential(
                nn.Conv2d(64, 32, 3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 1, 1),
            )

        def forward(self, A, B):
            with torch.no_grad():
                fa = self.stanet.netF(A)
                fb = self.stanet.netF(B)
                fa, fb = self.stanet.netA(fa, fb)
            diff = self.cosa(fa, fb)
            if diff.shape[2:] != A.shape[2:]:
                diff = F.interpolate(diff, size=A.shape[2:], mode="bilinear", align_corners=False)
            return self.pred_conv(diff), diff

    cosa_model = STANetWithCoSA(stanet_model, scales=[8, 16]).to(device)
    ckpt = torch.load(cosa_ckpt_path, map_location=device, weights_only=False)
    cosa_model.load_state_dict(ckpt["model_state_dict"])
    cosa_model.eval()

    # Dataset
    test_opt = copy.deepcopy(opt_net)
    test_opt.dataroot = str(dataroot_256)
    test_opt.phase = "val"
    test_opt.isTrain = False
    test_dataset = create_dataset(test_opt)
    all_samples = []
    n = len(test_dataset.dataset) if hasattr(test_dataset, "dataset") else len(test_dataset)
    it = iter(test_dataset)
    idx = 0
    while True:
        try:
            data = next(it)
        except StopIteration:
            break
        A = data["A"].to(device)
        B = data["B"].to(device)
        L = data["L"]
        if isinstance(L, torch.Tensor):
            L = L.numpy()
        else:
            L = np.asarray(L)
        if L.ndim == 3:
            L = L[0]
        L_bin = (L > 0.5).astype(np.uint8)
        name = data.get("A_paths", data.get("A_path", f"sample_{idx}"))
        if isinstance(name, (list, tuple)):
            name = name[0] if name else f"sample_{idx}"
        name = Path(name).name if hasattr(name, "__fspath__") else os.path.basename(str(name))

        with torch.no_grad():
            stanet_model.set_input({"A": A, "B": B, "L": data["L"], "A_paths": data.get("A_paths", data.get("A_path", name))})
            stanet_model.forward()
            pred_b = stanet_model.pred_L.long().cpu().numpy()
            pred_c, _ = cosa_model(A, B)
            pred_c = (torch.sigmoid(pred_c) > 0.5).long().cpu().numpy()
        if pred_b.ndim == 4:
            pred_b = pred_b.squeeze(1)
        if pred_c.ndim == 4:
            pred_c = pred_c.squeeze(1)
        if pred_b.ndim == 3:
            pred_b = pred_b[0]
        if pred_c.ndim == 3:
            pred_c = pred_c[0]
        m_b = _metrics_from_binary(pred_b, L_bin)
        m_c = _metrics_from_binary(pred_c, L_bin)
        imp = m_c["f1"] - m_b["f1"]
        # Check if no-change image (empty GT)
        gt_sum = L_bin.sum()
        is_no_change = (gt_sum == 0)
        all_samples.append({
            "name": name,
            "baseline_f1": m_b["f1"],
            "cosa_f1": m_c["f1"],
            "improvement": imp,
            "baseline_metrics": m_b,
            "cosa_metrics": m_c,
            "is_no_change": is_no_change,
        })
        idx += 1
        if idx % 100 == 0:
            print(f"  STANet: {idx} samples...")
    return all_samples


# -----------------------------------------------------------------------------
# BIT vs BIT+CoSA
# ------------------------------------------------------------------------------

def run_bit(device, dataroot_256, bit_ckpt_path, bit_cosa_ckpt_path):
    """Use LEVIR-CD_combined_256/test, 256x256, mean/std 0.5. BIT outputs [B,2,H,W]."""
    import os
    import torchvision.transforms.functional as TF
    from PIL import Image

    # Ensure change_detection is first so bit_cosa_wrapper's "from models.siamese_unet" resolves
    root_s = str(ROOT)
    while root_s in sys.path:
        sys.path.remove(root_s)
    sys.path.insert(0, root_s)
    from change_detection.models.bit_cosa_wrapper import load_bit_with_cosa

    BIT_DIR = ROOT / "checkpoints_official" / "BIT_LEVIR" / "BIT_CD-master"
    if str(BIT_DIR) not in sys.path:
        sys.path.insert(0, str(BIT_DIR))

    class LEVIRCDTestDataset(torch.utils.data.Dataset):
        def __init__(self, root, size=256):
            self.root = Path(root)
            a_dir = self.root / "A"
            self.names = sorted([f.name for f in a_dir.glob("*.png")])
            valid = []
            for n in self.names:
                if (self.root / "B" / n).exists() and (self.root / "label" / n).exists():
                    valid.append(n)
            self.names = valid
            self.size = size

        def __len__(self):
            return len(self.names)

        def __getitem__(self, i):
            n = self.names[i]
            a = Image.open(self.root / "A" / n).convert("RGB")
            b = Image.open(self.root / "B" / n).convert("RGB")
            l = Image.open(self.root / "label" / n).convert("L")
            if a.size != (self.size, self.size):
                a = TF.resize(a, [self.size, self.size], interpolation=Image.BICUBIC)
                b = TF.resize(b, [self.size, self.size], interpolation=Image.BICUBIC)
                l = TF.resize(l, [self.size, self.size], interpolation=Image.NEAREST)
            a = TF.to_tensor(a)
            b = TF.to_tensor(b)
            a = TF.normalize(a, [0.5] * 3, [0.5] * 3)
            b = TF.normalize(b, [0.5] * 3, [0.5] * 3)
            lab = torch.from_numpy(np.array(l, dtype=np.uint8))
            lab = (lab > 127).long()
            return a, b, lab, n

    cosa_model = load_bit_with_cosa(str(bit_ckpt_path), device, use_cosa=True, cosa_topk=32, gamma_init=0.0)
    ckpt_cosa = torch.load(bit_cosa_ckpt_path, map_location=device, weights_only=False)
    cosa_model.load_state_dict(ckpt_cosa["model_state_dict"], strict=True)
    cosa_model.eval()

    # Plain BIT for baseline (same checkpoint, no CoSA)
    bit_model = load_bit_with_cosa(str(bit_ckpt_path), device, use_cosa=False)
    bit_model.eval()

    ds = LEVIRCDTestDataset(dataroot_256, 256)
    dl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)
    all_samples = []
    for idx, (a, b, lab, name) in enumerate(dl):
        a = a.to(device)
        b = b.to(device)
        lab_np = lab.squeeze().numpy()
        lab_bin = (lab_np > 0).astype(np.uint8)
        with torch.no_grad():
            out_b = bit_model(a, b)
            out_c = cosa_model(a, b)
        if out_b.shape[1] == 2:
            pred_b = out_b.argmax(dim=1).cpu().numpy()
        else:
            pred_b = (torch.sigmoid(out_b) > 0.5).long().squeeze(1).cpu().numpy()
        if out_c.shape[1] == 2:
            pred_c = out_c.argmax(dim=1).cpu().numpy()
        else:
            pred_c = (torch.sigmoid(out_c) > 0.5).long().squeeze(1).cpu().numpy()
        if pred_b.ndim == 3:
            pred_b = pred_b[0]
        if pred_c.ndim == 3:
            pred_c = pred_c[0]
        # Check if this is a no-change image (empty GT) before computing metrics
        gt_sum = int(lab_bin.sum())
        is_no_change = (gt_sum == 0)
        m_b = _metrics_from_binary(pred_b, lab_bin)
        m_c = _metrics_from_binary(pred_c, lab_bin)
        imp = m_c["f1"] - m_b["f1"]
        nm = name[0] if isinstance(name, (list, tuple)) else name
        all_samples.append({
            "name": nm,
            "baseline_f1": m_b["f1"],
            "cosa_f1": m_c["f1"],
            "improvement": imp,
            "baseline_metrics": m_b,
            "cosa_metrics": m_c,
            "is_no_change": is_no_change,  # Track no-change images
        })
        if (idx + 1) % 100 == 0:
            print(f"  BIT: {idx+1}/{len(ds)}")
    return all_samples


# -----------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Quantitative tables for FC-Siam, STANet, BIT vs +CoSA")
    ap.add_argument("--pair", type=str, choices=["fc_siam", "stanet", "bit", "all"], default="all",
                    help="Which model pair(s) to run")
    ap.add_argument("--dataset_dir", type=str, default="datasets/to_check_dataset/LEVIR-CD_combined",
                    help="FC-Siam: LEVIR-CD_combined root")
    ap.add_argument("--dataset_256", type=str, default="datasets/to_check_dataset/LEVIR-CD_combined_256/test",
                    help="STANet/BIT: LEVIR-CD_256 test root")
    ap.add_argument("--baseline_bs8", type=str, default="results/baseline_bs8/checkpoint_best.pth")
    ap.add_argument("--cosa_v3", type=str, default="results/cosa_v3_residual_multiscale/checkpoint_best.pth")
    ap.add_argument("--stanet_ckpt_dir", type=str,
                    default="checkpoints_official/STANet_LEVIR/STANet-master/checkpoints/stanet_official_paper")
    ap.add_argument("--stanet_cosa_ckpt", type=str, default="results/stanet_cosa_finetune/best_checkpoint.pth")
    ap.add_argument("--bit_ckpt", type=str,
                    default="checkpoints_official/BIT_LEVIR/BIT_CD-master/checkpoints/BIT_LEVIR/best_ckpt.pt")
    ap.add_argument("--bit_cosa_ckpt", type=str, default="results/bit_cosa_finetune/best_checkpoint.pth")
    ap.add_argument("--output_dir", type=str, default="results/baseline_cosa_visualizations")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    device = torch.device(args.device)
    out_dir = ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset_dir = ROOT / args.dataset_dir
    dataset_256 = ROOT / args.dataset_256
    baseline_bs8 = ROOT / args.baseline_bs8
    cosa_v3 = ROOT / args.cosa_v3
    stanet_ckpt_dir = ROOT / args.stanet_ckpt_dir
    stanet_cosa_ckpt = ROOT / args.stanet_cosa_ckpt
    bit_ckpt = ROOT / args.bit_ckpt
    bit_cosa_ckpt = ROOT / args.bit_cosa_ckpt

    pairs = ["fc_siam", "stanet", "bit"] if args.pair == "all" else [args.pair]
    results = {}

    for pair in pairs:
        print("\n" + "=" * 70)
        print(f"Quantitative analysis: {pair.upper()}")
        print("=" * 70)
        all_samples = None
        if pair == "fc_siam":
            if not baseline_bs8.exists() or not cosa_v3.exists():
                print(f"  Skip: missing {baseline_bs8} or {cosa_v3}")
                continue
            all_samples = run_fc_siam(device, dataset_dir, baseline_bs8, cosa_v3)
        elif pair == "stanet":
            if not stanet_ckpt_dir.exists() or not stanet_cosa_ckpt.exists():
                print(f"  Skip: missing STANet checkpoints")
                continue
            all_samples = run_stanet(device, dataset_256, stanet_ckpt_dir, stanet_cosa_ckpt)
        elif pair == "bit":
            if not bit_ckpt.exists() or not bit_cosa_ckpt.exists():
                print(f"  Skip: missing BIT checkpoints")
                continue
            # Remove STANet dir from path so "models" resolves to change_detection.models in run_bit
            stanet_dir = str(ROOT / "checkpoints_official" / "STANet_LEVIR" / "STANet-master")
            while stanet_dir in sys.path:
                sys.path.remove(stanet_dir)
            all_samples = run_bit(device, dataset_256, bit_ckpt, bit_cosa_ckpt)
        if all_samples is None:
            continue
        results[pair] = all_samples
        name = {"fc_siam": "FC-Siam-diff vs FC-Siam-diff+CoSA",
                "stanet": "STANet vs STANet+CoSA",
                "bit": "BIT vs BIT+CoSA"}[pair]
        t1 = generate_table_1(all_samples, name)
        t2 = generate_table_2(all_samples, name)
        t3 = generate_table_3(all_samples, name)
        out_file = out_dir / f"quantitative_analysis_tables_{pair}.txt"
        with open(out_file, "w") as f:
            f.write(f"QUANTITATIVE ANALYSIS: {name}\n")
            f.write("=" * 70 + "\n\n")
            f.write(t1)
            f.write("\n\n")
            f.write(t2)
            f.write("\n\n")
            f.write(t3)
            f.write("\n\n")
            f.write("=" * 70 + "\n")
            f.write(f"Total samples: {len(all_samples)}\n")
            f.write("=" * 70 + "\n")
        print(f"  Saved: {out_file}")

    if args.pair == "all" and results:
        report_path = out_dir / "quantitative_analysis_report.txt"
        with open(report_path, "w") as f:
            f.write("QUANTITATIVE ANALYSIS REPORT: CoSA Across FC-Siam, STANet, BIT\n")
            f.write("=" * 70 + "\n\n")
            f.write("Star Example: BIT vs BIT+CoSA (full 3 tables)\n")
            f.write("-" * 70 + "\n\n")
            if "bit" in results:
                name = "BIT vs BIT+CoSA"
                t1 = generate_table_1(results["bit"], name)
                t2 = generate_table_2(results["bit"], name)
                t3 = generate_table_3(results["bit"], name)
                f.write(t1)
                f.write("\n\n")
                f.write(t2)
                f.write("\n\n")
                f.write(t3)
                f.write("\n\n")
            f.write("Summaries for other models\n")
            f.write("-" * 70 + "\n\n")
            if "stanet" in results:
                f.write(one_line_summary(results["stanet"], "STANet") + "\n\n")
            if "fc_siam" in results:
                f.write(one_line_summary(results["fc_siam"], "FC-Siam-diff") + "\n\n")
            f.write("=" * 70 + "\n")
        print(f"\n  Report saved: {report_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
