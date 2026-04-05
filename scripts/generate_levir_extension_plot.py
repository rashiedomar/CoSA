#!/usr/bin/env python3
"""
LEVIR-CD 150-epoch extension plot for Baseline vs CoSA.

Creates a single-panel figure showing training loss for:
  - Baseline (results/ablation_study/baseline_150e_nohup.log)
  - CoSA (results/ablation_study/cosa_v3_150e_nohup.log)

We use the epoch indices from these extension runs (which already include
the earlier epochs) and highlight the +50 epoch extension:
  - Vertical line at epoch 100 (original stopping point).
  - Markers at epoch 100 and at the final epoch for both curves.

Output:
  research_repo/paper/figures/levir_extension_50epochs.png
"""
from __future__ import annotations

from pathlib import Path
import re
import matplotlib.pyplot as plt
from matplotlib import rcParams

# Colors consistent with training_loss_all_datasets.py
BASELINE_COLOR = "#4A90E2"  # Blue
COSA_COLOR = "#F5A623"      # Orange


rcParams["font.size"] = 11
rcParams["font.family"] = "serif"
rcParams["axes.labelsize"] = 12
rcParams["axes.titlesize"] = 13
rcParams["xtick.labelsize"] = 10
rcParams["ytick.labelsize"] = 10
rcParams["legend.fontsize"] = 10


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]   # research_repo
    workspace_root = repo_root.parent                 # change_detection

    baseline_log = workspace_root / "results" / "ablation_study" / "baseline_150e_nohup.log"
    cosa_log = workspace_root / "results" / "ablation_study" / "cosa_v3_150e_nohup.log"

    def parse_training_log(log_path: Path):
        """Parse training.log for 'Epoch X/Y - Train Loss: v' entries."""
        if not log_path.exists():
            print(f"Warning: log not found: {log_path}")
            return None
        content = log_path.read_text()
        seen = set()
        epochs, losses = [], []
        pattern = r"Epoch (\d+)/\d+ - Train Loss: ([\d.]+)"
        for m in re.finditer(pattern, content):
            ep = int(m.group(1))
            if ep in seen:
                continue
            seen.add(ep)
            epochs.append(ep)
            losses.append(float(m.group(2)))
        if not epochs:
            return None
        pairs = sorted(zip(epochs, losses))
        epochs, losses = [p[0] for p in pairs], [p[1] for p in pairs]
        return {"epochs": epochs, "train_losses": losses}

    bl = parse_training_log(baseline_log)
    co = parse_training_log(cosa_log)

    if bl is None or co is None:
        raise SystemExit("Missing training logs for baseline or CoSA; cannot generate plot.")

    # Focus on epochs 50–150
    def slice_range(data, start_ep: int = 50, end_ep: int = 999):
        epochs = data["epochs"]
        losses = data["train_losses"]
        pairs = [(e, l) for e, l in zip(epochs, losses) if start_ep <= e <= end_ep]
        if not pairs:
            return [], []
        ep_s, lo_s = zip(*pairs)
        return list(ep_s), list(lo_s)

    bl_ep, bl_lo = slice_range(bl, 50, 200)
    co_ep, co_lo = slice_range(co, 50, 200)

    fig, ax = plt.subplots(1, 1, figsize=(6, 4))

    if bl_ep:
        ax.plot(bl_ep, bl_lo, color=BASELINE_COLOR, label="Baseline", linewidth=2)
    if co_ep:
        ax.plot(co_ep, co_lo, color=COSA_COLOR, label="CoSA", linewidth=2)

    # Vertical line at epoch 100 (end of original schedule)
    ax.axvline(x=100, color="gray", linestyle="--", linewidth=1.5, alpha=0.8)
    ax.text(100, ax.get_ylim()[0] * 1.1, "100 epochs\n(original)", rotation=90,
            va="bottom", ha="right", fontsize=9, color="gray")

    # Markers at epoch 100 and final epoch for both curves (if present)
    def mark_epoch(data, color, label_prefix: str):
        epochs = data["epochs"]
        losses = data["train_losses"]
        if not epochs:
            return
        # Find values at or closest to epoch 100 and the last epoch
        def closest(target):
            return min(zip(epochs, losses), key=lambda p: abs(p[0] - target))

        e100, l100 = closest(100)
        elast, llast = epochs[-1], losses[-1]
        ax.scatter([e100, elast], [l100, llast], color=color, s=35, zorder=5)

    mark_epoch(bl, BASELINE_COLOR, "Baseline")
    mark_epoch(co, COSA_COLOR, "CoSA")

    ax.set_xlabel("Epoch", fontweight="bold")
    ax.set_ylabel("Training Loss (log-scale)", fontweight="bold")
    ax.set_title("LEVIR-CD: Baseline vs CoSA with 150-epoch Extension", fontweight="bold")
    ax.set_xlim(48, max(bl_ep + co_ep) + 2)
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend(loc="upper right")

    out_dir = repo_root / "paper" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "levir_extension_50epochs.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"✅ Saved: {out_path}")


if __name__ == "__main__":
    main()

