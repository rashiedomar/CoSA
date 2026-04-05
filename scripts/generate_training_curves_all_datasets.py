#!/usr/bin/env python3
"""
Generate Training Curves Collage: All 4 Datasets

Creates a long vertical figure (4 rows x 3 columns) with training curves for:
- LEVIR-CD, S2Looking, DSIFN, CLCD

Each row: Training Loss | Validation F1-Score | Validation IoU
Style: Baseline (blue solid), CoSA (orange dashed)
X-axis: 0, 10, 20, 30, ..., 100
"""

import re
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams['font.size'] = 11
rcParams['font.family'] = 'serif'
rcParams['axes.labelsize'] = 12
rcParams['axes.titlesize'] = 13
rcParams['xtick.labelsize'] = 10
rcParams['ytick.labelsize'] = 10
rcParams['legend.fontsize'] = 10
rcParams['figure.dpi'] = 300
rcParams['savefig.dpi'] = 300
rcParams['savefig.bbox'] = 'tight'

BASELINE_COLOR = '#4A90E2'
COSA_COLOR = '#F5A623'
XTICKS = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]


def parse_training_log(log_path):
    """Parse training log and extract metrics."""
    if not Path(log_path).exists():
        print(f"Warning: Log file not found: {log_path}")
        return None

    with open(log_path, 'r') as f:
        content = f.read()

    epochs = []
    train_losses = []
    val_f1s = []
    val_ious = []

    pattern1 = r'Epoch (\d+)/\d+ - Train Loss: ([\d.]+), Val F1: ([\d.]+)%, IoU: ([\d.]+)%'
    pattern2 = r'Epoch (\d+)/\d+ - Train Loss: ([\d.]+), Val Precision: ([\d.]+)%, Recall: ([\d.]+)%, F1: ([\d.]+)%, IoU: ([\d.]+)%'

    matches1 = re.findall(pattern1, content)
    matches2 = re.findall(pattern2, content)

    if matches2:
        for m in matches2:
            epochs.append(int(m[0]))
            train_losses.append(float(m[1]))
            val_f1s.append(float(m[4]))
            val_ious.append(float(m[5]))
    elif matches1:
        for m in matches1:
            epochs.append(int(m[0]))
            train_losses.append(float(m[1]))
            val_f1s.append(float(m[2]))
            val_ious.append(float(m[3]))

    return {
        'epochs': epochs,
        'train_losses': train_losses,
        'val_f1s': val_f1s,
        'val_ious': val_ious,
    }


def plot_single_row(axes, baseline_data, cosa_data, dataset_name):
    """Plot one row: Loss, F1, IoU for a dataset."""
    bl = baseline_data
    co = cosa_data

    if bl is None or co is None:
        for ax in axes:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
        return

    # Loss
    ax1 = axes[0]
    ax1.plot(bl['epochs'], bl['train_losses'],
             color=BASELINE_COLOR, linestyle='-', marker='o',
             label='Baseline', linewidth=2, markersize=5, alpha=0.8, markevery=10)
    ax1.plot(co['epochs'], co['train_losses'],
             color=COSA_COLOR, linestyle='--', marker='s',
             label='CoSA', linewidth=2, markersize=5, alpha=0.8, markevery=10)
    ax1.set_ylabel(f'{dataset_name}\nTraining Loss', fontweight='bold')
    ax1.set_title('Training Loss', fontweight='bold', pad=10)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.legend(loc='upper right', fontsize=9)
    ax1.set_xlim([-2, 102])
    ax1.set_xticks(XTICKS)
    ax1.set_yscale('log')

    # F1
    ax2 = axes[1]
    ax2.plot(bl['epochs'], bl['val_f1s'],
             color=BASELINE_COLOR, linestyle='-', marker='o',
             label='Baseline', linewidth=2, markersize=5, alpha=0.8, markevery=10)
    ax2.plot(co['epochs'], co['val_f1s'],
             color=COSA_COLOR, linestyle='--', marker='s',
             label='CoSA', linewidth=2, markersize=5, alpha=0.8, markevery=10)
    ax2.set_ylabel('F1-Score (%)', fontweight='bold')
    ax2.set_title('Validation F1-Score', fontweight='bold', pad=10)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.legend(loc='lower right', fontsize=9)
    ax2.set_xlim([-2, 102])
    ax2.set_xticks(XTICKS)
    all_f1 = bl['val_f1s'] + co['val_f1s']
    if all_f1:
        ymin = max(0, min(all_f1) - 5)
        ymax = min(100, max(all_f1) + 5)
        ax2.set_ylim([ymin, ymax])

    # IoU
    ax3 = axes[2]
    ax3.plot(bl['epochs'], bl['val_ious'],
             color=BASELINE_COLOR, linestyle='-', marker='o',
             label='Baseline', linewidth=2, markersize=5, alpha=0.8, markevery=10)
    ax3.plot(co['epochs'], co['val_ious'],
             color=COSA_COLOR, linestyle='--', marker='s',
             label='CoSA', linewidth=2, markersize=5, alpha=0.8, markevery=10)
    ax3.set_ylabel('IoU (%)', fontweight='bold')
    ax3.set_title('Validation IoU', fontweight='bold', pad=10)
    ax3.grid(True, alpha=0.3, linestyle='--')
    ax3.legend(loc='lower right', fontsize=9)
    ax3.set_xlim([-2, 102])
    ax3.set_xticks(XTICKS)
    all_iou = bl['val_ious'] + co['val_ious']
    if all_iou:
        ymin = max(0, min(all_iou) - 5)
        ymax = min(100, max(all_iou) + 5)
        ax3.set_ylim([ymin, ymax])


def main():
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent
    workspace_root = repo_root.parent

    datasets = [
        ('LEVIR-CD', 'ablation_study/baseline', 'ablation_study/cosa'),
        ('S2Looking', 'baseline_s2looking', 'cosa_s2looking'),
        ('DSIFN', 'baseline_dsifn512', 'cosa_dsifn512'),
        ('CLCD', 'baseline_clcd', 'cosa_clcd'),
    ]

    results_dir = workspace_root / 'results'
    data = {}
    for name, bl_rel, co_rel in datasets:
        bl_path = results_dir / bl_rel / 'training.log'
        co_path = results_dir / co_rel / 'training.log'
        bl_data = parse_training_log(bl_path)
        co_data = parse_training_log(co_path)
        data[name] = (bl_data, co_data)
        if bl_data:
            print(f"  {name} Baseline: {len(bl_data['epochs'])} epochs")
        if co_data:
            print(f"  {name} CoSA: {len(co_data['epochs'])} epochs")

    # Create figure: 4 rows x 3 columns
    fig, axes_grid = plt.subplots(4, 3, figsize=(18, 20))

    for i, (name, _, _) in enumerate(datasets):
        bl_data, co_data = data[name]
        plot_single_row(axes_grid[i], bl_data, co_data, name)

    # Shared x-label for bottom row
    for ax in axes_grid[3]:
        ax.set_xlabel('Epoch', fontweight='bold')
    for ax in axes_grid[0]:
        ax.set_xlabel('')
    for ax in axes_grid[1]:
        ax.set_xlabel('')
    for ax in axes_grid[2]:
        ax.set_xlabel('')

    fig.suptitle('Training and Validation Metrics: Baseline vs CoSA (All Datasets)',
                 fontsize=16, fontweight='bold', y=1.01)

    plt.tight_layout()

    output_dir = repo_root / 'figures'
    output_dir.mkdir(parents=True, exist_ok=True)
    out_png = output_dir / 'training_validation_metrics_all_datasets.png'
    out_pdf = output_dir / 'training_validation_metrics_all_datasets.pdf'

    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.savefig(out_pdf, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\n✅ Saved: {out_png}")
    print(f"✅ Saved: {out_pdf}")


if __name__ == '__main__':
    main()
