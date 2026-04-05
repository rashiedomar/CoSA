#!/usr/bin/env python3
"""
Generate Training Loss Figure: 4 Datasets Side-by-Side

Creates a single horizontal figure (1 row x 4 columns) with training loss only:
LEVIR-CD | S2Looking | DSIFN | CLCD

Style (paper-style, like LightGCN/NGCF figure):
- 4 subplots connected in parallel
- Solid lines, no markers (for loss)
- Blue = Baseline, Orange = CoSA
- Legend top-right, subtle grey grid
- X-axis: 0, 10, 20, ..., 100
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
rcParams['axes.facecolor'] = 'white'
rcParams['figure.facecolor'] = 'white'

BASELINE_COLOR = '#4A90E2'   # Blue
COSA_COLOR = '#F5A623'       # Orange
XTICKS_10 = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
XTICKS_25 = [0, 25, 50, 75, 100]


def parse_training_log(log_path):
    """Parse training log and extract train loss. Keeps only first occurrence per epoch (avoids duplicates from progress bar output)."""
    if not Path(log_path).exists():
        print(f"Warning: Log file not found: {log_path}")
        return None

    with open(log_path, 'r') as f:
        content = f.read()

    seen_epochs = set()
    epochs = []
    train_losses = []

    pattern = r'Epoch (\d+)/\d+ - Train Loss: ([\d.]+)'
    for m in re.finditer(pattern, content):
        ep = int(m.group(1))
        if ep not in seen_epochs:
            seen_epochs.add(ep)
            epochs.append(ep)
            train_losses.append(float(m.group(2)))

    # Sort by epoch (in case log order is mixed)
    if epochs:
        sorted_pairs = sorted(zip(epochs, train_losses))
        epochs, train_losses = [p[0] for p in sorted_pairs], [p[1] for p in sorted_pairs]

    return {'epochs': epochs, 'train_losses': train_losses}


def plot_and_save(data, datasets, xticks, output_path, title_suffix=''):
    """Create and save training loss figure with given xticks."""
    fig, axes = plt.subplots(1, 4, figsize=(16, 5))

    for i, (name, _, _) in enumerate(datasets):
        ax = axes[i]
        bl_data, co_data = data[name]

        if bl_data is None and co_data is None:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            continue

        if bl_data:
            ax.plot(bl_data['epochs'], bl_data['train_losses'],
                    color=BASELINE_COLOR, linestyle='-', linewidth=2,
                    label='Baseline', alpha=0.9)
        if co_data:
            ax.plot(co_data['epochs'], co_data['train_losses'],
                    color=COSA_COLOR, linestyle='-', linewidth=2,
                    label='CoSA', alpha=0.9)

        ax.set_title(name, fontweight='bold', pad=10)
        ax.set_xlabel('Epoch', fontweight='bold')
        ax.set_ylabel('Training Loss', fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(loc='upper right', fontsize=9)
        ax.set_xlim([-2, 102])
        ax.set_xticks(xticks)
        ax.set_yscale('log')

    fig.suptitle(f'Training Loss: Baseline vs CoSA{title_suffix}',
                 fontsize=14, fontweight='bold', y=1.02)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


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

    output_dir = repo_root / 'figures'
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1) Original: 0, 10, 20, ..., 100
    plot_and_save(data, datasets, XTICKS_10,
                  output_dir / 'training_loss_all_datasets.png')
    plot_and_save(data, datasets, XTICKS_10,
                  output_dir / 'training_loss_all_datasets.pdf')
    print(f"\n✅ Saved: {output_dir / 'training_loss_all_datasets.png'}")
    print(f"✅ Saved: {output_dir / 'training_loss_all_datasets.pdf'}")

    # 2) New: 0, 25, 50, 75, 100
    plot_and_save(data, datasets, XTICKS_25,
                  output_dir / 'training_loss_all_datasets_02550.png')
    plot_and_save(data, datasets, XTICKS_25,
                  output_dir / 'training_loss_all_datasets_02550.pdf')
    print(f"✅ Saved: {output_dir / 'training_loss_all_datasets_02550.png'}")
    print(f"✅ Saved: {output_dir / 'training_loss_all_datasets_02550.pdf'}")


if __name__ == '__main__':
    main()
