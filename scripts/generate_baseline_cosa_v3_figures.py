#!/usr/bin/env python3
"""
Generate Training Figures for Baseline vs CoSA

Creates 1 publication-quality figure with 3 subplots:
1. Training Loss (both models)
2. Validation F1-Score (both models)
3. Validation IoU (both models)
"""

import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from pathlib import Path

# Set publication-quality style
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


def parse_training_log(log_path):
    """Parse training log and extract all metrics."""
    epochs = []
    train_losses = []
    val_f1s = []
    val_ious = []
    val_precisions = []
    val_recalls = []
    
    if not Path(log_path).exists():
        print(f"Warning: Log file not found: {log_path}")
        return None
    
    with open(log_path, 'r') as f:
        content = f.read()
    
    # Pattern: Epoch X/100 - Train Loss: Y, Val F1: Z%, IoU: W%, ...
    # Also handle: Epoch X/100 - Train Loss: Y, Val Precision: Z%, Recall: W%, F1: V%, IoU: U%
    pattern1 = r'Epoch (\d+)/\d+ - Train Loss: ([\d.]+), Val F1: ([\d.]+)%, IoU: ([\d.]+)%'
    pattern2 = r'Epoch (\d+)/\d+ - Train Loss: ([\d.]+), Val Precision: ([\d.]+)%, Recall: ([\d.]+)%, F1: ([\d.]+)%, IoU: ([\d.]+)%'
    
    matches1 = re.findall(pattern1, content)
    matches2 = re.findall(pattern2, content)
    
    # Use pattern2 if available (has more info), otherwise use pattern1
    if matches2:
        for epoch_str, loss_str, prec_str, rec_str, f1_str, iou_str in matches2:
            epochs.append(int(epoch_str))
            train_losses.append(float(loss_str))
            val_precisions.append(float(prec_str))
            val_recalls.append(float(rec_str))
            val_f1s.append(float(f1_str))
            val_ious.append(float(iou_str))
    elif matches1:
        for epoch_str, loss_str, f1_str, iou_str in matches1:
            epochs.append(int(epoch_str))
            train_losses.append(float(loss_str))
            val_f1s.append(float(f1_str))
            val_ious.append(float(iou_str))
            # For pattern1, we don't have precision/recall, so we'll calculate from F1 and IoU
            # Or leave them as None - we'll handle this in plotting
            val_precisions.append(None)
            val_recalls.append(None)
    
    return {
        'epochs': epochs,
        'train_losses': train_losses,
        'val_f1s': val_f1s,
        'val_ious': val_ious,
        'val_precisions': val_precisions,
        'val_recalls': val_recalls,
    }


def filter_epochs(epochs, values, target_epochs):
    """Filter to only show specific epochs (or closest available)."""
    filtered_epochs = []
    filtered_values = []
    
    for target_epoch in target_epochs:
        # Find closest epoch
        closest_idx = None
        min_diff = float('inf')
        for i, epoch in enumerate(epochs):
            diff = abs(epoch - target_epoch)
            if diff < min_diff:
                min_diff = diff
                closest_idx = i
        
        if closest_idx is not None and min_diff <= 2:  # Allow 2 epoch tolerance
            filtered_epochs.append(epochs[closest_idx])
            filtered_values.append(values[closest_idx])
    
    return filtered_epochs, filtered_values


def plot_combined_figure(baseline_data, cosa_data, output_path):
    """Single figure with 3 subplots: Training Loss, Validation F1, Validation IoU"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Colors
    baseline_color = '#4A90E2'  # Blue
    cosa_color = '#F5A623'       # Orange
    
    # Target epochs: 1, 25, 50, 75, 100
    target_epochs = [1, 25, 50, 75, 100]
    
    # Extract and filter data
    bl_epochs_all = baseline_data['epochs']
    bl_losses_all = baseline_data['train_losses']
    bl_f1s_all = baseline_data['val_f1s']
    bl_ious_all = baseline_data['val_ious']
    
    co_epochs_all = cosa_data['epochs']
    co_losses_all = cosa_data['train_losses']
    co_f1s_all = cosa_data['val_f1s']
    co_ious_all = cosa_data['val_ious']
    
    # Filter to target epochs
    bl_epochs_loss, bl_losses = filter_epochs(bl_epochs_all, bl_losses_all, target_epochs)
    bl_epochs_f1, bl_f1s = filter_epochs(bl_epochs_all, bl_f1s_all, target_epochs)
    bl_epochs_iou, bl_ious = filter_epochs(bl_epochs_all, bl_ious_all, target_epochs)
    
    co_epochs_loss, co_losses = filter_epochs(co_epochs_all, co_losses_all, target_epochs)
    co_epochs_f1, co_f1s = filter_epochs(co_epochs_all, co_f1s_all, target_epochs)
    co_epochs_iou, co_ious = filter_epochs(co_epochs_all, co_ious_all, target_epochs)
    
    # Subplot 1: Training Loss
    ax1 = axes[0]
    ax1.plot(bl_epochs_loss, bl_losses, 
            color=baseline_color, linestyle='-', marker='o', 
            label='Baseline', linewidth=2, markersize=8, alpha=0.8)
    ax1.plot(co_epochs_loss, co_losses, 
            color=cosa_color, linestyle='--', marker='s', 
            label='CoSA', linewidth=2, markersize=8, alpha=0.8)
    ax1.set_xlabel('Epoch', fontweight='bold')
    ax1.set_ylabel('Training Loss', fontweight='bold')
    ax1.set_title('Training Loss', fontweight='bold', pad=15)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.legend(loc='upper right')
    ax1.set_xlim([-5, 105])
    ax1.set_xticks([1, 25, 50, 75, 100])
    ax1.set_yscale('log')
    
    # Subplot 2: Validation F1-Score
    ax2 = axes[1]
    ax2.plot(bl_epochs_f1, bl_f1s, 
            color=baseline_color, linestyle='-', marker='o', 
            label='Baseline', linewidth=2, markersize=8, alpha=0.8)
    ax2.plot(co_epochs_f1, co_f1s, 
            color=cosa_color, linestyle='--', marker='s', 
            label='CoSA', linewidth=2, markersize=8, alpha=0.8)
    ax2.set_xlabel('Epoch', fontweight='bold')
    ax2.set_ylabel('F1-Score (%)', fontweight='bold')
    ax2.set_title('Validation F1-Score', fontweight='bold', pad=15)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.legend(loc='lower right')
    ax2.set_xlim([-5, 105])
    ax2.set_xticks([1, 25, 50, 75, 100])
    if bl_f1s and co_f1s:
        ax2.set_ylim([min(min(bl_f1s), min(co_f1s)) - 5, max(max(bl_f1s), max(co_f1s)) + 5])
    
    # Subplot 3: Validation IoU
    ax3 = axes[2]
    ax3.plot(bl_epochs_iou, bl_ious, 
            color=baseline_color, linestyle='-', marker='o', 
            label='Baseline', linewidth=2, markersize=8, alpha=0.8)
    ax3.plot(co_epochs_iou, co_ious, 
            color=cosa_color, linestyle='--', marker='s', 
            label='CoSA', linewidth=2, markersize=8, alpha=0.8)
    ax3.set_xlabel('Epoch', fontweight='bold')
    ax3.set_ylabel('IoU (%)', fontweight='bold')
    ax3.set_title('Validation IoU', fontweight='bold', pad=15)
    ax3.grid(True, alpha=0.3, linestyle='--')
    ax3.legend(loc='lower right')
    ax3.set_xlim([-5, 105])
    ax3.set_xticks([1, 25, 50, 75, 100])
    if bl_ious and co_ious:
        ax3.set_ylim([min(min(bl_ious), min(co_ious)) - 5, max(max(bl_ious), max(co_ious)) + 5])
    
    fig.suptitle('Training and Validation Metrics: Baseline vs CoSA', 
                 fontsize=16, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {output_path}")


def main():
    # Paths
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent
    workspace_root = repo_root.parent

    baseline_candidates = [
        repo_root / 'ablation' / 'ablation_study' / 'baseline' / 'training.log',
        workspace_root / 'results' / 'baseline_bs8' / 'training.log',
    ]
    cosa_candidates = [
        repo_root / 'ablation' / 'ablation_study' / 'cosa_v3' / 'training.log',
        workspace_root / 'results' / 'ablation_study' / 'cosa_v3' / 'training.log',
    ]

    baseline_log = next((p for p in baseline_candidates if p.exists()), baseline_candidates[0])
    cosa_log = next((p for p in cosa_candidates if p.exists()), cosa_candidates[0])

    output_dir = repo_root / 'figures'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Parse logs
    print("Parsing training logs...")
    baseline_data = parse_training_log(baseline_log)
    cosa_data = parse_training_log(cosa_log)
    
    if baseline_data is None or cosa_data is None:
        print("❌ Error: Could not parse one or both log files")
        return
    
    print(f"Baseline: {len(baseline_data['epochs'])} epochs")
    print(f"CoSA: {len(cosa_data['epochs'])} epochs")
    
    # Generate figure
    print("\nGenerating figure...")
    
    # Single combined figure - PDF and PNG
    plot_combined_figure(baseline_data, cosa_data, 
                        output_dir / 'training_validation_metrics_baseline_cosa.pdf')
    plot_combined_figure(baseline_data, cosa_data, 
                        output_dir / 'training_validation_metrics_baseline_cosa.png')
    
    print("\n✅ All figures generated successfully!")
    print(f"Output directory: {output_dir}")


if __name__ == '__main__':
    main()
