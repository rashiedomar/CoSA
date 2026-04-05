#!/usr/bin/env python3
"""
Visualize Ablation Study Results

Creates:
1. Training progression at epochs 0, 25, 50, 75, 100
2. Ablation component analysis showing contribution of each component
"""
import sys
from pathlib import Path
import re
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import matplotlib.patches as mpatches

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
    """Parse training log and extract metrics at each epoch."""
    epochs = []
    train_losses = []
    val_f1s = []
    val_ious = []
    val_precisions = []
    val_recalls = []
    
    if not Path(log_path).exists():
        return None
    
    with open(log_path, 'r') as f:
        content = f.read()
    
    # Pattern: Epoch X/100 - Train Loss: Y, Val Precision: Z%, Recall: W%, F1: V%, IoU: U%
    pattern = r'Epoch (\d+)/\d+ - Train Loss: ([\d.]+), Val Precision: ([\d.]+)%, Recall: ([\d.]+)%, F1: ([\d.]+)%, IoU: ([\d.]+)%'
    
    matches = re.findall(pattern, content)
    for epoch_str, loss_str, prec_str, rec_str, f1_str, iou_str in matches:
        epochs.append(int(epoch_str))
        train_losses.append(float(loss_str))
        val_precisions.append(float(prec_str))
        val_recalls.append(float(rec_str))
        val_f1s.append(float(f1_str))
        val_ious.append(float(iou_str))
    
    return {
        'epochs': epochs,
        'train_losses': train_losses,
        'val_f1s': val_f1s,
        'val_ious': val_ious,
        'val_precisions': val_precisions,
        'val_recalls': val_recalls,
    }


def get_metrics_at_epochs(data, target_epochs=[1, 25, 50, 75, 100]):
    """Extract metrics at specific epochs (or closest available)."""
    if data is None:
        return None
    
    epochs = data['epochs']
    result = {}
    
    for target_epoch in target_epochs:
        # Find closest epoch
        closest_idx = None
        closest_dist = float('inf')
        
        for i, epoch in enumerate(epochs):
            dist = abs(epoch - target_epoch)
            if dist < closest_dist:
                closest_dist = dist
                closest_idx = i
        
        if closest_idx is not None:
            actual_epoch = epochs[closest_idx]
            result[target_epoch] = {
                'actual_epoch': actual_epoch,
                'f1': data['val_f1s'][closest_idx],
                'iou': data['val_ious'][closest_idx],
                'precision': data['val_precisions'][closest_idx],
                'recall': data['val_recalls'][closest_idx],
                'loss': data['train_losses'][closest_idx],
            }
    
    return result


def plot_training_progression(all_data, output_path):
    """Plot 1: Training progression at epochs 1, 25, 50, 75, 100"""
    target_epochs = [1, 25, 50, 75, 100]  # Using epoch 1 instead of 0 (logs start at epoch 1)
    
    # Variant configuration
    variants = {
        'baseline': {'name': 'B0: Baseline', 'color': '#1f77b4', 'marker': 'o'},
        'attention': {'name': 'B1: Attention-only', 'color': '#ff7f0e', 'marker': 's'},
        'cosa': {'name': 'B2 (v2): CoSA single-scale', 'color': '#2ca02c', 'marker': '^'},
        'cosa_v3': {'name': 'B2 (v3): CoSA multi-scale', 'color': '#d62728', 'marker': 'D'},
        'cosa_multiscale_only': {'name': 'Ablation: Multi-scale only', 'color': '#9467bd', 'marker': 'v'},
        'cosa_learnable_gate_only': {'name': 'Ablation: Learnable gate only', 'color': '#8c564b', 'marker': '<'},
        'aligned': {'name': 'B3: Alignment-first', 'color': '#e377c2', 'marker': '>'},
    }
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: F1-Score progression
    ax1 = axes[0, 0]
    for variant, config in variants.items():
        if variant in all_data and all_data[variant] is not None:
            metrics = get_metrics_at_epochs(all_data[variant], target_epochs)
            if metrics:
                epochs_plot = []
                f1s_plot = []
                for epoch in target_epochs:
                    if epoch in metrics:
                        epochs_plot.append(epoch)
                        f1s_plot.append(metrics[epoch]['f1'])
                
                ax1.plot(epochs_plot, f1s_plot, 
                        color=config['color'], marker=config['marker'], 
                        label=config['name'], linewidth=2, markersize=8, alpha=0.8)
    
    ax1.set_xlabel('Epoch', fontweight='bold')
    ax1.set_ylabel('Validation F1-Score (%)', fontweight='bold')
    ax1.set_title('F1-Score Progression', fontweight='bold', pad=15)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.legend(loc='lower right', fontsize=8, ncol=1)
    ax1.set_xticks(target_epochs)
    
    # Plot 2: IoU progression
    ax2 = axes[0, 1]
    for variant, config in variants.items():
        if variant in all_data and all_data[variant] is not None:
            metrics = get_metrics_at_epochs(all_data[variant], target_epochs)
            if metrics:
                epochs_plot = []
                ious_plot = []
                for epoch in target_epochs:
                    if epoch in metrics:
                        epochs_plot.append(epoch)
                        ious_plot.append(metrics[epoch]['iou'])
                
                ax2.plot(epochs_plot, ious_plot, 
                        color=config['color'], marker=config['marker'], 
                        label=config['name'], linewidth=2, markersize=8, alpha=0.8)
    
    ax2.set_xlabel('Epoch', fontweight='bold')
    ax2.set_ylabel('Validation IoU (%)', fontweight='bold')
    ax2.set_title('IoU Progression', fontweight='bold', pad=15)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.legend(loc='lower right', fontsize=8, ncol=1)
    ax2.set_xticks(target_epochs)
    
    # Plot 3: Precision progression
    ax3 = axes[1, 0]
    for variant, config in variants.items():
        if variant in all_data and all_data[variant] is not None:
            metrics = get_metrics_at_epochs(all_data[variant], target_epochs)
            if metrics:
                epochs_plot = []
                precs_plot = []
                for epoch in target_epochs:
                    if epoch in metrics:
                        epochs_plot.append(epoch)
                        precs_plot.append(metrics[epoch]['precision'])
                
                ax3.plot(epochs_plot, precs_plot, 
                        color=config['color'], marker=config['marker'], 
                        label=config['name'], linewidth=2, markersize=8, alpha=0.8)
    
    ax3.set_xlabel('Epoch', fontweight='bold')
    ax3.set_ylabel('Validation Precision (%)', fontweight='bold')
    ax3.set_title('Precision Progression', fontweight='bold', pad=15)
    ax3.grid(True, alpha=0.3, linestyle='--')
    ax3.legend(loc='lower right', fontsize=8, ncol=1)
    ax3.set_xticks(target_epochs)
    
    # Plot 4: Recall progression
    ax4 = axes[1, 1]
    for variant, config in variants.items():
        if variant in all_data and all_data[variant] is not None:
            metrics = get_metrics_at_epochs(all_data[variant], target_epochs)
            if metrics:
                epochs_plot = []
                recalls_plot = []
                for epoch in target_epochs:
                    if epoch in metrics:
                        epochs_plot.append(epoch)
                        recalls_plot.append(metrics[epoch]['recall'])
                
                ax4.plot(epochs_plot, recalls_plot, 
                        color=config['color'], marker=config['marker'], 
                        label=config['name'], linewidth=2, markersize=8, alpha=0.8)
    
    ax4.set_xlabel('Epoch', fontweight='bold')
    ax4.set_ylabel('Validation Recall (%)', fontweight='bold')
    ax4.set_title('Recall Progression', fontweight='bold', pad=15)
    ax4.grid(True, alpha=0.3, linestyle='--')
    ax4.legend(loc='lower right', fontsize=8, ncol=1)
    ax4.set_xticks(target_epochs)
    
    plt.suptitle('Training Progression: Epochs 1, 25, 50, 75, 100', 
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {output_path}")


def plot_ablation_component_analysis(results_json, output_path):
    """Plot 2: Ablation Component Analysis - showing contribution of each component"""
    
    # Load final results
    with open(results_json, 'r') as f:
        results = json.load(f)
    
    # Define component contributions
    # Baseline = 0
    # + Attention = small gain
    # + Multi-scale only = negative (worse than baseline)
    # + Learnable gate only = small gain
    # + Both (CoSA v3) = best
    
    variants_order = [
        ('baseline', 'B0: Baseline', []),
        ('attention', 'B1: + Attention', ['Attention']),
        ('cosa_multiscale_only', 'Ablation: + Multi-scale', ['Multi-scale']),
        ('cosa_learnable_gate_only', 'Ablation: + Learnable Gate', ['Learnable Gate']),
        ('cosa_v3', 'B2 (v3): + Both', ['Multi-scale', 'Learnable Gate']),
    ]
    
    # Extract F1 scores
    f1_scores = []
    variant_names = []
    components = []
    
    for variant_key, variant_name, comps in variants_order:
        if variant_key in results:
            f1_scores.append(results[variant_key]['f1'])
            variant_names.append(variant_name)
            components.append(comps)
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Bar chart showing F1 scores with component annotations
    colors = ['#1f77b4', '#ff7f0e', '#9467bd', '#8c564b', '#d62728']
    bars = ax1.bar(range(len(variant_names)), f1_scores, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    
    # Add value labels on bars
    for i, (bar, score) in enumerate(zip(bars, f1_scores)):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.3,
                f'{score:.2f}%',
                ha='center', va='bottom', fontweight='bold', fontsize=10)
        
        # Add component annotations below x-axis
        if components[i]:
            comp_text = ' + '.join(components[i])
            ax1.text(bar.get_x() + bar.get_width()/2., -2,
                    comp_text,
                    ha='center', va='top', fontsize=8, style='italic', rotation=0)
    
    ax1.set_xticks(range(len(variant_names)))
    ax1.set_xticklabels(variant_names, rotation=15, ha='right', fontsize=10)
    ax1.set_ylabel('F1-Score (%)', fontweight='bold')
    ax1.set_title('Component Contribution Analysis', fontweight='bold', pad=15)
    ax1.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax1.set_ylim([85, 90.5])
    
    # Add improvement arrows
    baseline_f1 = f1_scores[0]
    for i in range(1, len(f1_scores)):
        improvement = f1_scores[i] - baseline_f1
        if improvement != 0:
            arrow_color = 'green' if improvement > 0 else 'red'
            ax1.annotate('', xy=(i, f1_scores[i]), xytext=(i, baseline_f1),
                        arrowprops=dict(arrowstyle='->', color=arrow_color, lw=2))
            ax1.text(i, (baseline_f1 + f1_scores[i])/2, 
                    f'{improvement:+.2f}%',
                    ha='center', va='center', fontsize=9, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=arrow_color, linewidth=1.5))
    
    # Plot 2: Stacked contribution visualization
    # Show how each component adds/subtracts from baseline
    baseline = f1_scores[0]
    contributions = []
    contribution_labels = []
    
    for i in range(1, len(f1_scores)):
        delta = f1_scores[i] - baseline
        contributions.append(delta)
        contribution_labels.append(variant_names[i].split(': ')[-1])
    
    colors_contrib = ['green' if c > 0 else 'red' for c in contributions]
    bars2 = ax2.barh(range(len(contribution_labels)), contributions, 
                     color=colors_contrib, alpha=0.7, edgecolor='black', linewidth=1.5)
    
    # Add value labels
    for i, (bar, contrib) in enumerate(zip(bars2, contributions)):
        width = bar.get_width()
        ax2.text(width + (0.1 if width > 0 else -0.1), bar.get_y() + bar.get_height()/2,
                f'{contrib:+.2f}%',
                ha='left' if width > 0 else 'right', va='center', 
                fontweight='bold', fontsize=10)
    
    ax2.set_yticks(range(len(contribution_labels)))
    ax2.set_yticklabels(contribution_labels, fontsize=10)
    ax2.set_xlabel('Δ F1-Score vs Baseline (%)', fontweight='bold')
    ax2.set_title('Component Impact (vs Baseline)', fontweight='bold', pad=15)
    ax2.axvline(x=0, color='black', linestyle='--', linewidth=1)
    ax2.grid(True, alpha=0.3, axis='x', linestyle='--')
    
    plt.suptitle('Ablation Component Analysis', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {output_path}")


def plot_final_performance_comparison(results_json, output_path):
    """Plot 3: Final performance comparison - all metrics"""
    
    with open(results_json, 'r') as f:
        results = json.load(f)
    
    variants_order = [
        'baseline',
        'attention',
        'cosa',
        'cosa_v3',
        'cosa_multiscale_only',
        'cosa_learnable_gate_only',
        'aligned',
    ]
    
    variant_names = []
    f1_scores = []
    iou_scores = []
    precision_scores = []
    recall_scores = []
    
    for variant in variants_order:
        if variant in results:
            variant_names.append(results[variant]['name'])
            f1_scores.append(results[variant]['f1'])
            iou_scores.append(results[variant]['iou'])
            precision_scores.append(results[variant]['precision'])
            recall_scores.append(results[variant]['recall'])
    
    x = np.arange(len(variant_names))
    width = 0.2
    
    fig, ax = plt.subplots(figsize=(16, 6))
    
    bars1 = ax.bar(x - 1.5*width, f1_scores, width, label='F1-Score', color='#1f77b4', alpha=0.8)
    bars2 = ax.bar(x - 0.5*width, iou_scores, width, label='IoU', color='#ff7f0e', alpha=0.8)
    bars3 = ax.bar(x + 0.5*width, precision_scores, width, label='Precision', color='#2ca02c', alpha=0.8)
    bars4 = ax.bar(x + 1.5*width, recall_scores, width, label='Recall', color='#d62728', alpha=0.8)
    
    # Highlight best performer
    best_idx = f1_scores.index(max(f1_scores))
    for bars in [bars1, bars2, bars3, bars4]:
        bars[best_idx].set_edgecolor('gold')
        bars[best_idx].set_linewidth(3)
    
    ax.set_xlabel('Model Variant', fontweight='bold')
    ax.set_ylabel('Score (%)', fontweight='bold')
    ax.set_title('Final Performance Comparison (Test Set)', fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(variant_names, rotation=15, ha='right', fontsize=9)
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax.set_ylim([70, 95])
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {output_path}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Visualize ablation study results')
    parser.add_argument('--ablation_dir', type=str,
                       default='results/ablation_study',
                       help='Directory containing ablation study results')
    parser.add_argument('--output_dir', type=str,
                       default='results/ablation_study/visualizations',
                       help='Output directory for visualizations')
    
    args = parser.parse_args()
    
    # Resolve paths
    project_root = Path(__file__).resolve().parents[1]
    ablation_dir = project_root / args.ablation_dir
    output_dir = project_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results_json = ablation_dir / 'ablation_results.json'
    
    print("=" * 70)
    print("Generating Ablation Study Visualizations")
    print("=" * 70)
    
    # Parse all training logs
    print("\nParsing training logs...")
    all_data = {}
    variants = ['baseline', 'attention', 'cosa', 'cosa_v3', 
                'cosa_multiscale_only', 'cosa_learnable_gate_only', 'aligned']
    
    for variant in variants:
        log_path = ablation_dir / variant / 'training.log'
        print(f"  Parsing {variant}...", end=' ')
        data = parse_training_log(log_path)
        if data:
            all_data[variant] = data
            print(f"✅ ({len(data['epochs'])} epochs)")
        else:
            print("❌ (not found)")
    
    # Generate visualizations
    print(f"\nGenerating visualizations...")
    
    plot_training_progression(all_data, output_dir / 'training_progression.png')
    plot_ablation_component_analysis(results_json, output_dir / 'component_analysis.png')
    plot_final_performance_comparison(results_json, output_dir / 'final_performance.png')
    
    print("\n" + "=" * 70)
    print("✅ All visualizations generated!")
    print(f"   Output directory: {output_dir}")
    print("=" * 70)


if __name__ == '__main__':
    main()

