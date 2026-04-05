#!/usr/bin/env python3
"""
Generate Quantitative Figures from Per-Sample Analysis Data

Creates three publication-quality figures:
1. Box Plot: F1 score distribution (Baseline vs CoSA)
2. Histogram: F1 Improvement distribution (Delta F1)
3. Bar Chart: Stratified F1 by difficulty (Easy/Medium/Hard)

Can either:
- Load saved per-sample data (JSON/pickle)
- Re-run analysis if data not available
"""
import sys
from pathlib import Path
import json
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import rcParams

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

# Add repo root to path
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

# Import the quantitative analysis functions
# Need to import from the same directory
import importlib.util
spec = importlib.util.spec_from_file_location(
    "quant_tables", 
    Path(__file__).parent / "generate_quantitative_tables_all_models.py"
)
quant_tables = importlib.util.module_from_spec(spec)
sys.path.insert(0, str(Path(__file__).parent.parent))
spec.loader.exec_module(quant_tables)
from generate_quantitative_tables_all_models import run_fc_siam, run_stanet, run_bit, ROOT
import torch


def load_or_compute_data(model_pair, device, dataset_dir, dataset_256, 
                         baseline_ckpt, cosa_ckpt, stanet_ckpt_dir=None, 
                         stanet_cosa_ckpt=None, bit_ckpt=None, bit_cosa_ckpt=None,
                         data_cache_dir=None):
    """Load saved per-sample data or compute it."""
    if data_cache_dir is None:
        data_cache_dir = ROOT / "results" / "baseline_cosa_visualizations" / "data_cache"
    data_cache_dir = Path(data_cache_dir)
    data_cache_dir.mkdir(parents=True, exist_ok=True)
    
    cache_file = data_cache_dir / f"per_sample_data_{model_pair}.pkl"
    
    # Try to load cached data
    if cache_file.exists():
        print(f"Loading cached data from {cache_file}")
        with open(cache_file, 'rb') as f:
            all_samples = pickle.load(f)
        print(f"✅ Loaded {len(all_samples)} samples")
        return all_samples
    
    # Compute data
    print(f"Computing per-sample data for {model_pair}...")
    all_samples = None
    
    if model_pair == "fc_siam":
        all_samples = run_fc_siam(device, dataset_dir, baseline_ckpt, cosa_ckpt)
    elif model_pair == "stanet":
        all_samples = run_stanet(device, dataset_256, stanet_ckpt_dir, stanet_cosa_ckpt)
    elif model_pair == "bit":
        # Remove STANet dir from path
        stanet_dir = str(ROOT / "checkpoints_official" / "STANet_LEVIR" / "STANet-master")
        while stanet_dir in sys.path:
            sys.path.remove(stanet_dir)
        all_samples = run_bit(device, dataset_256, bit_ckpt, bit_cosa_ckpt)
    
    if all_samples is None:
        raise ValueError(f"Failed to compute data for {model_pair}")
    
    # Save to cache
    print(f"Saving data to {cache_file}")
    with open(cache_file, 'wb') as f:
        pickle.dump(all_samples, f)
    
    return all_samples


def plot_boxplot_f1_distribution(all_samples, model_pair_name, output_path):
    """Figure 1: Box Plot comparing F1 score distribution."""
    baseline_f1s = [s['baseline_f1'] for s in all_samples]
    cosa_f1s = [s['cosa_f1'] for s in all_samples]
    
    fig, ax = plt.subplots(figsize=(6, 5))
    
    # Create box plot
    bp = ax.boxplot([baseline_f1s, cosa_f1s], 
                    tick_labels=['Baseline', 'CoSA'],
                    patch_artist=True,
                    widths=0.6,
                    showmeans=True,
                    meanline=True)
    
    # Color the boxes
    colors = ['#4A90E2', '#F5A623']  # Blue, Orange
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    # Style the median lines
    for median in bp['medians']:
        median.set_color('black')
        median.set_linewidth(2)
    
    # Style the mean lines
    for mean in bp['means']:
        mean.set_color('red')
        mean.set_linewidth(1.5)
        mean.set_linestyle('--')
    
    ax.set_ylabel('F1 Score', fontweight='bold')
    ax.set_title(f'F1 Score Distribution: {model_pair_name}', fontweight='bold', pad=15)
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax.set_ylim([0, 1.05])
    
    # Add statistics text
    baseline_mean = np.mean(baseline_f1s)
    cosa_mean = np.mean(cosa_f1s)
    baseline_std = np.std(baseline_f1s)
    cosa_std = np.std(cosa_f1s)
    
    stats_text = f'Baseline: μ={baseline_mean:.3f}, σ={baseline_std:.3f}\n'
    stats_text += f'CoSA: μ={cosa_mean:.3f}, σ={cosa_std:.3f}\n'
    stats_text += f'Improvement: {cosa_mean - baseline_mean:+.3f}'
    
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            verticalalignment='top', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {output_path}")


def plot_histogram_improvement(all_samples, model_pair_name, output_path):
    """Figure 2: Histogram of F1 Improvement (Delta F1)."""
    improvements = [s['improvement'] for s in all_samples]
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Create histogram
    bins = np.linspace(min(improvements) - 0.05, max(improvements) + 0.05, 50)
    n, bins, patches = ax.hist(improvements, bins=bins, edgecolor='black', alpha=0.7)
    
    # Color bars: negative (red), zero (gray), positive (green)
    for i, (patch, bin_left) in enumerate(zip(patches, bins[:-1])):
        if bin_left < 0:
            patch.set_facecolor('#E74C3C')  # Red for degradation
        elif bin_left < 0.02:  # Neutral zone
            patch.set_facecolor('#95A5A6')  # Gray
        else:
            patch.set_facecolor('#27AE60')  # Green for improvement
    
    # Add vertical line at zero
    ax.axvline(x=0, color='black', linestyle='--', linewidth=2, alpha=0.7, label='No Change')
    
    # Add mean line
    mean_improvement = np.mean(improvements)
    ax.axvline(x=mean_improvement, color='blue', linestyle='-', linewidth=2, 
               alpha=0.8, label=f'Mean: {mean_improvement:+.3f}')
    
    ax.set_xlabel('F1 Improvement (Δ F1)', fontweight='bold')
    ax.set_ylabel('Number of Images', fontweight='bold')
    ax.set_title(f'F1 Improvement Distribution: {model_pair_name}', fontweight='bold', pad=15)
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax.legend(loc='upper right')
    
    # Add statistics
    positive = sum(1 for i in improvements if i > 0.01)
    negative = sum(1 for i in improvements if i < -0.01)
    neutral = len(improvements) - positive - negative
    
    stats_text = f'Total: {len(improvements)} images\n'
    stats_text += f'Improved: {positive} ({positive/len(improvements)*100:.1f}%)\n'
    stats_text += f'Degraded: {negative} ({negative/len(improvements)*100:.1f}%)\n'
    stats_text += f'Neutral: {neutral} ({neutral/len(improvements)*100:.1f}%)'
    
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            verticalalignment='top', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {output_path}")


def plot_barchart_stratified(all_samples, model_pair_name, output_path):
    """Figure 3: Grouped Bar Chart of Stratified F1 (Easy/Medium/Hard)."""
    # Filter out no-change images
    samples_with_changes = [s for s in all_samples if not s.get('is_no_change', False)]
    if not samples_with_changes:
        samples_with_changes = all_samples
    
    # Stratify
    easy = [s for s in samples_with_changes if s['baseline_f1'] > 0.85]
    medium = [s for s in samples_with_changes if 0.60 <= s['baseline_f1'] <= 0.85]
    hard = [s for s in samples_with_changes if s['baseline_f1'] < 0.60]
    
    # Compute means
    def avg_f1(samples, key='baseline_f1'):
        if not samples:
            return 0.0
        return np.mean([s[key] for s in samples]) * 100
    
    easy_baseline = avg_f1(easy, 'baseline_f1')
    easy_cosa = avg_f1(easy, 'cosa_f1')
    medium_baseline = avg_f1(medium, 'baseline_f1')
    medium_cosa = avg_f1(medium, 'cosa_f1')
    hard_baseline = avg_f1(hard, 'baseline_f1')
    hard_cosa = avg_f1(hard, 'cosa_f1')
    
    # Create grouped bar chart
    fig, ax = plt.subplots(figsize=(8, 5))
    
    x = np.arange(3)
    width = 0.35
    
    baseline_means = [easy_baseline, medium_baseline, hard_baseline]
    cosa_means = [easy_cosa, medium_cosa, hard_cosa]
    
    bars1 = ax.bar(x - width/2, baseline_means, width, label='Baseline', 
                   color='#4A90E2', alpha=0.8, edgecolor='black', linewidth=1)
    bars2 = ax.bar(x + width/2, cosa_means, width, label='CoSA', 
                   color='#F5A623', alpha=0.8, edgecolor='black', linewidth=1)
    
    # Add value labels on bars
    def add_value_labels(bars):
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}%',
                   ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    add_value_labels(bars1)
    add_value_labels(bars2)
    
    ax.set_xlabel('Sample Difficulty', fontweight='bold')
    ax.set_ylabel('F1 Score (%)', fontweight='bold')
    ax.set_title(f'Stratified F1 Performance: {model_pair_name}', fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(['Easy\n(>85%)', 'Medium\n(60-85%)', 'Hard\n(<60%)'])
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax.set_ylim([0, 105])
    
    # Add counts
    counts_text = f'Easy: {len(easy)} | Medium: {len(medium)} | Hard: {len(hard)}'
    ax.text(0.5, 0.02, counts_text, transform=ax.transAxes,
            ha='center', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {output_path}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate Quantitative Figures')
    parser.add_argument('--pair', type=str, choices=['fc_siam', 'stanet', 'bit', 'all'], 
                       default='fc_siam', help='Model pair to analyze')
    parser.add_argument('--dataset_dir', type=str, 
                       default='datasets/to_check_dataset/LEVIR-CD_combined')
    parser.add_argument('--dataset_256', type=str, 
                       default='datasets/to_check_dataset/LEVIR-CD_combined_256/test')
    parser.add_argument('--baseline_bs8', type=str, 
                       default='results/baseline_bs8/checkpoint_best.pth')
    parser.add_argument('--cosa_v3', type=str, 
                       default='results/cosa_v3_residual_multiscale/checkpoint_best.pth')
    parser.add_argument('--stanet_ckpt_dir', type=str,
                       default='checkpoints_official/STANet_LEVIR/STANet-master/checkpoints/stanet_official_paper')
    parser.add_argument('--stanet_cosa_ckpt', type=str, 
                       default='results/stanet_cosa_finetune/best_checkpoint.pth')
    parser.add_argument('--bit_ckpt', type=str,
                       default='checkpoints_official/BIT_LEVIR/BIT_CD-master/checkpoints/BIT_LEVIR/best_ckpt.pt')
    parser.add_argument('--bit_cosa_ckpt', type=str, 
                       default='results/bit_cosa_finetune/best_checkpoint.pth')
    parser.add_argument('--output_dir', type=str, 
                       default='results/baseline_cosa_visualizations/figures')
    parser.add_argument('--data_cache_dir', type=str, 
                       default='results/baseline_cosa_visualizations/data_cache')
    parser.add_argument('--device', type=str, 
                       default='cuda' if torch.cuda.is_available() else 'cpu')
    
    args = parser.parse_args()
    
    device = torch.device(args.device)
    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    dataset_dir = ROOT / args.dataset_dir
    dataset_256 = ROOT / args.dataset_256
    baseline_bs8 = ROOT / args.baseline_bs8
    cosa_v3 = ROOT / args.cosa_v3
    stanet_ckpt_dir = ROOT / args.stanet_ckpt_dir
    stanet_cosa_ckpt = ROOT / args.stanet_cosa_ckpt
    bit_ckpt = ROOT / args.bit_ckpt
    bit_cosa_ckpt = ROOT / args.bit_cosa_ckpt
    
    pairs = ['fc_siam', 'stanet', 'bit'] if args.pair == 'all' else [args.pair]
    model_names = {
        'fc_siam': 'FC-Siam-diff vs FC-Siam-diff+CoSA',
        'stanet': 'STANet vs STANet+CoSA',
        'bit': 'BIT vs BIT+CoSA'
    }
    
    print("=" * 70)
    print("Generating Quantitative Figures")
    print("=" * 70)
    
    for pair in pairs:
        print(f"\nProcessing: {pair.upper()}")
        print("-" * 70)
        
        # Load or compute data
        try:
            all_samples = load_or_compute_data(
                pair, device, dataset_dir, dataset_256,
                baseline_bs8, cosa_v3, stanet_ckpt_dir, stanet_cosa_ckpt,
                bit_ckpt, bit_cosa_ckpt, args.data_cache_dir
            )
        except Exception as e:
            print(f"❌ Error loading/computing data for {pair}: {e}")
            continue
        
        model_pair_name = model_names[pair]
        
        # Generate figures
        print(f"\nGenerating figures for {model_pair_name}...")
        
        # Figure 1: Box Plot
        output_path = output_dir / f"figure1_boxplot_f1_{pair}.png"
        plot_boxplot_f1_distribution(all_samples, model_pair_name, output_path)
        
        # Figure 2: Histogram
        output_path = output_dir / f"figure2_histogram_improvement_{pair}.png"
        plot_histogram_improvement(all_samples, model_pair_name, output_path)
        
        # Figure 3: Bar Chart
        output_path = output_dir / f"figure3_barchart_stratified_{pair}.png"
        plot_barchart_stratified(all_samples, model_pair_name, output_path)
        
        print(f"✅ Completed {pair}")
    
    print("\n" + "=" * 70)
    print(f"✅ All figures saved to: {output_dir}")
    print("=" * 70)


if __name__ == '__main__':
    main()
