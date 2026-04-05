#!/usr/bin/env python3
"""
Ablation Study Script for CoSA-CD
Runs all model variants systematically and generates a summary report.
"""
import subprocess
import sys
from pathlib import Path
import argparse
from datetime import datetime
import json
import re


def resolve_training_script():
    """Find train_cd_fixed.py in common repository layouts."""
    script_dir = Path(__file__).resolve().parent
    candidates = [
        script_dir / 'train' / 'train_cd_fixed.py',
        script_dir.parent / 'scripts' / 'train' / 'train_cd_fixed.py',
        script_dir.parent.parent / 'scripts' / 'train' / 'train_cd_fixed.py',
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def extract_metrics_from_log(log_file):
    """Extract test metrics from training log file."""
    metrics = {
        'f1': None,
        'iou': None,
        'precision': None,
        'recall': None,
    }
    
    if not Path(log_file).exists():
        return metrics
    
    with open(log_file, 'r') as f:
        content = f.read()
    
    # Extract test metrics (look for last occurrence)
    patterns = {
        'f1': r'Test F1:\s*([0-9]+\.[0-9]+)',
        'iou': r'Test IoU:\s*([0-9]+\.[0-9]+)',
        'precision': r'Test Precision:\s*([0-9]+\.[0-9]+)',
        'recall': r'Test Recall:\s*([0-9]+\.[0-9]+)',
    }
    
    for key, pattern in patterns.items():
        matches = re.findall(pattern, content)
        if matches:
            metrics[key] = float(matches[-1])  # Get last occurrence
    
    return metrics


def run_training(train_script, variant, dataset_dir, output_dir, batch_size=8, epochs=100,
                 lr=1e-4, base_size=512, seed=42, num_workers=2, use_amp=True):
    """Run training for a single variant."""
    cmd = [
        sys.executable,
        str(train_script),
        '--variant', variant,
        '--dataset_dir', str(dataset_dir),
        '--batch_size', str(batch_size),
        '--num_workers', str(num_workers),
        '--epochs', str(epochs),
        '--lr', str(lr),
        '--base_size', str(base_size),
        '--seed', str(seed),
        '--output_dir', str(output_dir),
    ]
    
    if use_amp:
        cmd.append('--amp')
    
    print(f"\n{'='*70}")
    print(f"Running variant: {variant}")
    print(f"{'='*70}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*70}\n")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        return True, None
    except subprocess.CalledProcessError as e:
        return False, str(e)


def main():
    parser = argparse.ArgumentParser(description='Run ablation study for CoSA-CD')
    parser.add_argument('--dataset_dir', type=str, required=True,
                       help='Path to LEVIR-CD dataset root')
    parser.add_argument('--output_dir', type=str, default='ablation/ablation_study',
                       help='Base output directory for all results')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--base_size', type=int, default=512)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--skip_training', action='store_true',
                       help='Skip training, only generate summary from existing results')
    parser.add_argument('--variants', type=str, nargs='+',
                       default=['baseline', 'attention', 'cosa', 'cosa_v3', 
                               'cosa_multiscale_only', 'cosa_learnable_gate_only', 'aligned'],
                       help='List of variants to run')
    parser.add_argument('--no_amp', dest='use_amp', action='store_false')
    parser.set_defaults(use_amp=True)
    
    args = parser.parse_args()

    train_script = resolve_training_script()
    if train_script is None:
        print("ERROR: Could not find training script: train_cd_fixed.py")
        print("Expected one of:")
        print("  - scripts/train/train_cd_fixed.py (inside research_repo)")
        print("  - ../scripts/train/train_cd_fixed.py (parent workspace)")
        sys.exit(1)
    
    dataset_dir = Path(args.dataset_dir)
    if not dataset_dir.exists():
        print(f"ERROR: Dataset directory not found: {dataset_dir}")
        sys.exit(1)
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Variant descriptions
    variant_descriptions = {
        'baseline': 'B0: Siamese U-Net baseline',
        'attention': 'B1: Siamese U-Net + Attention-only',
        'cosa': 'B2 (v2): CoSA single-scale, fixed gate',
        'cosa_v3': 'B2 (v3): CoSA multi-scale, learnable gate',
        'cosa_multiscale_only': 'Ablation: Multi-scale only (no learnable gate)',
        'cosa_learnable_gate_only': 'Ablation: Learnable gate only (no multi-scale)',
        'aligned': 'B3: Alignment-first (aligns T2 to T1 before differencing)',
    }
    
    results = {}
    
    # Run training for each variant
    if not args.skip_training:
        for variant in args.variants:
            variant_output_dir = output_dir / variant
            variant_output_dir.mkdir(parents=True, exist_ok=True)
            
            success, error = run_training(
                train_script=train_script,
                variant=variant,
                dataset_dir=dataset_dir,
                output_dir=variant_output_dir,
                batch_size=args.batch_size,
                epochs=args.epochs,
                lr=args.lr,
                base_size=args.base_size,
                seed=args.seed,
                num_workers=args.num_workers,
                use_amp=args.use_amp,
            )
            
            if success:
                print(f"✅ Completed: {variant}")
            else:
                print(f"❌ Failed: {variant} - {error}")
    
    # Collect results
    print(f"\n{'='*70}")
    print("Collecting results...")
    print(f"{'='*70}\n")
    
    for variant in args.variants:
        variant_output_dir = output_dir / variant
        log_file = variant_output_dir / 'training.log'
        
        metrics = extract_metrics_from_log(log_file)
        results[variant] = {
            'description': variant_descriptions.get(variant, variant),
            'metrics': metrics,
            'log_file': str(log_file) if log_file.exists() else None,
        }
    
    # Generate summary report
    report_file = output_dir / 'ablation_summary.txt'
    json_file = output_dir / 'ablation_summary.json'
    
    with open(report_file, 'w') as f:
        f.write("="*70 + "\n")
        f.write("CoSA-CD Ablation Study Results\n")
        f.write("="*70 + "\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Seed: {args.seed}\n")
        f.write(f"Batch size: {args.batch_size}\n")
        f.write(f"Epochs: {args.epochs}\n")
        f.write(f"Learning rate: {args.lr}\n")
        f.write(f"Base size: {args.base_size}\n")
        f.write("="*70 + "\n\n")
        
        f.write(f"{'Variant':<30} | {'F1':>8} | {'IoU':>8} | {'Precision':>10} | {'Recall':>8}\n")
        f.write("-" * 70 + "\n")
        
        for variant, data in results.items():
            m = data['metrics']
            f1_str = f"{m['f1']:.2f}" if m['f1'] is not None else "N/A"
            iou_str = f"{m['iou']:.2f}" if m['iou'] is not None else "N/A"
            prec_str = f"{m['precision']:.2f}" if m['precision'] is not None else "N/A"
            recall_str = f"{m['recall']:.2f}" if m['recall'] is not None else "N/A"
            
            f.write(f"{variant:<30} | {f1_str:>8} | {iou_str:>8} | {prec_str:>10} | {recall_str:>8}\n")
        
        f.write("\n" + "="*70 + "\n")
        f.write("Variant Descriptions:\n")
        f.write("="*70 + "\n")
        for variant, data in results.items():
            f.write(f"\n{variant}:\n")
            f.write(f"  {data['description']}\n")
    
    # Save JSON results
    with open(json_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print("\n" + "="*70)
    print("Ablation Study Summary")
    print("="*70)
    print(f"\n{'Variant':<30} | {'F1':>8} | {'IoU':>8} | {'Precision':>10} | {'Recall':>8}")
    print("-" * 70)
    
    for variant, data in results.items():
        m = data['metrics']
        f1_str = f"{m['f1']:.2f}" if m['f1'] is not None else "N/A"
        iou_str = f"{m['iou']:.2f}" if m['iou'] is not None else "N/A"
        prec_str = f"{m['precision']:.2f}" if m['precision'] is not None else "N/A"
        recall_str = f"{m['recall']:.2f}" if m['recall'] is not None else "N/A"
        
        print(f"{variant:<30} | {f1_str:>8} | {iou_str:>8} | {prec_str:>10} | {recall_str:>8}")
    
    print("\n" + "="*70)
    print(f"✅ Results saved to:")
    print(f"   - Text report: {report_file}")
    print(f"   - JSON results: {json_file}")
    print("="*70)


if __name__ == '__main__':
    main()
