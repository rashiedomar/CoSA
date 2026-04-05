#!/usr/bin/env python3
"""
Generate comprehensive results table with all metrics:
- Method, Backbone, Lr Schd, Param (M), GFLOPS, Inf (fps), Precision, Recall, F1, IoU
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from collections import defaultdict

import torch
import numpy as np
from mmengine import Config
from mmengine.model.utils import revert_sync_batchnorm
from mmengine.registry import init_default_scope
from mmengine.runner import Runner, load_checkpoint

from opencd.registry import MODELS

try:
    from mmengine.analysis import get_model_complexity_info
    from mmengine.analysis.print_helper import _format_size
except ImportError:
    print("Warning: mmengine.analysis not available. GFLOPS calculation may fail.")


def get_model_params(config_path, checkpoint_path=None):
    """Get number of parameters in millions."""
    cfg = Config.fromfile(config_path)
    init_default_scope(cfg.get('default_scope', 'opencd'))
    
    cfg.model.train_cfg = None
    model = MODELS.build(cfg.model)
    
    if checkpoint_path and Path(checkpoint_path).exists():
        load_checkpoint(model, checkpoint_path, map_location='cpu')
    
    total_params = sum(p.numel() for p in model.parameters())
    return total_params / 1e6  # Convert to millions


def get_model_flops(config_path, input_shape=(256, 256)):
    """Get GFLOPS for model."""
    try:
        cfg = Config.fromfile(config_path)
        init_default_scope(cfg.get('default_scope', 'opencd'))
        
        input_shape_full = (6, input_shape[0], input_shape[1])
        model = MODELS.build(cfg.model)
        
        if hasattr(model, 'auxiliary_head'):
            model.auxiliary_head = None
        
        if torch.cuda.is_available():
            model.cuda()
        model = revert_sync_batchnorm(model)
        model.eval()
        
        from mmseg.structures import SegDataSample
        data_batch = {
            'inputs': [torch.rand(input_shape_full)],
            'data_samples': [SegDataSample(metainfo={'ori_shape': input_shape, 'pad_shape': input_shape})]
        }
        data = model.data_preprocessor(data_batch)
        
        outputs = get_model_complexity_info(
            model,
            None,
            inputs=data['inputs'],
            show_table=False,
            show_arch=False
        )
        
        # Convert FLOPs to GFLOPS
        flops = outputs['flops'] / 1e9  # Convert to GFLOPs
        return flops
    except Exception as e:
        print(f"Error computing FLOPs: {e}")
        return None


def get_inference_fps(config_path, checkpoint_path, num_iterations=120, num_warmup=5):
    """Get inference speed in FPS."""
    try:
        cfg = Config.fromfile(config_path)
        init_default_scope(cfg.get('default_scope', 'opencd'))
        
        cfg.model.train_cfg = None
        model = MODELS.build(cfg.model)
        
        if checkpoint_path and Path(checkpoint_path).exists():
            load_checkpoint(model, checkpoint_path, map_location='cpu')
        
        if torch.cuda.is_available():
            model = model.cuda()
        model = revert_sync_batchnorm(model)
        model.eval()
        
        # Build dataloader
        cfg.test_dataloader.batch_size = 1
        data_loader = Runner.build_dataloader(cfg.test_dataloader)
        
        torch.backends.cudnn.benchmark = False
        
        fps_list = []
        pure_inf_time = 0
        
        for i, data in enumerate(data_loader):
            data = model.data_preprocessor(data, True)
            inputs = data['inputs']
            data_samples = data['data_samples']
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            start_time = time.perf_counter()
            
            with torch.no_grad():
                model(inputs, data_samples, mode='predict')
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - start_time
            
            if i >= num_warmup:
                pure_inf_time += elapsed
                if (i + 1) == num_iterations:
                    fps = (i + 1 - num_warmup) / pure_inf_time
                    fps_list.append(fps)
                    break
        
        if fps_list:
            return np.mean(fps_list), np.std(fps_list) if len(fps_list) > 1 else 0.0
        return None, None
    except Exception as e:
        print(f"Error computing FPS: {e}")
        return None, None


def extract_test_metrics_from_json(json_path):
    """Extract test metrics from JSON file."""
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # Try different possible keys
        metrics = {}
        # Check for mIoU, mFscore, etc. (mean metrics)
        if 'mIoU' in data:
            metrics['IoU'] = data['mIoU']
        if 'mFscore' in data:
            metrics['F1'] = data['mFscore']
        if 'mPrecision' in data:
            metrics['Precision'] = data['mPrecision']
        if 'mRecall' in data:
            metrics['Recall'] = data['mRecall']
        
        # Check for lowercase keys (f1, iou, etc.)
        if 'iou' in data:
            val = data['iou']
            # Convert to percentage if value is between 0 and 1
            metrics['IoU'] = val * 100 if val < 1.0 else val
        if 'f1' in data:
            val = data['f1']
            metrics['F1'] = val * 100 if val < 1.0 else val
        if 'precision' in data:
            val = data['precision']
            metrics['Precision'] = val * 100 if val < 1.0 else val
        if 'recall' in data:
            val = data['recall']
            metrics['Recall'] = val * 100 if val < 1.0 else val
        
        return metrics
    except Exception as e:
        print(f"    Error reading JSON: {e}")
        return {}


def extract_test_metrics_from_log(log_path):
    """Extract test metrics from log file."""
    metrics = {}
    try:
        with open(log_path, 'r') as f:
            content = f.read()
        
        # Look for test results - per-class format first (more accurate)
        lines = content.split('\n')
        found_per_class = False
        for i, line in enumerate(lines):
            if '|' in line and 'changed' in line.lower() and 'Class' not in line:
                parts = [p.strip() for p in line.split('|')]
                # Filter out empty parts
                parts = [p for p in parts if p]
                # Skip header rows (check if first part is "Class" or second part is "Fscore")
                if len(parts) >= 5 and parts[0].lower() == 'changed':
                    try:
                        # Format: | changed | F1 | Precision | Recall | IoU | Acc |
                        # Index:    0         1       2          3       4     5
                        metrics['F1'] = float(parts[1])
                        metrics['Precision'] = float(parts[2])
                        metrics['Recall'] = float(parts[3])
                        metrics['IoU'] = float(parts[4])
                        found_per_class = True
                        break  # Found changed class metrics
                    except (ValueError, IndexError) as e:
                        # Debug: print error
                        pass
        
        # Fallback: look for Iter(test) summary ONLY if per-class not found
        if not found_per_class:
            for line in lines:
                if 'Iter(test)' in line and 'mIoU:' in line:
                    parts = line.split('mIoU:')
                    if len(parts) > 1:
                        try:
                            metrics['IoU'] = float(parts[1].split()[0])
                        except:
                            pass
                
                if 'Iter(test)' in line and 'mFscore:' in line:
                    parts = line.split('mFscore:')
                    if len(parts) > 1:
                        try:
                            metrics['F1'] = float(parts[1].split()[0])
                        except:
                            pass
                
                if 'Iter(test)' in line and 'mPrecision:' in line:
                    parts = line.split('mPrecision:')
                    if len(parts) > 1:
                        try:
                            metrics['Precision'] = float(parts[1].split()[0])
                        except:
                            pass
                
                if 'Iter(test)' in line and 'mRecall:' in line:
                    parts = line.split('mRecall:')
                    if len(parts) > 1:
                        try:
                            metrics['Recall'] = float(parts[1].split()[0])
                        except:
                            pass
    except Exception as e:
        print(f"    Error reading log: {e}")
        pass
    
    return metrics


def get_backbone_from_config(config_path):
    """Extract backbone information from config."""
    try:
        cfg = Config.fromfile(config_path)
        model = cfg.get('model', {})
        backbone = model.get('backbone', {})
        
        if 'type' in backbone:
            if 'ResNet' in backbone['type']:
                depth = backbone.get('depth', '')
                return f"ResNet-{depth}" if depth else "ResNet"
            elif 'VGG' in backbone['type']:
                return "VGG-16"
            elif 'TinyNet' in backbone['type']:
                arch = backbone.get('arch', '')
                return f"TinyNet-{arch}" if arch else "TinyNet"
            else:
                return backbone['type']
        return "-"
    except:
        return "-"


def main():
    parser = argparse.ArgumentParser(description='Generate comprehensive results table')
    parser.add_argument(
        '--output',
        type=str,
        default='docs/COMPREHENSIVE_TABLE.md',
        help='Output file path'
    )
    parser.add_argument(
        '--skip-flops',
        action='store_true',
        help='Skip FLOPs calculation (faster)'
    )
    parser.add_argument(
        '--skip-fps',
        action='store_true',
        help='Skip FPS calculation (faster)'
    )
    args = parser.parse_args()
    
    # Define all models to include
    models = [
        {
            'name': 'ChangerEx r18',
            'variant': 'baseline',
            'config': 'open-cd/configs/changer/changer_ex_r18_512x512_40k_levircd.py',
            'checkpoint': 'checkpoints_official/Open_CD/ChangerEx_r18-512x512_40k_levircd_20221223_120511.pth',
            'test_results': None,
            'test_log': 'results/opencd_integration/changer_ex_r18_baseline_official_test/20260209_132546/20260209_132546.log',
            'lr_schd': '40k'
        },
        {
            'name': 'ChangerEx r18',
            'variant': '+ CoSA',
            'config': 'open-cd/configs/changer/changer_ex_r18_512x512_20e_levircd_cosa.py',
            'checkpoint': 'results/opencd_integration/changer_ex_r18_cosa_x4_unfreeze_decode_20e_gamma1_lr10/best_mIoU_iter_11200.pth',
            'test_results': None,
            'test_log': 'results/opencd_integration/changer_ex_r18_cosa_test/20260207_193614/20260207_193614.log',
            'lr_schd': '20e'
        },
        {
            'name': 'ChangerEx s50',
            'variant': 'baseline',
            'config': 'open-cd/configs/changer/changer_ex_s50_512x512_40k_levircd.py',
            'checkpoint': 'checkpoints_official/Open_CD/ChangerEx_s50-512x512_40k_levircd_20220702-145628.pth',
            'test_results': None,
            'test_log': 'results/opencd_integration/changer_ex_s50_baseline_test/20260209_130405/20260209_130405.log',
            'lr_schd': '40k'
        },
        {
            'name': 'ChangerEx s50',
            'variant': '+ CoSA',
            'config': 'open-cd/configs/changer/changer_ex_s50_512x512_20e_levircd_cosa.py',
            'checkpoint': 'results/opencd_integration/changer_ex_s50_cosa_x4_unfreeze_decode_20e_gamma1_lr10/best_mIoU_iter_11200.pth',
            'test_results': None,
            'test_log': 'results/opencd_integration/changer_ex_s50_cosa_test.log',
            'lr_schd': '20e'
        },
        {
            'name': 'ChangerEx s101',
            'variant': 'baseline',
            'config': 'open-cd/configs/changer/changer_ex_s101_512x512_40k_levircd.py',
            'checkpoint': 'checkpoints_official/Open_CD/ChangerEx_s101-512x512_40k_levircd_20220710-082722.pth',
            'test_results': None,
            'test_log': 'results/opencd_integration/changer_ex_s101_baseline_test/20260209_130802/20260209_130802.log',
            'lr_schd': '40k'
        },
        {
            'name': 'ChangerEx s101',
            'variant': '+ CoSA',
            'config': 'open-cd/configs/changer/changer_ex_s101_512x512_20e_levircd_cosa.py',
            'checkpoint': 'results/opencd_integration/changer_ex_s101_cosa_x4_unfreeze_decode_20e_gamma1_lr10/best_mIoU_iter_11200.pth',
            'test_results': None,
            'test_log': 'results/opencd_integration/changer_ex_s101_cosa_test.log',
            'lr_schd': '20e'
        },
        {
            'name': 'STANet-BASE',
            'variant': 'baseline',
            'config': 'open-cd/configs/stanet/stanet_base_256x256_40k_levircd.py',
            'checkpoint': 'results/opencd_integration/stanet_baseline_retrain/best_mIoU_iter_40000.pth',
            'test_results': None,
            'test_log': 'results/opencd_integration/stanet_baseline_test/20260208_020217/20260208_020217.log',
            'lr_schd': '40k'
        },
        {
            'name': 'STANet-BASE',
            'variant': '+ CoSA',
            'config': 'open-cd/configs/stanet/stanet_base_256x256_20e_levircd_cosa.py',
            'checkpoint': 'results/opencd_integration/stanet_base_hparam_search/run_lr0.0005_gamma1.0/iter_1120.pth',
            'test_results': None,
            'test_log': 'results/opencd_integration/stanet_base_hparam_search/run_lr0.0005_gamma1.0_test.log',
            'lr_schd': '20e'
        },
        {
            'name': 'STANet-PAM',
            'variant': 'baseline',
            'config': 'open-cd/configs/stanet/stanet_pam_256x256_40k_levircd.py',
            'checkpoint': 'results/opencd_integration/stanet_pam_baseline/best_mIoU_iter_40000.pth',
            'test_results': None,
            'test_log': 'results/opencd_integration/stanet_pam_baseline_test/20260208_064342/20260208_064342.log',
            'lr_schd': '40k'
        },
        {
            'name': 'STANet-PAM',
            'variant': '+ CoSA',
            'config': 'open-cd/configs/stanet/stanet_pam_256x256_20e_levircd_cosa.py',
            'checkpoint': 'results/opencd_integration/stanet_pam_cosa_finetune_v2/best_mIoU_iter_11200.pth',
            'test_results': None,
            'test_log': 'results/opencd_integration/stanet_pam_cosa_test_v2.log',
            'lr_schd': '20e'
        },
        {
            'name': 'SNUNet',
            'variant': 'baseline',
            'config': 'open-cd/configs/snunet/snunet_c16_256x256_40k_levircd.py',
            'checkpoint': 'results/opencd_integration/snunet_baseline/best_mIoU_iter_40000.pth',
            'test_results': None,
            'test_log': 'results/opencd_integration/snunet_baseline_test/20260208_020218/20260208_020218.log',
            'lr_schd': '40k'
        },
        {
            'name': 'SNUNet',
            'variant': '+ CoSA',
            'config': 'open-cd/configs/snunet/snunet_c16_256x256_20e_levircd_cosa.py',
            'checkpoint': 'results/opencd_integration/snunet_cosa_finetune_v2/best_mIoU_iter_11200.pth',
            'test_results': None,
            'test_log': 'results/opencd_integration/snunet_cosa_test_v2.log',
            'lr_schd': '20e'
        },
        {
            'name': 'BIT',
            'variant': 'baseline',
            'config': 'open-cd/configs/bit/bit_r18_256x256_40k_levircd.py',
            'checkpoint': 'results/opencd_integration/bit_baseline_hparam_search/lr5em04_wd0.05/best_mIoU_iter_10000.pth',
            'test_results': None,
            'test_log': 'results/opencd_integration/bit_baseline_hparam_search/lr5em04_wd0.05_test.log',
            'lr_schd': '10k'
        },
        {
            'name': 'BIT',
            'variant': '+ CoSA',
            'config': 'open-cd/configs/bit/bit_r18_256x256_20e_levircd_cosa.py',
            'checkpoint': 'results/opencd_integration/bit_cosa_finetune_corrected/best_mIoU_iter_1120.pth',
            'test_results': None,
            'test_log': 'results/opencd_integration/bit_cosa_finetune_corrected/test_best.log',
            'lr_schd': '20e'
        },
        {
            'name': 'TinyCDv2-L',
            'variant': 'baseline',
            'config': 'open-cd/configs/tinycd_v2/tinycd_v2_l_256x256_40k_levircd.py',
            'checkpoint': 'results/opencd_integration/tinycdv2_l_baseline/best_mIoU_iter_40000.pth',
            'test_results': 'results/opencd_integration/tinycdv2_l_baseline/test_results/20260208_193431/20260208_193431.json',
            'test_log': None,
            'lr_schd': '40k'
        },
        {
            'name': 'TinyCDv2-L',
            'variant': 'Trial 7 (HParam)',
            'config': 'results/opencd_integration/tinycdv2_l_hyperparameter_tuning/trial_7/config.py',
            'checkpoint': 'results/opencd_integration/tinycdv2_l_hyperparameter_tuning/trial_7/best_mIoU_iter_20000.pth',
            'test_results': None,
            'test_log': 'results/opencd_integration/tinycdv2_l_hyperparameter_tuning/trial_7/test_best.log',
            'lr_schd': '20k'
        },
    ]
    
    print("=" * 70)
    print("Generating Comprehensive Results Table")
    print("=" * 70)
    print(f"Total models: {len(models)}\n")
    
    results = []
    
    for idx, model_info in enumerate(models, 1):
        name = model_info['name']
        variant = model_info['variant']
        config_path = Path(model_info['config'])
        checkpoint_path = Path(model_info['checkpoint']) if model_info.get('checkpoint') else None
        test_results_path = Path(model_info['test_results']) if model_info.get('test_results') else None
        
        print(f"[{idx}/{len(models)}] Processing: {name} ({variant})")
        
        if not config_path.exists():
            print(f"  ⚠️  Config not found: {config_path}")
            continue
        
        # Get backbone
        backbone = get_backbone_from_config(config_path)
        print(f"  Backbone: {backbone}")
        
        # Get parameters
        print("  Computing parameters...")
        try:
            params = get_model_params(config_path, str(checkpoint_path) if checkpoint_path else None)
            print(f"  Params: {params:.3f}M")
        except Exception as e:
            print(f"  ⚠️  Error computing params: {e}")
            params = None
        
        # Get GFLOPS
        flops = None
        if not args.skip_flops:
            print("  Computing GFLOPS...")
            try:
                flops = get_model_flops(config_path, input_shape=(256, 256))
                if flops:
                    print(f"  GFLOPS: {flops:.3f}")
                else:
                    print("  ⚠️  Could not compute GFLOPS")
            except Exception as e:
                print(f"  ⚠️  Error computing GFLOPS: {e}")
        
        # Get FPS
        fps = None
        fps_std = None
        if not args.skip_fps:
            if checkpoint_path and checkpoint_path.exists():
                print("  Computing inference speed...")
                try:
                    fps, fps_std = get_inference_fps(config_path, str(checkpoint_path))
                    if fps:
                        print(f"  FPS: {fps:.2f} ± {fps_std:.4f}")
                    else:
                        print("  ⚠️  Could not compute FPS")
                except Exception as e:
                    print(f"  ⚠️  Error computing FPS: {e}")
            else:
                print("  ⚠️  No checkpoint available for FPS computation")
        
        # Get test metrics
        print("  Extracting test metrics...")
        test_metrics = {}
        
        # Try JSON file first
        if model_info.get('test_results'):
            test_results_path = Path(model_info['test_results'])
            if test_results_path.exists():
                if test_results_path.suffix == '.json':
                    metrics = extract_test_metrics_from_json(test_results_path)
                    if metrics:
                        test_metrics.update(metrics)
                elif test_results_path.is_dir():
                    json_files = list(test_results_path.rglob('*.json'))
                    for json_file in json_files:
                        metrics = extract_test_metrics_from_json(json_file)
                        if metrics:
                            test_metrics.update(metrics)
                            break
        
        # Try log file
        if not test_metrics and model_info.get('test_log'):
            test_log_path = Path(model_info['test_log'])
            if test_log_path.exists():
                metrics = extract_test_metrics_from_log(test_log_path)
                if metrics:
                    test_metrics.update(metrics)
        
        # Fallback: try to find test results in directory
        if not test_metrics:
            test_results_val = model_info.get('test_results')
            if test_results_val and isinstance(test_results_val, str):
                test_results_dir = Path(test_results_val)
                if test_results_dir.exists() and test_results_dir.is_dir():
                    # Look for JSON files
                    json_files = list(test_results_dir.rglob('*.json'))
                    for json_file in json_files:
                        metrics = extract_test_metrics_from_json(json_file)
                        if metrics:
                            test_metrics.update(metrics)
                            break
                    
                    # Look for log files
                    if not test_metrics:
                        log_files = list(test_results_dir.rglob('*.log'))
                        for log_file in log_files:
                            metrics = extract_test_metrics_from_log(log_file)
                            if metrics:
                                test_metrics.update(metrics)
                                break
        
        if test_metrics:
            print(f"  Test metrics: Precision={test_metrics.get('Precision', 'N/A')}, "
                  f"Recall={test_metrics.get('Recall', 'N/A')}, "
                  f"F1={test_metrics.get('F1', 'N/A')}, "
                  f"IoU={test_metrics.get('IoU', 'N/A')}")
        else:
            print("  ⚠️  No test metrics found")
        
        results.append({
            'name': name,
            'variant': variant,
            'backbone': backbone,
            'lr_schd': model_info['lr_schd'],
            'params': params,
            'flops': flops,
            'fps': fps,
            'fps_std': fps_std,
            'precision': test_metrics.get('Precision'),
            'recall': test_metrics.get('Recall'),
            'f1': test_metrics.get('F1'),
            'iou': test_metrics.get('IoU'),
        })
        
        print()
    
    # Generate table
    print("=" * 70)
    print("Generating table...")
    print("=" * 70)
    
    # Create markdown table
    table_lines = [
        "# Comprehensive Results Table - LEVIR-CD Test Set",
        "",
        "| Method | Backbone | Lr Schd | Param (M) | GFLOPS | Inf (fps) | Precisionᶜ | Recallᶜ | F₁ᶜ | IoUᶜ |",
        "|--------|----------|---------|-----------|--------|-----------|-------------|---------|-----|------|"
    ]
    
    for r in results:
        method = f"{r['name']} ({r['variant']})"
        backbone = r['backbone']
        lr_schd = r['lr_schd']
        params = f"{r['params']:.3f}" if r['params'] else "N/A"
        flops = f"{r['flops']:.3f}" if r['flops'] else "N/A"
        
        if r['fps']:
            fps_str = f"{r['fps']:.2f}±{r['fps_std']:.4f}" if r['fps_std'] else f"{r['fps']:.2f}"
        else:
            fps_str = "N/A"
        
        precision = f"{r['precision']:.2f}" if r['precision'] else "N/A"
        recall = f"{r['recall']:.2f}" if r['recall'] else "N/A"
        f1 = f"{r['f1']:.2f}" if r['f1'] else "N/A"
        iou = f"{r['iou']:.2f}" if r['iou'] else "N/A"
        
        table_lines.append(
            f"| {method} | {backbone} | {lr_schd} | {params} | {flops} | {fps_str} | {precision} | {recall} | {f1} | {iou} |"
        )
    
    table_lines.extend([
        "",
        "**Notes:**",
        "- All metrics are for the **changed** class (Precisionᶜ, Recallᶜ, F₁ᶜ, IoUᶜ)",
        "- Inference speed (fps) measured on NVIDIA RTX A6000",
        "- Input size: 256×256 for training, 1024×1024 for testing",
        "- GFLOPS computed for 256×256 input",
    ])
    
    # Save table
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        f.write('\n'.join(table_lines))
    
    print(f"\n✅ Table saved to: {output_path}")
    
    # Also update FINAL_TABLE.md
    final_table_path = Path('docs/FINAL_TABLE.md')
    if final_table_path.exists():
        with open(final_table_path, 'w') as f:
            f.write('\n'.join(table_lines))
        print(f"✅ Updated: {final_table_path}")
    
    print("\n" + "=" * 70)
    print("Summary:")
    print("=" * 70)
    for r in results:
        print(f"{r['name']} ({r['variant']}): "
              f"IoU={r['iou']:.2f}% " if r['iou'] else "IoU=N/A ",
              f"F1={r['f1']:.2f}% " if r['f1'] else "F1=N/A ",
              f"Params={r['params']:.3f}M" if r['params'] else "Params=N/A")


if __name__ == '__main__':
    main()
