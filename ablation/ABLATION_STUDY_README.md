# Ablation Study Guide

This guide explains how to run the ablation study for CoSA-CD models.

## Overview

The ablation study compares different model variants:
- **B0 (baseline)**: Siamese U-Net baseline
- **B1 (attention)**: Siamese U-Net + Attention-only (no correlation)
- **B2 (cosa)**: CoSA v2 - single-scale, fixed gate
- **B2 (cosa_v3)**: CoSA v3 - multi-scale, learnable residual gate
- **Ablation: cosa_multiscale_only**: Multi-scale only (without learnable gate)
- **Ablation: cosa_learnable_gate_only**: Learnable gate only (without multi-scale)
- **B3 (aligned)**: Alignment-first (aligns T2 to T1 before differencing)

## Quick Start

### Option 1: Python Script (Recommended)

```bash
# Set your dataset directory
export DATASET_DIR="/path/to/LEVIR-CD"

# Run all variants
python scripts/run_ablation_study.py \
    --dataset_dir "$DATASET_DIR" \
    --output_dir ablation/ablation_study \
    --batch_size 8 \
    --epochs 100 \
    --lr 1e-4

# Or run specific variants only
python scripts/run_ablation_study.py \
    --dataset_dir "$DATASET_DIR" \
    --output_dir ablation/ablation_study \
    --variants baseline attention cosa_v3

# Skip training and only generate summary from existing results
python scripts/run_ablation_study.py \
    --dataset_dir "$DATASET_DIR" \
    --output_dir ablation/ablation_study \
    --skip_training
```

## Output Structure

After running, you'll have:

```
ablation/ablation_study/
├── baseline/
│   ├── checkpoint_best.pth
│   └── training.log
├── attention/
│   ├── checkpoint_best.pth
│   └── training.log
├── cosa_v3/
│   ├── checkpoint_best.pth
│   └── training.log
├── ...
├── ablation_summary.txt      # Human-readable summary
└── ablation_summary.json     # Machine-readable results
```

## Results Summary

The script automatically generates:
1. **ablation_summary.txt**: Formatted table with all metrics
2. **ablation_summary.json**: JSON file with detailed results for further analysis

Example summary:
```
Variant                        |      F1 |     IoU |  Precision |   Recall
----------------------------------------------------------------------
baseline                       |   85.23 |   74.56 |     82.45 |   88.12
attention                      |   86.45 |   75.89 |     83.67 |   89.34
cosa_v3                        |   89.12 |   78.45 |     86.23 |   92.15
...
```

## Customization

### Training Parameters

You can customize training parameters:

```bash
python scripts/run_ablation_study.py \
    --dataset_dir "$DATASET_DIR" \
    --output_dir ablation/ablation_study \
    --batch_size 16 \
    --epochs 150 \
    --lr 5e-5 \
    --base_size 256 \
    --seed 123
```

### Running Specific Variants

```bash
# Run only baseline and CoSA v3
python scripts/run_ablation_study.py \
    --dataset_dir "$DATASET_DIR" \
    --variants baseline cosa_v3
```

## Notes

- All variants use the same random seed (default: 42) for reproducibility
- Training uses mixed precision (AMP) by default for faster training
- Each variant is trained independently - you can run them in parallel if you have multiple GPUs
- Results are saved after each variant completes, so you can resume if interrupted
- The script auto-detects `train_cd_fixed.py` from either `scripts/train/` (inside this repo) or `../scripts/train/` (parent workspace).

## Troubleshooting

### Dataset not found
Make sure your dataset directory contains the standard LEVIR-CD structure:
```
LEVIR-CD/
├── splits/
│   ├── train.txt
│   ├── val.txt
│   └── test.txt
├── train/
│   ├── A/
│   ├── B/
│   └── label/
├── val/
└── test/
```

### Out of Memory
Reduce batch size:
```bash
python scripts/run_ablation_study.py \
    --dataset_dir "$DATASET_DIR" \
    --batch_size 4
```

### Resume from existing results
If training was interrupted, you can generate a summary from existing results:
```bash
python scripts/run_ablation_study.py \
    --dataset_dir "$DATASET_DIR" \
    --output_dir ablation/ablation_study \
    --skip_training
```
