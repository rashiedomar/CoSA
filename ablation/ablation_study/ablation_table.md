# Ablation Study Results

## LEVIR-CD Dataset

| Variant | F1 (%) | IoU (%) | Precision (%) | Recall (%) |
|---------|--------|---------|---------------|------------|
| B0: Baseline | 87.70 | 78.09 | 90.23 | 85.31 |
| B1: Attention-only | 87.88 | 78.38 | 90.27 | 85.61 |
| B2 (v2): CoSA single-scale | 88.23 | 78.94 | 93.88 | 83.22 |
| **B2 (v3): CoSA multi-scale** | **89.45** | **80.91** | 90.39 | **88.52** |
| Ablation: Multi-scale only | 87.35 | 77.54 | 91.35 | 83.68 |
| Ablation: Learnable gate only | 87.68 | 78.06 | 93.33 | 82.67 |
| B3: Alignment-first | 88.70 | 79.69 | 92.00 | 85.62 |

**Note:** All experiments used batch_size=8 for fair comparison.
