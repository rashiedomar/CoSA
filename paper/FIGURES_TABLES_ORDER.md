# Figures and Tables Order (Quick Guide)

This file lists all figures and tables **in paper order** (first to last), so you can review structure without reading the full manuscript.

Note: page numbers below are from the current `paper/main.pdf` and may shift slightly after recompilation.

## Figures (in order)

| Order | Figure No. | Section | Caption (short) | Source file | Current page |
|---|---|---|---|---|---|
| 1 | Fig. 1 | Introduction | CoSA qualitative intuition | `paper/sections/01_introduction.tex` (`figures/example.png`) | 2 |
| 2 | Fig. 2 | Method | FC-Siam-diff + CoSA architecture | `paper/sections/02_method.tex` (`figures/model_architecture.png`) | 4 |
| 3 | Fig. 3 | Experimental Setup | Training dynamics (baseline vs CoSA) | `paper/sections/03_experiments.tex` (`figures/training_validation_metrics_baseline_cosa.png`) | 4 |
| 4 | Fig. 4 | Experimental Setup | Qualitative baseline vs CoSA collage | `paper/sections/03_experiments.tex` (`figures/hero_collage_fcsiam.png`) | 5 |
| 5 | Fig. 5 | Ablation Study | Quantitative ablation summary (F1-only) | `paper/sections/05_ablation.tex` (`figures/ablation_quanitiative.png`) | 6 |
| 6 | Fig. 6 | Ablation Study | Qualitative ablation comparison | `paper/sections/05_ablation.tex` (`figures/ablation_qualitative.png`) | 6 |

**Alternative ablation qualitative layouts** (choose one if you prefer a different visualization):

- `figures/ablation_qualitative_alt1_row_per_variant.png` — 6×3: one row per variant (GT, Baseline, …, CoSA v3), 3 samples per row.
- `figures/ablation_qualitative_alt2_compact.png` — 3×3: only GT | Baseline | CoSA v3 (main comparison).
- `figures/ablation_qualitative_alt3_errors_only.png` — same 3×6 as current, but only FP/FN (no green TP).
- `figures/ablation_qualitative_alt4_improvement.png` — 3×2: CoSA v3 pred | (CoSA v3 − Baseline) difference (green = gain, red = FP).

Generate with:  
`python scripts/visualize_ablation_alternatives.py --dataset_dir /path/to/LEVIR-CD [--output_dir paper/figures] [--methods alt1 alt2 alt3 alt4]`

## Tables (in order)

| Order | Table No. | Section | Caption (short) | Source file | Current page |
|---|---|---|---|---|---|
| 1 | Table I | Experimental Setup | FC-Siam baseline vs FC-Siam+CoSA | `paper/tables/table1_fcsiam_baseline_vs_cosa.tex` | 3 |
| 2 | Table II | Ablation Study | Ablation variants on LEVIR-CD | `paper/tables/table2_ablation_variants.tex` | 3 |
| 3 | Table III | Standalone Hero Models | STANet/BIT under native pipelines | `paper/tables/table3_hero_models.tex` | 7 |
| 4 | Table IV | Analysis | Efficiency overhead of adding CoSA | `paper/tables/table6_efficiency_overhead.tex` | 7 |
| 5 | Table V | Analysis | Error mechanics + patch-level consistency (combined) | `paper/tables/table4_error_mechanics_and_table5_consistency_combined.tex` | 7 |
| 6 | Table VI | Open-CD Benchmarks | Open-CD benchmark subset (LEVIR-CD, changed class) | `paper/tables/comprehensive_results_table.tex` | 8 |

## Not currently included in main paper

These table files exist but are **not currently referenced** by `paper/main.tex`:

- `paper/tables/table4_error_mechanics.tex`
- `paper/tables/table5_consistency.tex`
- `paper/tables/table6_changerex_focused.tex`
