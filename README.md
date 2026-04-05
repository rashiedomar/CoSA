# CoSA — Correlation-Guided Change Attention

Official **paper + experiment configs** release for **CoSA** (Context Sampling Attention): a lightweight decoder-side refinement module for remote sensing change detection.

**Repository:** [https://github.com/rashiedomar/CoSA](https://github.com/rashiedomar/CoSA)

This tree is a **clean export** for **public GitHub** and **submission**: LaTeX paper, **configuration files** for training/benchmarks, scripts, and compact ablation summaries—**no** bundled model Python package or checkpoints.

---

## Layout

| Path | Contents |
|------|-----------|
| **`paper/`** | IEEE Access LaTeX project: `main.tex`, `sections/`, `tables/`, `refs.bib`, `figures/`, class files. **Use this for Overleaf** (see `OVERLEAF.md`). |
| **`configs/`** | **All experiment definitions** used with Open-CD / MMEngine-style runs: `configs/custom/` (baseline + CoSA) and `configs/opencd/` (model recipes). These are Python config files (not a separate `models/` tree). |
| **`scripts/`** | Figure/table generation and analysis helpers (optional; may expect a local Open-CD checkout for imports). |
| **`figures/`** | Extra plots used by scripts (not all are in `paper/figures/`). |
| **`ablation/`** | README + compact summaries (`*.json`, `*.md`, `visualizations/`). |
| **`LICENSE`** | MIT |
| **`CITATION.cff`** | Citation metadata |

---

## Quick start

- **Paper only:** see **`OVERLEAF.md`** and zip **`paper/`** for Overleaf.
- **Reproduce training:** use an [Open-CD](https://github.com/likyoo/open-cd) (or compatible) environment, point dataset paths in **`configs/`**, and run with the same framework versions you used in the lab. **Checkpoints and full training code are not shipped** in this repo to keep it small and license-clean.

```bash
pip install -r requirements.txt   # minimal; add torch / mmcv per your stack
```

---

## What is intentionally excluded

- **`models/`** (standalone training scripts and local checkpoints from development).  
- Raw datasets, `work_dirs/`, and `*.pth` weights.  
- Multi‑gigabyte ablation run directories (only summaries + small visuals are included).

---

## Citation

See **`CITATION.cff`**. Cite the IEEE Access article when DOI/issue are final.

---

## Authors

Abdirashid Omar · Jonghyuk Park — Kookmin University, Seoul, Republic of Korea
