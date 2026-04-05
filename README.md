# CoSA — Correlation-Guided Change Attention

Official **code + paper** release for **CoSA** (Context Sampling Attention): a lightweight decoder-side refinement module for remote sensing change detection.

**Repository:** [https://github.com/rashiedomar/CoSA](https://github.com/rashiedomar/CoSA)

This tree is a **clean export** (paper, configs, models, scripts, small ablation summaries). It is meant to replace a cluttered development checkout for **public GitHub** and **submission**.

---

## Layout

| Path | Contents |
|------|-----------|
| **`paper/`** | IEEE Access LaTeX project: `main.tex`, `sections/`, `tables/`, `refs.bib`, `figures/`, class files. **Use this for Overleaf** (see `OVERLEAF.md`). |
| **`configs/`** | Training configs (`configs/custom/` baseline + CoSA; `configs/opencd/` model recipes). |
| **`models/`** | CoSA-related code: FC-Siam / Open-CD paths, STANet/BIT standalone experiments. |
| **`scripts/`** | Figure/table generation and analysis helpers. |
| **`figures/`** | Extra plots used by scripts (not all are in `paper/figures/`). |
| **`ablation/`** | README + **compact** ablation summaries (`*.json`, `*.md`, `visualizations/`). Full per-run checkpoints and multi‑GB masks are **not** shipped here. |
| **`LICENSE`** | MIT |
| **`CITATION.cff`** | Citation metadata |

---

## Quick start (code)

```bash
# Python deps (adjust torch for your CUDA)
pip install -r requirements.txt
```

Point your experiments at **`configs/custom/`** and the **`models/`** package layout you use with Open-CD (or your fork). Paths assume you clone datasets separately; see configs for dataset roots.

---

## Paper (Overleaf / Monday)

1. Zip **`paper/`** or drag the folder into Overleaf.  
2. Main document: **`paper/main.tex`**.  
3. Details: **`OVERLEAF.md`**.

---

## What is intentionally excluded

- Raw datasets and `work_dirs/` / checkpoints (large binaries).  
- Multi‑gigabyte ablation run directories from the lab machine (only summaries + small visuals are included).  
- Scratch drafts and duplicate `paper/` trees from development repos.

---

## Citation

See **`CITATION.cff`**. Cite the IEEE Access article when DOI/issue are final.

---

## Authors

Abdirashid Omar · Jonghyuk Park — Kookmin University, Seoul, Republic of Korea
