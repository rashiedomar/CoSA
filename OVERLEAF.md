# Overleaf — IEEE Access paper

## Upload

Use the **`paper/`** directory as a **single self-contained** project:

- Drag **`paper/`** into Overleaf, **or** zip it:

```bash
zip -r CoSA-paper.zip paper
```

## Settings

1. **Main document:** `main.tex`
2. **Compiler:** **pdfLaTeX** (recommended).
3. **Build:** pdfLaTeX → **BibTeX** → pdfLaTeX → pdfLaTeX.

Graphics live under **`paper/figures/`** (`\graphicspath{{figures/}}` in `main.tex`).

## Local build (optional)

```bash
cd paper
tectonic main.tex
# or pdflatex + bibtex + pdflatex ×2
```

Do not reference files **outside** `paper/` — Overleaf will not see them.
