# IEEE Access manuscript (CoSA)

This folder is **self-contained** for **Overleaf** or local `pdflatex` / Tectonic.

## Files

- **`main.tex`** — root document (set this as the Overleaf main file).
- **`sections/`** — `\input` sections.
- **`tables/`** — table sources.
- **`figures/`** — all `\includegraphics` assets used by the PDF.
- **`refs.bib`** — bibliography.
- **`ieeeaccess.cls`**, **`IEEEtran.cls`**, **`IEEEtran.bst`**, font metric files — required for IEEE Access layout.

## Build

```bash
tectonic main.tex
# or: pdflatex → bibtex → pdflatex ×2
```

## Repo URL in the PDF

The title-page footnote points to: **https://github.com/rashiedomar/CoSA**

## Order of figures/tables

See **`FIGURES_TABLES_ORDER.md`** (editorial reference only).
