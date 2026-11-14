# E4 report section (IEEE format)

| File | What it is |
|---|---|
| `e4_section.tex` | The section body, no preamble. This is the file to merge. |
| `e4_refs.bib` | Its ten references. |
| `figures/*.pdf` | Six vector figures. |
| `make_figures.py` | Regenerates the figures from `results/`. |
| `main.tex`, `main.pdf` | Standalone IEEEtran wrapper and its output (7 pages), for reading the section by itself. |

## Merging into the team report

1. Copy `e4_section.tex`, `e4_refs.bib` and `figures/` next to the main file (or into a subfolder).
2. The preamble needs `cite`, `amsmath`, `graphicx`, `booktabs` and `url`.
3. Where the section should appear:

   ```latex
   \newcommand{\EFourFigDir}{e4/figures}   % only if the figures are not in ./figures
   \input{e4/e4_section}
   ```

4. Add `e4_refs` to the `\bibliography{...}` list, or paste its entries into the team's `.bib`.

All labels start with `sec:e4`, `tab:e4`, `fig:e4` or `eq:`, and the two helper macros (`\EFourFigDir`, `\cfg`) are defined with `\providecommand`, so nothing clashes if the main file already defines them. The section opens with one sentence that refers to "the previous experiments"; adjust it if E4 is not placed after E1 to E3.

## Rebuilding

```bash
python report/e4/make_figures.py
cd report/e4 && pdflatex main && bibtex main && pdflatex main && pdflatex main
```
