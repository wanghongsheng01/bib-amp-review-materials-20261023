# Figure rendering

The four Python files separately render Figures 1–4 in `../figures/` and create diagnostic previews in `../qa/`. They do not run experiments or edit manuscript prose. Figures 1–3 require matplotlib and Pillow. Figure 4 requires ReportLab, Pillow, pypdf, pdfplumber, DejaVu Sans, and the optional Poppler command-line tools for its built-in visual preflight. Set FIGURE_FONT_DIR if DejaVu Sans is not available in a default location. The empirical values are bound to the accompanying computed tables; Figure 4 maps three separate evaluation tasks to their inputs, measurements and outcomes. It does not link the experimental panel to a CE21/MSE21 sampling run.

Run one renderer at a time from any directory, for example `python3 manuscript/figure_source/render_figure3.py`. Image regeneration may change file metadata without changing numeric content.

Figure 2 reads `supplement/tables/Table_S4e_AR_information.csv` and cross-checks it against the packaged latent-AR summary; it writes `fig2_latent_controls.pdf` and `.svg`. It shows reconstruction scores, not generation improvements. Set `FIGURE2_QA_DIR` or `FIGURE4_QA_DIR` to keep local previews outside the release directory.

Version v6 uses deterministic point marks for Figures 1 and 2. The Figure 2 first three scores are eight-draw equal-sequence residue accuracies (S8); comparator C is one plain decode without latent input and has unmatched capacity. All plotted numerical values are preserved. No generative image tool was used.
