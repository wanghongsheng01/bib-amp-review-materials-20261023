# Figure rendering

The four Python files separately render Figures 1–4 in `../figures/` and create diagnostic previews in `../qa/`. They do not run experiments or edit manuscript prose. Figures 1–3 require matplotlib and Pillow. Figure 4 requires ReportLab, Pillow, pypdf, pdfplumber, DejaVu Sans, and the optional Poppler command-line tools for its built-in visual preflight. Set FIGURE_FONT_DIR if DejaVu Sans is not available in a default location. The empirical values are bound to the accompanying computed tables; Figure 4 preserves distinct reported and retrospective endpoints.

Run one renderer at a time from any directory, for example `python3 manuscript/figure_source/render_figure3.py`. Image regeneration may change file metadata without changing numeric content.
