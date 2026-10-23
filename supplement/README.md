# Supplementary material

`main.tex` is the editable English source. `supplement.pdf` is the compiled supplement. `table_S1.tex` through `table_S6.tex` are deterministic display tables; the full-precision editable CSVs are in `tables/`. The tables are grouped by S1–S6; suffixes identify complete components rather than new studies.

| Family | Complete contents |
|---|---|
| S1a–S1c | 6 reconstruction aggregate rows, 960 per-peptide records, 120 per-amino-acid rows |
| S2a–S2e | 96 original distances, 24 aligned comparisons, 96 recomputed distances, 8 eligibility/entropy rows, 8 validity/copying rows |
| S3a–S3e | 6 native activity rows, 8 censored activity rows, 12 contrasts, 2 variance rows, 14 independently recomputed rows |
| S4a–S4g | 7 route classifications, 96 source-substitution distances, 5 warmup guards, 10 paired warmup statistics, 8 information entries, 2 direct-latent training rows, 12 conditioning-context rows |
| S5a–S5h | 30 order/assay rows, 108 MIC comparisons, 3 differences, 3,240 OD rows, 27 supplier fact rows, 320 threshold comparisons, 108 N-semantics rows, 6 design-group rows |
| S6a–S6b | 8 population/endpoint comparison rows, 27 peptide-specific molar-to-mass conversions |

The PDF prints compact complete comparisons and clearly identifies the accompanying full row-level CSV files. The 960 reconstruction rows and 3,240 OD rows remain complete in the CSV attachments; they are not silently sampled for presentation.

## Reproduce

From the repository root:

```bash
python3 verify_all.py
python3 supplement/verify_supplement.py
```

The first command verifies the computational and laboratory data packages. The second verifies all supplementary CSV hashes, source bindings, six architecture-source hashes and row-level reconstruction/OD arithmetic. The supplied supplement check passes 4,802 checks. It does not train, infer, deserialize checkpoints or connect to a network.

To regenerate the tables from the repository's `compute/` and `wetlab/` records:

```bash
python3 supplement/build_tables.py
```

To compile from editable sources:

```bash
cd supplement
pdflatex -interaction=nonstopmode -halt-on-error -jobname=supplement main.tex
pdflatex -interaction=nonstopmode -halt-on-error -jobname=supplement main.tex
```

The document uses standard LaTeX packages and Latin Modern fonts. `source/architecture_facts.json` records the executed encoder, decoder, prior transform and DDPM structure with source-line and hash bindings. `table-provenance.json` contains exact row counts, original paths, byte hashes and transformations. Historical source paths are retained as provenance; the verifier maps the original preparation layout to the repository's `compute/` and `wetlab/` layout. `source/AMPGen_publisher_facts.json` contains only targeted published facts and an official source locator, not a downloaded full article.

## Interpretation

Original reported MIC values are preserved. The retrospective OD rule is a separate algorithmic calculation; it does not adjudicate true MIC. Three differences are located, with original and algorithm values both retained. Source N, no-endpoint-majority and no-consensus are distinguished. No model checkpoint, configuration and seed is assigned to an assayed peptide without evidence.

Computational checks reproduce saved-output arithmetic, not a new training run. The seven follow-up routes keep their original diagnostic, guard, information, generation and mechanism-screen meanings. The AMPGen initial 31/38 result uses a different assessment population from its nine-peptide four-strain follow-up.
