# Supplementary material

`main.tex` is the editable English source for **Separating reconstruction, generation and antibacterial activity in antimicrobial peptide design**. `supplement.pdf` is the compiled supplement. `table_S1.tex` through `table_S6.tex` are deterministic display tables. Full-precision scientific data and retained source archives are in `tables/`.

The current paper displays all 27 assayed peptides in Table S5a and both MIC representations for all 108 peptide–strain combinations in S5b. Table S5c gives the three differing endpoints and their replicate calls. Table S6 describes the published AMPGen workflow as a supporting experimental-design example. The current computational comparator is the locally trained LSTM; AMPGen is not a reproduced baseline.

| Family | Complete contents |
|---|---|
| S1a–S1c | 6 reconstruction aggregate rows, 960 per-peptide records, 120 per-amino-acid rows |
| S2a–S2e | 96 original distances, 24 aligned comparisons, 96 recomputed distances, 8 eligibility/entropy rows, 8 validity/copying rows |
| S3a–S3e | 6 native activity rows, 8 censored activity rows, 12 contrasts, 2 variance rows, 14 independently recomputed rows |
| S4a–S4g | 7 route classifications, 96 source-substitution distances, 5 archived warmup/decision entries, 10 paired warmup statistics, 8 information entries, 2 direct-latent training rows, 12 conditioning-context rows; the PDF displays the four scientific reconstruction criteria |
| S5a–S5h source archive | Complete 30-candidate source inventory, 108 paired MIC cells, 3 differences, 3,240 OD rows, 27 characterization rows, 320 threshold rows, 108 N-semantics rows and 6 design-group rows; the PDF displays the complete 27-peptide assayed cohort |
| Legacy S6a–S6b CSV archive | 8 prior-revision population/endpoint comparison rows and 27 prior-revision molar-to-mass conversions; retained contextual calculations, not the current paper's comparison or its displayed Table S6 |

The original 30-candidate source inventory (`Table_S5a_order_delivery_manifest.csv`) is preserved byte-for-byte for provenance. The assayed-cohort table filters only on the existing `assayed == True` field; no assayed peptide is excluded. Procurement history is not an endpoint of the scientific paper. Original design-group labels and historical denominator calculations remain archived source data, without being used as activity or specificity claims in the current paper.

The legacy `Table_S6a_AMPGen_comparison.csv` and `Table_S6b_molar_mass_equivalents.csv` are archived prior-revision contextual calculations. Their historical filenames and values are retained for reproducibility; the current Table S6 is the workflow presentation rendered from the targeted published AMPGen facts. It does not juxtapose AMPGen and present-study hit rates, and it does not treat molecular-weight conversions for present-study peptides as AMPGen measurements.

The 960 reconstruction rows and 3,240 OD rows remain complete in the CSV attachments. `table-provenance.json` and all 30 CSV files are unchanged by the editorial presentation revision.

## Reproduce

From the repository root:

```bash
python3 verify_all.py
python3 supplement/verify_supplement.py
```

The first command verifies the computational and laboratory data packages. The second verifies all supplementary CSV hashes, source bindings, six architecture-source hashes and row-level reconstruction/OD arithmetic. The supplied supplement check passes 4,802 checks. It does not train, infer, deserialize checkpoints or connect to a network.

To regenerate only the revised S4–S6 presentation tables without writing any CSV or provenance files:

```bash
python3 supplement/build_tables.py --editorial-display-only
```

To regenerate the full data archive and all display tables from the repository's `compute/` and `wetlab/` records:

```bash
python3 supplement/build_tables.py
```

To compile from editable sources:

```bash
cd supplement
pdflatex -interaction=nonstopmode -halt-on-error -jobname=supplement main.tex
pdflatex -interaction=nonstopmode -halt-on-error -jobname=supplement main.tex
```

The document uses standard LaTeX packages and Latin Modern fonts. `source/architecture_facts.json` records the executed encoder, decoder, prior transform and DDPM structure with source-line and hash bindings. `table-provenance.json` contains exact archive row counts, original paths, byte hashes and transformations. Historical source paths are retained as provenance; the verifier maps the original preparation layout to the repository's `compute/` and `wetlab/` layout. `source/AMPGen_publisher_facts.json` contains targeted published facts and an official source locator.

The model implementation derives from [3DHumanRehab/AMP](https://github.com/3DHumanRehab/AMP), commit `e7486fb714cbf5c663cc01052e42fb63d4ba3a40`; wrappers and configurations derive from [wanghongsheng01/amp-claude-handoff-20261013](https://github.com/wanghongsheng01/amp-claude-handoff-20261013), commit `f1b748d21386e1d1683a0b6e92cbcf9a208a4d2b`.

## Interpretation

Original MIC values and retrospective endpoints are preserved separately. Source N, no-endpoint-majority and no-consensus retain their distinct meanings. Activity fractions in the current paper use all 27 assayed candidates. The assay panel is a separate experiment from the reconstruction-objective comparison and is not attributed to an evaluated checkpoint.

Computational checks reproduce saved-output arithmetic, not a new training run. The seven follow-up routes keep their original scientific metrics and route-specific interpretation. The source archive retains historical project-decision fields and the incomplete safe-loader check for direct-latent diffusion; those execution records are not scientific endpoints in the displayed table. AMPGen's initial 31/38 result and nine-peptide four-strain assessment remain distinct stages of its published workflow.
