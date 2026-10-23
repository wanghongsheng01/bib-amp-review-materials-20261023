# AMP laboratory evidence

`mic_summary_reported.csv` preserves the supplied MIC summary. `od600_raw.csv` contains observed optical densities. `mic_algorithm_calls.csv` is a retrospective algorithmic interpretation, not an adjudication of the true biological MIC. Its legacy N code is used only to reproduce the historical 105/108 comparison. `algorithm_status` distinguishes numeric consensus, no-endpoint majority and no consensus. No source measurements were overwritten.

`threshold_counts_all.csv` reports both strict and inclusive comparisons for every dilution concentration plus manuscript thresholds, for all four strains and any-strain activity. Numerators count qualifying numeric endpoints; N is not imputed as zero or a verified negative.

`sample_manifest_30.csv` joins the unchanged 30-sequence order file to 27 COAs and 27 assay identifiers. Ordered groups contain 8/8/14; assayed groups contain 7/8/12. The order source encodes group numbers. Biological group labels are retained as manuscript assignments with their evidence level explicit. Missing IDs do not establish the reason for non-delivery.

`coa_27_facts.csv` and `solubility_27_facts.csv` contain supplier-reported scientific facts. Ten peptides were insoluble in water and eighteen in DPBS in the supplier's qualitative test. These results do not establish the actual assay solvent. The repository includes extracted scientific certificate facts; original signed supplier PDFs are retained by the authors.

The original XLSX is included unchanged. It contains no financial or contact fields. File hashes and source descriptions are in MANIFEST.json. Source discussion logs, cloud account IDs, auth receipts and private filesystem paths are excluded.

This package does not establish an assay-to-model checkpoint link, prospective choice of assay criteria, full CLSI conformity, or clinical safety. It contains no new laboratory measurements.
