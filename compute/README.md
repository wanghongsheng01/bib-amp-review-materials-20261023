# AMP saved-output computational verification

Run from any directory with Python 3 (standard library only):

```bash
python3 verify.py
```

The command validates all 37 packaged input hashes, independently recalculates saved-output statistics, writes CSV tables and JSON receipts to `results/`, and exits nonzero if a check fails. It does not import the historical model implementation, train models, perform inference, load checkpoints, access the network, or open a reserved evaluation partition.

The supplied run passes **2,630 checks**. The input is a minimal, field-selected derivative of fixed source commit `f1b748d21386e1d1683a0b6e92cbcf9a208a4d2b` of `wanghongsheng01/amp-claude-handoff-20261013`. `input_manifest.json` binds each packaged input to its original source URL, original byte hash, original Git blob hash, packaged byte hash, and transformation. Original numerical and sequence field values are preserved. Historical machine paths and operational instructions are omitted.

## Scope of reproduction

- Reconstruct residue counts, per-amino-acid counts, length errors, and exact-match counts from 960 saved train/development reconstruction records.
- Recompute all 96 distribution distances from 128 saved generated sequences and the saved train/development reference sequences. Produce the 24 aligned comparator rows. The maximum absolute difference from the source is `2.4424906541753444e-15`.
- Recompute validity, uniqueness, exact training-copy counts, and pooled entropy differences.
- Check matched initialization metadata and all 1,000 paired VAE batch-index, forward-RNG-hash, and learning-rate records.
- Recompute all six error statistics per model/unit group from 255 native-unit and 588 censored-experiment development predictions.
- Extract all seven follow-up routes without conflating nonapplicable diagnostic comparisons, reconstruction guards, information gates, generation gates, and mechanism screens. The saved EB ledger contains 384 attempts, including three empty outputs; attempts are not silently equated with valid samples.

This is **saved-output reproducibility**, not independent training reproduction. Source metadata, training success, prospective timing, model checkpoint contents, data acquisition, and original cohort construction have not been independently recreated by this package. One fitted configuration/seed and reused development data do not support independent-replicate inference. Activity predictions concern existing activity observations, not validation of newly generated peptides.

## Outputs

| Files under `results/` | Content |
|---|---|
| `QA.json`, `verification.json`, `sequence_prediction_verification.json` | Check totals, individual outcomes, source hashes |
| `S1_reconstruction_train_dev.csv`, `S1_reconstruction_per_amino_acid.csv` | Reconstruction by split and residue class |
| `S2_generation_all_96_distances.csv`, `S2_generation_24_comparisons.csv` | Complete reported distribution comparisons |
| `independent_generation_96_distances.csv` | Independent sequence-derived values and absolute differences |
| `S3_generation_complementary.csv`, `independent_generation_complementary.csv` | Eligibility, copying, uniqueness, entropy diagnostics |
| `S4_seven_routes.csv` | Actual route-specific gate meanings and outcomes |
| Other `S4_*.csv/json` | Full source substitution, warmup, AR-latent, direct-z, and EB metrics |
| `S7_*.csv`, `S8_*.csv`, `S9_*.csv` | Native activity, censored likelihood, variance decomposition |
| `independent_activity_predictions_aggregated.csv` | All activity-error statistics independently recalculated |
| `model_parameter_scope.csv`, `LSTM_development_NLL_by_update.csv` | Capacity scope and fixed-endpoint monitor trajectory |

`reference/` contains static source excerpts documenting the implemented loss, decoder and comparator architecture. These excerpts are never executed. `audit/` contains the independently written verification programs executed by the single entry point.
