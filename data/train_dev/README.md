# Train and development inputs

These four CSVs are byte-identical to the inputs bound in the executed formal configuration. Training contains 279 peptides; development contains 41. `train_descriptors.csv` and `dev_descriptors.csv` are model inputs. `train_cohort.csv` and `dev_cohort.csv` supply sequence, terminal-state and source provenance for the corresponding rows. There is no reserved test partition in this directory.

| Column | Meaning |
|---|---|
| `idx` | Input row index (training begins at 0; development at 279) |
| `protein` | Amino-acid sequence |
| `describe` | Per-residue triplets of numeric descriptors, preserved in original encoding |
| `point_describe` | Nine finite global descriptors in JSON-array encoding |
| `class` | Terminal class; the implementation maps FREE to 0 and AMD to 1 |
| `molecule_sha256`, `sequence_sha256` | Original molecule/sequence identifiers |
| `sequence`, `length`, `terminal_state` | Cohort sequence, residue count, terminal chemistry |
| `component_id`, `source_ids` | Original cohort component and source-record provenance |

The strict input parser is retained at `model/source/corrected_code/AMP_VAEflow/tasks/dataset_loader/dataset_utils/amp_input_contract.py`. Training-derived feature scaling is implemented in `model/implementation/vae_pipeline_formal_v2.py`; the recorded scaler files are in `model/scalers/`. Data provenance, descriptor definitions and the study's scientific eligibility criteria are described in the manuscript and supplement; row-format validation alone does not establish those criteria.
