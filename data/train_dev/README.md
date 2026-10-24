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

The strict input parser is retained at `model/source/corrected_code/AMP_VAEflow/tasks/dataset_loader/dataset_utils/amp_input_contract.py`. Training-derived feature scaling is implemented in `model/implementation/vae_pipeline_formal_v2.py`; the recorded scaler files are in `model/scalers/`. Data provenance and the study's scientific eligibility criteria are described in the manuscript and supplement; row-format validation alone does not establish those criteria. The ordered descriptor definitions are recorded in [descriptor_dictionary.json](descriptor_dictionary.json) and the hash-bound [molecular representation source](source/molecular_representation.py). The source was inspected without execution; the four frozen CSV inputs were not recomputed or altered in this revision.

The residue triplet order is `MolWt`, `MolLogP`, `TPSA`. Each canonical L residue is represented as an isolated free amino acid; these RDKit descriptor outputs are rounded to two decimal places and the same residue lookup is used for both peptide terminal states.

The global order is `RingCount`, `NumAromaticRings`, `NumSaturatedRings`, `FractionCSP3`, `HallKierAlpha`, `LabuteASA`, `MaxPartialCharge`, `MinPartialCharge`, `TPSA`. These are native outputs of the correspondingly named `rdkit.Chem.Descriptors` functions on the full peptide, without manual unit conversion or rounding in this source. `Chem.MolFromSequence(..., flavor=0)` creates the free-terminal L-peptide; AMD replaces the terminal OXT oxygen with nitrogen and validates the amide, element counts, exact-mass change and chirality. The source represents database-declared chemistry rather than an independent chemical measurement.

The descriptor source SHA256 is `4d0d6abb6cb7c49efe69a387270dbd44e49e6749c8515ce2d844ccdc4a717dd4`. The formal reconstruction and warmup scoring contracts bind that source. This source-level documentation provides descriptor semantics; validation in this revision did not independently regenerate each frozen descriptor value.
