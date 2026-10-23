# Model implementation and frozen settings

The six files in `implementation/` retain their original bytes and hashes. `vae_pipeline_formal_v2.py` implements the categorical-CE, raw-one-hot-MSE and scalar VAE/DDPM arms; `ar_pipeline.py` with `ar_entry.py` implements the terminal-conditioned LSTM. `evaluate_h5.py` and `frozen_metrics_v1.py` define the original evaluation. `ce_checkpoint_recovery_v1.py` is retained as historical source for the CE checkpoint-based generation provenance; it references its original operational paths and is not the portable training entry point.

`source/corrected_code/` contains the 60 non-test source/configuration text files recovered against the executed source hashes. `SOURCE_SELECTION.json` lists the 60 included files and 31 exclusions: 26 legacy bytecode files, three format-example CSVs, one reserved test CSV and one image. The reserved test CSV was not opened. The runtime manifest under `source/runner/` contains exactly these files. Its subset selection removes legacy bytecode, images and example CSVs, including the reserved test file, without changing the included source bytes. `ORIGINAL_SOURCE_MANIFEST.json` preserves the historical per-file hashes without machine-specific root paths. `upstream_to_corrected.patch` records the four changed upstream files and the added input-contract module. All original source comments and author markings are retained.

The numerical settings in `configs/` are unchanged from the executed configurations. Only source and data paths are made relative. The following commands materialize absolute paths in a new `replay_configs/` directory, then request fresh training and generation:

```bash
python3 model/prepare_configs.py
python3 model/implementation/vae_pipeline_formal_v2.py run --config replay_configs/categorical_ce.json --output replay_runs/categorical_ce
python3 model/implementation/vae_pipeline_formal_v2.py run --config replay_configs/raw_onehot_mse.json --output replay_runs/raw_onehot_mse
python3 model/implementation/vae_pipeline_formal_v2.py run --config replay_configs/scalar_legacy.json --output replay_runs/scalar_legacy
python3 model/implementation/ar_entry.py --config replay_configs/terminal_lstm.json --output replay_runs/terminal_lstm
```

Training is opt-in and is not called by `verify_all.py`. These commands were checked against the source command-line interface and configurations, but were not executed as training reproduction for this release. The recorded environment is Python 3.12.13, PyTorch 2.2.2, NumPy 1.26.4, CPU, one thread; the full observed package inventory is in `environment/requirements.lock.txt`. The original elapsed-time caps remain part of the protocol and may stop a run on different hardware.

The final inference weights are in `weights/`. The VAE states and DDPM EMA states preserve original hashes; the LSTM checkpoint also contains its optimizer state according to its save routine. The release verifier hashes their bytes without loading or deserializing them. The original package was not rebuilt and checkpoints were not loaded in producing this release, so the offline numerical checks are not a claim of training or inference replay.
