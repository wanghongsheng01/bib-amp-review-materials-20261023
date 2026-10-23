# AMP study materials

This repository accompanies the AMP manuscript and supplement. It contains the computational evidence, laboratory measurements, train/development inputs, implemented model code and fixed model weights used to examine sequence representation, reconstruction, generation and activity-related analyses.

## Reproduce the reported calculations

From the repository root, using Python 3:

```bash
python3 verify_all.py
```

The command checks file integrity, reruns the offline computational verification and recalculates the laboratory endpoint summaries. The computational verifier passes **2,630 checks**, including sequence-derived reconstruction counts, all **96 distribution distances**, activity-error aggregates and matched-training records. The laboratory verifier checks **3,240 OD600 measurements**, **324 replicate series**, **108 peptide–strain endpoints** and the complete concentration-threshold table. No network connection or third-party Python package is needed for these calculations.

## Contents

| Directory | Contents |
|---|---|
| `compute/` | Saved computational observations, independent calculation code, source bindings and derived tables |
| `wetlab/` | Original order workbook, scientific certificate facts, OD600 observations, reported MIC values and separately labelled algorithmic endpoints |
| `data/train_dev/` | Four original CSV inputs: 279 training and 41 development peptides, with descriptor and cohort representations |
| `model/` | H5 implementations, corrected source dependencies, frozen numerical configurations, scalers and source-change provenance |
| `weights/` | Three final VAE states, three final DDPM EMA states and the terminal LSTM final checkpoint |
| `environment/` | Original package inventory and recorded runtime versions |
| `manuscript/`, `supplement/` | Version-aligned article and supplementary materials when included in the release |

`MANIFEST.json` binds packaged files to SHA256 hashes. The computational input manifest additionally records the source repository commit, original Git blob/hash, packaged hash and any field-selection transformation. `weights/MANIFEST.json` identifies all seven checkpoint files by model arm and role.

## Scientific scope

The four-arm computational analysis uses one fitted configuration/seed and training/development data. The paired 21-class CE/MSE comparison and the scalar and terminal-LSTM comparators have the different scopes described in the manuscript. Development reuse and a single fitted run do not establish independent-replicate inference. Activity regressions concern recorded activity observations. The laboratory series characterizes 27 assayed peptides from a 30-peptide order across four strains; its provenance is documented separately from the model comparison.

Reported MIC values and retrospective OD600-based endpoint calls remain separate datasets. The original `N` code is preserved and is not imputed as zero, a numeric MIC or a verified negative. Certificate data and solubility observations retain their supplier-reported status.

The verified commands reproduce calculations from saved outputs. They do not retrain the models or regenerate samples from the checkpoints. Code, inputs, configurations and checkpoints are supplied for further computational reproduction; [model/README.md](model/README.md) provides the training commands and recorded execution settings. Training and checkpoint loading were not rerun for this release.

## Provenance and use

The early implementation is pinned to [3DHumanRehab/AMP, e7486fb](https://github.com/3DHumanRehab/AMP/tree/e7486fb714cbf5c663cc01052e42fb63d4ba3a40). The subsequent computational source is pinned to [the author-provided H5 snapshot, f1b748d](https://github.com/wanghongsheng01/amp-claude-handoff-20261013/tree/f1b748d21386e1d1683a0b6e92cbcf9a208a4d2b). See [COPYRIGHT_AND_USE.md](COPYRIGHT_AND_USE.md) for retained source rights and confidential review use.
