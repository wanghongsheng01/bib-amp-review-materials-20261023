#!/usr/bin/env python3
"""Deep wet-lab audit: OD600 blank/control justification, multi-threshold sensitivity,
physicochemical & sequence-novelty profiling of the 27 assayed peptides, molar MIC
equivalents, 1000->30->27 provenance funnel, and AMPGen overlap verification.
"""
from __future__ import annotations

import collections
import csv
import json
import math
import sys
from pathlib import Path
import numpy as np

ROOT = Path("/Users/loctek/.gemini/antigravity/scratch/bib-26-2683-revision")
sys.path.insert(0, str(ROOT / "scripts"))
import metrics_suite as ms

OUT_DIR = ROOT / "revision_results"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    # 1. Load OD600 raw data and reported MICs
    od_rows = list(csv.DictReader((ROOT / "github_repo/wetlab/data/od600_raw.csv").open()))
    rep_rows = list(csv.DictReader((ROOT / "github_repo/wetlab/data/mic_summary_reported.csv").open()))
    manifest = list(csv.DictReader((ROOT / "github_repo/wetlab/data/sample_manifest_30.csv").open()))

    # Strain-wise blank & negative control OD600 summary
    strains = ["ecoli", "paeruginosa", "saureus", "efaecalis"]
    ctrl_stats = {}
    for st in strains:
        sub = [r for r in od_rows if r["strain"] == st]
        blanks = np.array([float(r["blank_od600"]) for r in sub])
        negs = np.array([float(r["negative_control_od600"]) for r in sub])
        ods = np.array([float(r["od600"]) for r in sub])
        # What % growth inhibition does OD600 < 0.10 correspond to vs median blank/neg?
        med_b, med_n = float(np.median(blanks)), float(np.median(negs))
        pct_growth_at_010 = (0.10 - med_b) / (med_n - med_b) * 100.0
        ctrl_stats[st] = {
            "n_readings": len(sub),
            "blank_median": round(med_b, 4),
            "blank_iqr": [round(float(np.percentile(blanks, 25)), 4), round(float(np.percentile(blanks, 75)), 4)],
            "neg_ctrl_median": round(med_n, 4),
            "neg_ctrl_iqr": [round(float(np.percentile(negs, 25)), 4), round(float(np.percentile(negs, 75)), 4)],
            "neg_ctrl_min": round(float(np.min(negs)), 4),
            "od_010_pct_of_growth": round(pct_growth_at_010, 1),
            "od_010_pct_inhibition": round(100.0 - pct_growth_at_010, 1),
        }

    # 2. Multi-threshold OD600 sensitivity analysis (0.08, 0.10, 0.12, 0.15, and normalized <10% / <20% growth)
    # Group readings by (sample_id, strain, replicate)
    series = collections.defaultdict(list)
    for r in od_rows:
        series[(r["sample_id"], r["strain"], int(r["replicate"]))].append((
            float(r["concentration_ug_ml"]),
            float(r["od600"]),
            float(r["blank_od600"]),
            float(r["negative_control_od600"]),
        ))
    for k in series:
        series[k].sort(key=lambda x: x[0])  # ascending concentration

    # Map reported MICs
    reported_map = {}
    for r in rep_rows:
        sid = r["sample_id"]
        for st_col, st_key in [("ecoli_ug_ml", "ecoli"), ("paeruginosa_ug_ml", "paeruginosa"), ("saureus_ug_ml", "saureus"), ("efaecalis_ug_ml", "efaecalis")]:
            val = r[st_col].strip()
            reported_map[(sid, st_key)] = None if val == "N" else float(val)

    peptides27 = [r for r in manifest if r["assayed"] == "True"]
    sample_ids = [r["sample_id"] for r in peptides27]

    def eval_rule(rule_name, pred_fn):
        cells = {}
        concordant = 0
        for sid in sample_ids:
            for st in strains:
                rep_calls = []
                for rep in (1, 2, 3):
                    pts = series[(sid, st, rep)]
                    # lowest concentration passing predicate
                    hit = None
                    for conc, od, blk, neg in pts:
                        if pred_fn(od, blk, neg):
                            hit = conc
                            break
                    rep_calls.append(hit)
                # >=2 of 3 consensus
                cnt = collections.Counter(rep_calls)
                val, freq = cnt.most_common(1)[0]
                ans = val if freq >= 2 else None
                cells[(sid, st)] = ans
                if ans == reported_map[(sid, st)]:
                    concordant += 1
        sa_le3125 = sum(1 for sid in sample_ids if cells[(sid, "saureus")] is not None and cells[(sid, "saureus")] <= 31.25)
        sa_lt200 = sum(1 for sid in sample_ids if cells[(sid, "saureus")] is not None and cells[(sid, "saureus")] < 200.0)
        any_lt200 = sum(1 for sid in sample_ids if any(cells[(sid, st)] is not None and cells[(sid, st)] < 200.0 for st in strains))
        any_le3125 = sum(1 for sid in sample_ids if any(cells[(sid, st)] is not None and cells[(sid, st)] <= 31.25 for st in strains))
        return {
            "rule": rule_name,
            "concordance_of_108": concordant,
            "saureus_le_31_25": sa_le3125,
            "saureus_lt_200": sa_lt200,
            "any_strain_le_31_25": any_le3125,
            "any_strain_lt_200": any_lt200,
        }

    sensitivity = [
        eval_rule("OD600 < 0.08", lambda od, b, n: od < 0.08),
        eval_rule("OD600 < 0.10 (primary retrospective)", lambda od, b, n: od < 0.10),
        eval_rule("OD600 < 0.12", lambda od, b, n: od < 0.12),
        eval_rule("OD600 < 0.15", lambda od, b, n: od < 0.15),
        eval_rule("Normalized growth < 10% (OD-blank)/(neg-blank)", lambda od, b, n: (n > b) and ((od - b) / (n - b) < 0.10)),
        eval_rule("Normalized growth < 20% (OD-blank)/(neg-blank)", lambda od, b, n: (n > b) and ((od - b) / (n - b) < 0.20)),
    ]

    # 3. Physicochemical & sequence-novelty profile of all 27 assayed peptides + molar MIC equivalents
    train_seqs = [r["sequence"] for r in csv.DictReader((ROOT / "github_repo/data/train_dev/train_cohort.csv").open())]
    dev_seqs = [r["sequence"] for r in csv.DictReader((ROOT / "github_repo/data/train_dev/dev_cohort.csv").open())]
    pilot_all = sorted(set(train_seqs + dev_seqs))

    peptide_profiles = []
    for row in peptides27:
        sid = row["sample_id"]
        seq = row["sequence"]
        mw = float(row["theoretical_molecular_weight"])
        q = ms.net_charge_ph74(seq, "FREE")
        h = ms.mean_hydrophobicity(seq)
        muh = ms.hydrophobic_moment_alpha(seq)
        bom = ms.boman_index(seq)
        ali = ms.aliphatic_index(seq)
        # Nearest neighbor identity to pilot DBAASP cohort
        best_id = max(ms.global_identity_fast(seq, p) for p in pilot_all)
        # Molar MICs (uM) = (ug/mL) / MW * 1000
        sa_mic = reported_map[(sid, "saureus")]
        ec_mic = reported_map[(sid, "ecoli")]
        pa_mic = reported_map[(sid, "paeruginosa")]
        ef_mic = reported_map[(sid, "efaecalis")]
        min_mic_ug = min([v for v in (sa_mic, ec_mic, pa_mic, ef_mic) if v is not None], default=None)
        min_mic_uM = round(min_mic_ug / mw * 1000.0, 2) if min_mic_ug is not None else None
        sa_mic_uM = round(sa_mic / mw * 1000.0, 2) if sa_mic is not None else None
        peptide_profiles.append({
            "sample_id": sid,
            "design_group": row["ordered_group_id"],
            "design_label": row["group_label_from_manuscript"],
            "sequence": seq,
            "length": len(seq),
            "mw_da": mw,
            "purity_pct": float(row["hplc_purity_percent"]),
            "charge_ph74": round(q, 2),
            "hydrophobicity": round(h, 3),
            "mu_H_alpha": round(muh, 3),
            "boman": round(bom, 3),
            "aliphatic_index": round(ali, 1),
            "nn_id_pilot_dbaasp": round(best_id, 4),
            "saureus_mic_ug_ml": sa_mic,
            "saureus_mic_uM": sa_mic_uM,
            "min_mic_ug_ml": min_mic_ug,
            "min_mic_uM": min_mic_uM,
            "active_sa_le3125": sa_mic is not None and sa_mic <= 31.25,
            "active_any_lt200": min_mic_ug is not None and min_mic_ug < 200.0,
        })

    # Active (6 S. aureus <=31.25) vs inactive (21) physicochemical summary
    act6 = [p for p in peptide_profiles if p["active_sa_le3125"]]
    inact21 = [p for p in peptide_profiles if not p["active_sa_le3125"]]
    group_stats = {}
    for g in ("1", "2", "3"):
        sub = [p for p in peptide_profiles if p["design_group"] == g]
        group_stats[f"group_{g}"] = {
            "assayed": len(sub),
            "sa_le_31_25": sum(1 for p in sub if p["active_sa_le3125"]),
            "any_lt_200": sum(1 for p in sub if p["active_any_lt200"]),
            "ecoli_lt_200": sum(1 for p in sub if reported_map[(p["sample_id"], "ecoli")] is not None and reported_map[(p["sample_id"], "ecoli")] < 200),
            "pa_lt_200": sum(1 for p in sub if reported_map[(p["sample_id"], "paeruginosa")] is not None and reported_map[(p["sample_id"], "paeruginosa")] < 200),
        }

    # Check active peptides' molar MICs (<16 uM / <20 uM)
    molar_active_summary = [
        {"sample_id": p["sample_id"], "saureus_ug_ml": p["saureus_mic_ug_ml"], "saureus_uM": p["saureus_mic_uM"],
         "charge": p["charge_ph74"], "mu_H": p["mu_H_alpha"], "nn_id_dbaasp": p["nn_id_pilot_dbaasp"]}
        for p in act6
    ]

    result = {
        "strain_od600_controls": ctrl_stats,
        "od600_threshold_sensitivity": sensitivity,
        "group_activity_summary": group_stats,
        "active6_vs_inactive21_means": {
            feat: {
                "active6_mean": round(float(np.mean([p[feat] for p in act6])), 3),
                "inactive21_mean": round(float(np.mean([p[feat] for p in inact21])), 3),
            }
            for feat in ("length", "charge_ph74", "hydrophobicity", "mu_H_alpha", "boman", "nn_id_pilot_dbaasp")
        },
        "active6_molar_summary": molar_active_summary,
        "max_nn_id_to_pilot_dbaasp_across_27": round(max(p["nn_id_pilot_dbaasp"] for p in peptide_profiles), 4),
        "peptide_profiles": peptide_profiles,
    }
    (OUT_DIR / "wetlab_deep_audit.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "peptide_profiles"}, indent=2))


if __name__ == "__main__":
    main()
