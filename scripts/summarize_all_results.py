#!/usr/bin/env python3
"""Aggregate Suite 1, Suite 2, Expanded DBAASP split audit, and Wet-lab deep audit
into a single verified master_summary.json and print clean summary tables.
"""
from __future__ import annotations

import collections
import csv
import json
import sys
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr

ROOT = Path("/Users/loctek/.gemini/antigravity/scratch/bib-26-2683-revision")
sys.path.insert(0, str(ROOT / "scripts"))
import metrics_suite as ms

s1 = json.load(open(ROOT / "revision_results/suite1_statistical_ar_gruvae_v7replay.json"))
s2 = json.load(open(ROOT / "revision_results/suite2_vae_ddpm_ablations_and_expanded.json"))
wl = json.load(open(ROOT / "revision_results/wetlab_deep_audit.json"))
sp = json.load(open(ROOT / "revision_data/dbaasp_expanded/split_audit.json"))

# Also compute v7 1000-step CE21 per-AA support vs recall Spearman correlation directly from github_repo S1c
v7_s1c = list(csv.DictReader((ROOT / "github_repo/supplement/tables/Table_S1c_reconstruction_by_amino_acid.csv").open()))
tr_sup = {}
dev_rec = {}
for row in v7_s1c:
    if row["method"] != "CE21_DDPM":
        continue
    aa = row["amino_acid"]
    if row["split"] == "train":
        tr_sup[aa] = int(row["target_residues"])
    elif row["split"] == "development":
        dev_rec[aa] = float(row["recall"])
aas_sorted = sorted(dev_rec.keys())
v7_rho, v7_p = spearmanr([tr_sup[a] for a in aas_sorted], [dev_rec[a] for a in aas_sorted])

# Group Suite 2 runs by condition key
by_cond = collections.defaultdict(list)
for r in s2["runs"]:
    j = r["job"]
    key = (j["group"], j["cohort"], j["arm"], j["kl_beta"], j.get("mse_scale", 0.5), j.get("lr", 5e-4), j.get("mask_pad", False))
    by_cond[key].append(r)

print("================ SUITE 2 SUMMARY (3 seeds each) ================")
s2_summary = {}
for key in sorted(by_cond.keys()):
    runs = sorted(by_cond[key], key=lambda x: x["job"]["seed"])
    grp, coh, arm, beta, msc, lr, mpad = key
    label = f"{coh}_{arm}_b{beta}_sc{msc}_lr{lr}_pm{int(mpad)}"
    rec_micro = ms.summarize_seeds([r["recon"]["micro_pct"] for r in runs])
    rec_macro = ms.summarize_seeds([r["recon"]["macro_pct"] for r in runs])
    rec_lmae = ms.summarize_seeds([r["recon"]["length_mae"] for r in runs])
    rec_exact = ms.summarize_seeds([r["recon"]["exact_pct"] for r in runs])
    rec_om = ms.summarize_seeds([r["recon"]["oracle_len_micro_pct"] for r in runs])
    pr_f = ms.summarize_seeds([r["prior_gen"]["FREE"]["MMD2"] for r in runs])
    pr_a = ms.summarize_seeds([r["prior_gen"]["AMD"]["MMD2"] for r in runs])
    entry = {
        "group": grp, "cohort": coh, "arm": arm, "kl_beta": beta, "mse_scale": msc, "lr": lr, "mask_pad": mpad,
        "micro_pct": rec_micro, "macro_pct": rec_macro, "length_mae": rec_lmae, "exact_pct": rec_exact,
        "oracle_len_micro_pct": rec_om, "prior_FREE_MMD2": pr_f, "prior_AMD_MMD2": pr_a,
    }
    ddpm_str = ""
    if runs[0]["ddpm_gen"] is not None:
        for tname in ("FREE", "AMD"):
            for mk in ("MMD2", "MMD2_5_48", "AA_JS", "Pair_JS", "Tri_JS", "Length_KS", "w1_charge", "w1_mu_H", "nn_mean", "frac_ge70", "exact_copy_frac", "internal_diversity", "frac_len_lt5"):
                entry[f"ddpm_{tname}_{mk}"] = ms.summarize_seeds([r["ddpm_gen"][tname][mk] for r in runs])
        ddpm_str = (f" | DDPM F={entry['ddpm_FREE_MMD2']['mean']:.5f}±{entry['ddpm_FREE_MMD2']['sd']:.5f} "
                    f"A={entry['ddpm_AMD_MMD2']['mean']:.5f}±{entry['ddpm_AMD_MMD2']['sd']:.5f} "
                    f"TriF={entry['ddpm_FREE_Tri_JS']['mean']:.4f} W1muH={entry['ddpm_FREE_w1_mu_H']['mean']:.4f} "
                    f"NN={entry['ddpm_FREE_nn_mean']['mean']:.3f} Div={entry['ddpm_FREE_internal_diversity']['mean']:.3f}")
    s2_summary[label] = entry
    print(f"{label:42s} | micro={rec_micro['mean']:5.2f}±{rec_micro['sd']:4.2f}% (oracle={rec_om['mean']:5.2f}%) macro={rec_macro['mean']:5.2f}±{rec_macro['sd']:4.2f}% lenMAE={rec_lmae['mean']:5.2f} | PrF={pr_f['mean']:.4f}{ddpm_str}")

print(f"\nv7 1000-step CE21 AA train-support vs dev-recall Spearman: rho={v7_rho:.4f}, p={v7_p:.4g}")
print(f"Suite 2 220-step CE21 AA train-support vs dev-recall Spearman: rho={s2['aa_support_vs_recall_spearman']['rho']:.4f}, p={s2['aa_support_vs_recall_spearman']['p_value']:.4g}")

# Key paired tests on Suite 2
def get_runs(coh, arm, beta, msc=0.5, lr=5e-4, mpad=False):
    for key, runs in by_cond.items():
        if key[1:] == (coh, arm, beta, msc, lr, mpad):
            return sorted(runs, key=lambda x: x["job"]["seed"])
    raise KeyError((coh, arm, beta, msc, lr, mpad))

print("\n================ SUITE 2 KEY PAIRED TESTS (3 seeds) ================")
tests = [
    ("Pilot CE21 vs MSE21 b=0.01 micro", get_runs("pilot","CE21",0.01), get_runs("pilot","MSE21",0.01), lambda r: r["recon"]["micro_pct"]),
    ("Pilot CE21 vs MSE21_tuned b=0.01 macro", get_runs("pilot","CE21",0.01), get_runs("pilot","MSE21",0.01,10.5,2e-3), lambda r: r["recon"]["macro_pct"]),
    ("Pilot CE21_padmask vs CE21_unmasked oracle_micro", get_runs("pilot","CE21",1.0,mask_pad=True) if False else get_runs("pilot","CE21",1.0,0.5,5e-4,True), get_runs("pilot","CE21",1.0,0.5,5e-4,False), lambda r: r["recon"]["oracle_len_micro_pct"]),
    ("Expanded CE21 b0.1 vs MSE21_tuned b0.1 micro", get_runs("expanded","CE21",0.1), get_runs("expanded","MSE21",0.1,10.5,2e-3), lambda r: r["recon"]["micro_pct"]),
    ("Expanded CE21 b0.1 vs MSE21_tuned b0.1 macro", get_runs("expanded","CE21",0.1), get_runs("expanded","MSE21",0.1,10.5,2e-3), lambda r: r["recon"]["macro_pct"]),
]
s2_tests = {}
for name, ra, rb, fn in tests:
    va = [fn(r) for r in ra]
    vb = [fn(r) for r in rb]
    t = ms.paired_seed_test(va, vb)
    s2_tests[name] = t
    print(f"  {name:50s}: diff={t['mean_diff_a_minus_b']:+.3f} (t={t['paired_t']:.2f}, p_t={t['paired_t_p']:.4g})")

master = {
    "expanded_split_audit": sp,
    "v7_aa_support_recall_spearman": {"rho": round(float(v7_rho), 4), "p_value": float(v7_p)},
    "suite1_metric_correlation": s1["metric_correlation_audit"],
    "suite2_summary": s2_summary,
    "suite2_tests": s2_tests,
    "wetlab_audit": {k: v for k, v in wl.items() if k != "peptide_profiles"},
}
(ROOT / "revision_results/master_summary.json").write_text(json.dumps(master, indent=2) + "\n")
print("\nSaved revision_results/master_summary.json")
