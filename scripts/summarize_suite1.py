#!/usr/bin/env python3
import json, sys
from pathlib import Path
import numpy as np

ROOT = Path("/Users/loctek/.gemini/antigravity/scratch/bib-26-2683-revision")
sys.path.insert(0, str(ROOT / "scripts"))
import metrics_suite as ms

d = json.load(open(ROOT / "revision_results/suite1_statistical_ar_gruvae_v7replay.json"))
order = ["Unigram", "Bigram_Markov", "Trigram_Markov", "LSTM_small", "LSTM_matched", "Transformer_LM", "SeqGRUVAE_CE21", "SeqGRUVAE_MSE21"]

for cname in ("pilot", "expanded"):
    print(f"\n================ {cname.upper()} (5 seeds x 256 samples/terminal) ================")
    print(f"{'Model':16s} {'Params':>9s} | {'FREE MMD2 (mean±sd)':>19s} {'AMD MMD2 (mean±sd)':>19s} | {'F_PairJS':>8s} {'F_TriJS':>8s} {'F_W1_q':>7s} {'F_W1_muH':>8s} | {'NN_id':>6s} {'ge70%':>6s} {'Exact%':>6s} {'Div':>6s} {'<5aa%':>6s}")
    for mname in order:
        mdata = d[cname]["models"][mname]
        def msd(term, key, fmt=".5f"):
            vals = [r["eval"][key] for r in mdata["runs"][term]]
            return f"{np.mean(vals):{fmt}}±{np.std(vals, ddof=1):{fmt}}"
        def mv(term, key):
            return float(np.mean([r["eval"][key] for r in mdata["runs"][term]]))
        print(f"{mname:16s} {mdata['params']:9,d} | {msd('FREE','MMD2'):>19s} {msd('AMD','MMD2'):>19s} | "
              f"{mv('FREE','Pair_JS'):8.4f} {mv('FREE','Tri_JS'):8.4f} {mv('FREE','w1_charge'):7.3f} {mv('FREE','w1_mu_H'):8.4f} | "
              f"{mv('FREE','nn_mean'):6.3f} {100*mv('FREE','frac_ge70'):5.1f}% {100*mv('FREE','exact_copy_frac'):5.1f}% "
              f"{mv('FREE','internal_diversity'):6.3f} {100*mv('FREE','frac_len_lt5'):5.1f}%")
    # Also print AMD Pair_JS, Tri_JS, W1_charge, W1_mu_H, NN, ge70, Exact, Div
    print(f"--- {cname.upper()} AMD higher-order / physchem / memorization ---")
    for mname in order:
        mdata = d[cname]["models"][mname]
        def mv(term, key): return float(np.mean([r["eval"][key] for r in mdata["runs"][term]]))
        print(f"  {mname:16s} | A_PairJS={mv('AMD','Pair_JS'):.4f} A_TriJS={mv('AMD','Tri_JS'):.4f} A_W1_q={mv('AMD','w1_charge'):.3f} A_W1_muH={mv('AMD','w1_mu_H'):.4f} NN={mv('AMD','nn_mean'):.3f} ge70={100*mv('AMD','frac_ge70'):.1f}% Exact={100*mv('AMD','exact_copy_frac'):.1f}% Div={mv('AMD','internal_diversity'):.3f} <5aa={100*mv('AMD','frac_len_lt5'):.1f}% MMD5-48={mv('AMD','MMD2_5_48'):.5f}")
    # GRU-VAE reconstruction summary
    print(f"--- {cname.upper()} SeqGRUVAE Reconstruction (5 seeds) ---")
    for mname in ("SeqGRUVAE_CE21", "SeqGRUVAE_MSE21"):
        recs = d[cname]["models"][mname]["recons"]
        for k in ("micro_pct", "macro_pct", "length_mae", "exact_pct"):
            s = ms.summarize_seeds([r[k] for r in recs])
            print(f"  {mname:16s} {k:12s}: {s['mean']:.3f} ± {s['sd']:.3f} (95% CI [{s['ci95_lo']:.3f}, {s['ci95_hi']:.3f}])")
    # Paired seed test: SeqGRUVAE_CE21 vs MSE21 & LSTM_small vs Transformer_LM & LSTM_small vs Bigram_Markov
    print(f"--- {cname.upper()} Paired 5-seed Statistical Tests (FREE MMD2 & Tri_JS & Recon) ---")
    for a, b, term, key in [
        ("SeqGRUVAE_CE21", "SeqGRUVAE_MSE21", "FREE", "MMD2"),
        ("SeqGRUVAE_CE21", "SeqGRUVAE_MSE21", "AMD", "MMD2"),
        ("LSTM_small", "SeqGRUVAE_CE21", "FREE", "MMD2"),
        ("LSTM_small", "Bigram_Markov", "FREE", "MMD2"),
        ("LSTM_small", "Unigram", "FREE", "Tri_JS"),
        ("Transformer_LM", "Unigram", "FREE", "Tri_JS"),
        ("LSTM_matched", "Unigram", "FREE", "Tri_JS"),
        ("LSTM_small", "Unigram", "FREE", "w1_mu_H"),
    ]:
        va = [r["eval"][key] for r in d[cname]["models"][a]["runs"][term]]
        vb = [r["eval"][key] for r in d[cname]["models"][b]["runs"][term]]
        t = ms.paired_seed_test(va, vb)
        print(f"  {a} vs {b} ({term} {key}): diff={t['mean_diff_a_minus_b']:+.5f}, t={t['paired_t']:.2f}, p_t={t['paired_t_p']:.4g}, p_perm={t['exact_perm_p']:.4g}")

print("\n================ V7 FROZEN CHECKPOINT REPLAY (256 x 5 gen seeds) ================")
for k in ("v7_terminal_lstm_256x5", "v7_categorical_ce_ddpm_256x5", "v7_raw_onehot_mse_ddpm_256x5", "v7_scalar_legacy_ddpm_256x5"):
    v = d["v7_checkpoint_replay"][k]
    for term in ("FREE", "AMD"):
        s_mmd = ms.summarize_seeds([r["dev"]["MMD2"] for r in v[term]])
        s_548 = ms.summarize_seeds([r["dev"]["MMD2_5_48"] for r in v[term]])
        s_pjs = ms.summarize_seeds([r["dev"]["Pair_JS"] for r in v[term]])
        s_tjs = ms.summarize_seeds([r["dev"]["Tri_JS"] for r in v[term]])
        s_nn = ms.summarize_seeds([r["dev"]["nn_mean"] for r in v[term]])
        s_div = ms.summarize_seeds([r["dev"]["internal_diversity"] for r in v[term]])
        s_lt5 = ms.summarize_seeds([100*r["dev"]["frac_len_lt5"] for r in v[term]])
        print(f"  {k:28s} {term:4s} | MMD2={s_mmd['mean']:.5f}±{s_mmd['sd']:.5f} [{s_mmd['ci95_lo']:.5f},{s_mmd['ci95_hi']:.5f}] | 5-48={s_548['mean']:.5f} | PairJS={s_pjs['mean']:.4f} TriJS={s_tjs['mean']:.4f} | NN={s_nn['mean']:.3f} Div={s_div['mean']:.3f} <5aa={s_lt5['mean']:.1f}%")

print("\n================ METRIC CORRELATION AUDIT ================")
print(json.dumps(d["metric_correlation_audit"], indent=2))
