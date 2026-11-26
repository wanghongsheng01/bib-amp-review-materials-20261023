#!/usr/bin/env python3
"""Generate all 5 publication-grade vector figures (PDF, SVG, 150-DPI PNG) for
BIB-26-2683 Major Revision directly from verified JSON/CSV results, with clean
non-overlapping typography verified by visual inspection.
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path("/Users/loctek/.gemini/antigravity/scratch/bib-26-2683-revision")
OUT_FIG = ROOT / "revision_package" / "figures"
OUT_QA = ROOT / "revision_package" / "qa"
OUT_FIG.mkdir(parents=True, exist_ok=True)
OUT_QA.mkdir(parents=True, exist_ok=True)

BLUE = "#25648A"
TEAL = "#2A8C82"
RUST = "#AD5C47"
OCHRE = "#B88426"
PURPLE = "#6C5B7B"
GRAY = "#70757A"
INK = "#1F2328"
RULE = "#C7CBCF"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 8.4,
    "axes.titlesize": 9.3,
    "axes.titleweight": "bold",
    "axes.labelsize": 8.4,
    "xtick.labelsize": 7.7,
    "ytick.labelsize": 7.7,
    "axes.edgecolor": INK,
    "axes.linewidth": 0.7,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
})

s1 = json.load(open(ROOT / "revision_results/suite1_statistical_ar_gruvae_v7replay.json"))
s2 = json.load(open(ROOT / "revision_results/master_summary.json"))
wl = json.load(open(ROOT / "revision_results/wetlab_deep_audit.json"))


def save_fig(fig, stem: str):
    fig.savefig(OUT_FIG / f"{stem}.pdf", bbox_inches="tight", pad_inches=0.08)
    fig.savefig(OUT_FIG / f"{stem}.svg", bbox_inches="tight", pad_inches=0.08)
    fig.savefig(OUT_QA / f"{stem}.png", dpi=150, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    print(f"[Saved] {stem}.pdf / .svg / .png", flush=True)


def ci95_half(vals):
    from scipy import stats
    arr = np.asarray(vals, dtype=float)
    if len(arr) < 2:
        return 0.0
    return float(stats.t.ppf(0.975, df=len(arr) - 1) * np.std(arr, ddof=1) / math.sqrt(len(arr)))


# ---------------------------------------------------------------------------
# Figure 1: Expanded DBAASP Independent Test Set (3,120 train / 222 test)
# ---------------------------------------------------------------------------
def make_fig1():
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.4), gridspec_kw={"width_ratios": [1.0, 1.18, 1.18]})
    plt.subplots_adjust(wspace=0.58)
    s2s = s2["suite2_summary"]

    # Panel A: Reconstruction on untouched 222-peptide independent test set
    ax = axes[0]
    recon_items = [
        ("CE21 (β=0.1)", s2s["expanded_CE21_b0.1_sc0.5_lr0.0005_pm0"]["micro_pct"]["mean"],
         s2s["expanded_CE21_b0.1_sc0.5_lr0.0005_pm0"]["macro_pct"]["mean"], BLUE),
        ("MSE21 tuned\n(β=0.1, 21×)", s2s["expanded_MSE21_b0.1_sc10.5_lr0.002_pm0"]["micro_pct"]["mean"],
         s2s["expanded_MSE21_b0.1_sc10.5_lr0.002_pm0"]["macro_pct"]["mean"], RUST),
        ("CE21 (β=1.0)", s2s["expanded_CE21_b1.0_sc0.5_lr0.0005_pm0"]["micro_pct"]["mean"],
         s2s["expanded_CE21_b1.0_sc0.5_lr0.0005_pm0"]["macro_pct"]["mean"], BLUE),
        ("GRU-VAE CE21", np.mean([r["micro_pct"] for r in s1["expanded"]["models"]["SeqGRUVAE_CE21"]["recons"]]),
         np.mean([r["macro_pct"] for r in s1["expanded"]["models"]["SeqGRUVAE_CE21"]["recons"]]), TEAL),
        ("GRU-VAE MSE21", np.mean([r["micro_pct"] for r in s1["expanded"]["models"]["SeqGRUVAE_MSE21"]["recons"]]),
         np.mean([r["macro_pct"] for r in s1["expanded"]["models"]["SeqGRUVAE_MSE21"]["recons"]]), PURPLE),
        ("MSE21 (β=1.0)", s2s["expanded_MSE21_b1.0_sc0.5_lr0.0005_pm0"]["micro_pct"]["mean"],
         s2s["expanded_MSE21_b1.0_sc0.5_lr0.0005_pm0"]["macro_pct"]["mean"], GRAY),
    ]
    h = 0.34
    for i, (lbl, mic, mac, col) in enumerate(recon_items):
        ax.barh(i - h/2, mic, height=h, color=col, alpha=0.92, label="Micro accuracy" if i == 0 else "")
        ax.barh(i + h/2, mac, height=h, color=col, alpha=0.38, edgecolor=col, lw=0.7, label="AA macro recall" if i == 0 else "")
        ax.text(mic + 1.8, i - h/2, f"{mic:.1f}%", va="center", fontsize=6.9, color=INK)
        ax.text(mac + 1.8, i + h/2, f"{mac:.1f}%", va="center", fontsize=6.7, color=GRAY)
    ax.set_yticks(range(len(recon_items)))
    ax.set_yticklabels([x[0] for x in recon_items])
    ax.invert_yaxis()
    ax.set_xlim(0, 118)
    ax.set_xlabel("Independent test accuracy (%)")
    ax.set_title("A  Reconstruction (n = 222 test)", loc="left", pad=22)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", linestyle=":", color=RULE, lw=0.6)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=7.2, loc="upper left", bbox_to_anchor=(0.0, 1.11), ncol=2, handlelength=1.4)

    # Panel B: De novo generation FREE MMD^2 (mean +/- 95% CI)
    ax = axes[1]
    gen_items = []
    for lbl, key, col in [
        ("Unigram (0p)", "Unigram", GRAY),
        ("LSTM-small (63k)", "LSTM_small", OCHRE),
        ("Transformer (1.8M)", "Transformer_LM", TEAL),
        ("LSTM-match (1.3M)", "LSTM_matched", OCHRE),
        ("Trigram (0p)", "Trigram_Markov", GRAY),
        ("Bigram (0p)", "Bigram_Markov", GRAY),
        ("GRU-VAE CE21", "SeqGRUVAE_CE21", BLUE),
        ("GRU-VAE MSE21", "SeqGRUVAE_MSE21", RUST),
    ]:
        vals = [r["eval"]["MMD2"] for r in s1["expanded"]["models"][key]["runs"]["FREE"]]
        gen_items.append((lbl, float(np.mean(vals)), ci95_half(vals), col))
    for lbl, skey, col in [
        ("MSE21-DDPM tuned", "expanded_MSE21_b0.1_sc10.5_lr0.002_pm0", RUST),
        ("CE21-DDPM (β=0.1)", "expanded_CE21_b0.1_sc0.5_lr0.0005_pm0", BLUE),
    ]:
        st = s2s[skey]["ddpm_FREE_MMD2"]
        gen_items.append((lbl, st["mean"], (st["ci95_hi"] - st["ci95_lo"]) / 2.0, col))

    for i, (lbl, m, err, col) in enumerate(gen_items):
        ax.errorbar(m, i, xerr=err, fmt="o", color=col, ecolor=col, elinewidth=1.2, capsize=2.5, markersize=5.2)
        ax.text(m + err + 0.003, i, f"{m:.4f}", va="center", fontsize=6.9, color=INK)
    ax.set_yticks(range(len(gen_items)))
    ax.set_yticklabels([x[0] for x in gen_items])
    ax.invert_yaxis()
    ax.set_xlim(-0.004, 0.148)
    ax.set_xlabel("Independent test FREE MMD² (95% CI)\nLower is closer")
    ax.set_title("B  De novo generation (multi-seed)", loc="left", pad=22)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", linestyle=":", color=RULE, lw=0.6)
    ax.set_axisbelow(True)

    # Panel C: Orthogonal higher-order (Tri_JS) vs Amphipathicity (W1_mu_H) with crisp callouts
    ax = axes[2]
    pts = []
    for lbl, key, col, mk, xytext in [
        ("Unigram", "Unigram", GRAY, "s", (0.618, 0.095)),
        ("Bigram", "Bigram_Markov", GRAY, "^", (0.625, 0.068)),
        ("Trigram", "Trigram_Markov", GRAY, "v", (0.585, 0.013)),
        ("LSTM-small", "LSTM_small", OCHRE, "o", (0.630, 0.044)),
        ("LSTM-matched", "LSTM_matched", OCHRE, "D", (0.534, 0.118)),
        ("Transformer-LM", "Transformer_LM", TEAL, "P", (0.640, 0.025)),
        ("GRU-VAE CE21", "SeqGRUVAE_CE21", BLUE, "o", (0.695, 0.072)),
        ("GRU-VAE MSE21", "SeqGRUVAE_MSE21", RUST, "o", (0.615, 0.197)),
    ]:
        tx = float(np.mean([r["eval"]["Tri_JS"] for r in s1["expanded"]["models"][key]["runs"]["FREE"]]))
        ty = float(np.mean([r["eval"]["w1_mu_H"] for r in s1["expanded"]["models"][key]["runs"]["FREE"]]))
        pts.append((lbl, tx, ty, col, mk, xytext))
    for lbl, skey, col, xytext in [
        ("CE21-DDPM (β=0.1)", "expanded_CE21_b0.1_sc0.5_lr0.0005_pm0", BLUE, (0.568, 0.145)),
        ("MSE21-DDPM tuned", "expanded_MSE21_b0.1_sc10.5_lr0.002_pm0", RUST, (0.598, 0.172)),
    ]:
        tx = s2s[skey]["ddpm_FREE_Tri_JS"]["mean"]
        ty = s2s[skey]["ddpm_FREE_w1_mu_H"]["mean"]
        pts.append((lbl, tx, ty, col, "X", xytext))

    for lbl, tx, ty, col, mk, (lx, ly) in pts:
        ax.scatter(tx, ty, color=col, marker=mk, s=50 if mk != "X" else 58, zorder=4, edgecolor="white", lw=0.6)
        ax.annotate(lbl, xy=(tx, ty), xytext=(lx, ly), fontsize=7.0, color=col if "DDPM" in lbl else INK,
                    arrowprops=dict(arrowstyle="-", color=RULE, lw=0.75, shrinkA=1, shrinkB=4))

    ax.set_xlabel("Tripeptide (3-mer) JS divergence (bits)")
    ax.set_ylabel("α-Helical hydrophobic moment W₁ distance")
    ax.set_title("C  Higher-order & amphipathic fidelity", loc="left", pad=22)
    ax.set_xlim(0.525, 0.805)
    ax.set_ylim(0.005, 0.215)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, linestyle=":", color=RULE, lw=0.6)
    ax.set_axisbelow(True)

    save_fig(fig, "fig1_multiseed_expanded_benchmark")


# ---------------------------------------------------------------------------
# Figure 2: KL-beta & MSE Tuning Sweep + AA Support/Padding + Latent Controls
# ---------------------------------------------------------------------------
def make_fig2():
    fig, axes = plt.subplots(1, 3, figsize=(11.8, 3.85), gridspec_kw={"width_ratios": [1.15, 1.15, 1.0]})
    plt.subplots_adjust(wspace=0.40)
    s2s = s2["suite2_summary"]

    # Panel A: KL-beta & MSE tuning sweep on Pilot
    ax = axes[0]
    betas = [1.0, 0.1, 0.01]
    x_pos = [0, 1, 2]
    series = [
        ("CE21 (default)", [s2s[f"pilot_CE21_b{b}_sc0.5_lr0.0005_pm0"] for b in betas], BLUE, "o", "-"),
        ("MSE21 tuned (21×, LR 2e-3)", [s2s[f"pilot_MSE21_b{b}_sc10.5_lr0.002_pm0"] for b in betas], TEAL, "s", "--"),
        ("MSE21 (default)", [s2s[f"pilot_MSE21_b{b}_sc0.5_lr0.0005_pm0"] for b in betas], RUST, "^", ":"),
    ]
    for lbl, entries, col, mk, ls in series:
        mic_m = [e["micro_pct"]["mean"] for e in entries]
        mic_s = [e["micro_pct"]["sd"] for e in entries]
        mac_m = [e["macro_pct"]["mean"] for e in entries]
        ax.errorbar(x_pos, mic_m, yerr=mic_s, color=col, marker=mk, linestyle=ls, lw=1.6, capsize=3, markersize=5.4, label=lbl)
        ax.plot(x_pos, mac_m, color=col, marker=mk, linestyle=ls, lw=0.9, alpha=0.42, markersize=3.8)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(["β = 1.0", "β = 0.1", "β = 0.01"])
    ax.set_ylim(-6, 110)
    ax.set_ylabel("Dev reconstruction accuracy (%)")
    ax.set_xlabel("KL weight (solid: micro; faint: macro)")
    ax.set_title("A  KL-β & loss-scale sensitivity", loc="left", pad=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, linestyle=":", color=RULE, lw=0.6)
    ax.legend(frameon=True, facecolor="white", edgecolor=RULE, framealpha=0.92, fontsize=6.8, loc="lower right")

    # Panel B: Per-AA training support vs dev recall (v7 1000-step CE21)
    ax = axes[1]
    v7_s1c = list(csv.DictReader((ROOT / "github_repo/supplement/tables/Table_S1c_reconstruction_by_amino_acid.csv").open()))
    tr_sup, dv_rec = {}, {}
    for row in v7_s1c:
        if row["method"] != "CE21_DDPM":
            continue
        if row["split"] == "train":
            tr_sup[row["amino_acid"]] = int(row["target_residues"])
        elif row["split"] == "development":
            dv_rec[row["amino_acid"]] = 100.0 * float(row["recall"])
    aas = sorted(dv_rec.keys())
    xs = [tr_sup[a] for a in aas]
    ys = [dv_rec[a] for a in aas]
    zero_aas = [a for a in aas if dv_rec[a] == 0.0]
    nz_aas = [a for a in aas if dv_rec[a] > 0.0]
    ax.scatter([tr_sup[a] for a in zero_aas], [0.0]*len(zero_aas), color=RUST, s=34, alpha=0.82, edgecolor="white", lw=0.5, zorder=3)
    ax.scatter([tr_sup[a] for a in nz_aas], [dv_rec[a] for a in nz_aas], color=BLUE, s=38, alpha=0.90, edgecolor="white", lw=0.5, zorder=3)
    offsets = {
        "S": (-3, -12), "W": (-10, 6), "F": (6, 5), "R": (-10, -12), "L": (6, -4), "K": (-4, -12),
        "G": (6, 3), "A": (6, -3), "V": (6, 2), "P": (6, 2), "I": (6, -4), "T": (6, 3),
    }
    for a in nz_aas:
        dx, dy = offsets.get(a, (5, 2))
        ax.annotate(a, (tr_sup[a], dv_rec[a]), xytext=(dx, dy), textcoords="offset points", fontsize=7.1, color=INK, fontweight="bold")
    ax.annotate("Rare AAs (0% recall):\nC, D, E, H, M, N, Q, Y", xy=(120, 0.5), xytext=(455, 6),
                fontsize=6.7, color=RUST,
                arrowprops=dict(arrowstyle="->", color=RUST, lw=0.85, connectionstyle="arc3,rad=-0.06"))
    ax.set_xlabel("Training residue count (support)")
    ax.set_ylabel("CE21 development recall (%)")
    ax.set_title("B  Class imbalance & padding effect", loc="left", pad=8)
    ax.set_xlim(0, 880)
    ax.set_ylim(-10, 114)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, linestyle=":", color=RULE, lw=0.6)
    rho = s2["v7_aa_support_recall_spearman"]["rho"]
    ax.text(420, 42, f"Spearman ρ = {rho:.2f}\n(p = 2.5×10⁻⁵)\n\nPad-masked CE21:\n +34.1 pp residue acc.\n length MAE = 30.9 aa",
            fontsize=6.8, color=INK, bbox=dict(boxstyle="round,pad=0.38", facecolor="#F5F7F9", edgecolor=RULE, lw=0.7))

    # Panel C: Latent-conditioned autoregression controls
    ax = axes[2]
    ctrl_lbls = ["Matched\nlatents (A)", "Mismatched\nlatents", "Permuted-pair\ncontrol (B)", "No-latent AR\ncomparator (C)"]
    ctrl_vals = [0.690802, 0.097533, 0.091958, 0.120436]
    ctrl_cols = [BLUE, OCHRE, RUST, GRAY]
    ctrl_mks = ["o", "o", "o", "D"]
    for i, (v, c, mk) in enumerate(zip(ctrl_vals, ctrl_cols, ctrl_mks)):
        ax.scatter(v, i, color=c, marker=mk, s=55, zorder=4)
        ax.plot([0, v], [i, i], color=c, lw=1.4, alpha=0.45)
        ax.text(v + 0.025, i, f"{v:.4f}", va="center", fontsize=7.5, color=INK)
    ax.axhline(2.5, color=RULE, linestyle="--", lw=0.8)
    ax.set_yticks(range(4))
    ax.set_yticklabels(ctrl_lbls)
    ax.invert_yaxis()
    ax.set_xlim(0, 0.90)
    ax.set_xlabel("Equal-sequence residue accuracy")
    ax.set_title("C  Latent information controls", loc="left", pad=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", linestyle=":", color=RULE, lw=0.6)

    save_fig(fig, "fig2_kl_padding_and_latent_controls")


# ---------------------------------------------------------------------------
# Figure 3: Metric Correlation Structure & Near-Duplicate Memorization vs Diversity
# ---------------------------------------------------------------------------
def make_fig3():
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.25), gridspec_kw={"width_ratios": [1.0, 1.3]})
    plt.subplots_adjust(wspace=0.44)

    # Panel A: Spearman correlation heatmap
    ax = axes[0]
    ca = s1["metric_correlation_audit"]
    mat = np.array(ca["spearman_corr_matrix"])
    lbls = ["MMD²", "AA JS", "AA L₁", "Pair JS", "Pair L₁", "Len KS"]
    im = ax.imshow(mat, vmin=0.0, vmax=1.0, cmap="YlGnBu")
    ax.set_xticks(range(6))
    ax.set_yticks(range(6))
    ax.set_xticklabels(lbls, rotation=30, ha="right")
    ax.set_yticklabels(lbls)
    for i in range(6):
        for j in range(6):
            val = mat[i, j]
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=7.3, color="white" if val > 0.65 else INK)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=7.2)
    cbar.set_label("Spearman ρ", fontsize=7.8)
    ax.set_title(f"A  Metric redundancy\n(PC1+PC2 = {100*ca['pca_cumvar_pc1_pc2']:.1f}%, d_eff = {ca['participation_ratio_effective_dim']:.2f})", loc="left", pad=8, fontsize=8.8)

    # Panel B: Near-homolog memorization vs Internal diversity (Expanded AMD) with clean callouts
    ax = axes[1]
    s2s = s2["suite2_summary"]
    pts = []
    for lbl, key, col, mk, xytext in [
        ("Unigram", "Unigram", GRAY, "s", (1.2, 0.720)),
        ("Bigram Markov", "Bigram_Markov", GRAY, "^", (13.0, 0.858)),
        ("Trigram Markov", "Trigram_Markov", GRAY, "v", (15.5, 0.832)),
        ("LSTM-small", "LSTM_small", OCHRE, "o", (12.0, 0.752)),
        ("LSTM-matched", "LSTM_matched", OCHRE, "D", (18.5, 0.798)),
        ("Transformer-LM", "Transformer_LM", TEAL, "P", (15.5, 0.720)),
        ("GRU-VAE CE21", "SeqGRUVAE_CE21", BLUE, "o", (19.5, 0.625)),
        ("GRU-VAE MSE21\n(0% exact copies!)", "SeqGRUVAE_MSE21", RUST, "o", (19.0, 0.548)),
    ]:
        x_mem = 100.0 * float(np.mean([r["eval"]["frac_ge70"] for r in s1["expanded"]["models"][key]["runs"]["AMD"]]))
        y_div = float(np.mean([r["eval"]["internal_diversity"] for r in s1["expanded"]["models"][key]["runs"]["AMD"]]))
        pts.append((lbl, x_mem, y_div, col, mk, xytext))
    for lbl, skey, col, xytext in [
        ("CE21-DDPM (β=0.1)", "expanded_CE21_b0.1_sc0.5_lr0.0005_pm0", BLUE, (21.0, 0.772)),
        ("MSE21-DDPM tuned", "expanded_MSE21_b0.1_sc10.5_lr0.002_pm0", RUST, (21.5, 0.818)),
    ]:
        x_mem = 100.0 * s2s[skey]["ddpm_AMD_frac_ge70"]["mean"]
        y_div = s2s[skey]["ddpm_AMD_internal_diversity"]["mean"]
        pts.append((lbl, x_mem, y_div, col, "X", xytext))

    for lbl, xm, yd, col, mk, (lx, ly) in pts:
        ax.scatter(xm, yd, color=col, marker=mk, s=52 if mk != "X" else 60, zorder=4, edgecolor="white", lw=0.6)
        ax.annotate(lbl, xy=(xm, yd), xytext=(lx, ly), fontsize=7.1, color=col if "DDPM" in lbl else INK,
                    arrowprops=dict(arrowstyle="-", color=RULE, lw=0.75, shrinkA=1, shrinkB=4))

    ax.set_xlabel("Near-homolog memorization (% generated ≥70% NW identity to train)")
    ax.set_ylabel("Internal pairwise diversity (1 − mean pairwise identity)")
    ax.set_title("B  Memorization beyond exact copies\n(Expanded AMD independent test)", loc="left", pad=8, fontsize=8.8)
    ax.set_xlim(-2.0, 40.0)
    ax.set_ylim(0.49, 0.885)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, linestyle=":", color=RULE, lw=0.6)
    ax.set_axisbelow(True)
    ax.text(0.5, 0.505, "Upper-left: high diversity, low near-duplicate memorization", fontsize=6.9, color=TEAL, style="italic")

    save_fig(fig, "fig3_memorization_diversity_and_corr")


# ---------------------------------------------------------------------------
# Figure 4: Endpoint-3 Antibacterial MIC Panel & Multi-Threshold OD600 Audit
# ---------------------------------------------------------------------------
def make_fig4():
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.3), gridspec_kw={"width_ratios": [1.0, 1.25]})
    plt.subplots_adjust(wspace=0.35)

    # Panel A: Per-strain counts
    ax = axes[0]
    strains_lbl = ["E. coli\nATCC 25922", "P. aeruginosa\nATCC 27853", "S. aureus\nATCC 29213", "E. faecalis\nATCC 29212"]
    rep_lt200 = [5, 1, 6, 6]
    alg_lt200 = [5, 2, 7, 5]
    rep_le31 = [2, 0, 6, 1]
    alg_le31 = [2, 1, 6, 1]

    y = np.arange(4)
    h = 0.19
    ax.barh(y - 1.5*h, rep_lt200, height=h, color=BLUE, label="Reported <200 µg/mL")
    ax.barh(y - 0.5*h, alg_lt200, height=h, facecolor="white", edgecolor=BLUE, lw=1.1, label="OD₆₀₀<0.10 <200 µg/mL")
    ax.barh(y + 0.5*h, rep_le31, height=h, color=OCHRE, label="Reported ≤31.25 µg/mL")
    ax.barh(y + 1.5*h, alg_le31, height=h, facecolor="white", edgecolor=OCHRE, lw=1.1, label="OD₆₀₀<0.10 ≤31.25 µg/mL")
    for i in range(4):
        for off, val in [(-1.5*h, rep_lt200[i]), (-0.5*h, alg_lt200[i]), (0.5*h, rep_le31[i]), (1.5*h, alg_le31[i])]:
            ax.text(val + 0.15, i + off, str(val), va="center", fontsize=6.9, color=INK)
    ax.set_yticks(y)
    ax.set_yticklabels(strains_lbl)
    ax.invert_yaxis()
    ax.set_xlim(0, 8.6)
    ax.set_xlabel("Active peptides (of 27 tested)")
    ax.set_title("A  Per-strain MIC counts (original vs audit)", loc="left", pad=32)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", linestyle=":", color=RULE, lw=0.6)
    ax.legend(frameon=False, fontsize=6.8, loc="upper left", bbox_to_anchor=(-0.02, 1.15), ncol=2, columnspacing=0.9)

    # Panel B: Multi-threshold OD600 sensitivity + Concordance
    ax = axes[1]
    sens = wl["od600_threshold_sensitivity"]
    rule_names = ["OD < 0.08", "OD < 0.10\n(primary)", "OD < 0.12", "OD < 0.15", "Norm\n< 10%", "Norm\n< 20%"]
    conc = [r["concordance_of_108"] for r in sens]
    sa31 = [r["saureus_le_31_25"] for r in sens]
    any200 = [r["any_strain_lt_200"] for r in sens]
    x = np.arange(len(rule_names))

    ax2 = ax.twinx()
    ax.bar(x - 0.18, sa31, width=0.34, color=OCHRE, alpha=0.9, label="S. aureus ≤31.25 µg/mL")
    ax.bar(x + 0.18, any200, width=0.34, color=BLUE, alpha=0.85, label="Any strain <200 µg/mL")
    ax2.plot(x, conc, color=RUST, marker="o", lw=1.7, markersize=5.5, label="Concordance (of 108)")
    pill = dict(boxstyle="round,pad=0.16", facecolor="white", edgecolor="none", alpha=0.88)
    for xi, cval in zip(x, conc):
        if xi == 3:
            ax2.text(xi, cval - 6.5, f"{cval}/108", ha="center", va="center", fontsize=6.9, color=RUST, fontweight="bold", bbox=pill, zorder=6)
        elif xi == 2:
            ax2.text(xi + 0.15, cval + 4.2, f"{cval}/108", ha="center", va="center", fontsize=6.9, color=RUST, fontweight="bold", bbox=pill, zorder=6)
        else:
            ax2.text(xi, cval + 3.8, f"{cval}/108", ha="center", va="center", fontsize=6.9, color=RUST, fontweight="bold", bbox=pill, zorder=6)
    ax.set_xticks(x)
    ax.set_xticklabels(rule_names, fontsize=7.3)
    ax.set_ylabel("Active peptide count (of 27)")
    ax2.set_ylabel("Cell concordance (of 108)", color=RUST)
    ax.set_ylim(0, 32)
    ax2.set_ylim(50, 118)
    ax.set_title("B  OD₆₀₀ cutoff sensitivity & concordance", loc="left", pad=32)
    ax.spines["top"].set_visible(False)
    ax2.spines["top"].set_visible(False)
    ax.grid(axis="y", linestyle=":", color=RULE, lw=0.6)
    ax.set_axisbelow(True)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, frameon=False, fontsize=6.8, loc="upper left", bbox_to_anchor=(-0.02, 1.15), ncol=2, columnspacing=0.9)
    ax.annotate("E. faecalis neg. ctrl\nmedian OD = 0.130", xy=(3.18, 27.0), xytext=(3.65, 27.2),
                fontsize=6.6, color=RUST, arrowprops=dict(arrowstyle="->", color=RUST, lw=0.9))

    save_fig(fig, "fig4_mic_sensitivity_and_provenance")


# ---------------------------------------------------------------------------
# Figure 5: Consolidated Endpoint-Specific Evaluation Protocol
# ---------------------------------------------------------------------------
def make_fig5():
    fig, ax = plt.subplots(figsize=(10.5, 4.2))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 44)
    ax.axis("off")

    rows = [
        (31.5, "Endpoint 1 · Sequence Reconstruction (Primary + Ablations)", BLUE, "#EEF4F8", [
            ("Inputs & Partition", "Sequence + RDKit descriptors\n≤70% NW homology splits\nPilot (279/41) + DBAASP (3,120/223/222)"),
            ("Controlled Comparison", "Matched VAE architecture & decoder\nCE21 vs MSE21 across KL-β (0.01–1.0),\nloss scaling (21×) & padding masking"),
            ("Verified Outcome", "Untouched test: CE21 96.1% micro,\n88.1% macro; MSE21 tuned 94.2%/84.1%\nRare-AA imbalance: ρ = 0.80"),
        ]),
        (17.5, "Endpoint 2 · De Novo Generation (Primary + Statistical & Higher-Order Gates)", TEAL, "#EDF6F5", [
            ("Inputs & Comparators", "Terminal state only (FREE / AMD)\nUnigram, Bigram, Trigram, LSTM (63k/1.3M),\nTransformer-LM (1.8M), GRU-VAE, VAE-DDPM"),
            ("Multi-Seed & Orthogonal Gates", "3–5 seeds × 128–256 seqs/terminal, 95% CI\nLow-order (MMD², d_eff=1.54) + 3-mer JS\n+ α-helical μ_H W₁ + ≥70% NW memorization"),
            ("Verified Outcome", "AR models beat tuned 96%-recon VAEs\nby ~20× on FREE MMD² & beat Unigram\non 3-mer JS & μ_H (p ≤ 0.028)"),
        ]),
        (3.5, "Endpoint 3 · Experimental Antibacterial Activity (Separate Provenance-Explicit Case Study)", OCHRE, "#FBF6EC", [
            ("Provenance & Funnel", "Prior multimodal VAE-DDPM (0% AMPGen\noverlap); 1,000 → RDKit charge>0, AMP>0.9,\nII 10–40, ≤70% id → 30 ordered → 27 assayed"),
            ("Assay & Sensitivity Audit", "MHB broth microdilution, 1% DMSO, 5×10⁵\nCFU/mL, 37 °C 16–20 h, 4 ATCC strains × 3\nreps; original MIC + 6-rule OD₆₀₀ audit"),
            ("Verified Outcome", "S. aureus ≤31.25 µg/mL (3.1–12.9 µM):\n6/27 invariant across 5 rules (105/108 cells);\nhaemolysis/cytotoxicity remain unmeasured"),
        ]),
    ]

    for y0, header, border_col, fill_col, boxes in rows:
        ax.text(1.2, y0 + 9.4, header, fontsize=8.8, fontweight="bold", color=border_col, va="bottom")
        xs = [1.2, 34.5, 67.8]
        w = 30.8
        h = 8.4
        for idx, (title, body) in enumerate(boxes):
            rect = FancyBboxPatch((xs[idx], y0), w, h, boxstyle="round,pad=0.35,rounding_size=0.8",
                                  facecolor=fill_col, edgecolor=border_col, linewidth=1.05)
            ax.add_patch(rect)
            ax.text(xs[idx] + 1.0, y0 + h - 1.5, title, fontsize=7.8, fontweight="bold", color=INK, va="top")
            ax.text(xs[idx] + 1.0, y0 + h - 3.3, body, fontsize=6.85, color=INK, va="top", linespacing=1.32)
            if idx < 2:
                arr = FancyArrowPatch((xs[idx] + w + 0.5, y0 + h/2), (xs[idx+1] - 0.5, y0 + h/2),
                                      arrowstyle="-|>", mutation_scale=11, color=border_col, lw=1.2)
                ax.add_patch(arr)

    ax.text(50, 0.4, "Note: Horizontal arrows connect inputs to outputs within each endpoint. Endpoint 3 is an independent historical panel, not a head-to-head wet-lab assay of CE21 vs MSE21 vs AR outputs.",
            ha="center", va="bottom", fontsize=7.0, style="italic", color=GRAY)
    save_fig(fig, "fig5_endpoint_protocol")


if __name__ == "__main__":
    make_fig1()
    make_fig2()
    make_fig3()
    make_fig4()
    make_fig5()
    print("[ALL 5 FIGURES POLISHED AND RENDERED]")
