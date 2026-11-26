#!/usr/bin/env python3
"""Comprehensive evaluation metrics, physicochemical descriptors, memorization/diversity
audits, and statistical uncertainty tools for BIB-26-2683 Major Revision.

All original six distributional metrics match frozen_metrics_v1.py / supplement S2 exactly.
"""
from __future__ import annotations

import collections
import math
from typing import Dict, List, Sequence, Tuple
import numpy as np
from scipy import stats

AA = "ACDEFGHIKLMNPQRSTVWY"
AA_SET = frozenset(AA)
AA_INDEX = {a: i for i, a in enumerate(AA)}
DIPEPTIDES = [a + b for a in AA for b in AA]
DIPEP_INDEX = {dp: i for i, dp in enumerate(DIPEPTIDES)}
MAX_LEN = 48

# Eisenberg consensus hydrophobicity scale
EISENBERG = {
    "A": 0.62, "R": -2.53, "N": -0.78, "D": -0.90, "C": 0.29,
    "Q": -0.85, "E": -0.74, "G": 0.48, "H": -0.40, "I": 1.38,
    "L": 1.06, "K": -1.50, "M": 0.64, "F": 1.19, "P": 0.12,
    "S": -0.18, "T": -0.05, "W": 0.81, "Y": 0.26, "V": 1.08,
}

# Boman (Radzicka-Wolfenden) solubility-to-cyclohexane transfer free energies (kcal/mol)
BOMAN = {
    "L": -4.92, "I": -4.92, "V": -4.04, "F": -2.98, "M": -2.35,
    "W": -2.33, "A": -1.81, "C": -1.28, "G": -0.94, "Y": 0.14,
    "T": 2.57, "S": 3.40, "H": 4.66, "Q": 5.54, "K": 5.55,
    "N": 6.64, "E": 6.81, "D": 8.72, "R": 14.92, "P": 0.00,
}

# Standard pKa values (EMBOSS/Bjellqvist) for Henderson-Hasselbalch charge at pH 7.4
PKA_POS = {"K": 10.54, "R": 12.48, "H": 6.04}
PKA_NEG = {"D": 3.90, "E": 4.07, "C": 8.37, "Y": 10.46}
PKA_NTERM = 8.00
PKA_CTERM = 3.10

# Average residue molecular weights (Da)
RES_MW = {
    "A": 89.094, "R": 174.203, "N": 132.119, "D": 133.104, "C": 121.154,
    "E": 147.131, "Q": 146.146, "G": 75.067, "H": 155.156, "I": 131.175,
    "L": 131.175, "K": 146.189, "M": 149.208, "F": 165.192, "P": 115.132,
    "S": 105.093, "T": 119.120, "W": 204.228, "Y": 181.191, "V": 117.148,
}
H2O_MW = 18.01528


def net_charge_ph74(seq: str, terminal: str = "FREE") -> float:
    pH = 7.4
    pos = 1.0 / (1.0 + 10.0 ** (pH - PKA_NTERM))
    for ch in seq:
        if ch in PKA_POS:
            pos += 1.0 / (1.0 + 10.0 ** (pH - PKA_POS[ch]))
    neg = 0.0 if terminal == "AMD" else 1.0 / (1.0 + 10.0 ** (PKA_CTERM - pH))
    for ch in seq:
        if ch in PKA_NEG:
            neg += 1.0 / (1.0 + 10.0 ** (PKA_NEG[ch] - pH))
    return pos - neg


def mean_hydrophobicity(seq: str) -> float:
    return sum(EISENBERG[c] for c in seq) / len(seq) if seq else 0.0


def hydrophobic_moment_alpha(seq: str, angle_deg: float = 100.0) -> float:
    """Eisenberg alpha-helical hydrophobic moment (mu_H) at 100 degrees per residue."""
    if not seq:
        return 0.0
    rad = math.radians(angle_deg)
    sx = sum(EISENBERG[c] * math.sin(i * rad) for i, c in enumerate(seq))
    cx = sum(EISENBERG[c] * math.cos(i * rad) for i, c in enumerate(seq))
    return math.hypot(sx, cx) / len(seq)


def boman_index(seq: str) -> float:
    return sum(BOMAN[c] for c in seq) / len(seq) if seq else 0.0


def aliphatic_index(seq: str) -> float:
    if not seq:
        return 0.0
    n = len(seq)
    return 100.0 * (seq.count("A") + 2.9 * seq.count("V") + 3.9 * (seq.count("I") + seq.count("L"))) / n


def aromaticity(seq: str) -> float:
    if not seq:
        return 0.0
    return sum(seq.count(c) for c in "FWY") / len(seq)


def peptide_mw(seq: str, terminal: str = "FREE") -> float:
    if not seq:
        return 0.0
    mw = sum(RES_MW[c] for c in seq) - (len(seq) - 1) * H2O_MW
    if terminal == "AMD":
        mw -= 0.984016
    return mw


def physchem_summary(seqs: Sequence[str], terminal: str = "FREE") -> Dict[str, np.ndarray]:
    return {
        "charge": np.array([net_charge_ph74(s, terminal) for s in seqs], dtype=np.float64),
        "hydrophobicity": np.array([mean_hydrophobicity(s) for s in seqs], dtype=np.float64),
        "mu_H": np.array([hydrophobic_moment_alpha(s) for s in seqs], dtype=np.float64),
        "boman": np.array([boman_index(s) for s in seqs], dtype=np.float64),
        "aliphatic": np.array([aliphatic_index(s) for s in seqs], dtype=np.float64),
        "aromaticity": np.array([aromaticity(s) for s in seqs], dtype=np.float64),
    }


def _js_divergence_bits(p: np.ndarray, q: np.ndarray) -> float:
    m = 0.5 * (p + q)
    def _kl(a, b):
        mask = a > 0
        return float(np.sum(a[mask] * np.log2(a[mask] / b[mask])))
    return 0.5 * (_kl(p, m) + _kl(q, m))


def _feature_matrix(seqs: Sequence[str]) -> np.ndarray:
    w = 1.0 / math.sqrt(3.0)
    mat = np.zeros((len(seqs), 1 + 20 + 400), dtype=np.float64)
    for i, s in enumerate(seqs):
        L = len(s)
        mat[i, 0] = (L / float(MAX_LEN)) * w
        for ch in s:
            mat[i, 1 + AA_INDEX[ch]] += w / L
        if L >= 2:
            denom = L - 1
            for k in range(denom):
                mat[i, 21 + DIPEP_INDEX[s[k:k+2]]] += w / denom
    return mat


def _pooled_aa_dist(seqs: Sequence[str]) -> np.ndarray:
    cnt = np.zeros(20, dtype=np.float64)
    for s in seqs:
        for ch in s:
            cnt[AA_INDEX[ch]] += 1.0
    tot = cnt.sum()
    return cnt / tot if tot > 0 else cnt


def _pooled_kmer_dist(seqs: Sequence[str], k: int) -> Dict[str, float]:
    cnt = collections.Counter()
    tot = 0
    for s in seqs:
        if len(s) >= k:
            for i in range(len(s) - k + 1):
                cnt[s[i:i+k]] += 1
                tot += 1
    if tot == 0:
        return {}
    return {km: v / tot for km, v in cnt.items()}


def _dict_js_l1(d1: Dict[str, float], d2: Dict[str, float]) -> Tuple[float, float]:
    keys = set(d1.keys()) | set(d2.keys())
    if not keys:
        return 0.0, 0.0
    p = np.array([d1.get(k, 0.0) for k in keys], dtype=np.float64)
    q = np.array([d2.get(k, 0.0) for k in keys], dtype=np.float64)
    return _js_divergence_bits(p, q), float(np.sum(np.abs(p - q)))


def _entropy_bits(probs: Sequence[float]) -> float:
    return float(-sum(p * math.log2(p) for p in probs if p > 0))


def global_identity_fast(a: str, b: str) -> float:
    """Exact Needleman-Wunsch (+2, -1, -2), matches / all alignment columns."""
    la, lb = len(a), len(b)
    if la == 0 and lb == 0:
        return 1.0
    prev = [(-2 * j, 0, -j) for j in range(lb + 1)]
    for i, aa in enumerate(a, 1):
        curr = [(-2 * i, 0, -i)]
        for j, bb in enumerate(b, 1):
            m = 1 if aa == bb else 0
            diag = (prev[j - 1][0] + (2 if m else -1), prev[j - 1][1] + m, prev[j - 1][2] - 1)
            dele = (prev[j][0] - 2, prev[j][1], prev[j][2] - 1)
            inse = (curr[j - 1][0] - 2, curr[j - 1][1], curr[j - 1][2] - 1)
            curr.append(max(diag, dele, inse))
        prev = curr
    _, matches, neg_cols = prev[-1]
    return matches / (-neg_cols) if neg_cols else 1.0


def nearest_neighbor_identities(gen_seqs: Sequence[str], ref_seqs: Sequence[str], max_eval: int = 128) -> Dict[str, float]:
    """Compute exact nearest-neighbor global NW identity to reference set (pruned upper bound)."""
    if not gen_seqs or not ref_seqs:
        return {"nn_mean": float("nan"), "nn_median": float("nan"), "nn_p95": float("nan"),
                "nn_max": float("nan"), "frac_ge70": float("nan"), "frac_ge80": float("nan"),
                "exact_copy_frac": float("nan")}
    ref_set = set(ref_seqs)
    exact_copies = sum(1 for s in gen_seqs if s in ref_set) / len(gen_seqs)
    # Deterministic subsample if > max_eval for speed, preserving exact order
    sub = list(gen_seqs[:max_eval])
    # Precompute AA count tuples for ref
    ref_list = sorted(ref_set)
    ref_lens = [len(s) for s in ref_list]
    ref_cnts = []
    for s in ref_list:
        v = [0] * 20
        for ch in s:
            v[AA_INDEX[ch]] += 1
        ref_cnts.append(tuple(v))

    nn_vals = []
    for s in sub:
        if s in ref_set:
            nn_vals.append(1.0)
            continue
        ls = len(s)
        cs = [0] * 20
        for ch in s:
            cs[AA_INDEX[ch]] += 1
        # Order ref by length ratio descending
        order = sorted(range(len(ref_list)), key=lambda idx: -min(ls, ref_lens[idx]) / max(ls, ref_lens[idx]))
        best = 0.0
        for idx in order:
            lr = ref_lens[idx]
            upper_len = min(ls, lr) / max(ls, lr)
            if upper_len <= best:
                break
            cr = ref_cnts[idx]
            common = 0
            for k in range(20):
                va, vb = cs[k], cr[k]
                common += va if va < vb else vb
            if common / max(ls, lr) <= best:
                continue
            val = global_identity_fast(s, ref_list[idx])
            if val > best:
                best = val
        nn_vals.append(best)
    arr = np.array(nn_vals, dtype=np.float64)
    return {
        "nn_mean": float(np.mean(arr)),
        "nn_median": float(np.median(arr)),
        "nn_p95": float(np.percentile(arr, 95)),
        "nn_max": float(np.max(arr)),
        "frac_ge70": float(np.mean(arr >= 0.70)),
        "frac_ge80": float(np.mean(arr >= 0.80)),
        "exact_copy_frac": float(exact_copies),
    }


def internal_diversity(gen_seqs: Sequence[str], max_eval: int = 64) -> Dict[str, float]:
    """Compute unique fraction, 3-mer richness, and internal pairwise global identity (1 - diversity)."""
    if not gen_seqs:
        return {"unique_frac": 0.0, "internal_mean_id": float("nan"), "internal_diversity": float("nan"), "trimer_unique_ratio": 0.0}
    uniq_frac = len(set(gen_seqs)) / len(gen_seqs)
    trimers = []
    for s in gen_seqs:
        if len(s) >= 3:
            for i in range(len(s) - 2):
                trimers.append(s[i:i+3])
    trimer_ratio = len(set(trimers)) / len(trimers) if trimers else 0.0
    sub = list(gen_seqs[:max_eval])
    m = len(sub)
    if m < 2:
        return {"unique_frac": uniq_frac, "internal_mean_id": float("nan"), "internal_diversity": float("nan"), "trimer_unique_ratio": trimer_ratio}
    pair_ids = []
    for i in range(m):
        for j in range(i + 1, m):
            pair_ids.append(global_identity_fast(sub[i], sub[j]))
    mean_id = float(np.mean(pair_ids))
    return {
        "unique_frac": float(uniq_frac),
        "internal_mean_id": mean_id,
        "internal_diversity": float(1.0 - mean_id),
        "trimer_unique_ratio": float(trimer_ratio),
    }


def compute_all_generation_metrics(
    gen_seqs: Sequence[str],
    ref_seqs: Sequence[str],
    train_seqs: Sequence[str],
    terminal: str = "FREE",
    compute_nn: bool = True,
    nn_max_eval: int = 128,
) -> Dict[str, float]:
    """Compute original 6 distances + higher-order 3-mer + physchem Wasserstein-1 + memorization/diversity."""
    valid_1_48 = [s for s in gen_seqs if isinstance(s, str) and 1 <= len(s) <= MAX_LEN and set(s) <= AA_SET]
    valid_5_48 = [s for s in valid_1_48 if len(s) >= 5]
    if not valid_1_48 or not ref_seqs:
        return {"n_valid_1_48": 0, "n_valid_5_48": 0}

    # Original 6 distances on length 1-48
    p_aa = _pooled_aa_dist(valid_1_48)
    q_aa = _pooled_aa_dist(ref_seqs)
    aa_js = _js_divergence_bits(p_aa, q_aa)
    aa_l1 = float(np.sum(np.abs(p_aa - q_aa)))

    p_dp = _pooled_kmer_dist(valid_1_48, 2)
    q_dp = _pooled_kmer_dist(ref_seqs, 2)
    pair_js, pair_l1 = _dict_js_l1(p_dp, q_dp)

    p_tri = _pooled_kmer_dist(valid_1_48, 3)
    q_tri = _pooled_kmer_dist(ref_seqs, 3)
    tri_js, tri_l1 = _dict_js_l1(p_tri, q_tri)

    len_gen = np.array([len(s) for s in valid_1_48], dtype=np.float64)
    len_ref = np.array([len(s) for s in ref_seqs], dtype=np.float64)
    len_ks = float(stats.ks_2samp(len_gen, len_ref).statistic)

    fg = _feature_matrix(valid_1_48)
    fr = _feature_matrix(ref_seqs)
    diff = fg.mean(axis=0) - fr.mean(axis=0)
    mmd2 = float(np.dot(diff, diff))

    # Entropy gaps
    aa_ent_gap = abs(_entropy_bits(p_aa) - _entropy_bits(q_aa))
    pair_ent_gap = abs(_entropy_bits(p_dp.values()) - _entropy_bits(q_dp.values()))

    # Short-sequence filtered (length 5-48) sensitivity on MMD2 & AA JS & Pair JS
    if valid_5_48:
        fg5 = _feature_matrix(valid_5_48)
        diff5 = fg5.mean(axis=0) - fr.mean(axis=0)
        mmd2_5_48 = float(np.dot(diff5, diff5))
        aa_js_5_48 = _js_divergence_bits(_pooled_aa_dist(valid_5_48), q_aa)
        pair_js_5_48, _ = _dict_js_l1(_pooled_kmer_dist(valid_5_48, 2), q_dp)
    else:
        mmd2_5_48 = float("nan")
        aa_js_5_48 = float("nan")
        pair_js_5_48 = float("nan")

    # Physicochemical & structural Wasserstein-1 distances
    pc_gen = physchem_summary(valid_1_48, terminal)
    pc_ref = physchem_summary(ref_seqs, terminal)
    pc_metrics = {}
    for k in pc_gen:
        w1 = float(stats.wasserstein_distance(pc_gen[k], pc_ref[k]))
        ks = float(stats.ks_2samp(pc_gen[k], pc_ref[k]).statistic)
        pc_metrics[f"w1_{k}"] = w1
        pc_metrics[f"ks_{k}"] = ks
        pc_metrics[f"mean_{k}"] = float(np.mean(pc_gen[k]))

    out = {
        "n_requests": len(gen_seqs),
        "n_valid_1_48": len(valid_1_48),
        "n_valid_5_48": len(valid_5_48),
        "frac_len_lt5": float(1.0 - len(valid_5_48) / len(valid_1_48)),
        "mean_length": float(np.mean(len_gen)),
        "MMD2": mmd2,
        "AA_JS": aa_js,
        "AA_L1": aa_l1,
        "Pair_JS": pair_js,
        "Pair_L1": pair_l1,
        "Length_KS": len_ks,
        "Tri_JS": tri_js,
        "Tri_L1": tri_l1,
        "AA_ent_gap": aa_ent_gap,
        "Pair_ent_gap": pair_ent_gap,
        "MMD2_5_48": mmd2_5_48,
        "AA_JS_5_48": aa_js_5_48,
        "Pair_JS_5_48": pair_js_5_48,
    }
    out.update(pc_metrics)

    if compute_nn:
        out.update(nearest_neighbor_identities(valid_1_48, train_seqs, max_eval=nn_max_eval))
        out.update(internal_diversity(valid_1_48, max_eval=64))
    return out


def summarize_seeds(values: Sequence[float]) -> Dict[str, float]:
    arr = np.array([v for v in values if v is not None and math.isfinite(v)], dtype=np.float64)
    n = len(arr)
    if n == 0:
        return {"n": 0, "mean": float("nan"), "sd": float("nan"), "ci95_lo": float("nan"), "ci95_hi": float("nan")}
    mean = float(np.mean(arr))
    sd = float(np.std(arr, ddof=1)) if n > 1 else 0.0
    if n > 1:
        t_crit = float(stats.t.ppf(0.975, df=n - 1))
        half = t_crit * sd / math.sqrt(n)
    else:
        half = 0.0
    return {"n": n, "mean": mean, "sd": sd, "ci95_lo": mean - half, "ci95_hi": mean + half}


def paired_seed_test(vals_a: Sequence[float], vals_b: Sequence[float]) -> Dict[str, float]:
    a = np.array(vals_a, dtype=np.float64)
    b = np.array(vals_b, dtype=np.float64)
    diff = a - b
    t_stat, t_p = stats.ttest_rel(a, b)
    try:
        w_stat, w_p = stats.wilcoxon(a, b)
    except Exception:
        w_stat, w_p = float("nan"), float("nan")
    # Exact sign-flip permutation p-value for small n (e.g. n=5 -> 32 sign combinations)
    n = len(diff)
    obs = abs(float(np.mean(diff)))
    if n <= 12:
        ge = 0
        total = 1 << n
        for mask in range(total):
            s = sum(diff[i] if (mask >> i) & 1 else -diff[i] for i in range(n))
            if abs(s / n) >= obs - 1e-12:
                ge += 1
        perm_p = ge / total
    else:
        perm_p = float("nan")
    return {
        "mean_diff_a_minus_b": float(np.mean(diff)),
        "sd_diff": float(np.std(diff, ddof=1)) if n > 1 else 0.0,
        "paired_t": float(t_stat),
        "paired_t_p": float(t_p),
        "wilcoxon_p": float(w_p),
        "exact_perm_p": float(perm_p),
    }
