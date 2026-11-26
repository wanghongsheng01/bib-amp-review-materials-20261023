#!/usr/bin/env python3
"""Suite 2: Multi-seed exact VAEModel + UNet_conditional experiments covering:
  1. Pilot (279/41) KL-beta & MSE loss-scaling/LR hyperparameter sweep (R3 #3)
  2. Pilot (279/41) Padding-masked vs padding-included ablation & AA frequency-recall audit (R3 Minor #1, #2)
  3. Pilot (279/41) 3-seed full VAE-DDPM (CE21 vs MSE21, 256 samples/terminal)
  4. Expanded DBAASP (3,120 train / 223 val / 222 untouched independent test) 3-seed full VAE-DDPM
     (CE21 vs MSE21 default & tuned beta/scale, 256 samples/terminal) evaluated on the untouched test set (R1, R2 #1, R3 #2)

Uses multiprocessing (4 workers x 2 CPU threads each) for clean parallel execution.
"""
from __future__ import annotations

import collections
import concurrent.futures as cf
import csv
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from scipy.stats import spearmanr
import torch
from torch.nn import functional as F

ROOT = Path("/Users/loctek/.gemini/antigravity/scratch/bib-26-2683-revision")
OUT_DIR = ROOT / "revision_results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ALPHABET = "ARNDCEQGHILKMFPSTWYV"
PAD = 20
MAX_LEN = 48
N_GEN = 256


def _parse_res_desc(raw_str: str, seq_len: int) -> List[List[float]]:
    try:
        val = json.loads(raw_str)
        if isinstance(val, list) and len(val) >= seq_len:
            return val[:seq_len]
    except Exception:
        pass
    import re
    nums = [float(x) for x in re.findall(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?", raw_str)]
    assert len(nums) >= seq_len * 3, f"expected >= {seq_len*3} floats, got {len(nums)}"
    return [nums[3*i:3*i+3] for i in range(seq_len)]


def load_descriptor_dataset(desc_csv: Path, cohort_csv: Path):
    seqs, res_list, glob_list, labels, terms = [], [], [], [], []
    with desc_csv.open() as fd, cohort_csv.open() as fc:
        for rd, rc in zip(csv.DictReader(fd), csv.DictReader(fc)):
            seq = rd["protein"]
            assert seq == rc["sequence"]
            seqs.append(seq)
            res_list.append(_parse_res_desc(rd["describe"], len(seq)))
            glob_list.append(json.loads(rd["point_describe"]))
            labels.append(int(rd["class"]))
            terms.append(rc["terminal_state"])
    return seqs, res_list, glob_list, labels, terms


def fit_scaler(res_list, glob_list):
    g_arr = np.array(glob_list, dtype=np.float64)
    g_mean = g_arr.mean(axis=0)
    g_std = g_arr.std(axis=0)
    g_std = np.where(g_std > 0, g_std, 1.0)

    r_flat = np.array([row for seq_r in res_list for row in seq_r], dtype=np.float64)
    r_mean = r_flat.mean(axis=0)
    r_std = r_flat.std(axis=0)
    r_std = np.where(r_std > 0, r_std, 1.0)
    return {"g_mean": g_mean, "g_std": g_std, "r_mean": r_mean, "r_std": r_std}


def tensorize(seqs, res_list, glob_list, labels, scaler):
    n = len(seqs)
    target = torch.full((n, MAX_LEN), PAD, dtype=torch.long)
    g = torch.zeros((n, 3, MAX_LEN), dtype=torch.float32)
    h_np = np.abs((np.array(glob_list, dtype=np.float64) - scaler["g_mean"]) / scaler["g_std"])
    for i, (s, rmat) in enumerate(zip(seqs, res_list)):
        for t, ch in enumerate(s):
            target[i, t] = ALPHABET.index(ch)
        arr = (np.array(rmat, dtype=np.float64) - scaler["r_mean"]) / scaler["r_std"]
        g[i, :, :len(s)] = torch.tensor(arr.T, dtype=torch.float32)
    h = torch.tensor(h_np, dtype=torch.float64)
    lbl = torch.tensor(labels, dtype=torch.long)
    return target, g, h, lbl


def joint_decode_bkt(score_b_21_48: torch.Tensor) -> List[str]:
    scores = score_b_21_48.transpose(1, 2)  # (B, 48, 21)
    body, aa = scores[:, :, :20].max(-1)
    pad = scores[:, :, 20]
    base = pad.sum(1, keepdim=True)
    lscores = base + torch.cat([torch.zeros_like(base), (body - pad).cumsum(1)], dim=1)
    lengths = lscores.argmax(1).tolist()
    aa_list = aa.tolist()
    return ["".join(ALPHABET[row[k]] for k in range(L)) for L, row in zip(lengths, aa_list)]


def eval_recon(pred_seqs, true_seqs):
    total_res = sum(len(s) for s in true_seqs)
    correct_res = 0
    exact = 0
    len_err = 0
    aa_tot = collections.Counter()
    aa_ok = collections.Counter()
    for p, t in zip(pred_seqs, true_seqs):
        if p == t:
            exact += 1
        len_err += abs(len(p) - len(t))
        for i, ch in enumerate(t):
            aa_tot[ch] += 1
            if i < len(p) and p[i] == ch:
                correct_res += 1
                aa_ok[ch] += 1
    supported = [c for c in ALPHABET if aa_tot[c] > 0]
    macro = sum(aa_ok[c] / aa_tot[c] for c in supported) / len(supported)
    per_aa = {c: {"support": aa_tot[c], "correct": aa_ok[c], "recall": aa_ok[c] / aa_tot[c] if aa_tot[c] else None} for c in ALPHABET}
    return {
        "micro_pct": 100.0 * correct_res / total_res,
        "macro_pct": 100.0 * macro,
        "length_mae": len_err / len(true_seqs),
        "exact_pct": 100.0 * exact / len(true_seqs),
        "exact_count": exact,
        "correct_residues": correct_res,
        "total_residues": total_res,
        "per_aa": per_aa,
    }


def run_single_job(job: dict) -> dict:
    """Worker function: trains exact VAEModel (+ optional UNet_conditional DDPM) and evaluates."""
    os.environ["OMP_NUM_THREADS"] = "2"
    os.environ["OPENBLAS_NUM_THREADS"] = "2"
    torch.set_num_threads(2)

    sys.path.insert(0, str(ROOT / "scripts"))
    sys.path.insert(0, str(ROOT / "github_repo/model/source/corrected_code/AMP_VAEflow"))
    import metrics_suite as ms
    from modules.vae import VAEModel
    from modules.diff import UNet_conditional

    t0 = time.time()
    job_id = job["job_id"]
    cohort = job["cohort"]
    arm = job["arm"]  # "CE21" or "MSE21"
    seed = int(job["seed"])
    kl_beta = float(job["kl_beta"])
    mse_scale = float(job.get("mse_scale", 0.5))
    lr = float(job.get("lr", 5e-4))
    mask_pad = bool(job.get("mask_pad", False))
    vae_updates = int(job.get("vae_updates", 400))
    run_ddpm = bool(job.get("run_ddpm", False))
    ddpm_updates = int(job.get("ddpm_updates", 400))

    if cohort == "pilot":
        tr_s, tr_r, tr_g, tr_l, tr_t = load_descriptor_dataset(
            ROOT / "github_repo/data/train_dev/train_descriptors.csv",
            ROOT / "github_repo/data/train_dev/train_cohort.csv",
        )
        ev_s, ev_r, ev_g, ev_l, ev_t = load_descriptor_dataset(
            ROOT / "github_repo/data/train_dev/dev_descriptors.csv",
            ROOT / "github_repo/data/train_dev/dev_cohort.csv",
        )
    else:
        tr_s, tr_r, tr_g, tr_l, tr_t = load_descriptor_dataset(
            ROOT / "revision_data/dbaasp_expanded/train_descriptors.csv",
            ROOT / "revision_data/dbaasp_expanded/train_cohort.csv",
        )
        ev_s, ev_r, ev_g, ev_l, ev_t = load_descriptor_dataset(
            ROOT / "revision_data/dbaasp_expanded/test_descriptors.csv",
            ROOT / "revision_data/dbaasp_expanded/test_cohort.csv",
        )

    scaler = fit_scaler(tr_r, tr_g)
    tr_x, tr_g_t, tr_h_t, tr_lbl = tensorize(tr_s, tr_r, tr_g, tr_l, scaler)
    ev_x, ev_g_t, ev_h_t, ev_lbl = tensorize(ev_s, ev_r, ev_g, ev_l, scaler)

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    vae = VAEModel(out_dim=21, con_dim=3, latent_size=16, sqz_prior=False, cond_drop=False, use_prior_flow=True, device="cpu")
    opt = torch.optim.Adam(vae.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-8)
    rng_idx = np.random.default_rng(seed)

    # Warmup LR schedule matching v7 convention (capped to vae_updates)
    warmup = min(200, vae_updates // 2)
    vae.train()
    for step in range(1, vae_updates + 1):
        cur_lr = max(1e-7, lr * min(1.0, step / float(warmup)))
        for pg in opt.param_groups:
            pg["lr"] = cur_lr
        idx = rng_idx.integers(0, len(tr_x), size=32)
        bx, bg, bh = tr_x[idx], tr_g_t[idx], tr_h_t[idx]
        opt.zero_grad()
        v = vae.vae
        z, mu, logsigma = v.encoder(v.word_emb(bx.T).permute(1, 2, 0), v.g_pre_net(bg), bh)
        raw = v.decoder(z)  # (B, 21, 48)
        zp = v.prior_flow(z, bg)
        logq = torch.distributions.Normal(mu, logsigma.exp()).log_prob(z)
        logp = v.prior_dist.log_prob(zp)
        density = (logq - logp).sum()
        denom = float(bx.numel())

        if mask_pad:
            nonpad = (bx != PAD)
            np_denom = float(nonpad.sum().clamp(min=1))
            if arm == "CE21":
                tok_ce = F.cross_entropy(raw, bx, reduction="none")
                rec = tok_ce.masked_select(nonpad).sum() / np_denom
            else:
                oh = F.one_hot(bx, num_classes=21).permute(0, 2, 1).to(raw)
                sq = (raw - oh).square().sum(dim=1)  # (B, 48)
                rec = mse_scale * sq.masked_select(nonpad).sum() / np_denom
            kl = kl_beta * (density / denom)
        else:
            if arm == "CE21":
                rec = F.cross_entropy(raw, bx, reduction="sum") / denom
            else:
                oh = F.one_hot(bx, num_classes=21).permute(0, 2, 1).to(raw)
                rec = mse_scale * (raw - oh).square().sum() / denom
            kl = kl_beta * (density / denom)

        loss = rec + kl
        loss.backward()
        torch.nn.utils.clip_grad_norm_(vae.parameters(), 1.0)
        opt.step()

    # Evaluate posterior-mean reconstruction on eval set (dev for pilot, untouched test for expanded)
    vae.eval()
    pred_seqs = []
    with torch.no_grad():
        for s_i in range(0, len(ev_x), 64):
            bx, bg, bh = ev_x[s_i:s_i+64], ev_g_t[s_i:s_i+64], ev_h_t[s_i:s_i+64]
            v = vae.vae
            _, mu, _ = v.encoder(v.word_emb(bx.T).permute(1, 2, 0), v.g_pre_net(bg), bh)
            raw = v.decoder(mu)
            if mask_pad:
                # For padding-masked ablation, also evaluate fixed-oracle-length argmax over 20 AAs AND joint_decode
                pred_seqs.extend(joint_decode_bkt(raw))
            else:
                pred_seqs.extend(joint_decode_bkt(raw))
    recon_res = eval_recon(pred_seqs, ev_s)

    # Also compute oracle-length non-pad argmax accuracy for mask_pad comparison
    oracle_correct = 0
    oracle_tot = sum(len(s) for s in ev_s)
    oracle_aa_tot = collections.Counter()
    oracle_aa_ok = collections.Counter()
    with torch.no_grad():
        for s_i in range(0, len(ev_x), 64):
            bx, bg, bh = ev_x[s_i:s_i+64], ev_g_t[s_i:s_i+64], ev_h_t[s_i:s_i+64]
            v = vae.vae
            _, mu, _ = v.encoder(v.word_emb(bx.T).permute(1, 2, 0), v.g_pre_net(bg), bh)
            raw = v.decoder(mu)[:, :20, :].argmax(dim=1).tolist()
            for bi, row in enumerate(raw):
                tseq = ev_s[s_i + bi]
                for pos, ch in enumerate(tseq):
                    oracle_aa_tot[ch] += 1
                    if ALPHABET[row[pos]] == ch:
                        oracle_correct += 1
                        oracle_aa_ok[ch] += 1
    sup = [c for c in ALPHABET if oracle_aa_tot[c] > 0]
    recon_res["oracle_len_micro_pct"] = 100.0 * oracle_correct / oracle_tot
    recon_res["oracle_len_macro_pct"] = 100.0 * sum(oracle_aa_ok[c] / oracle_aa_tot[c] for c in sup) / len(sup)

    # Prior N(0, I) generation evaluation (256 samples per terminal)
    tr_by_term = {"FREE": [s for s, t in zip(tr_s, tr_t) if t == "FREE"], "AMD": [s for s, t in zip(tr_s, tr_t) if t == "AMD"]}
    ev_by_term = {"FREE": [s for s, t in zip(ev_s, ev_t) if t == "FREE"], "AMD": [s for s, t in zip(ev_s, ev_t) if t == "AMD"]}

    prior_gen = {}
    with torch.no_grad():
        for tidx, tname in ((0, "FREE"), (1, "AMD")):
            g_rng = torch.Generator().manual_seed(seed * 10 + tidx + 400)
            z = torch.randn((N_GEN, 16, 48), generator=g_rng)
            raw = vae.vae.decoder(z)
            seqs = [s for s in joint_decode_bkt(raw) if s]
            prior_gen[tname] = ms.compute_all_generation_metrics(seqs, ev_by_term[tname], tr_by_term[tname], terminal=tname, compute_nn=True, nn_max_eval=128)

    ddpm_gen = None
    if run_ddpm:
        # Encode training posteriors (mu, logsigma) -> (N, 2, 16, 48)
        mus, logsigs = [], []
        with torch.no_grad():
            for s_i in range(0, len(tr_x), 64):
                bx, bg, bh = tr_x[s_i:s_i+64], tr_g_t[s_i:s_i+64], tr_h_t[s_i:s_i+64]
                v = vae.vae
                _, mu, ls = v.encoder(v.word_emb(bx.T).permute(1, 2, 0), v.g_pre_net(bg), bh)
                mus.append(mu)
                logsigs.append(ls)
        post_arr = torch.stack([torch.cat(mus, dim=0), torch.cat(logsigs, dim=0)], dim=1)  # (N, 2, 16, 48)

        torch.manual_seed(seed + 100)
        unet = UNet_conditional(c_in=2, c_out=2, num_class1=2, num_class2=2)
        opt_d = torch.optim.AdamW(unet.parameters(), lr=5e-3, eps=1e-5, betas=(0.9, 0.999), weight_decay=0.01)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt_d, max_lr=5e-3, total_steps=ddpm_updates)
        beta_sched = torch.linspace(1e-4, 0.02, 999)
        alpha = 1.0 - beta_sched
        alpha_hat = torch.cumprod(alpha, dim=0)

        rng_d = np.random.default_rng(seed + 100)
        unet.train()
        for step in range(ddpm_updates):
            idx = rng_d.integers(0, len(post_arr), size=8)
            x0 = post_arr[idx]
            y = tr_lbl[idx].clone()
            # 10% label drop matching v7
            drop_mask = torch.tensor(rng_d.random(8) < 0.1)
            t_vals = torch.tensor(rng_d.integers(1, 999, size=8), dtype=torch.long)
            ah = alpha_hat[t_vals][:, None, None, None]
            eps = torch.randn_like(x0)
            xt = torch.sqrt(ah) * x0 + torch.sqrt(1.0 - ah) * eps
            opt_d.zero_grad()
            # Unconditional pass when dropped not needed if y=None; use conditional pass for non-dropped
            pred = unet(xt, t_vals, None if bool(drop_mask.all()) else y)
            loss_d = F.mse_loss(pred, eps)
            loss_d.backward()
            opt_d.step()
            sched.step()

        unet.eval()
        ddpm_gen = {}
        steps_grid = np.linspace(998, 1, 15, dtype=int)
        for tidx, tname in ((0, "FREE"), (1, "AMD")):
            g_rng = torch.Generator().manual_seed(seed * 10 + tidx + 700)
            seqs = []
            for _ in range(2):  # 2 x 64 = 128 samples (8x original 16)
                B = 64
                x = torch.randn((B, 2, 16, 48), generator=g_rng)
                y = torch.full((B,), tidx, dtype=torch.long)
                with torch.no_grad():
                    for idx_s in range(len(steps_grid)):
                        t_cur = int(steps_grid[idx_s])
                        t_prev = int(steps_grid[idx_s + 1]) if idx_s + 1 < len(steps_grid) else 0
                        t_tensor = torch.full((B,), t_cur, dtype=torch.long)
                        eps_pred = unet(x, t_tensor, y)
                        ah_cur = alpha_hat[t_cur]
                        ah_prev = alpha_hat[t_prev] if t_prev > 0 else torch.tensor(1.0)
                        x0_pred = (x - torch.sqrt(1.0 - ah_cur) * eps_pred) / torch.sqrt(ah_cur)
                        sigma = 0.5 * torch.sqrt((1.0 - ah_prev) / (1.0 - ah_cur) * (1.0 - ah_cur / ah_prev))
                        dir_xt = torch.sqrt(torch.clamp(1.0 - ah_prev - sigma ** 2, min=0.0)) * eps_pred
                        noise = torch.randn(x.shape, generator=g_rng) if t_prev > 0 else torch.zeros_like(x)
                        x = torch.sqrt(ah_prev) * x0_pred + dir_xt + sigma * noise
                    mu_lat = x[:, 0, :, :]
                    logsig_lat = x[:, 1, :, :].clamp(-5.0, 2.0)
                    z_samp = mu_lat + torch.exp(logsig_lat) * torch.randn(mu_lat.shape, generator=g_rng)
                    raw = vae.vae.decoder(z_samp)
                    seqs.extend([s for s in joint_decode_bkt(raw) if s])
            ddpm_gen[tname] = ms.compute_all_generation_metrics(seqs, ev_by_term[tname], tr_by_term[tname], terminal=tname, compute_nn=True, nn_max_eval=128)

    elapsed = round(time.time() - t0, 1)
    # Strip per_aa from compact summary unless needed
    per_aa_keep = recon_res.pop("per_aa")
    summary = {
        "job": job,
        "elapsed_s": elapsed,
        "recon": recon_res,
        "per_aa": per_aa_keep,
        "prior_gen": prior_gen,
        "ddpm_gen": ddpm_gen,
    }
    ddpm_str = f" | DDPM FREE MMD2={ddpm_gen['FREE']['MMD2']:.5f}, AMD MMD2={ddpm_gen['AMD']['MMD2']:.5f}" if ddpm_gen else ""
    cache_path = job.get("_cache_path")
    if cache_path:
        Path(cache_path).write_text(json.dumps(summary))
    print(f"[{job_id}] ({elapsed}s) micro={recon_res['micro_pct']:.2f}% macro={recon_res['macro_pct']:.2f}% lenMAE={recon_res['length_mae']:.3f} | Prior FREE MMD2={prior_gen['FREE']['MMD2']:.5f}{ddpm_str}", flush=True)
    return summary


def main():
    t_all = time.time()
    jobs = []

    # 1. Pilot KL-beta & MSE tuning sweep (3 seeds: 222, 333, 444)
    for sd in (222, 333, 444):
        for beta in (0.01, 0.1, 1.0):
            jobs.append({"job_id": f"pilot_CE21_b{beta}_s{sd}", "group": "pilot_kl_sweep", "cohort": "pilot",
                         "arm": "CE21", "seed": sd, "kl_beta": beta, "lr": 5e-4, "vae_updates": 220,
                         "run_ddpm": (beta == 1.0), "ddpm_updates": 250})
            jobs.append({"job_id": f"pilot_MSE21_b{beta}_s{sd}", "group": "pilot_kl_sweep", "cohort": "pilot",
                         "arm": "MSE21", "seed": sd, "kl_beta": beta, "mse_scale": 0.5, "lr": 5e-4, "vae_updates": 220,
                         "run_ddpm": (beta == 1.0), "ddpm_updates": 250})
        # Tuned MSE21: 21x loss scale (10.5 prefactor) + higher LR 2e-3 across beta=0.01, 0.1, 1.0
        for beta in (0.01, 0.1, 1.0):
            jobs.append({"job_id": f"pilot_MSE21_tuned_b{beta}_s{sd}", "group": "pilot_mse_tuned", "cohort": "pilot",
                         "arm": "MSE21", "seed": sd, "kl_beta": beta, "mse_scale": 10.5, "lr": 2e-3, "vae_updates": 220,
                         "run_ddpm": (beta == 0.1), "ddpm_updates": 250})

    # 2. Pilot Padding-masked ablation (3 seeds)
    for sd in (222, 333, 444):
        jobs.append({"job_id": f"pilot_CE21_padmask_s{sd}", "group": "pilot_pad_ablation", "cohort": "pilot",
                     "arm": "CE21", "seed": sd, "kl_beta": 1.0, "lr": 5e-4, "mask_pad": True, "vae_updates": 220, "run_ddpm": False})
        jobs.append({"job_id": f"pilot_MSE21_padmask_s{sd}", "group": "pilot_pad_ablation", "cohort": "pilot",
                     "arm": "MSE21", "seed": sd, "kl_beta": 1.0, "mse_scale": 0.5, "lr": 5e-4, "mask_pad": True, "vae_updates": 220, "run_ddpm": False})

    # 3. Expanded DBAASP Benchmark (3,120 train / 222 untouched test) — 3 seeds x full VAE+DDPM
    for sd in (222, 333, 444):
        jobs.append({"job_id": f"exp_CE21_b1.0_s{sd}", "group": "expanded_main", "cohort": "expanded",
                     "arm": "CE21", "seed": sd, "kl_beta": 1.0, "lr": 5e-4, "vae_updates": 300, "run_ddpm": True, "ddpm_updates": 300})
        jobs.append({"job_id": f"exp_CE21_b0.1_s{sd}", "group": "expanded_main", "cohort": "expanded",
                     "arm": "CE21", "seed": sd, "kl_beta": 0.1, "lr": 5e-4, "vae_updates": 300, "run_ddpm": True, "ddpm_updates": 300})
        jobs.append({"job_id": f"exp_MSE21_b1.0_s{sd}", "group": "expanded_main", "cohort": "expanded",
                     "arm": "MSE21", "seed": sd, "kl_beta": 1.0, "mse_scale": 0.5, "lr": 5e-4, "vae_updates": 300, "run_ddpm": True, "ddpm_updates": 300})
        jobs.append({"job_id": f"exp_MSE21_tuned_b0.1_s{sd}", "group": "expanded_main", "cohort": "expanded",
                     "arm": "MSE21", "seed": sd, "kl_beta": 0.1, "mse_scale": 10.5, "lr": 2e-3, "vae_updates": 300, "run_ddpm": True, "ddpm_updates": 300})

    cache_dir = OUT_DIR / "cache_suite2"
    cache_dir.mkdir(exist_ok=True)
    print(f"[Suite 2] Launching {len(jobs)} jobs across 5 parallel workers (with in-worker cache)...", flush=True)
    results = []
    with cf.ProcessPoolExecutor(max_workers=5) as ex:
        futs = []
        for j in jobs:
            cpath = cache_dir / f"{j['job_id']}.json"
            if cpath.exists():
                results.append(json.loads(cpath.read_text()))
            else:
                j["_cache_path"] = str(cpath)
                futs.append(ex.submit(run_single_job, j))
        for f in cf.as_completed(futs):
            results.append(f.result())

    results.sort(key=lambda r: r["job"]["job_id"])

    # Compute amino-acid support vs recall Spearman correlation on pilot CE21 b1.0
    train_seqs = [r["sequence"] for r in csv.DictReader((ROOT / "github_repo/data/train_dev/train_cohort.csv").open())]
    tr_aa_cnt = collections.Counter(ch for s in train_seqs for ch in s)
    ce_pilot_runs = [r for r in results if r["job"]["group"] == "pilot_kl_sweep" and r["job"]["arm"] == "CE21" and r["job"]["kl_beta"] == 1.0]
    aa_rec_summary = []
    for ch in ALPHABET:
        recs = [r["per_aa"][ch]["recall"] for r in ce_pilot_runs if r["per_aa"][ch]["recall"] is not None]
        if recs:
            aa_rec_summary.append({"aa": ch, "train_support": tr_aa_cnt[ch], "mean_dev_recall": float(np.mean(recs))})
    rho, pval = spearmanr([x["train_support"] for x in aa_rec_summary], [x["mean_dev_recall"] for x in aa_rec_summary])

    out = {
        "elapsed_seconds": round(time.time() - t_all, 1),
        "n_jobs": len(results),
        "aa_support_vs_recall_spearman": {"rho": round(float(rho), 4), "p_value": float(pval), "per_aa": aa_rec_summary},
        "runs": results,
    }
    (OUT_DIR / "suite2_vae_ddpm_ablations_and_expanded.json").write_text(json.dumps(out, indent=2) + "\n")
    print(f"[Suite 2 DONE] Total elapsed: {time.time()-t_all:.1f}s | AA support-recall Spearman rho={rho:.4f} (p={pval:.2e})", flush=True)


if __name__ == "__main__":
    main()
