#!/usr/bin/env python3
"""Parallelized Suite 1 (5 workers x 2 CPU threads):
Multi-seed (5 seeds, 256 samples/terminal) evaluation across BOTH:
  (A) Pilot cohort (279 train / 41 dev)
  (B) Expanded homology-partitioned DBAASP cohort (3,120 train / 223 val / 222 untouched independent test)

Models:
  1. Unigram (empirical length + 0-order AA frequency)
  2. Bigram Markov (1st-order Markov, Laplace alpha=0.5)
  3. Trigram Markov (2nd-order Markov with bigram backoff)
  4. Terminal-LSTM Small (62,501 params — exact v7 architecture)
  5. Terminal-LSTM Matched (~1.34M params — hidden 320, 2 layers)
  6. Transformer-LM (~1.80M params — 4 causal layers, d_model=192, 4 heads)
  7. Sequence GRU-VAE CE21 (~1.6M params, non-diffusion VAE comparator, CE21)
  8. Sequence GRU-VAE MSE21 (~1.6M params, matched non-diffusion VAE comparator, MSE21)
  9. Frozen v7 checkpoint large-sample replay (CE21_DDPM, MSE21_DDPM, scalar_DDPM, terminal_lstm -> 256/terminal x 5 gen seeds)
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
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path("/Users/loctek/.gemini/antigravity/scratch/bib-26-2683-revision")
OUT_DIR = ROOT / "revision_results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ALPHABET = "ARNDCEQGHILKMFPSTWYV"
PAD = 20
BOS = 21
MAX_LEN = 48
SEEDS = [222, 333, 444, 555, 666]
N_GEN = 256


def load_cohort(path: Path) -> List[Tuple[str, str]]:
    with path.open() as fh:
        return [(r["sequence"], r["terminal_state"]) for r in csv.DictReader(fh)]


def sample_unigram(train_seqs: Sequence[str], n: int, seed: int) -> List[str]:
    rng = random.Random(seed)
    lens = [len(s) for s in train_seqs]
    cnt = collections.Counter(ch for s in train_seqs for ch in s)
    aas = list(ALPHABET)
    weights = [cnt[a] + 0.5 for a in aas]
    return ["".join(rng.choices(aas, weights=weights, k=rng.choice(lens))) for _ in range(n)]


def sample_bigram_markov(train_seqs: Sequence[str], n: int, seed: int, alpha: float = 0.5) -> List[str]:
    rng = random.Random(seed)
    syms = list(ALPHABET) + ["<EOS>"]
    counts = {ctx: {s: alpha for s in syms} for ctx in ["<BOS>"] + list(ALPHABET)}
    for seq in train_seqs:
        prev = "<BOS>"
        for ch in seq:
            counts[prev][ch] += 1.0
            prev = ch
        counts[prev]["<EOS>"] += 1.0
    out = []
    for _ in range(n):
        prev = "<BOS>"
        res = []
        for _ in range(MAX_LEN):
            w = [counts[prev][s] for s in syms]
            nxt = rng.choices(syms, weights=w, k=1)[0]
            if nxt == "<EOS>":
                break
            res.append(nxt)
            prev = nxt
        out.append("".join(res) if res else rng.choice(ALPHABET))
    return out


def sample_trigram_markov(train_seqs: Sequence[str], n: int, seed: int, alpha2: float = 0.2, alpha1: float = 0.5) -> List[str]:
    rng = random.Random(seed)
    syms = list(ALPHABET) + ["<EOS>"]
    bi = {ctx: {s: alpha1 for s in syms} for ctx in ["<BOS>"] + list(ALPHABET)}
    tri = collections.defaultdict(lambda: {s: alpha2 for s in syms})
    tri_obs = collections.Counter()
    for seq in train_seqs:
        p2, p1 = "<BOS>", "<BOS>"
        for ch in seq:
            bi[p1][ch] += 1.0
            tri[(p2, p1)][ch] += 1.0
            tri_obs[(p2, p1)] += 1
            p2, p1 = p1, ch
        bi[p1]["<EOS>"] += 1.0
        tri[(p2, p1)]["<EOS>"] += 1.0
        tri_obs[(p2, p1)] += 1
    out = []
    for _ in range(n):
        p2, p1 = "<BOS>", "<BOS>"
        res = []
        for _ in range(MAX_LEN):
            w = [tri[(p2, p1)][s] for s in syms] if tri_obs[(p2, p1)] >= 3 else [bi[p1][s] for s in syms]
            nxt = rng.choices(syms, weights=w, k=1)[0]
            if nxt == "<EOS>":
                break
            res.append(nxt)
            p2, p1 = p1, nxt
        out.append("".join(res) if res else rng.choice(ALPHABET))
    return out


def make_ar_tensors(rows: Sequence[Tuple[str, str]]):
    n = len(rows)
    targets = torch.full((n, MAX_LEN), PAD, dtype=torch.long)
    inputs = torch.full((n, MAX_LEN), PAD, dtype=torch.long)
    inputs[:, 0] = BOS
    mask = torch.zeros((n, MAX_LEN), dtype=torch.bool)
    labels = torch.tensor([0 if t == "FREE" else 1 for _, t in rows], dtype=torch.long)
    for i, (seq, _) in enumerate(rows):
        for j, ch in enumerate(seq):
            targets[i, j] = ALPHABET.index(ch)
        mask[i, :min(len(seq) + 1, MAX_LEN)] = True
    inputs[:, 1:] = targets[:, :-1]
    return inputs, targets, mask, labels


class FlexTerminalLSTM(nn.Module):
    def __init__(self, tok_dim=32, term_dim=8, hidden_dim=64, layers=2, dropout=0.0):
        super().__init__()
        self.tok_emb = nn.Embedding(22, tok_dim)
        self.term_emb = nn.Embedding(2, term_dim)
        self.lstm = nn.LSTM(tok_dim + term_dim, hidden_dim, num_layers=layers, batch_first=True, dropout=dropout if layers > 1 else 0.0)
        self.out = nn.Linear(hidden_dim, 21)

    def forward(self, x, labels, hidden=None):
        term = self.term_emb(labels).unsqueeze(1).expand(-1, x.size(1), -1)
        h, hidden = self.lstm(torch.cat([self.tok_emb(x), term], dim=-1), hidden)
        return self.out(h), hidden


class TerminalTransformerLM(nn.Module):
    def __init__(self, d_model=192, nhead=4, num_layers=4, dim_ff=768, dropout=0.15):
        super().__init__()
        self.tok_emb = nn.Embedding(22, d_model)
        self.pos_emb = nn.Embedding(MAX_LEN, d_model)
        self.term_emb = nn.Embedding(2, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
            dropout=dropout, batch_first=True, activation="gelu", norm_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)
        self.out = nn.Linear(d_model, 21)

    def forward(self, x, labels):
        B, T = x.shape
        pos = torch.arange(T, device=x.device).unsqueeze(0).expand(B, T)
        h = self.tok_emb(x) + self.pos_emb(pos) + self.term_emb(labels).unsqueeze(1)
        causal = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
        h = self.encoder(h, mask=causal)
        return self.out(self.norm(h))


class SeqGRUVAE(nn.Module):
    def __init__(self, tok_dim=48, term_dim=16, hidden_dim=224, latent_dim=64):
        super().__init__()
        self.latent_dim = latent_dim
        self.tok_emb = nn.Embedding(21, tok_dim)
        self.term_emb = nn.Embedding(2, term_dim)
        self.enc = nn.GRU(tok_dim + term_dim, hidden_dim, num_layers=2, batch_first=True, bidirectional=True)
        self.to_mu = nn.Linear(hidden_dim * 2, latent_dim)
        self.to_logvar = nn.Linear(hidden_dim * 2, latent_dim)
        self.dec_proj = nn.Linear(latent_dim + term_dim, hidden_dim)
        self.dec_pos = nn.Embedding(MAX_LEN, hidden_dim)
        self.dec = nn.GRU(hidden_dim, hidden_dim, num_layers=2, batch_first=True)
        self.out = nn.Linear(hidden_dim, 21)

    def encode(self, x, labels):
        term = self.term_emb(labels).unsqueeze(1).expand(-1, x.size(1), -1)
        _, h = self.enc(torch.cat([self.tok_emb(x), term], dim=-1))
        h_cat = torch.cat([h[-2], h[-1]], dim=-1)
        return self.to_mu(h_cat), self.to_logvar(h_cat).clamp(-8.0, 4.0)

    def decode(self, z, labels):
        B = z.size(0)
        z_cond = self.dec_proj(torch.cat([z, self.term_emb(labels)], dim=-1)).unsqueeze(1).expand(B, MAX_LEN, -1)
        pos = self.dec_pos(torch.arange(MAX_LEN, device=z.device)).unsqueeze(0).expand(B, MAX_LEN, -1)
        h, _ = self.dec(z_cond + pos)
        return self.out(h)


def joint_decode_bkt(logits_b_t_21: torch.Tensor) -> List[str]:
    body, aa = logits_b_t_21[:, :, :20].max(-1)
    pad = logits_b_t_21[:, :, 20]
    base = pad.sum(1, keepdim=True)
    lscores = base + torch.cat([torch.zeros_like(base), (body - pad).cumsum(1)], dim=1)
    lengths = lscores.argmax(1).tolist()
    aa_list = aa.tolist()
    return ["".join(ALPHABET[row[i]] for i in range(L)) for L, row in zip(lengths, aa_list)]


def eval_reconstruction_seqs(pred_seqs: Sequence[str], true_seqs: Sequence[str]) -> Dict[str, float]:
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
    return {
        "micro_pct": 100.0 * correct_res / total_res,
        "macro_pct": 100.0 * macro,
        "length_mae": len_err / len(true_seqs),
        "exact_pct": 100.0 * exact / len(true_seqs),
        "exact_count": exact,
        "correct_residues": correct_res,
        "total_residues": total_res,
    }


def worker_job(job: dict) -> dict:
    os.environ["OMP_NUM_THREADS"] = "2"
    os.environ["OPENBLAS_NUM_THREADS"] = "2"
    torch.set_num_threads(2)
    sys.path.insert(0, str(ROOT / "scripts"))
    import metrics_suite as ms

    t0 = time.time()
    cohort = job["cohort"]
    mname = job["model"]
    seed = job["seed"]

    if cohort == "pilot":
        train_rows = load_cohort(ROOT / "github_repo/data/train_dev/train_cohort.csv")
        val_rows = load_cohort(ROOT / "github_repo/data/train_dev/dev_cohort.csv")
        test_rows = val_rows
        ar_updates, vae_updates = 500, 400
    else:
        train_rows = load_cohort(ROOT / "revision_data/dbaasp_expanded/train_cohort.csv")
        val_rows = load_cohort(ROOT / "revision_data/dbaasp_expanded/val_cohort.csv")
        test_rows = load_cohort(ROOT / "revision_data/dbaasp_expanded/test_cohort.csv")
        ar_updates, vae_updates = 800, 600

    tr_by_term = {"FREE": [s for s, t in train_rows if t == "FREE"], "AMD": [s for s, t in train_rows if t == "AMD"]}
    ev_by_term = {"FREE": [s for s, t in test_rows if t == "FREE"], "AMD": [s for s, t in test_rows if t == "AMD"]}

    params = 0
    recon = None
    best_val_nll = None
    best_step = None
    sampled = {}

    if mname in ("Unigram", "Bigram_Markov", "Trigram_Markov"):
        fn = {"Unigram": sample_unigram, "Bigram_Markov": sample_bigram_markov, "Trigram_Markov": sample_trigram_markov}[mname]
        for tidx, tname in ((0, "FREE"), (1, "AMD")):
            sampled[tname] = fn(tr_by_term[tname], N_GEN, seed * 10 + tidx)
    elif mname in ("LSTM_small", "LSTM_matched", "Transformer_LM"):
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        if mname == "LSTM_small":
            model = FlexTerminalLSTM(32, 8, 64, 2, 0.0); lr = 1e-3; wd = 0.0
        elif mname == "LSTM_matched":
            model = FlexTerminalLSTM(64, 16, 320, 2, 0.15); lr = 8e-4; wd = 1e-2
        else:
            model = TerminalTransformerLM(192, 4, 4, 768, 0.15); lr = 5e-4; wd = 1e-2
        params = sum(p.numel() for p in model.parameters())
        tr_x, tr_y, tr_m, tr_l = make_ar_tensors(train_rows)
        va_x, va_y, va_m, va_l = make_ar_tensors(val_rows)
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
        rng_idx = np.random.default_rng(seed)
        best_val_nll = float("inf")
        best_state = None
        eval_every = max(50, ar_updates // 8)

        def eval_nll():
            model.eval()
            tot_l, tot_t = 0.0, 0
            with torch.no_grad():
                for s in range(0, len(va_x), 128):
                    bx, by, bm, bl = va_x[s:s+128], va_y[s:s+128], va_m[s:s+128], va_l[s:s+128]
                    logits = model(bx, bl)[0] if mname != "Transformer_LM" else model(bx, bl)
                    tl = F.cross_entropy(logits.transpose(1, 2), by, reduction="none")
                    tot_l += float(tl.masked_select(bm).sum())
                    tot_t += int(bm.sum())
            model.train()
            return tot_l / max(1, tot_t)

        model.train()
        for step in range(1, ar_updates + 1):
            idx = rng_idx.integers(0, len(tr_x), size=32)
            bx, by, bm, bl = tr_x[idx], tr_y[idx], tr_m[idx], tr_l[idx]
            opt.zero_grad()
            logits = model(bx, bl)[0] if mname != "Transformer_LM" else model(bx, bl)
            loss = F.cross_entropy(logits.transpose(1, 2), by, reduction="none").masked_select(bm).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            if step % eval_every == 0 or step == ar_updates:
                vn = eval_nll()
                if vn < best_val_nll:
                    best_val_nll = vn
                    best_step = step
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if best_state is not None:
            model.load_state_dict(best_state)
        model.eval()
        for tidx, tname in ((0, "FREE"), (1, "AMD")):
            gen = torch.Generator(device="cpu").manual_seed(seed * 10 + tidx + 100)
            lbls = torch.full((N_GEN,), tidx, dtype=torch.long)
            if mname != "Transformer_LM":
                inp = torch.full((N_GEN, 1), BOS, dtype=torch.long)
                fin = torch.zeros(N_GEN, dtype=torch.bool)
                toks = torch.full((N_GEN, MAX_LEN), PAD, dtype=torch.long)
                hid = None
                with torch.no_grad():
                    for pos in range(MAX_LEN):
                        logits, hid = model(inp, lbls, hid)
                        draw = torch.multinomial(torch.softmax(logits[:, 0, :], -1), 1, generator=gen).squeeze(1)
                        act = torch.where(fin, torch.tensor(PAD), draw)
                        toks[:, pos] = act
                        fin = fin | (act == PAD)
                        inp = act.unsqueeze(1)
                        if bool(fin.all()):
                            break
            else:
                seq_in = torch.full((N_GEN, 1), BOS, dtype=torch.long)
                fin = torch.zeros(N_GEN, dtype=torch.bool)
                toks = torch.full((N_GEN, MAX_LEN), PAD, dtype=torch.long)
                with torch.no_grad():
                    for pos in range(MAX_LEN):
                        logits = model(seq_in, lbls)[:, -1, :]
                        draw = torch.multinomial(torch.softmax(logits, -1), 1, generator=gen).squeeze(1)
                        act = torch.where(fin, torch.tensor(PAD), draw)
                        toks[:, pos] = act
                        fin = fin | (act == PAD)
                        if pos + 1 < MAX_LEN:
                            seq_in = torch.cat([seq_in, act.unsqueeze(1)], dim=1)
                        if bool(fin.all()):
                            break
            seqs = []
            for row in toks.tolist():
                s = []
                for t in row:
                    if t == PAD: break
                    s.append(ALPHABET[t])
                if s: seqs.append("".join(s))
            sampled[tname] = seqs
    else:
        # SeqGRUVAE_CE21 or SeqGRUVAE_MSE21
        arm = "CE21" if mname.endswith("CE21") else "MSE21"
        torch.manual_seed(seed)
        np.random.seed(seed)
        model = SeqGRUVAE()
        params = sum(p.numel() for p in model.parameters())
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        def to_xy(rows):
            x = torch.full((len(rows), MAX_LEN), PAD, dtype=torch.long)
            l = torch.tensor([0 if t == "FREE" else 1 for _, t in rows], dtype=torch.long)
            for i, (s, _) in enumerate(rows):
                for j, ch in enumerate(s): x[i, j] = ALPHABET.index(ch)
            return x, l
        tr_x, tr_l = to_xy(train_rows)
        ev_x, ev_l = to_xy(test_rows)
        rng_idx = np.random.default_rng(seed)
        model.train()
        for _ in range(vae_updates):
            idx = rng_idx.integers(0, len(tr_x), size=32)
            bx, bl = tr_x[idx], tr_l[idx]
            opt.zero_grad()
            mu, logvar = model.encode(bx, bl)
            z = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)
            logits = model.decode(z, bl)
            if arm == "CE21":
                rec = F.cross_entropy(logits.reshape(-1, 21), bx.reshape(-1), reduction="mean")
            else:
                oh = F.one_hot(bx, num_classes=21).float()
                rec = 0.5 * (logits - oh).square().sum() / bx.numel()
            kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum() / bx.numel()
            (rec + 0.1 * kl).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        model.eval()
        with torch.no_grad():
            mu, _ = model.encode(ev_x, ev_l)
            pred_seqs = joint_decode_bkt(model.decode(mu, ev_l))
        recon = eval_reconstruction_seqs(pred_seqs, [s for s, _ in test_rows])
        with torch.no_grad():
            for tidx, tname in ((0, "FREE"), (1, "AMD")):
                gen = torch.Generator(device="cpu").manual_seed(seed * 10 + tidx + 200)
                z = torch.randn((N_GEN, model.latent_dim), generator=gen)
                lbls = torch.full((N_GEN,), tidx, dtype=torch.long)
                sampled[tname] = [s for s in joint_decode_bkt(model.decode(z, lbls)) if s]

    eval_metrics = {}
    for tname in ("FREE", "AMD"):
        eval_metrics[tname] = ms.compute_all_generation_metrics(
            sampled[tname], ev_by_term[tname], tr_by_term[tname], terminal=tname, compute_nn=True, nn_max_eval=128
        )
    elapsed = round(time.time() - t0, 1)
    rstr = f" recon={recon['micro_pct']:.1f}%" if recon else ""
    print(f"  [{cohort:8s}] {mname:15s} s{seed} ({elapsed:4.1f}s){rstr} | FREE MMD2={eval_metrics['FREE']['MMD2']:.5f} AMD MMD2={eval_metrics['AMD']['MMD2']:.5f}", flush=True)
    return {"cohort": cohort, "model": mname, "seed": seed, "params": params, "best_val_nll": best_val_nll,
            "best_step": best_step, "recon": recon, "eval": eval_metrics}


def replay_v7_worker(arm_dir: str) -> Tuple[str, dict]:
    os.environ["OMP_NUM_THREADS"] = "2"
    torch.set_num_threads(2)
    sys.path.insert(0, str(ROOT / "scripts"))
    sys.path.insert(0, str(ROOT / "github_repo/model/source/corrected_code/AMP_VAEflow"))
    sys.path.insert(0, str(ROOT / "github_repo/model/implementation"))
    import metrics_suite as ms
    from modules.vae import VAEModel
    from modules.diff import UNet_conditional
    import ar_pipeline as arp

    pilot_train = load_cohort(ROOT / "github_repo/data/train_dev/train_cohort.csv")
    pilot_dev = load_cohort(ROOT / "github_repo/data/train_dev/dev_cohort.csv")
    tr_by_term = {"FREE": [s for s, t in pilot_train if t == "FREE"], "AMD": [s for s, t in pilot_train if t == "AMD"]}
    dv_by_term = {"FREE": [s for s, t in pilot_dev if t == "FREE"], "AMD": [s for s, t in pilot_dev if t == "AMD"]}

    if arm_dir == "terminal_lstm":
        lstm = arp.TerminalLSTM(arp.ARConfig())
        ckpt = torch.load(ROOT / "github_repo/weights/terminal_lstm/final.pt", map_location="cpu", weights_only=False)
        sd = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        lstm.load_state_dict(sd)
        lstm.eval()
        runs = {"FREE": [], "AMD": []}
        for gs in [422, 522, 622, 722, 822]:
            for tidx, tname in ((0, "FREE"), (1, "AMD")):
                res = arp.sample_batch(lstm, [tidx] * N_GEN, seed=gs + tidx, requests=N_GEN)
                seqs = [s for s in res["sequences"] if s]
                m_dev = ms.compute_all_generation_metrics(seqs, dv_by_term[tname], tr_by_term[tname], terminal=tname, compute_nn=True, nn_max_eval=128)
                runs[tname].append({"seed": gs, "dev": m_dev})
        print(f"  [v7 replay] terminal_lstm DONE FREE={np.mean([r['dev']['MMD2'] for r in runs['FREE']]):.5f} AMD={np.mean([r['dev']['MMD2'] for r in runs['AMD']]):.5f}", flush=True)
        return "v7_terminal_lstm_256x5", runs

    vae = VAEModel(out_dim=1 if arm_dir == "scalar_legacy" else 21, con_dim=3, latent_size=16,
                   sqz_prior=False, cond_drop=False, use_prior_flow=True, device="cpu")
    vsd = torch.load(ROOT / f"github_repo/weights/{arm_dir}/vae_state.pt", map_location="cpu", weights_only=False)
    vae.load_state_dict(vsd["state_dict"] if "state_dict" in vsd else vsd)
    vae.eval()

    unet = UNet_conditional(c_in=2, c_out=2, num_class1=2, num_class2=2)
    dsd = torch.load(ROOT / f"github_repo/weights/{arm_dir}/ddpm_ema_state.pt", map_location="cpu", weights_only=False)
    unet.load_state_dict(dsd["state_dict"] if "state_dict" in dsd else dsd)
    unet.eval()

    beta = torch.linspace(1e-4, 0.02, 999)
    alpha_hat = torch.cumprod(1.0 - beta, dim=0)
    steps_grid = np.linspace(998, 1, 60, dtype=int)

    ddpm_runs = {"FREE": [], "AMD": []}
    for gs in [422, 522, 622, 722, 822]:
        for tidx, tname in ((0, "FREE"), (1, "AMD")):
            g_rng = torch.Generator().manual_seed(gs + tidx)
            seqs = []
            for _ in range(4):  # 4 x 64 = 256 samples
                B = 64
                x = torch.randn((B, 2, 16, 48), generator=g_rng)
                y = torch.full((B,), tidx, dtype=torch.long)
                with torch.no_grad():
                    for idx_s in range(len(steps_grid)):
                        t_cur = int(steps_grid[idx_s])
                        t_prev = int(steps_grid[idx_s + 1]) if idx_s + 1 < len(steps_grid) else 0
                        eps = unet(x, torch.full((B,), t_cur, dtype=torch.long), y)
                        ah_cur = alpha_hat[t_cur]
                        ah_prev = alpha_hat[t_prev] if t_prev > 0 else torch.tensor(1.0)
                        x0_pred = (x - torch.sqrt(1.0 - ah_cur) * eps) / torch.sqrt(ah_cur)
                        sigma = 0.5 * torch.sqrt((1.0 - ah_prev) / (1.0 - ah_cur) * (1.0 - ah_cur / ah_prev))
                        dir_xt = torch.sqrt(torch.clamp(1.0 - ah_prev - sigma ** 2, min=0.0)) * eps
                        noise = torch.randn(x.shape, generator=g_rng) if t_prev > 0 else torch.zeros_like(x)
                        x = torch.sqrt(ah_prev) * x0_pred + dir_xt + sigma * noise
                    mu_lat = x[:, 0, :, :]
                    logsig_lat = x[:, 1, :, :].clamp(-5.0, 2.0)
                    z_samp = mu_lat + torch.exp(logsig_lat) * torch.randn(mu_lat.shape, generator=g_rng)
                    raw = vae.vae.decoder(z_samp)
                    if arm_dir == "scalar_legacy":
                        toks = raw.squeeze(1).clamp(0, 20).round().long().tolist()
                        for row in toks:
                            L = row.index(20) if 20 in row else 48
                            s = "".join(ALPHABET[row[k]] for k in range(L) if row[k] < 20)
                            if s: seqs.append(s)
                    else:
                        seqs.extend([s for s in joint_decode_bkt(raw.transpose(1, 2)) if s])
            m_dev = ms.compute_all_generation_metrics(seqs, dv_by_term[tname], tr_by_term[tname], terminal=tname, compute_nn=True, nn_max_eval=128)
            ddpm_runs[tname].append({"seed": gs, "dev": m_dev})
    print(f"  [v7 replay] {arm_dir} DONE FREE={np.mean([r['dev']['MMD2'] for r in ddpm_runs['FREE']]):.5f} AMD={np.mean([r['dev']['MMD2'] for r in ddpm_runs['AMD']]):.5f}", flush=True)
    return f"v7_{arm_dir}_ddpm_256x5", ddpm_runs


def main():
    t0 = time.time()
    models = ["Unigram", "Bigram_Markov", "Trigram_Markov", "LSTM_small", "LSTM_matched", "Transformer_LM", "SeqGRUVAE_CE21", "SeqGRUVAE_MSE21"]
    jobs = [{"cohort": c, "model": m, "seed": s} for c in ("pilot", "expanded") for m in models for s in SEEDS]
    print(f"[Suite 1 Parallel] Running {len(jobs)} jobs + 4 v7 replays across 6 workers...", flush=True)

    cache_dir = OUT_DIR / "cache_suite1"
    cache_dir.mkdir(exist_ok=True)
    results = []
    v7_res = {}
    with cf.ProcessPoolExecutor(max_workers=6) as ex:
        futs = {}
        for j in jobs:
            cpath = cache_dir / f"{j['cohort']}_{j['model']}_s{j['seed']}.json"
            if cpath.exists():
                results.append(json.loads(cpath.read_text()))
            else:
                futs[ex.submit(worker_job, j)] = cpath
        v7_futs = {}
        for a in ("terminal_lstm", "categorical_ce", "raw_onehot_mse", "scalar_legacy"):
            cpath = cache_dir / f"v7_{a}.json"
            if cpath.exists():
                d = json.loads(cpath.read_text())
                v7_res[d["key"]] = d["val"]
            else:
                v7_futs[ex.submit(replay_v7_worker, a)] = cpath
        for f in cf.as_completed(futs):
            r = f.result()
            futs[f].write_text(json.dumps(r))
            results.append(r)
        for f in cf.as_completed(v7_futs):
            k, v = f.result()
            v7_futs[f].write_text(json.dumps({"key": k, "val": v}))
            v7_res[k] = v

    # Organize by cohort -> model
    organized = {"pilot": {"models": {}}, "expanded": {"models": {}}}
    for r in results:
        c, m = r["cohort"], r["model"]
        entry = organized[c]["models"].setdefault(m, {"params": r["params"], "runs": {"FREE": [], "AMD": []}, "recons": []})
        if r["recon"] is not None:
            entry["recons"].append({"seed": r["seed"], **r["recon"]})
        for tname in ("FREE", "AMD"):
            entry["runs"][tname].append({"seed": r["seed"], "eval": r["eval"][tname], "best_val_nll": r["best_val_nll"], "best_step": r["best_step"]})
    for c in organized:
        for m in organized[c]["models"]:
            for tname in ("FREE", "AMD"):
                organized[c]["models"][m]["runs"][tname].sort(key=lambda x: x["seed"])
            organized[c]["models"][m]["recons"].sort(key=lambda x: x["seed"])

    # Correlation & PCA effective dimensionality audit across all 6 metrics (R3 #4)
    metric_keys = ["MMD2", "AA_JS", "AA_L1", "Pair_JS", "Pair_L1", "Length_KS"]
    rows = []
    for r in results:
        for tname in ("FREE", "AMD"):
            ev = r["eval"][tname]
            if all(math.isfinite(ev.get(k, float("nan"))) for k in metric_keys):
                rows.append([ev[k] for k in metric_keys])
    mat = np.array(rows, dtype=np.float64)
    from scipy.stats import spearmanr
    corr, _ = spearmanr(mat, axis=0)
    z = (mat - mat.mean(axis=0)) / (mat.std(axis=0) + 1e-12)
    _, s_vals, _ = np.linalg.svd(z, full_matrices=False)
    eigs = (s_vals ** 2) / len(mat)
    var_exp = eigs / eigs.sum()
    eff_dim = float((eigs.sum() ** 2) / np.sum(eigs ** 2))
    corr_audit = {
        "n_points": int(len(mat)),
        "metrics": metric_keys,
        "spearman_corr_matrix": [[round(float(x), 4) for x in row] for row in corr],
        "pca_variance_explained": [round(float(v), 4) for v in var_exp],
        "pca_cumvar_pc1_pc2": round(float(var_exp[0] + var_exp[1]), 4),
        "participation_ratio_effective_dim": round(eff_dim, 3),
    }

    out = {
        "elapsed_seconds": round(time.time() - t0, 1),
        "seeds": SEEDS,
        "samples_per_terminal": N_GEN,
        "pilot": organized["pilot"],
        "expanded": organized["expanded"],
        "v7_checkpoint_replay": v7_res,
        "metric_correlation_audit": corr_audit,
    }
    (OUT_DIR / "suite1_statistical_ar_gruvae_v7replay.json").write_text(json.dumps(out, indent=2) + "\n")
    print(f"\n[Suite 1 Parallel DONE] Elapsed {time.time()-t0:.1f}s | PC1+PC2={corr_audit['pca_cumvar_pc1_pc2']}, eff_dim={corr_audit['participation_ratio_effective_dim']}", flush=True)


if __name__ == "__main__":
    main()
