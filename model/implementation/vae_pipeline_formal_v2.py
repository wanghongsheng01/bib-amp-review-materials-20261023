#!/usr/bin/env python3
"""AMP H5 fresh VAE/DDPM arms. No data/model work occurs on import.

The CE21/MSE21 pair differs only in the reconstruction objective. Scalar keeps
the inherited objective and is a whole-pipeline comparator. Test is unsupported.
Run from a hash-bound JSON configuration; benchmark uses synthetic tensors only.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time
import traceback

AA = "ARNDCEQGHILKMFPSTWYV"
ARMS = {"categorical_ce": "CE21_DDPM", "raw_onehot_mse": "MSE21_DDPM",
        "scalar_legacy": "scalar_DDPM"}
DEFAULT_SOURCE = "/Users/loctek/Documents/Codex/2026-10-09/task-8/amp_commsbio_20261011/rebuild_experiments_20261011"
GENERATED_FIELDS = ["sample_id", "method", "terminal", "sequence", "status",
                    "length", "seed", "terminal_sample_id"]
SUCCESS_STATUSES = ("ok", "empty_sequence", "no_eos_at_limit")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, obj):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def log(path, obj):
    line = json.dumps(obj, allow_nan=False)
    with Path(path).open("a") as fh:
        fh.write(line + "\n")
    print(line, flush=True)


def write_csv(path, rows, fields):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def init_runtime(source_root):
    """Only original source is imported; bytecode writes are disabled."""
    global torch, np, VAEModel, UNet_conditional, EMA, validate_file
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                 "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "1"
    sys.dont_write_bytecode = True
    source_root = Path(source_root).resolve()
    code = source_root / "corrected_code" / "AMP_VAEflow"
    manifest_path = source_root / "runner" / "source_copy_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    checked = []
    for entry in manifest["files"]:
        path = source_root / "corrected_code" / entry["relative_path"]
        digest = sha(path)
        if digest != entry["corrected_sha256"]:
            raise ValueError("frozen corrected source drift: " + str(path))
        checked.append({"path": str(path), "sha256": digest})
    sys.path.insert(0, str(code))
    import torch
    import numpy as np
    from modules.vae import VAEModel
    from modules.diff import UNet_conditional, EMA
    from tasks.dataset_loader.dataset_utils.amp_input_contract import validate_file
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        if torch.get_num_interop_threads() != 1:
            raise
    return {"torch": torch.__version__, "numpy": np.__version__,
            "python": sys.version, "threads": torch.get_num_threads(),
            "interop_threads": torch.get_num_interop_threads(), "device": "cpu",
            "source_manifest_sha256": sha(manifest_path), "source_files": checked}


def finite(tensor, name):
    if not bool(torch.isfinite(tensor).all()):
        raise FloatingPointError("nonfinite " + name + "; no clipping or retry")


def state_hash(model):
    h = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        a = value.detach().cpu().contiguous().numpy()
        h.update(name.encode()); h.update(str(a.dtype).encode())
        h.update(str(a.shape).encode()); h.update(a.tobytes())
    return h.hexdigest()


def batch_hash(*values):
    h = hashlib.sha256()
    for value in values:
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def check_parameters(model, optimizer=None):
    for name, p in model.named_parameters():
        finite(p, "parameter " + name)
        if p.grad is not None:
            finite(p.grad, "gradient " + name)
    if optimizer is not None:
        for st in optimizer.state.values():
            for name, value in st.items():
                if torch.is_tensor(value):
                    finite(value, "optimizer " + name)


def read_records(path, expected_sha, expected_rows, cohort_path, cohort_sha, idx_start):
    path = Path(path).resolve()
    if any("test" in p.lower() for p in path.parts):
        raise ValueError("test paths are prohibited")
    if sha(path) != expected_sha:
        raise ValueError("input hash differs from prelocked configuration")
    result = validate_file(path, stage="diffusion", num_classes=2)
    if not result.valid or len(result.records) != expected_rows:
        raise ValueError("input contract/count failure: " + json.dumps(result.summary()))
    cohort_path = Path(cohort_path).resolve()
    if any("test" in p.lower() for p in cohort_path.parts):
        raise ValueError("test cohort paths are prohibited")
    if sha(cohort_path) != cohort_sha:
        raise ValueError("cohort input hash differs from prelocked configuration")
    with cohort_path.open() as fh:
        cohort = list(csv.DictReader(fh))
    if len(cohort) != expected_rows:
        raise ValueError("cohort count differs")
    for i, (r, row) in enumerate(zip(result.records, cohort)):
        if (r.idx != idx_start + i or r.protein != row["sequence"] or r.class_index != int(row["class"])
                or row["terminal_state"] != ("FREE", "AMD")[r.class_index]):
            raise ValueError("cohort/descriptor row identity differs at " + str(i))
    return result.records


class FeatureScaler:
    """Same train-only population zscore before inherited abs(global)."""
    def __init__(self, train):
        g = [[float(v) for v in r.global_descriptor] for r in train]
        s = [[float(v) for v in row] for r in train for row in r.residue_descriptors]
        self.stats = {"global": self._stats(g), "residue": self._stats(s),
                      "fit_record_ids": [r.idx for r in train], "fit_scope": "train only",
                      "method": "train_zscore_before_original_model_abs"}

    @staticmethod
    def _stats(rows):
        means = [sum(r[j] for r in rows) / len(rows) for j in range(len(rows[0]))]
        scales = [math.sqrt(sum((r[j] - means[j]) ** 2 for r in rows) / len(rows))
                  for j in range(len(means))]
        return {"mean": means, "scale": [s if s > 0 else 1.0 for s in scales], "n": len(rows)}

    def tensorize(self, records):
        target = torch.tensor([[AA.index(a) for a in r.protein] + [20] * (48-len(r.protein))
                               for r in records], dtype=torch.long)
        g = torch.zeros(len(records), 3, 48, dtype=torch.float32)
        globals_ = []
        for i, r in enumerate(records):
            st = self.stats["global"]
            # Float64 abs then float32 in the encoder matches the inherited boundary.
            globals_.append([abs((float(v)-st["mean"][j])/st["scale"][j])
                             for j, v in enumerate(r.global_descriptor)])
            st = self.stats["residue"]
            for t, row in enumerate(r.residue_descriptors):
                g[i, :, t] = torch.tensor([(float(v)-st["mean"][j])/st["scale"][j]
                                          for j, v in enumerate(row)], dtype=torch.float32)
        h = torch.tensor(globals_, dtype=torch.float64)
        labels = torch.tensor([r.class_index for r in records], dtype=torch.long)
        return target, g, h, labels


def make_model(arm, seed):
    torch.manual_seed(seed)
    return VAEModel(out_dim=1 if arm == "scalar_legacy" else 21, con_dim=3, latent_size=16,
                    sqz_prior=False, cond_drop=False, use_prior_flow=True, device="cpu")


def objective(model, target, g, h, arm):
    """Retain the encoder/flow/sample rule; normalize CE21/MSE21 by B*48."""
    v = model.vae
    z, mu, logsigma = v.encoder(v.word_emb(target.T).permute(1, 2, 0), v.g_pre_net(g), h)
    raw = v.decoder(z)
    zp = v.prior_flow(z, g)
    logq = torch.distributions.Normal(mu, logsigma.exp()).log_prob(z)
    logp = v.prior_dist.log_prob(zp)
    density = (logq-logp).sum()
    denom = target.numel()
    if arm == "scalar_legacy":
        reconstruction = (raw-target).square().sum() / (target != 20).sum()
        kl = density / 16
    else:
        if raw.shape != (len(target), 21, 48):
            raise ValueError("21-head output shape")
        if arm == "categorical_ce":
            reconstruction = torch.nn.functional.cross_entropy(raw, target, reduction="sum") / denom
        elif arm == "raw_onehot_mse":
            onehot = torch.nn.functional.one_hot(target, num_classes=21).permute(0, 2, 1).to(raw)
            reconstruction = 0.5 * (raw-onehot).square().sum() / denom
        else:
            raise ValueError("unknown arm")
        kl = density / denom
    for name, value in (("raw", raw), ("mu", mu), ("logsigma", logsigma),
                        ("density", density), ("reconstruction", reconstruction)):
        finite(value, name)
    return {"loss": reconstruction+kl, "reconstruction": reconstruction, "kl": kl,
            "density_sum": density, "mu": mu, "logsigma": logsigma, "raw": raw}


def joint_decode(score_bkt):
    """Same legal-prefix J for both heads; empty/full48 retained, earliest tie."""
    scores = score_bkt.transpose(1, 2)
    body, aa = scores[:, :, :20].max(-1)
    pad = scores[:, :, 20]
    base = pad.sum(1, keepdim=True)
    length_scores = base + torch.cat((torch.zeros_like(base), (body-pad).cumsum(1)), dim=1)
    length = length_scores.argmax(1)
    tokens = torch.where(torch.arange(48)[None, :] < length[:, None], aa, torch.full_like(aa, 20))
    return tokens, length, length_scores


def decode(model, z, arm, observations=None):
    if observations is not None:
        observations["sampled_z"] = z
    finite(z, "decoder input")
    raw = model.vae.decoder(z)
    if observations is not None:
        observations["decoder_raw"] = raw
    finite(raw, "decoder output")
    if arm == "scalar_legacy":
        tokens = raw.clamp(0, 20).round().long()
        lengths = []
        for row in tokens.tolist():
            lengths.append(row.index(20) if 20 in row else 48)
        lengths = torch.tensor(lengths)
        length_scores = None
    else:
        tokens, lengths, length_scores = joint_decode(raw)
    if observations is not None:
        observations.update(tokens=tokens, lengths=lengths)
        if length_scores is not None:
            observations["length_scores"] = length_scores
    sequences = ["".join(AA[t] for t in row[:length])
                 for row, length in zip(tokens.tolist(), lengths.tolist())]
    return sequences, tokens, lengths, raw, length_scores


def cycling_indices(n, batch_size, seed):
    rng = random.Random(seed)
    while True:
        order = list(range(n)); rng.shuffle(order)
        for i in range(0, n, batch_size):
            yield order[i:i+batch_size]


def inherited_vae_lr(step):
    if step <= 1000:
        return max(0.0005 * step / 1000, 1e-7)
    return max(0.0005 * 0.1 ** (step / 250000), 1e-7)


class Budget:
    """Check before each atomic update; at most one update may cross a cap."""
    def __init__(self, seconds, method_started, method_cap):
        self.started = time.perf_counter()
        self.seconds = seconds
        self.method_started = method_started
        self.method_cap = method_cap

    def exhausted(self):
        now = time.perf_counter()
        return now-self.started >= self.seconds or now-self.method_started >= self.method_cap


def save_model(out, name, model, extra):
    path = out / (name + ".pt")
    torch.save(model.state_dict(), path)
    dump(out / (name + ".json"), {"sha256": sha(path), "tensor_state_sha256": state_hash(model), **extra})


def train_vae(config, records, scaler, out, method_started, dev=None):
    budget = Budget(config["vae_seconds_cap"], method_started, config["training_seconds_cap"])
    model = make_model(config["arm"], config["initialization_seed"])
    initial = state_hash(model)
    save_model(out, "vae_initial_state", model, {"seed": config["initialization_seed"], "fresh": True})
    dump(out / "vae_initialization.json", {"tensor_state_sha256": initial,
         "seed": config["initialization_seed"], "parameters": sum(p.numel() for p in model.parameters()),
         "decoder_head_parameters": sum(p.numel() for p in model.vae.decoder.out_proj.parameters())})
    opt = torch.optim.Adam(model.parameters(), lr=inherited_vae_lr(0), betas=(0.9, 0.999))
    stream = cycling_indices(len(records), config["vae_batch_size"], config["shuffle_seed"])
    torch.manual_seed(config["vae_noise_seed"])
    done = 0
    for step in range(1, config["vae_updates"]+1):
        if budget.exhausted():
            break
        tic = time.perf_counter()
        idx = next(stream)
        target, g, h, _ = scaler.tensorize([records[i] for i in idx])
        model.train(); opt.zero_grad(set_to_none=True)
        for group in opt.param_groups:
            group["lr"] = inherited_vae_lr(max(0, step-2))
        forward_rng_sha256 = batch_hash(torch.get_rng_state())
        ret = objective(model, target, g, h, config["arm"])
        finite(ret["loss"], "VAE loss")
        ret["loss"].backward()
        check_parameters(model)
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        opt.step(); check_parameters(model, opt)
        done = step
        log(out / "vae_train.jsonl", {"step": step, "rows": len(idx), "record_indices": idx,
            "loss": float(ret["loss"].detach()), "reconstruction": float(ret["reconstruction"].detach()),
            "kl": float(ret["kl"].detach()), "density_sum": float(ret["density_sum"].detach()),
            "lr": opt.param_groups[0]["lr"], "gradient_norm_before_clip": float(norm),
            "forward_rng_sha256": forward_rng_sha256,
            "duration_seconds": time.perf_counter()-tic,
            "elapsed_stage_seconds": time.perf_counter()-budget.started})
        if dev is not None and config["monitor_interval"] and step % config["monitor_interval"] == 0:
            monitor_mean(config, model, dev, scaler, out, step)
    info = {"updates": done, "requested_updates": config["vae_updates"],
            "fixed_endpoint_reached": done == config["vae_updates"],
            "initial_tensor_state_sha256": initial,
            "elapsed_seconds_before_checkpoint_io": time.perf_counter()-budget.started}
    save_model(out, "vae_state", model, info)
    torch.save({"optimizer": opt.state_dict(), "torch_rng": torch.get_rng_state(),
                "completed_updates": done}, out / "vae_training_state.pt")
    info["elapsed_seconds"] = time.perf_counter()-budget.started
    dump(out / "vae_training_summary.json", info)
    return model, info


def posterior_parameters(model, records, scaler, batch_size, budget=None):
    model.eval()
    xs, ys = [], []
    with torch.random.fork_rng(), torch.no_grad():
        torch.manual_seed(0)
        for i in range(0, len(records), batch_size):
            if budget is not None and budget.exhausted():
                raise TimeoutError("training budget exhausted before posterior encoding batch")
            target, g, h, y = scaler.tensorize(records[i:i+batch_size])
            v = model.vae
            _, mu, logs = v.encoder(v.word_emb(target.T).permute(1, 2, 0), v.g_pre_net(g), h)
            value = torch.stack((mu, logs), dim=1).contiguous()
            finite(value, "posterior parameters")
            if value.shape[1:] != (2, 16, 48):
                raise ValueError("posterior shape")
            xs.append(value); ys.append(y)
    return torch.cat(xs), torch.cat(ys)


def noise_schedule(steps):
    beta = torch.linspace(1e-4, 0.02, steps)
    alpha = 1-beta
    return beta, alpha, torch.cumprod(alpha, dim=0)


def train_ddpm(config, model, train, scaler, out, method_started):
    budget = Budget(config["ddpm_seconds_cap"], method_started, config["training_seconds_cap"])
    if budget.exhausted():
        raise TimeoutError("training budget exhausted at DDPM stage entry")
    values, labels = posterior_parameters(model, train, scaler, config["vae_batch_size"], budget)
    np.savez_compressed(out / "train_posterior_parameters.npz", values=values.numpy(), labels=labels.numpy(),
                        row_ids=np.asarray([r.idx for r in train]))
    if budget.exhausted():
        raise TimeoutError("training budget exhausted before DDPM construction")
    torch.manual_seed(config["ddpm_seed"]); random.seed(config["ddpm_seed"])
    diff = UNet_conditional(c_in=2, c_out=2, time_dim=256, num_class1=2, num_class2=2)
    initial = state_hash(diff)
    ema = copy.deepcopy(diff).eval().requires_grad_(False)
    helper = EMA(0.995)
    opt = torch.optim.AdamW(diff.parameters(), lr=config["ddpm_lr"], eps=1e-5)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=config["ddpm_lr"],
                                                  total_steps=config["ddpm_updates"])
    beta, alpha, abar = noise_schedule(config["noise_steps"])
    generator = torch.Generator().manual_seed(config["ddpm_seed"])
    order, cursor, done = [], 0, 0
    setup_seconds = time.perf_counter()-budget.started
    for step in range(1, config["ddpm_updates"]+1):
        if budget.exhausted():
            break
        tic = time.perf_counter()
        if cursor >= len(order):
            order = torch.randperm(len(values), generator=generator).tolist(); cursor = 0
        idx = order[cursor:cursor+config["ddpm_batch_size"]]; cursor += len(idx)
        value, label = values[idx], labels[idx]
        t = torch.randint(1, config["noise_steps"], (len(idx),))
        noise = torch.randn_like(value)
        a = abar[t][:, None, None, None]
        xt = a.sqrt()*value+(1-a).sqrt()*noise
        dropped = random.random() < 0.1
        diff.train(); opt.zero_grad(set_to_none=True)
        pred = diff(xt, t, None if dropped else label)
        finite(pred, "DDPM prediction")
        loss = torch.nn.functional.mse_loss(pred, noise)
        finite(loss, "DDPM loss"); loss.backward(); check_parameters(diff)
        opt.step(); check_parameters(diff, opt); helper.step_ema(ema, diff); scheduler.step()
        done = step
        log(out / "ddpm_train.jsonl", {"step": step, "record_indices": idx, "rows": len(idx),
            "mse": float(loss.detach()), "lr": opt.param_groups[0]["lr"], "label_dropped": dropped,
            "noise_timestep_sha256": batch_hash(noise, t), "duration_seconds": time.perf_counter()-tic,
            "elapsed_stage_seconds": time.perf_counter()-budget.started})
    info = {"updates": done, "requested_updates": config["ddpm_updates"],
            "fixed_endpoint_reached": done == config["ddpm_updates"],
            "initial_tensor_state_sha256": initial, "posterior_encoding_and_setup_seconds": setup_seconds,
            "elapsed_seconds_before_checkpoint_io": time.perf_counter()-budget.started,
            "parameters": sum(p.numel() for p in diff.parameters()), "ema_step": helper.step,
            "ema_semantics": "inherited reset/copy until 2000 updates; beta=.995 thereafter"}
    save_model(out, "ddpm_state", diff, info); save_model(out, "ddpm_ema_state", ema, info)
    torch.save({"optimizer": opt.state_dict(), "scheduler": scheduler.state_dict(),
                "torch_rng": torch.get_rng_state(), "python_rng": random.getstate(),
                "ema_step": helper.step, "completed_updates": done}, out / "ddpm_training_state.pt")
    info["elapsed_seconds"] = time.perf_counter()-budget.started
    dump(out / "ddpm_training_summary.json", info)
    return ema, (beta, alpha, abar), info


def allocated_requests(config):
    rows = []
    for label, terminal in enumerate(("FREE", "AMD")):
        for i in range(config["samples_per_terminal"]):
            rows.append({"sample_id": f"{ARMS[config['arm']]}_{terminal}_{i:03d}",
                         "method": ARMS[config["arm"]], "terminal": terminal, "sequence": "",
                         "status": "not_executed", "length": 0, "seed": config["sampling_seeds"][label],
                         "terminal_sample_id": i})
    return rows


def tensor_observation_summary(tensor):
    """Diagnostics only: preserve nonfinite values in NPZ, use JSON-safe extrema."""
    array = tensor.detach().cpu().numpy()
    mask = np.isfinite(array)
    finite_values = array[mask]
    return {"shape": list(array.shape), "dtype": str(array.dtype),
            "elements": int(array.size), "finite_elements": int(mask.sum()),
            "all_finite": bool(mask.all()), "nan_elements": int(np.isnan(array).sum()),
            "positive_inf_elements": int(np.isposinf(array).sum()),
            "negative_inf_elements": int(np.isneginf(array).sum()),
            "finite_min": float(finite_values.min()) if finite_values.size else None,
            "finite_max": float(finite_values.max()) if finite_values.size else None}


def save_generation_observations(out, terminal, observations, phase, timestep, error):
    """Save only already-computed tensors, with no RNG or model calls."""
    arrays = {name: value.detach().cpu().numpy() for name, value in observations.items()
              if torch.is_tensor(value)}
    np.savez_compressed(out / f"generation_failure_{terminal}.npz", **arrays)
    dump(out / f"generation_failure_{terminal}.json", {"terminal": terminal,
        "phase": phase, "reverse_timestep": timestep, "error": error,
        "tensors": {name: tensor_observation_summary(value) for name, value in observations.items()
                    if torch.is_tensor(value)},
        "scope": "latest available tensor per name at failure; no retry, clipping or replacement"})


def generate(config, model, ddpm, schedule, out, rows):
    started = time.perf_counter()
    beta, alpha, abar = schedule
    model.eval(); ddpm.eval()
    for label, terminal in enumerate(("FREE", "AMD")):
        selected = [r for r in rows if r["terminal"] == terminal]
        if time.perf_counter()-started >= config["generation_seconds_cap"]:
            break
        observations, phase, timestep = {}, "initial_noise", None
        try:
            with torch.no_grad():
                torch.manual_seed(config["sampling_seeds"][label])
                n = len(selected)
                labels = torch.full((n,), label, dtype=torch.long)
                x = torch.randn(n, 2, 16, 48)
                observations.update(labels=labels, posterior_parameters=x)
                phase = "reverse_ddpm"
                for timestep in reversed(range(1, config["noise_steps"])):
                    if time.perf_counter()-started >= config["generation_seconds_cap"]:
                        raise TimeoutError("generation cap reached; no retry")
                    t = torch.full((n,), timestep, dtype=torch.long)
                    observations["timestep_tensor"] = t
                    prediction = ddpm(x, t, labels)
                    observations["latest_prediction"] = prediction
                    noise = torch.randn_like(x) if timestep > 1 else torch.zeros_like(x)
                    observations["latest_noise"] = noise
                    x = (x-(1-alpha[timestep])/(1-abar[timestep]).sqrt()*prediction)/alpha[timestep].sqrt()
                    x = x+beta[timestep].sqrt()*noise
                    observations["posterior_parameters"] = x
                    finite(x, "reverse DDPM")
                    if timestep % 100 == 0 or timestep == 1:
                        log(out / "generation_progress.jsonl", {"terminal": terminal,
                            "timestep": timestep, "elapsed_seconds": time.perf_counter()-started})
                mu, logsigma = x[:, 0], x[:, 1]
                observations.update(mu=mu, logsigma=logsigma)
                phase = "final_posterior_before_exp"
                # Retain generated posterior parameters even if exp(logsigma) later fails.
                np.savez_compressed(out / f"generated_posterior_{terminal}.npz", values=x.numpy())
                dump(out / f"generated_posterior_summary_{terminal}.json", {"terminal": terminal,
                    "mu": tensor_observation_summary(mu), "logsigma": tensor_observation_summary(logsigma),
                    "posterior_parameters": tensor_observation_summary(x), "recorded_before_exp": True})
                phase = "posterior_exp"
                sigma = logsigma.exp()
                observations["sigma"] = sigma
                finite(sigma, "generated exp(logsigma)")
                if not bool((sigma > 0).all()):
                    raise FloatingPointError("generated exp(logsigma) underflow")
                z = mu+torch.randn_like(mu)*sigma
                phase = "sampled_z_and_decode"
                seqs, tokens, lengths, raw, length_scores = decode(model, z, config["arm"], observations)
                phase = "successful_export"
                arrays = {"posterior_parameters": x.numpy(), "z": z.numpy(), "raw": raw.numpy(),
                          "tokens": tokens.numpy(), "lengths": lengths.numpy()}
                if length_scores is not None:
                    arrays["length_scores"] = length_scores.numpy()
                np.savez_compressed(out / f"generated_{terminal}.npz", **arrays)
                for row, seq, length in zip(selected, seqs, lengths.tolist()):
                    status = "empty_sequence" if length == 0 else ("ok" if length < 48 else "no_eos_at_limit")
                    row.update(sequence=seq, length=length, status=status)
        except Exception as exc:
            failure_traceback = traceback.format_exc()
            for row in selected:
                row["status"] = "error:" + type(exc).__name__
            observation_error = None
            try:
                save_generation_observations(out, terminal, observations, phase, timestep, repr(exc))
            except Exception as observation_exc:
                # A diagnostic IO failure must not replace the original model error.
                observation_error = repr(observation_exc)
            log(out / "generation_errors.jsonl", {"terminal": terminal,
                "error": repr(exc), "traceback": failure_traceback,
                "observation_write_error": observation_error})
        write_csv(out / "generated.csv", rows, GENERATED_FIELDS)
    info = {"elapsed_seconds": time.perf_counter()-started, "requested": len(rows),
            "successful_execution": sum(r["status"] in SUCCESS_STATUSES for r in rows),
            "failed_execution": sum(r["status"].startswith("error:") for r in rows),
            "not_executed": sum(r["status"] == "not_executed" for r in rows),
            "condition_inputs": ["terminal class only"], "no_retry_or_filter": True}
    dump(out / "generation_summary.json", info)
    return info


def monitor_mean(config, model, dev, scaler, out, step):
    """Fixed-interval, no-selection development diagnostics; preserve training RNG."""
    started = time.perf_counter()
    was_training = model.training
    before_rng = batch_hash(torch.get_rng_state())
    sums = {"correct_nonpad": 0, "nonpad": 0, "absolute_length_error": 0, "exact_sequences": 0}
    reconstruction_sum = 0.0
    try:
        with torch.random.fork_rng(), torch.no_grad():
            values, _ = posterior_parameters(model, dev, scaler, config["vae_batch_size"])
            for i in range(0, len(dev), config["vae_batch_size"]):
                block = dev[i:i+config["vae_batch_size"]]
                target, _, _, _ = scaler.tensorize(block)
                seqs, _, lengths, raw, _ = decode(model, values[i:i+len(block), 0], config["arm"])
                visible = torch.tensor([[AA.index(a) for a in s]+[20]*(48-len(s)) for s in seqs])
                mask = target != 20
                sums["correct_nonpad"] += int(((visible == target) & mask).sum())
                sums["nonpad"] += int(mask.sum())
                sums["absolute_length_error"] += sum(abs(int(n)-len(r.protein)) for n, r in zip(lengths, block))
                sums["exact_sequences"] += sum(s == r.protein for s, r in zip(seqs, block))
                if config["arm"] == "scalar_legacy":
                    reconstruction_sum += float((raw-target).square().sum())
                elif config["arm"] == "categorical_ce":
                    reconstruction_sum += float(torch.nn.functional.cross_entropy(raw, target, reduction="sum"))
                else:
                    onehot = torch.nn.functional.one_hot(target, 21).permute(0, 2, 1).to(raw)
                    reconstruction_sum += float(0.5*(raw-onehot).square().sum())
    finally:
        model.train(was_training)
    after_rng = batch_hash(torch.get_rng_state())
    if before_rng != after_rng:
        raise RuntimeError("development monitor changed training RNG")
    log(out / "dev_monitor.jsonl", {"step": step, "rows": len(dev), **sums,
        "nonpad_token_accuracy": sums["correct_nonpad"]/sums["nonpad"],
        "length_mae": sums["absolute_length_error"]/len(dev),
        "mean_reconstruction_objective": reconstruction_sum / (sums["nonpad"] if config["arm"] == "scalar_legacy" else len(dev)*48),
        "rng_preserved": True, "rng_sha256": before_rng, "seconds": time.perf_counter()-started,
        "checkpoint_selection": False})


def reconstruction_export(config, model, records, scaler, out, role):
    """No fitting/selection here. Export deterministic mean rows for independent evaluation."""
    values, _ = posterior_parameters(model, records, scaler, config["vae_batch_size"])
    rows = []
    with torch.no_grad():
        for i in range(0, len(records), config["vae_batch_size"]):
            block = records[i:i+config["vae_batch_size"]]
            seqs, tokens, _, _, _ = decode(model, values[i:i+len(block), 0], config["arm"])
            for j, (r, seq, tok) in enumerate(zip(block, seqs, tokens.tolist())):
                visible_tokens = [AA.index(a) for a in seq]+[20]*(48-len(seq))
                rows.append({"row_id": i+j, "terminal": ("FREE", "AMD")[r.class_index],
                    "truth_sequence": r.protein, "prediction_sequence": seq,
                    "prediction_tokens48": json.dumps(visible_tokens),
                    "raw_prediction_tokens48": json.dumps(tok),
                    "token_encoding": "ARNDCEQGHILKMFPSTWYV=0..19;PAD=20", "status": "ok"})
    write_csv(out / f"{role}_reconstruction.csv", rows,
              ["row_id", "terminal", "truth_sequence", "prediction_sequence", "prediction_tokens48",
               "raw_prediction_tokens48", "token_encoding", "status"])


def validate_config(config):
    required = ["arm", "source_root", "train", "train_sha256", "train_rows", "purpose",
                "train_cohort", "train_cohort_sha256", "train_idx_start",
                "initialization_seed", "shuffle_seed", "vae_noise_seed", "ddpm_seed", "sampling_seeds",
                "vae_updates", "ddpm_updates", "vae_batch_size", "ddpm_batch_size", "ddpm_lr",
                "noise_steps", "training_seconds_cap", "vae_seconds_cap", "ddpm_seconds_cap",
                "generation_seconds_cap", "samples_per_terminal", "do_eval"]
    required.append("monitor_interval")
    for key in required:
        if key not in config:
            raise ValueError("missing config key: " + key)
    if config["arm"] not in ARMS or config["purpose"] not in ("train_only_smoke", "formal_pilot"):
        raise ValueError("arm/purpose not registered")
    if config["train_rows"] != 279 or len(config["sampling_seeds"]) != 2:
        raise ValueError("frozen train count or terminal seeds")
    if type(config["train_idx_start"]) is not int or config["train_idx_start"] != 0:
        raise ValueError("frozen train source indices must start at 0")
    if config["do_eval"]:
        for k in ("dev", "dev_sha256", "dev_rows", "dev_cohort", "dev_cohort_sha256", "dev_idx_start"):
            if k not in config:
                raise ValueError("missing development binding")
        if config["dev_rows"] != 41:
            raise ValueError("development count")
        if type(config["dev_idx_start"]) is not int or config["dev_idx_start"] != 279:
            raise ValueError("frozen development source indices must start at 279")
    if config["purpose"] == "train_only_smoke":
        if config["do_eval"] or config["samples_per_terminal"] != 1 or config["monitor_interval"] != 0:
            raise ValueError("smoke is train-only and two generation attempts")
        if any(k.startswith("dev") for k in config):
            raise ValueError("smoke may not provide development paths")
    elif not config["do_eval"] or config["samples_per_terminal"] != 16:
        raise ValueError("formal pilot requires deterministic dev export and all 32 requests")
    elif config["monitor_interval"] != 100:
        raise ValueError("formal pilot monitor interval is fixed at 100")
    for key in ("vae_updates", "ddpm_updates", "vae_batch_size", "ddpm_batch_size"):
        if not isinstance(config[key], int) or config[key] <= 0:
            raise ValueError("positive integer required: " + key)
    if config["noise_steps"] < 3 or config["ddpm_updates"] < 3:
        raise ValueError("DDPM schedule too short")
    for key in ("training_seconds_cap", "vae_seconds_cap", "ddpm_seconds_cap", "generation_seconds_cap"):
        if not math.isfinite(config[key]) or config[key] <= 0:
            raise ValueError("invalid positive cap: " + key)
    if config["vae_seconds_cap"]+config["ddpm_seconds_cap"] > config["training_seconds_cap"]:
        raise ValueError("phase allocations exceed total training cap")
    if "implementation_sha256" not in config or sha(__file__) != config["implementation_sha256"]:
        raise ValueError("runner hash must match prelocked configuration")
    # No checkpoint inputs or test options accepted, including disguised extras.
    if any("checkpoint" in k.lower() or "test" in k.lower() for k in config):
        raise ValueError("warm-start/checkpoint/test inputs prohibited")


def run(config_path, output):
    config = json.loads(Path(config_path).read_text())
    validate_config(config)
    out = Path(output).resolve()
    if out.exists() and any(out.iterdir()):
        raise ValueError("output must be absent or empty; no overwrite or implicit resume")
    out.mkdir(parents=True, exist_ok=True)
    dump(out / "config.json", config)
    rows = allocated_requests(config)
    write_csv(out / "generated.csv", rows, GENERATED_FIELDS)
    process_started = time.perf_counter()
    status = {"state": "running", "purpose": config["purpose"], "config_sha256": sha(config_path),
              "runner_sha256": sha(__file__), "test_read": False, "development_read": False,
              "train_read_attempted": False, "train_read_validated": False,
              "development_read_attempted": False, "development_read_validated": False}
    dump(out / "status.json", status)
    try:
        dump(out / "runtime_and_source.json", init_runtime(config["source_root"]))
        status["train_read_attempted"] = True
        dump(out / "status.json", status)
        train = read_records(config["train"], config["train_sha256"], config["train_rows"],
                             config["train_cohort"], config["train_cohort_sha256"], config["train_idx_start"])
        status["train_read_validated"] = True
        dump(out / "status.json", status)
        dev = None
        if config["do_eval"]:
            status["development_read"] = True
            status["development_read_attempted"] = True
            dump(out / "status.json", status)
            dev = read_records(config["dev"], config["dev_sha256"], config["dev_rows"],
                               config["dev_cohort"], config["dev_cohort_sha256"], config["dev_idx_start"])
            status["development_read_validated"] = True
            dump(out / "status.json", status)
            if {r.protein for r in train} & {r.protein for r in dev}:
                raise ValueError("train/development sequence overlap")
        method_started = time.perf_counter()
        scaler = FeatureScaler(train)
        dump(out / "train_feature_scaler.json", scaler.stats)
        model, vae_info = train_vae(config, train, scaler, out, method_started, dev)
        if not vae_info["fixed_endpoint_reached"]:
            raise TimeoutError("VAE fixed update endpoint not reached under cap")
        ddpm, schedule, ddpm_info = train_ddpm(config, model, train, scaler, out, method_started)
        status["training_seconds"] = time.perf_counter()-method_started
        if not ddpm_info["fixed_endpoint_reached"]:
            raise TimeoutError("DDPM fixed update endpoint not reached under cap")
        status["training_cap_exceeded"] = status["training_seconds"] > config["training_seconds_cap"]
        generation = generate(config, model, ddpm, schedule, out, rows)
        if config["do_eval"]:
            reconstruction_export(config, model, train, scaler, out, "train")
            reconstruction_export(config, model, dev, scaler, out, "dev")
        status.update(state="completed" if generation["successful_execution"] == len(rows)
                      and not status["training_cap_exceeded"] else "completed_with_failure",
                      vae_updates=vae_info["updates"], ddpm_updates=ddpm_info["updates"], generation=generation)
    except Exception as exc:
        status.update(state="failed", error=repr(exc), traceback=traceback.format_exc())
        print(status["traceback"], file=sys.stderr)
    finally:
        status["process_seconds"] = time.perf_counter()-process_started
        dump(out / "status.json", status)
        dump(out / "manifest.json", {"files": [{"path": p.name, "bytes": p.stat().st_size,
              "sha256": sha(p)} for p in sorted(out.iterdir()) if p.is_file() and p.name != "manifest.json"]})
    return 0 if status["state"] == "completed" else 2


def benchmark(source_root, output, steps):
    """Synthetic model/gradient/decoder checks, no natural data read or sampling."""
    out = Path(output).resolve(); out.mkdir(parents=True, exist_ok=False)
    runtime = init_runtime(source_root)
    torch.manual_seed(812)
    target = torch.randint(0, 20, (32, 48)); target[:, 25:] = 20
    g = torch.randn(32, 3, 48); g[:, :, 25:] = 0
    h = torch.randn(32, 9).abs().double()
    results = []
    for arm in ARMS:
        model = make_model(arm, 222)
        initial = state_hash(model)
        opt = torch.optim.Adam(model.parameters(), lr=0.0005)
        seconds = []
        for step in range(steps):
            tic = time.perf_counter(); opt.zero_grad(set_to_none=True)
            ret = objective(model, target, g, h, arm)
            ret["loss"].backward(); check_parameters(model)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            opt.step(); check_parameters(model, opt)
            seconds.append(time.perf_counter()-tic)
        with torch.no_grad():
            _, tok, lengths, _, _ = decode(model, ret["mu"], arm)
        results.append({"arm": arm, "initial_state": initial, "step_seconds": seconds,
                        "token_shape": list(tok.shape), "length_range": [int(lengths.min()), int(lengths.max())]})
        del model, opt, ret
    score = torch.zeros(2, 21, 48); score[1, 0, :5] = 2
    tokens, lengths, _ = joint_decode(score)
    assert lengths.tolist() == [0, 5]
    assert results[0]["initial_state"] == results[1]["initial_state"]
    torch.manual_seed(322)
    diff = UNet_conditional(c_in=2, c_out=2, time_dim=256, num_class1=2, num_class2=2)
    opt = torch.optim.AdamW(diff.parameters(), lr=0.005, eps=1e-5)
    durations = []
    for step in range(steps):
        tic = time.perf_counter()
        x = torch.randn(8, 2, 16, 48); t = torch.randint(1, 999, (8,)); y = torch.arange(8) % 2
        opt.zero_grad(set_to_none=True)
        loss = diff(x, t, y).square().mean(); loss.backward(); check_parameters(diff)
        opt.step(); check_parameters(diff, opt); durations.append(time.perf_counter()-tic)
    report = {"scope": "synthetic-only; no natural train/dev/test data read", "vae": results,
              "ddpm_step_seconds": durations, "joint_decoder_tie_and_prefix_checks": True,
              "matched_21head_initialization": True, "runtime": runtime}
    dump(out / "synthetic_benchmark.json", report)
    print(json.dumps({k: v for k, v in report.items() if k != "runtime"}), flush=True)
    return 0


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    runp = sub.add_parser("run"); runp.add_argument("--config", required=True); runp.add_argument("--output", required=True)
    bench = sub.add_parser("benchmark"); bench.add_argument("--source-root", default=DEFAULT_SOURCE)
    bench.add_argument("--output", required=True); bench.add_argument("--steps", type=int, default=3)
    args = p.parse_args()
    if args.command == "run":
        return run(args.config, args.output)
    return benchmark(args.source_root, args.output, args.steps)


if __name__ == "__main__":
    raise SystemExit(main())
