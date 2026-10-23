"""Fixed 2x64 terminal-conditional LSTM reference model for AMP H5.

No file is read and no training starts on import. The only generation condition
is terminal class. This is a local canonical AR implementation, not a claimed
reproduction of a named paper. Root protocol owns budgets, splits, and seeds.
"""
from dataclasses import asdict, dataclass
import csv
import hashlib
import json
import math
from pathlib import Path
import time
import traceback
from typing import Sequence

import torch
from torch import nn
from torch.nn import functional as F

ALPHABET = "ARNDCEQGHILKMFPSTWYV"
PAD = 20  # PAD is also the first-stop/EOS outcome.
BOS = 21  # Input only, never an output class.
MAX_LENGTH = 48
TERMINAL_CLASS = {"FREE": 0, "AMD": 1}


@dataclass(frozen=True)
class ARConfig:
    token_embedding_dim: int = 32
    terminal_embedding_dim: int = 8
    hidden_dim: int = 64
    layers: int = 2
    max_length: int = MAX_LENGTH

    def validate(self):
        if asdict(self) != asdict(ARConfig()):
            raise ValueError("H5 AR architecture is fixed; a change needs a new prelock")

    def parameter_count(self):
        e, c, h = self.token_embedding_dim, self.terminal_embedding_dim, self.hidden_dim
        return 22 * e + 2 * c + 4 * h * (e + c + h + 2) + 4 * h * (h + h + 2) + 21 * h + 21


def check_labels(labels, count):
    if labels.ndim != 1 or labels.shape[0] != count or labels.dtype != torch.long:
        raise ValueError("terminal labels must be a one-dimensional long tensor")
    if not bool(torch.all((labels == 0) | (labels == 1))):
        raise ValueError("terminal labels must be FREE=0 or AMD=1")


def make_teacher_forcing_batch(sequences: Sequence[str], labels, *, allow_empty=False):
    """Return BOS/shifted inputs, targets, and loss mask, never oracle length input.

    A sequence shorter than 48 contributes its residues and exactly its first
    PAD/EOS target. Later PAD positions are excluded. A 48-residue sequence has
    no observed EOS inside this finite observation window (48 AA targets).
    """
    if not sequences:
        raise ValueError("empty batch")
    labels = torch.as_tensor(labels, dtype=torch.long, device="cpu")
    check_labels(labels, len(sequences))
    targets = torch.full((len(sequences), MAX_LENGTH), PAD, dtype=torch.long)
    inputs = torch.full_like(targets, PAD)
    inputs[:, 0] = BOS
    mask = torch.zeros_like(targets, dtype=torch.bool)
    lengths = []
    for row, sequence in enumerate(sequences):
        if not isinstance(sequence, str) or not set(sequence) <= set(ALPHABET):
            raise ValueError("noncanonical sequence")
        if not (0 if allow_empty else 1) <= len(sequence) <= MAX_LENGTH:
            raise ValueError("sequence length outside contract")
        lengths.append(len(sequence))
        if sequence:
            targets[row, :len(sequence)] = torch.tensor([ALPHABET.index(a) for a in sequence])
        mask[row, :min(len(sequence) + 1, MAX_LENGTH)] = True
    inputs[:, 1:] = targets[:, :-1]
    return {"inputs": inputs, "targets": targets, "mask": mask,
            "labels": labels, "lengths_for_accounting_only": torch.tensor(lengths)}


class TerminalLSTM(nn.Module):
    def __init__(self, config=ARConfig()):
        super().__init__()
        config.validate()
        self.config = config
        self.token_embedding = nn.Embedding(22, config.token_embedding_dim)
        self.terminal_embedding = nn.Embedding(2, config.terminal_embedding_dim)
        self.lstm = nn.LSTM(config.token_embedding_dim + config.terminal_embedding_dim,
                           config.hidden_dim, num_layers=2, batch_first=True, dropout=0.0)
        self.output = nn.Linear(config.hidden_dim, 21)
        if sum(p.numel() for p in self.parameters()) != config.parameter_count():
            raise AssertionError("parameter accounting mismatch")

    def forward(self, inputs, labels, hidden=None):
        if inputs.ndim != 2 or inputs.dtype != torch.long:
            raise ValueError("inputs must be batch by time long tokens")
        if not bool(torch.all((inputs >= 0) & (inputs <= BOS))):
            raise ValueError("input token outside vocabulary")
        check_labels(labels, inputs.shape[0])
        terminal = self.terminal_embedding(labels).unsqueeze(1).expand(-1, inputs.shape[1], -1)
        embedded = torch.cat((self.token_embedding(inputs), terminal), dim=-1)
        states, hidden = self.lstm(embedded, hidden)
        return self.output(states), hidden


def sequence_nll(logits, targets, mask):
    """Mean observed-token CE; AA and first-EOS sums are logged separately."""
    if logits.shape != (*targets.shape, 21) or targets.shape != mask.shape or mask.dtype != torch.bool:
        raise ValueError("logits/targets/mask shape mismatch")
    if not bool(torch.any(mask)) or not bool(torch.all((targets >= 0) & (targets <= PAD))):
        raise ValueError("empty observation mask or out-of-vocabulary target")
    token_loss = F.cross_entropy(logits.transpose(1, 2), targets, reduction="none")
    observed = token_loss.masked_select(mask)
    aa_mask, eos_mask = mask & (targets != PAD), mask & (targets == PAD)
    return {"loss": observed.mean(), "nll_sum": observed.sum(), "observed_tokens": mask.sum(),
            "aa_nll_sum": token_loss.masked_select(aa_mask).sum(), "aa_tokens": aa_mask.sum(),
            "eos_nll_sum": token_loss.masked_select(eos_mask).sum(), "eos_tokens": eos_mask.sum(),
            "correct_tokens": ((logits.argmax(-1) == targets) & mask).sum()}


def tokens_to_sequence(tokens):
    sequence = []
    stopped = False
    for value in tokens:
        token = int(value)
        if not 0 <= token <= PAD:
            raise ValueError("output token outside vocabulary")
        if token == PAD:
            stopped = True
            break
        sequence.append(ALPHABET[token])
    if len(sequence) > MAX_LENGTH:
        raise ValueError("output exceeds fixed maximum")
    return "".join(sequence), stopped


@torch.no_grad()
def sample_batch(model, labels, *, seed, requests, cap_seconds=None):
    """Exactly the requested attempts, temperature 1, no length/sample rejection.

    The seed owns an isolated CPU torch.Generator. Draws are batched at each
    position, including absorbing finished rows; this RNG rule is fixed. Saved
    active-before masks distinguish real next-token scores from finished rows.
    There is no reference-sequence argument or length condition.
    """
    if not isinstance(requests, int) or requests < 1:
        raise ValueError("an explicit positive request count is mandatory")
    if cap_seconds is not None and (not math.isfinite(cap_seconds) or cap_seconds < 0):
        raise ValueError("optional sampling cap must be finite and nonnegative")
    started = time.perf_counter()
    labels = torch.as_tensor(labels, dtype=torch.long, device="cpu")
    check_labels(labels, requests)
    if any(p.device.type != "cpu" for p in model.parameters()):
        raise ValueError("H5 pilot is CPU only")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    was_training = model.training
    model.eval()
    inputs = torch.full((requests, 1), BOS, dtype=torch.long)
    finished = torch.zeros(requests, dtype=torch.bool)
    tokens = torch.full((requests, MAX_LENGTH), PAD, dtype=torch.long)
    draws = torch.full_like(tokens, PAD)
    scores = torch.zeros((requests, MAX_LENGTH, 21), dtype=torch.float32)
    probabilities = torch.zeros_like(scores)
    active_before = torch.zeros((requests, MAX_LENGTH), dtype=torch.bool)
    hidden = None
    steps = 0
    failure, capped = None, False
    try:
        try:
            for position in range(MAX_LENGTH):
                if cap_seconds is not None and time.perf_counter() - started >= cap_seconds:
                    capped = True
                    break
                logits, hidden = model(inputs, labels, hidden)
                logits = logits[:, 0, :]
                if not bool(torch.isfinite(logits).all()):
                    raise FloatingPointError("nonfinite AR sampling logits")
                probability = torch.softmax(logits, -1)
                proposal = torch.multinomial(probability, 1, generator=generator).squeeze(1)
                active_before[:, position] = ~finished
                scores[:, position, :] = logits
                probabilities[:, position, :] = probability
                draws[:, position] = proposal
                actual = torch.where(finished, PAD, proposal)
                tokens[:, position] = actual
                finished = finished | (actual == PAD)
                inputs = actual.unsqueeze(1)
                steps += 1
                if bool(finished.all()):
                    break
        except Exception as exc:
            failure = {"type": type(exc).__name__, "message": str(exc), "position": steps,
                       "traceback": traceback.format_exc()}
        sequences, stopped = zip(*(tokens_to_sequence(row.tolist()) for row in tokens))
        # A PAD beyond the evaluated positions is padding, not a sampled EOS.
        terminated = [(len(s) < steps) for s in sequences]
        statuses = []
        for row, sequence in enumerate(sequences):
            if bool(finished[row]) or steps == MAX_LENGTH:
                statuses.append("empty_sequence" if not sequence else "ok")
            elif failure:
                statuses.append("error")
            else:
                statuses.append("partial_capped" if steps else "not_executed")
        return {"sequences": list(sequences), "tokens": tokens, "raw_draws": draws,
                "logits": scores, "probabilities": probabilities, "active_before": active_before,
                "terminated": terminated, "max_length_hit": [len(s) == MAX_LENGTH for s in sequences],
                "status": statuses, "failure": failure, "capped": capped,
                "steps": steps, "seed": int(seed), "requests": requests,
                "observed_sampling_tokens": int(active_before.sum()), "generator_state": generator.get_state()}
    finally:
        model.train(was_training)


def configure_cpu():
    """One CPU thread; called only on an explicit train/evaluate/generate call."""
    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)


def dump(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def file_record(path):
    path = Path(path)
    return {"path": str(path.resolve()), "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def state_digest(model):
    digest = hashlib.sha256()
    for key, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(key.encode("utf-8") + b"\0")
        digest.update(str(value.dtype).encode("ascii") + b"\0")
        digest.update(str(tuple(value.shape)).encode("ascii") + b"\0")
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def read_split(path, *, role, expected_sha256, expected_rows):
    """Explicit hash-bound train/dev loader. Reject test before file access."""
    if role not in ("train", "development"):
        raise ValueError("only train and reused development data may be read")
    path = Path(path).resolve()
    if any("test" in part.lower() for part in path.parts):
        raise ValueError("test path prohibited")
    payload = path.read_bytes()
    actual_sha256 = hashlib.sha256(payload).hexdigest()
    if actual_sha256 != expected_sha256:
        raise ValueError("split hash differs from prelocked input")
    rows = list(csv.DictReader(payload.decode("utf-8").splitlines()))
    if len(rows) != expected_rows:
        raise ValueError("split size mismatch")
    seen = set()
    for row in rows:
        if None in row or any(v is None for v in row.values()):
            raise ValueError("ragged row")
        s, terminal = row["sequence"], row["terminal_state"]
        if not 1 <= len(s) <= 48 or not set(s) <= set(ALPHABET) or str(TERMINAL_CLASS.get(terminal)) != row["class"]:
            raise ValueError("invalid canonical sequence or terminal class")
        sha = hashlib.sha256(s.encode("ascii")).hexdigest()
        molecule = hashlib.sha256(("AMP_MOLECULE_V1\n" + s + "\n" + terminal).encode("ascii")).hexdigest()
        if (row["sequence_sha256"] != sha or row["molecule_sha256"] != molecule
                or row["length"] != str(len(s)) or not row["component_id"] or molecule in seen):
            raise ValueError("invalid identity, length, component, or duplicate")
        for field in ("split", "partition", "role"):
            allowed = {"train"} if role == "train" else {"val", "validation", "development"}
            if field in row and row[field].lower() not in allowed:
                raise ValueError("row split conflicts with input role")
        seen.add(molecule)
    return {"role": role, "path": str(path), "sha256": actual_sha256, "rows": rows}


def assert_split_isolation(train, development):
    if train["role"] != "train" or development["role"] != "development":
        raise ValueError("train/dev role mismatch")
    for field in ("sequence", "molecule_sha256", "component_id"):
        if {r[field] for r in train["rows"]} & {r[field] for r in development["rows"]}:
            raise ValueError("train/development overlap: " + field)
    return True


def _checkpoint(model, optimizer, generator, *, seed, updates, order, cursor, epoch):
    return {"architecture": asdict(ARConfig()), "model": model.state_dict(),
            "optimizer": optimizer.state_dict(), "seed": seed, "updates": updates,
            "batch_generator_state": generator.get_state(), "batch_order": order,
            "batch_cursor": cursor, "batch_epoch": epoch, "tensor_state_sha256": state_digest(model)}


def run_train(cap_seconds, fixed_steps, seed, out, *, train_bundle, batch_size=32, shuffle_seed=None,
              dev_bundle=None, monitor_every=None):
    """One recipe, final state only, no development access or retry.

    Root supplies the prelocked train bundle and budget after independent audit.
    Optional dev_bundle + monitor_every=100 permits descriptive train/dev NLL
    monitoring at exactly 100-update boundaries, at most ten observations. It
    never affects optimization, checkpoint choice, hyperparameters, or stopping.
    Omit both in smoke: then no development row is accessed.
    This function has no data-path default. cap_seconds starts before setup and
    is checked between optimizer updates. One update and final checkpoint I/O
    may overrun: both actual time and overrun are reported. fixed_steps is a hard
    update upper bound, not a dev-selected convergence stopping rule. A separate
    train-only throughput smoke must pass its separate bundle/budget/output and
    its weights must be discarded by the caller.
    """
    started = time.perf_counter()
    if (isinstance(fixed_steps, bool) or not isinstance(fixed_steps, int) or fixed_steps < 1
            or not math.isfinite(cap_seconds) or cap_seconds <= 0 or batch_size != 32
            or isinstance(seed, bool) or not isinstance(seed, int)):
        raise ValueError("positive finite cap, positive fixed_steps, integer seed, fixed batch32 required")
    if train_bundle.get("role") != "train" or not train_bundle.get("rows"):
        raise ValueError("nonempty explicit train role required")
    if dev_bundle is None and monitor_every is not None:
        raise ValueError("monitor interval requires explicit development bundle")
    if dev_bundle is not None:
        if monitor_every != 100 or fixed_steps > 1000:
            raise ValueError("optional monitors fixed every100 steps, at most1000 total updates")
        assert_split_isolation(train_bundle, dev_bundle)
    shuffle_seed = seed if shuffle_seed is None else shuffle_seed
    if isinstance(shuffle_seed, bool) or not isinstance(shuffle_seed, int):
        raise ValueError("integer shuffle seed required")
    configure_cpu()
    rows = train_bundle["rows"]
    batch = make_teacher_forcing_batch([r["sequence"] for r in rows], [int(r["class"]) for r in rows])
    output = Path(out)
    output.mkdir(parents=True, exist_ok=False)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = TerminalLSTM().cpu()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999),
                                 eps=1e-8, weight_decay=0.0, foreach=False, fused=False)
    generator = torch.Generator(device="cpu").manual_seed(shuffle_seed)
    initial_digest = state_digest(model)
    bound = {"method": "terminal_LSTM", "architecture": asdict(ARConfig()),
             "parameter_count": ARConfig().parameter_count(), "seed": seed, "shuffle_seed": shuffle_seed,
             "cap_seconds": cap_seconds, "fixed_steps": fixed_steps, "batch_size": batch_size,
             "optimizer": {"name": "Adam", "lr": 0.001, "betas": [0.9, 0.999], "eps": 1e-8,
                           "weight_decay": 0.0, "gradient_clip_l2": 1.0},
             "train_objects": len(rows), "train_path": train_bundle.get("path"),
             "train_sha256": train_bundle.get("sha256"),
             "row_order_molecule_sha256": [r.get("molecule_sha256") for r in rows],
             "objective": "mean observed-token CE: AA + first EOS if L<48; no EOS49",
             "input_information": ["terminal", "BOS", "past tokens (teacher forcing on train)"],
             "checkpoint_selection": "final only", "development_accessed": dev_bundle is not None, "test_accessed": False,
             "monitor_every": monitor_every, "development_sha256": dev_bundle.get("sha256") if dev_bundle else None,
             "development_path": dev_bundle.get("path") if dev_bundle else None,
             "torch": torch.__version__, "threads": torch.get_num_threads(),
             "interop_threads": torch.get_num_interop_threads(),
             "code": file_record(__file__)}
    dump(output / "BOUND_TRAINING_INPUT.json", bound)
    order = torch.arange(len(rows))
    cursor, epoch = len(rows), -1
    torch.save(_checkpoint(model, optimizer, generator, seed=seed, updates=0, order=order,
                           cursor=cursor, epoch=epoch), output / "initial.pt")
    status, failure, completed, attempted = "RUNNING", None, 0, 0
    tokens_seen = aa_seen = eos_seen = examples_seen = 0
    batch_indices = []
    train_seconds = 0.0
    monitors, monitoring_seconds = [], 0.0
    with (output / "steps.jsonl").open("x") as stream:
        try:
            model.train()
            for step in range(1, fixed_steps + 1):
                if time.perf_counter() - started >= cap_seconds:
                    status = "CAPPED_BEFORE_NEXT_UPDATE"
                    break
                update_started = time.perf_counter()
                if cursor >= len(rows):
                    order = torch.randperm(len(rows), generator=generator)
                    cursor = 0
                    epoch += 1
                indices = order[cursor:cursor + batch_size]
                cursor += len(indices)
                batch_indices = indices.tolist()
                attempted = step
                optimizer.zero_grad(set_to_none=True)
                logits, _ = model(batch["inputs"][indices], batch["labels"][indices])
                terms = sequence_nll(logits, batch["targets"][indices], batch["mask"][indices])
                if not bool(torch.isfinite(terms["loss"])):
                    raise FloatingPointError("nonfinite training objective")
                terms["loss"].backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
                optimizer.step()
                if not all(bool(torch.isfinite(p).all()) for p in model.parameters()):
                    raise FloatingPointError("nonfinite parameters after update")
                step_seconds = time.perf_counter() - update_started
                train_seconds += step_seconds
                completed = step
                tokens, aa, eos = (int(terms[k]) for k in ("observed_tokens", "aa_tokens", "eos_tokens"))
                tokens_seen += tokens
                aa_seen += aa
                eos_seen += eos
                examples_seen += len(indices)
                row = {"step": step, "epoch_zero_based": epoch, "row_indices": batch_indices,
                       "examples": len(indices), "loss": float(terms["loss"].detach()),
                       "nll_sum": float(terms["nll_sum"].detach()), "observed_tokens": tokens,
                       "aa_tokens": aa, "eos_tokens": eos,
                       "preclip_grad_l2": float(grad_norm), "seconds": step_seconds,
                       "total_elapsed_seconds": time.perf_counter() - started}
                stream.write(json.dumps(row, allow_nan=False) + "\n")
                stream.flush()
                if (dev_bundle is not None and step % monitor_every == 0
                        and time.perf_counter() - started < cap_seconds):
                    monitoring_started = time.perf_counter()
                    entry = {"step": step, "used_for_selection": False,
                             "train": teacher_forcing_evaluation(model, train_bundle),
                             "development": teacher_forcing_evaluation(model, dev_bundle)}
                    monitoring_seconds += time.perf_counter() - monitoring_started
                    monitors.append(entry)
                    dump(output / "MONITORS.json", monitors)
            else:
                status = "COMPLETED_FIXED_STEPS"
        except Exception as exc:
            status = "FAILED_NO_RETRY"
            failure = {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc(),
                       "attempted_step": attempted, "row_indices": batch_indices}
    final_digest = state_digest(model)
    torch.save(_checkpoint(model, optimizer, generator, seed=seed, updates=completed, order=order,
                           cursor=cursor, epoch=epoch), output / "final.pt")
    total = time.perf_counter() - started
    summary = {**bound, "status": status, "failure": failure, "completed_steps": completed,
               "attempted_steps": attempted, "examples_presented": examples_seen,
               "observed_tokens_presented": tokens_seen, "aa_tokens_presented": aa_seen,
               "eos_tokens_presented": eos_seen, "train_update_seconds": train_seconds,
               "monitor_count": len(monitors), "monitoring_seconds": monitoring_seconds,
               "total_pipeline_seconds_before_manifest": total, "cap_overrun_seconds": max(0.0, total - cap_seconds),
               "updates_per_train_second": completed / train_seconds if train_seconds else None,
               "initial_tensor_state_sha256": initial_digest, "final_tensor_state_sha256": final_digest,
               "initial_checkpoint": file_record(output / "initial.pt"),
               "final_checkpoint": file_record(output / "final.pt"),
               "steps": file_record(output / "steps.jsonl"), "retries": 0}
    dump(output / "RUN.json", summary)
    return model, summary


@torch.no_grad()
def teacher_forcing_evaluation(model, bundle):
    """Descriptive AR likelihood only; never comparable to VAE objective totals."""
    if bundle.get("role") not in ("train", "development") or not bundle.get("rows"):
        raise ValueError("nonempty explicit train/development role required")
    configure_cpu()
    rows = bundle["rows"]
    was_training = model.training
    before = state_digest(model)
    model.eval()
    totals = {key: 0 for key in ("nll_sum", "aa_nll_sum", "eos_nll_sum", "observed_tokens", "aa_tokens", "eos_tokens", "correct_tokens")}
    begun = time.perf_counter()
    try:
        for start in range(0, len(rows), 32):
            part = rows[start:start + 32]
            batch = make_teacher_forcing_batch([r["sequence"] for r in part], [int(r["class"]) for r in part])
            logits, _ = model(batch["inputs"], batch["labels"])
            terms = sequence_nll(logits, batch["targets"], batch["mask"])
            for key in totals:
                totals[key] += float(terms[key]) if key.endswith("nll_sum") else int(terms[key])
    finally:
        model.train(was_training)
    if before != state_digest(model):
        raise AssertionError("evaluation changed model parameters")
    return {"method": "terminal_LSTM", "role": bundle["role"], "objects": len(rows), **totals,
            "token_count": totals["observed_tokens"], "mean_nll": totals["nll_sum"] / totals["observed_tokens"],
            "teacher_forced_token_accuracy": totals["correct_tokens"] / totals["observed_tokens"],
            "wall_seconds": time.perf_counter() - begun, "tensor_state_sha256": before,
            "test_accessed": False, "is_generation_metric": False,
            "objective_comparable_with_VAE": False}


def load_checkpoint(path):
    """Load this run's trusted local checkpoint; never a downloaded pickle."""
    record = torch.load(Path(path), map_location="cpu", weights_only=True)
    if record["architecture"] != asdict(ARConfig()):
        raise ValueError("checkpoint architecture mismatch")
    model = TerminalLSTM().cpu()
    model.load_state_dict(record["model"])
    if state_digest(model) != record["tensor_state_sha256"]:
        raise ValueError("checkpoint tensor digest mismatch")
    return model


def generate(model, seed, out, *, labels=None, smoke=False, cap_seconds=450.0):
    """Formal FREE16/AMD16; explicit smoke=True permits only FREE1/AMD1.

    seed is either an integer (FREE=seed, AMD=seed+1) or a two-key dictionary
    {'FREE': int, 'AMD': int}. Each terminal has a separate CPU generator.
    There is no sequence, length, residue-property, or reference input.
    """
    started = time.perf_counter()
    if not math.isfinite(cap_seconds) or cap_seconds <= 0:
        raise ValueError("positive finite generation cap required")
    configure_cpu()
    count = 1 if smoke else 16
    prescribed = [0] * count + [1] * count
    if labels is None:
        labels = prescribed
    labels = torch.as_tensor(labels, dtype=torch.long, device="cpu")
    if labels.tolist() != prescribed:
        raise ValueError("generation labels differ from formal16/smoke1 per terminal plan")
    seeds = {"FREE": seed, "AMD": seed + 1} if isinstance(seed, int) else dict(seed)
    if set(seeds) != {"FREE", "AMD"} or any(isinstance(s, bool) or not isinstance(s, int) for s in seeds.values()):
        raise ValueError("explicit integer FREE/AMD sampling seeds required")
    output = Path(out)
    output.mkdir(parents=True, exist_ok=False)
    before = state_digest(model)
    parts = []
    for terminal, name in enumerate(("FREE", "AMD")):
        remaining = max(0.0, cap_seconds - (time.perf_counter() - started))
        parts.append(sample_batch(model, [terminal] * count, seed=seeds[name], requests=count,
                                  cap_seconds=remaining))
    sampled = {key: parts[0][key] + parts[1][key]
               for key in ("sequences", "status", "terminated", "max_length_hit")}
    rows = []
    for index, sequence in enumerate(sampled["sequences"]):
        terminal = "FREE" if labels[index] == 0 else "AMD"
        sample_id = f"terminal_LSTM_{terminal}_{index % count:03d}"
        rows.append({"sample_id": sample_id, "attempt_id": sample_id, "method": "terminal_LSTM",
                     "terminal": terminal, "terminal_state": terminal, "class": int(labels[index]),
                     "sequence": sequence, "length": len(sequence), "status": sampled["status"][index],
                     "terminated": sampled["terminated"][index], "max_length_hit": sampled["max_length_hit"][index],
                     "seed": seeds[terminal], "temperature": 1.0, "source": "autoregressive_terminal",
                     "complete_attempt": sampled["status"][index] in ("ok", "empty_sequence")})
    if before != state_digest(model):
        raise AssertionError("generation changed model parameters")
    raw = {key: torch.cat([part[key] for part in parts], dim=0)
           for key in ("tokens", "raw_draws", "logits", "probabilities", "active_before")}
    raw["generator_states"] = torch.stack([part["generator_state"] for part in parts])
    raw.update({"labels": labels, "sample_ids": [r["sample_id"] for r in rows],
                "temperature": 1.0, "seeds": seeds, "evaluated_steps": [part["steps"] for part in parts],
                "tensor_state_sha256": before})
    torch.save(raw, output / "raw_sampling.pt")
    with (output / "samples.jsonl").open("x") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    with (output / "samples.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {"method": "terminal_LSTM", "requests": 2 * count, "terminal_counts": {"FREE": count, "AMD": count},
               "smoke": smoke, "seeds": seeds, "temperature": 1.0, "no_rejection_or_resampling": True,
               "cap_seconds": cap_seconds, "capped_by_terminal": {name: part["capped"] for name, part in zip(("FREE", "AMD"), parts)},
               "failures_by_terminal": {name: part["failure"] for name, part in zip(("FREE", "AMD"), parts)},
               "observed_sampling_tokens": sum(part["observed_sampling_tokens"] for part in parts),
               "evaluated_steps_by_terminal": {name: part["steps"] for name, part in zip(("FREE", "AMD"), parts)},
               "wall_seconds_before_manifest": time.perf_counter() - started,
               "cap_overrun_seconds_before_manifest": max(0.0, time.perf_counter() - started - cap_seconds),
               "tensor_state_sha256": before, "source_sequence_accessed": False,
               "true_length_accessed": False, "test_accessed": False,
               "raw": file_record(output / "raw_sampling.pt"), "rows": file_record(output / "samples.jsonl"),
               "csv": file_record(output / "samples.csv"),
               "rng_rule": "separate CPU generator per terminal; each step draws entire terminal batch including finished rows; stop group when all ended"}
    dump(output / "GENERATION.json", summary)
    return rows, summary
