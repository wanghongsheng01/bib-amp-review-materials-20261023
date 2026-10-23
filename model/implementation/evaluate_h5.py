#!/usr/bin/env python3
"""AMP H5 train/development evaluation, standard library only; no model calls.

The distribution and Needleman–Wunsch formulas are reused byte-for-byte from
the independently reviewed H4 dependency. Empty/failed requests remain visible.
No test role exists. CLI inputs bind exact train/development SHA256 and counts.
"""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import csv
import hashlib
import importlib.util
import io
import json
import math
from pathlib import Path
import sys
import time

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
METRICS = HERE / "frozen_metrics_v1.py"
METRICS_SHA256 = "f65eec16391d98a382d415e7b879f99555a33b33398f8be704661d1c7c027480"
if hashlib.sha256(METRICS.read_bytes()).hexdigest() != METRICS_SHA256:
    raise ValueError("frozen H4 metric dependency changed")
spec = importlib.util.spec_from_file_location("h5_frozen_metrics", METRICS)
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)
METHODS = ("CE21_DDPM", "MSE21_DDPM", "scalar_DDPM", "terminal_LSTM")
TERMINALS = ("FREE", "AMD")
COMPLETED_GENERATION_STATUSES = frozenset(("ok", "no_eos_at_limit", "empty_sequence"))
TRAIN_SHA = "a232401fb46bb9419cf8d690f21476447b853862fd23622e81247b88add0b660"
DEV_SHA = "65fdc7d402240235698d30aa769de3ea45c577cbcb28931970571ee808098ec1"
POLICY = {
    "scope": "exploratory single-fit train/development pilot; no test access",
    "mmd": "biased linear MMD squared on 421 frozen features; valid canonical 1..48 only",
    "feature_blocks": "length/48,20 AA fractions,400 dipeptide fractions; each scaled by 1/sqrt(3)",
    "reconstruction": "posterior mean only; rendered sequence aligned to target positions; no nearest-neighbor truth",
    "macro": "mean recall over amino-acid classes supported by target; PAD excluded",
    "missing_prediction": "counts incorrect at every unrendered target position; never removes truth denominator",
    "ar_nll": "teacher-forced diagnostic reported separately; not comparable with VAE objective",
    "uncertainty": "no p-values/CI; draws are not independent training replicates; development has been reused",
    "claims": "no activity, wetlab validation, significance, novelty, general method superiority or submission-readiness claim",
    "engineering_prerequisite": "screens require complete request execution here and independently verified fixed-step training/budget upstream",
}


def dump(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def guarded_read(path):
    base.reject_test_path(path)
    return Path(path).read_bytes()


def read_rows(path):
    payload = guarded_read(path).decode("utf-8-sig")
    if Path(path).suffix == ".jsonl":
        rows = [json.loads(line) for line in payload.splitlines() if line.strip()]
        if any(not isinstance(r, dict) for r in rows):
            raise ValueError("JSONL rows must be objects")
    else:
        reader = csv.DictReader(io.StringIO(payload, newline=""))
        base.validate_headers(reader.fieldnames)
        rows = list(reader)
    if any(None in r or any(v is None for v in r.values()) for r in rows):
        raise ValueError("ragged or null row; empty sequence must be explicit string")
    return rows


def dataset(path, role, expected_sha, expected_rows):
    if role not in ("train", "development"):
        raise ValueError("only train/development roles")
    if hashlib.sha256(guarded_read(path)).hexdigest() != expected_sha:
        raise ValueError("prelocked split hash mismatch")
    rows = read_rows(path)
    if len(rows) != expected_rows:
        raise ValueError("split row count mismatch")
    seen = set()
    for row in rows:
        s, terminal = row["sequence"], row["terminal_state"]
        if not 1 <= len(s) <= 48 or not set(s) <= base.AA_SET or terminal not in TERMINALS:
            raise ValueError("invalid canonical reference or terminal")
        if str(row["class"]) != str(TERMINALS.index(terminal)):
            raise ValueError("terminal class mismatch")
        for field in ("split", "partition", "role"):
            allowed = {"train"} if role == "train" else {"val", "validation", "development", "dev"}
            if field in row and row[field].lower() not in allowed:
                raise ValueError("row role conflicts with declared reference")
        sh = hashlib.sha256(s.encode("ascii")).hexdigest()
        mh = hashlib.sha256(("AMP_MOLECULE_V1\n" + s + "\n" + terminal).encode("ascii")).hexdigest()
        if (row["sequence_sha256"] != sh or row["molecule_sha256"] != mh or
                int(row["length"]) != len(s) or not row["component_id"] or mh in seen):
            raise ValueError("reference identity/length/duplicate mismatch")
        seen.add(mh)
    if {r["terminal_state"] for r in rows} != set(TERMINALS):
        raise ValueError("reference must represent both terminal states")
    return rows


def split_isolation(train, development):
    for field in ("sequence", "molecule_sha256", "component_id"):
        if {r[field] for r in train} & {r[field] for r in development}:
            raise ValueError("train/development overlap: " + field)


def sample_groups(paths, methods, requests, expected_seeds=None):
    expected = {(m, t) for m in methods for t in TERMINALS}
    groups, seen = defaultdict(list), set()
    for path in paths:
        for original in read_rows(path):
            row = dict(original)
            # Both collaborators emit the shared keys; aliases allow JSONL raw ledgers.
            row.setdefault("sample_id", row.get("attempt_id"))
            row.setdefault("terminal", row.get("terminal_state"))
            if any(not isinstance(row.get(k), (str, int)) or str(row[k]) == "" for k in
                   ("sample_id", "method", "terminal", "status", "seed")):
                raise ValueError("missing sample identity/status/seed")
            if not isinstance(row.get("sequence"), str):
                raise ValueError("sequence must be string, including empty output")
            key = row["method"], row["terminal"]
            if key not in expected:
                raise ValueError("unexpected method/terminal")
            if "terminal_state" in row and row["terminal_state"] != row["terminal"]:
                raise ValueError("terminal aliases disagree")
            if "class" in row and str(row["class"]) != str(TERMINALS.index(row["terminal"])):
                raise ValueError("sample class disagrees")
            if "length" in row and int(row["length"]) != len(row["sequence"]):
                raise ValueError("sample declared length disagrees")
            identity = key + (str(row["sample_id"]),)
            if identity in seen:
                raise ValueError("duplicate sample identity")
            seen.add(identity)
            row["sample_id"], row["seed"] = str(row["sample_id"]), str(row["seed"])
            if expected_seeds is not None and row["seed"] != str(expected_seeds[key[0]][key[1]]):
                raise ValueError("sample seed differs from prelocked schedule")
            groups[key].append(row)
    if set(groups) != expected or any(len(r) != requests for r in groups.values()):
        raise ValueError("missing group or request rows; failed requests may not be filtered")
    for (method, terminal), rows in groups.items():
        expected_ids = {f"{method}_{terminal}_{index:03d}" for index in range(requests)}
        if {row["sample_id"] for row in rows} != expected_ids:
            raise ValueError("sample IDs differ from exact preallocated request slots")
        for row in rows:
            if "terminal_sample_id" in row and row["sample_id"] != f"{method}_{terminal}_{int(row['terminal_sample_id']):03d}":
                raise ValueError("terminal sample index and identity disagree")
    return dict(sorted(groups.items()))


def by_terminal(rows, terminal):
    selected = [r for r in rows if r["terminal_state"] == terminal]
    return [r["sequence"] for r in selected], [r["molecule_sha256"] for r in selected]


def evaluate_generation(groups, train, development):
    all_train = sorted({r["sequence"] for r in train})
    all_ids = [hashlib.sha256(s.encode("ascii")).hexdigest() for s in all_train]
    conditional_cache = {t: {} for t in TERMINALS}
    global_cache, reports, details = {}, [], []
    references = {}
    for terminal in TERMINALS:
        tr, _ = by_terminal(train, terminal)
        dev, _ = by_terminal(development, terminal)
        references[terminal] = {"train": base.sequence_summary(tr), "development": base.sequence_summary(dev)}
    for (method, terminal), rows in sorted(groups.items()):
        tr, ids = by_terminal(train, terminal)
        dev, _ = by_terminal(development, terminal)
        summary, detail = base.evaluate_group(rows, tr, ids, dev, conditional_cache[terminal])
        summary["valid_1_48_vs_development"] = summary.pop("valid_1_48_vs_validation")
        for row, original in zip(detail, rows):
            seq = row["sequence"]
            row["terminal"] = terminal
            row["raw_sample_metadata"] = original
            row["exact_sequence_copy_any_train_terminal"] = seq in all_train
            row["exact_sequence_plus_requested_terminal_copy"] = row.pop("exact_copy_train")
            if row["valid_1_48"] and seq not in global_cache:
                global_cache[seq] = base.nearest_train(seq, all_train, all_ids)
            row["nearest_any_train_terminal"] = global_cache.get(seq) if row["valid_1_48"] else None
        copies = sum(r["exact_sequence_copy_any_train_terminal"] for r in detail)
        summary.update({
            "method": method, "terminal": terminal,
            "execution_completed_count": sum(r["status"] in COMPLETED_GENERATION_STATUSES for r in rows),
            "execution_completed_fraction_of_requests": base.ratio(sum(r["status"] in COMPLETED_GENERATION_STATUSES for r in rows), len(rows)),
            "seeds": sorted({r["seed"] for r in rows}),
            "train_same_terminal_n": len(tr), "development_same_terminal_n": len(dev),
            "valid_unique_count_per_request": base.ratio(summary["valid_1_48"]["unique_count"], len(rows)),
            "exact_sequence_copy_any_train_terminal_count": copies,
            "exact_sequence_copy_any_train_terminal_fraction_of_requests": base.ratio(copies, len(rows)),
            "nearest_any_train_terminal_identity_valid_1_48": base.summarize_numbers([
                r["nearest_any_train_terminal"]["identity"] for r in detail if r["valid_1_48"]]),
            "nearest_any_train_terminal_at_least_70_percent_count_valid_1_48": sum(
                r["nearest_any_train_terminal"]["edge_at_70_percent"] for r in detail if r["valid_1_48"]),
            "nearest_and_copy_scope": "unqualified frozen keys mean same requested terminal; any_train_terminal explicitly pools distinct train sequences",
        })
        for feature in ("aa", "dipeptide"):
            field = "pooled_" + feature + "_entropy_bits"
            observed = summary["valid_1_48"][field]
            reference = references[terminal]["train"][field]
            summary[field + "_absolute_gap_to_train"] = abs(observed - reference) if observed is not None and reference is not None else None
        reports.append(summary)
        details.extend(detail)
    return {"policy": POLICY, "groups": reports, "references_by_terminal": references}, details


def reconstruction_metrics(rows, truth):
    """Rendered-position accuracy, with target class macro as in frozen H2."""
    if len(rows) != len(truth):
        raise ValueError("reconstruction count mismatch")
    ordered = {}
    for row in rows:
        key = int(row["row_id"])
        if key in ordered or not 0 <= key < len(truth):
            raise ValueError("duplicate/outside reconstruction row_id")
        expected = truth[key]
        if row["truth_sequence"] != expected["sequence"] or row["terminal"] != expected["terminal_state"]:
            raise ValueError("reconstruction ground truth row binding mismatch")
        pred = row["prediction_sequence"]
        if not isinstance(pred, str) or len(pred) > 48 or not set(pred) <= base.AA_SET:
            raise ValueError("invalid reconstruction prediction")
        if not row.get("status"):
            raise ValueError("reconstruction status absent")
        ordered[key] = row
    if set(ordered) != set(range(len(truth))):
        raise ValueError("reconstruction missing rows")
    support, correct = Counter(), Counter()
    length_errors, per_sequence, details = [], [], []
    exact, length_exact, empty = 0, 0, 0
    for index, expected in enumerate(truth):
        row, target = ordered[index], expected["sequence"]
        pred = row["prediction_sequence"]
        row_correct = 0
        for position, aa in enumerate(target):
            support[aa] += 1
            match = position < len(pred) and pred[position] == aa
            correct[aa] += int(match)
            row_correct += int(match)
        error = abs(len(pred) - len(target))
        length_errors.append(error)
        per_sequence.append(row_correct / len(target))
        exact += int(pred == target)
        length_exact += int(len(pred) == len(target))
        empty += int(not pred)
        details.append({**row, "correct_target_positions": row_correct, "target_residues": len(target),
                        "rendered_nonpad_accuracy": per_sequence[-1], "length_absolute_error": error,
                        "whole_rendered_exact": pred == target})
    total = sum(support.values())
    supported = [aa for aa in base.AA if support[aa]]
    n = len(rows)
    return {
        "rows": n, "mode": "posterior_mean", "target_residues": total,
        "rendered_aligned_correct": sum(correct.values()),
        "rendered_nonpad_micro": base.ratio(sum(correct.values()), total),
        "rendered_nonpad_macro": base.ratio(sum(correct[a] / support[a] for a in supported), len(supported)),
        "macro_definition": "unweighted mean recall among target-supported amino acid classes, excluding PAD",
        "per_sequence_mean_nonpad_accuracy": base.ratio(sum(per_sequence), n),
        "target_support_by_aa": {a: support[a] for a in base.AA},
        "rendered_correct_by_aa": {a: correct[a] for a in base.AA},
        "supported_aa_classes": supported,
        "length_absolute_error_sum": sum(length_errors), "length_mae": base.ratio(sum(length_errors), n),
        "length_exact_count": length_exact, "whole_rendered_exact_count": exact,
        "whole_rendered_exact_fraction": base.ratio(exact, n), "empty_count": empty,
        "status_counts": dict(sorted(Counter(r["status"] for r in rows).items())),
    }, details


def comparator(left, right, operator):
    if left is None or right is None:
        return {"left": left, "right": right, "operator": operator, "pass": False, "reason": "undefined metric"}
    passed = {"lt": left < right, "le": left <= right, "gt": left > right, "ge": left >= right, "eq": left == right}[operator]
    return {"left": left, "right": right, "operator": operator, "pass": bool(passed)}


def exploratory_gates(generation, reconstruction, ce="CE21_DDPM", mse="MSE21_DDPM", ar="terminal_LSTM"):
    groups = {(g["method"], g["terminal"]): g for g in generation["groups"]}
    rec_checks = {}
    rec = {(g["method"], g["split"]): g for g in reconstruction}
    if (ce, "development") in rec and (mse, "development") in rec:
        a, b = rec[ce, "development"], rec[mse, "development"]
        for key, op in (("rendered_nonpad_micro", "gt"), ("rendered_nonpad_macro", "gt"),
                        ("length_mae", "le"), ("whole_rendered_exact_count", "ge")):
            rec_checks[key] = comparator(a[key], b[key], op)
    generation_checks = {}
    for terminal in TERMINALS:
        if (ce, terminal) not in groups or (ar, terminal) not in groups:
            generation_checks[terminal] = {"missing": {"pass": False}}
            continue
        a, b = groups[ce, terminal], groups[ar, terminal]
        checks = {
            "development_mmd2_strictly_lower": comparator(a["valid_1_48_vs_development"]["mmd2_linear_biased_fixed_features"], b["valid_1_48_vs_development"]["mmd2_linear_biased_fixed_features"], "lt"),
            "ce_all_requests_valid1": comparator(a["valid_1_48_count"], a["requests"], "eq"),
            "ar_all_requests_valid1": comparator(b["valid_1_48_count"], b["requests"], "eq"),
            "valid5_count_not_lower": comparator(a["valid_5_48_count"], b["valid_5_48_count"], "ge"),
            "valid_unique_count_not_lower": comparator(a["valid_1_48"]["unique_count"], b["valid_1_48"]["unique_count"], "ge"),
            "same_terminal_train_copies_not_higher": comparator(a["exact_copy_train_count_all_requests"], b["exact_copy_train_count_all_requests"], "le"),
            "any_terminal_train_copies_not_higher": comparator(a["exact_sequence_copy_any_train_terminal_count"], b["exact_sequence_copy_any_train_terminal_count"], "le"),
        }
        for feature in ("aa", "dipeptide"):
            key = "pooled_" + feature + "_entropy_bits_absolute_gap_to_train"
            checks[feature + "_entropy_train_gap_not_larger"] = comparator(a[key], b[key], "le")
        generation_checks[terminal] = checks
    rec_complete = all(set(g.get("status_counts", {"ok": g["rows"]})) <= {"ok"} for g in reconstruction)
    generation_complete = all(g["execution_completed_count"] == g["requests"] for g in generation["groups"])
    rec_pass = bool(rec_checks) and rec_complete and all(v["pass"] for v in rec_checks.values())
    gen_pass = generation_complete and all(v["pass"] for t in generation_checks.values() for v in t.values())
    return {"scope": "predeclared exploratory development screen; never a significance/activity/novelty/publication gate",
            "engineering_prerequisite": "upstream fixed-step training, common budget and final checkpoint identity must independently pass; this file only checks output execution",
            "all_generation_requests_completed": generation_complete,
            "reconstruction_ce_vs_matched_mse": {"checks": rec_checks, "pass": rec_pass,
                "status": ("EVALUATED" if rec_complete else "NOT_EVALUABLE_FAILED_RECONSTRUCTION") if rec_checks else "NOT_EVALUATED_MISSING_RECONSTRUCTION"},
            "generation_ce_vs_terminal_ar": {"checks": generation_checks, "pass": gen_pass,
                "status": "EVALUATED" if generation_complete else "NOT_EVALUABLE_FAILED_GENERATION"},
            "both_exploratory_screens_pass": rec_pass and gen_pass}


def parse_bindings(items):
    output = {}
    for text in items:
        method, separator, path = text.partition("=")
        if not separator or not method or not path or method in output:
            raise ValueError("binding must be unique METHOD=PATH")
        output[method] = path
    return output


def ar_nll_diagnostics(paths):
    output = []
    for path in paths:
        value = json.loads(guarded_read(path))
        if isinstance(value, dict) and "train" in value and "development" in value:
            entries = [value["train"], value["development"]]
        else:
            entries = value if isinstance(value, list) else [value]
        for entry in entries:
            if entry.get("role") not in ("train", "development", "dev", "val", "validation"):
                raise ValueError("AR NLL role must be train/development")
            nll, count = entry["nll_sum"], entry["token_count"]
            if not isinstance(nll, (int, float)) or not math.isfinite(nll) or nll < 0 or not isinstance(count, int) or count <= 0:
                raise ValueError("invalid AR NLL diagnostic")
            output.append({**entry, "nll_per_target_token_nats": nll / count, "source": base.record(path),
                           "comparison": "teacher forced AR only; do not compare to VAE objectives or reconstruction accuracy"})
    return output


def run(args):
    started = time.perf_counter()
    train = dataset(args.train, "train", args.train_sha256, args.train_rows)
    development = dataset(args.development, "development", args.development_sha256, args.development_rows)
    split_isolation(train, development)
    methods = args.methods.split(",")
    if not all(methods) or len(set(methods)) != len(methods):
        raise ValueError("methods must be nonempty and unique")
    seeds = json.loads(guarded_read(args.sampling_seeds_json)) if args.sampling_seeds_json else None
    groups = sample_groups(args.samples, methods, args.requests, seeds)
    generation, details = evaluate_generation(groups, train, development)
    recon, recon_details, recon_inputs = [], [], []
    for split, bindings, truth in (("train", args.reconstruction_train, train),
                                   ("development", args.reconstruction_development, development)):
        for method, path in parse_bindings(bindings).items():
            if method not in methods or method == "terminal_LSTM":
                raise ValueError("reconstruction must name a VAE method in generation")
            metrics, rows = reconstruction_metrics(read_rows(path), truth)
            recon.append({"method": method, "split": split, **metrics})
            recon_details.extend({"method": method, "split": split, **r} for r in rows)
            recon_inputs.append(base.record(path))
    ar_nll = ar_nll_diagnostics(args.ar_nll)
    gates = exploratory_gates(generation, recon)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    dump(out / "generation_metrics.json", generation)
    dump(out / "generation_sample_audit.json", details)
    dump(out / "reconstruction_metrics.json", {"policy": POLICY, "groups": recon})
    dump(out / "reconstruction_row_audit.json", recon_details)
    dump(out / "ar_teacher_forced_nll.json", ar_nll)
    dump(out / "exploratory_gates.json", gates)
    dump(out / "run.json", {"status": "COMPLETE", "created_utc": datetime.now(timezone.utc).isoformat(),
         "source": base.record(__file__), "frozen_metrics_dependency": base.record(METRICS),
         "train": base.record(args.train), "development": base.record(args.development),
         "sample_inputs": [base.record(p) for p in args.samples], "reconstruction_inputs": recon_inputs,
         "ar_nll_inputs": [base.record(p) for p in args.ar_nll],
         "sampling_seed_schedule": base.record(args.sampling_seeds_json) if args.sampling_seeds_json else None,
         "seed_check": "prelocked_schedule_checked" if seeds is not None else "metadata_reported_only_not_prelocked",
         "requests_per_method_terminal": args.requests, "methods": methods, "rows_retained": len(details),
         "model_calls": 0, "test_accessed": False, "wall_seconds": time.perf_counter() - started,
         "argv": sys.argv, "python": sys.version, "policy": POLICY, "backup": "BACKUP_BLOCKED"})
    dump(out / "MANIFEST.json", {"files": [base.record(p) for p in sorted(out.iterdir()) if p.is_file()]})
    print(json.dumps({"output": str(out), "rows_retained": len(details), "groups": len(groups), "model_calls": 0, "test_accessed": False}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True)
    parser.add_argument("--development", required=True)
    parser.add_argument("--train-sha256", default=TRAIN_SHA)
    parser.add_argument("--development-sha256", default=DEV_SHA)
    parser.add_argument("--train-rows", type=int, default=279)
    parser.add_argument("--development-rows", type=int, default=41)
    parser.add_argument("--samples", nargs="+", required=True)
    parser.add_argument("--methods", default=",".join(METHODS))
    parser.add_argument("--requests", type=int, default=16)
    parser.add_argument("--sampling-seeds-json")
    parser.add_argument("--reconstruction-train", action="append", default=[], metavar="METHOD=PATH")
    parser.add_argument("--reconstruction-development", action="append", default=[], metavar="METHOD=PATH")
    parser.add_argument("--ar-nll", action="append", default=[])
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.requests <= 0 or args.train_rows <= 0 or args.development_rows <= 0:
        parser.error("request/reference counts must be positive")
    run(args)


if __name__ == "__main__":
    main()
