#!/usr/bin/env python3
"""New reconstruction-only baselines and descriptive validation metrics.

Python standard library only. No model/research-code imports, activity predictor,
wetlab labels, or test-set access. See PROTOCOL.md for all denominators.
"""
import argparse
import collections
import csv
import datetime
import hashlib
import json
import math
from pathlib import Path
import random
import re
import sys
import time

AA = "ACDEFGHIKLMNPQRSTVWY"
AA_SET = frozenset(AA)
EOS = "<EOS>"
BOS = "<BOS>"
ALPHA = 1.0
MAX_LENGTH = 48
OUTPUT_FIELDS = ["method", "seed", "sample_id", "sequence", "status"]
BASELINE_METHODS = ("markov_alpha1", "memorize_train", "shuffle_train")
POLICY = {
    "version": "2026-10-11.validation-pilot.2",
    "purpose": "New reconstruction, validation pilot only; not historical replication or activity validation",
    "test_access": "unsupported; path/role guard rejects test; no test CLI exists",
    "canonical_alphabet": AA,
    "valid_length_ranges": [[1, 48], [5, 48]],
    "generation_limit_token_decisions": 48,
    "markov": {"order": 1, "alpha": 1.0, "fit": "train only, one unique sequence per input row",
        "outcomes": list(AA) + [EOS], "contexts": [BOS] + list(AA),
        "smoothing": "add one to all 21 outcomes, including EOS in BOS context",
        "budget": "closed-form fit; no gradient updates claimed"},
    "success_status": "ok",
    "sampling": "independent RNG per method/seed; memorize and shuffle share bootstrap-selection stream",
    "nearest_identity": {"method": "global Needleman-Wunsch", "match": 2, "mismatch": -1,
        "gap": -2, "tie": "score, then more matches, then fewer alignment columns",
        "denominator": "all alignment columns, including terminal/internal gaps",
        "rows": "all canonical sequences of length 1..48, regardless of status"},
    "feature_kernel": "linear kernel, equal 1/3 weight to length/48, 20 AA fractions, 400 dipeptide fractions",
    "mmd": "biased squared distance of feature means, descriptive low-order distributions only; no activity implication",
    "uncertainty": "no p-values or confidence intervals; generated sequences within one fitted seed are not independent experimental repeats",
}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def record(path):
    path = Path(path).resolve()
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False) + "\n")


def reject_test_path(path):
    # Check before reading contents, and again on the resolved symlink target.
    for text in (str(path), str(Path(path).resolve())):
        if re.search(r"(^|[/_.\-])test($|[/_.\-])", text.lower()):
            raise ValueError("test access is prohibited in this validation-pilot tool: " + text)


def validate_headers(fields):
    if not fields or any(not field or field != field.strip() for field in fields):
        raise ValueError("CSV headers must be nonempty and have no surrounding whitespace")
    if len(fields) != len(set(fields)):
        raise ValueError("duplicate CSV header")


def read_dataset(path, role, column="sequence"):
    reject_test_path(path)
    if role not in ("train", "validation"):
        raise ValueError("only train and validation roles supported")
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        validate_headers(reader.fieldnames)
        if column not in (reader.fieldnames or []):
            raise ValueError("missing sequence column: " + column)
        rows = list(reader)
    if not rows:
        raise ValueError("empty " + role + " dataset")
    sequences = []
    identifiers = []
    for line, row in enumerate(rows, 2):
        if None in row or any(value is None for value in row.values()):
            raise ValueError("ragged data row at line " + str(line))
        for field in ("split", "partition", "role"):
            if field in row and row[field].lower() not in ({"train"} if role == "train" else {"val", "validation"}):
                raise ValueError("row role disagrees with declared " + role + ": " + row[field])
        seq = row[column]
        if not (1 <= len(seq) <= MAX_LENGTH) or set(seq) - AA_SET:
            raise ValueError("invalid " + role + " sequence at line " + str(line) + "; no trimming or coercion")
        sequences.append(seq)
        identifiers.append(row.get("sequence_id", row.get("sequence_sha256", row.get("record_id", row.get("id", str(line - 2))))))
    if len(set(sequences)) != len(sequences):
        raise ValueError("dataset must already be exact-sequence deduplicated")
    return sequences, identifiers


def derive_seed(seed, stream):
    return int.from_bytes(hashlib.sha256((str(seed) + ":" + stream).encode()).digest()[:16], "big")


def fit_markov(sequences):
    counts = {context: {symbol: ALPHA for symbol in list(AA) + [EOS]} for context in [BOS] + list(AA)}
    for seq in sequences:
        previous = BOS
        for symbol in seq:
            counts[previous][symbol] += 1
            previous = symbol
        counts[previous][EOS] += 1
    return counts


def sample_markov(counts, rng):
    previous = BOS
    result = []
    symbols = list(AA) + [EOS]
    for _ in range(MAX_LENGTH):
        # Explicit inverse CDF fixes semantics independent of random.choices.
        weights = counts[previous]
        draw = rng.random() * sum(weights.values())
        cumulative = 0.0
        choice = symbols[-1]
        for symbol in symbols:
            cumulative += weights[symbol]
            if draw < cumulative:
                choice = symbol
                break
        if choice == EOS:
            return "".join(result), "ok"
        result.append(choice)
        previous = choice
    return "".join(result), "no_eos_at_limit"


def generate_baselines(sequences, seed, requests, counts=None):
    if requests <= 0:
        raise ValueError("requests must be positive")
    counts = fit_markov(sequences) if counts is None else counts
    markov_rng = random.Random(derive_seed(seed, "markov_alpha1"))
    selection_rng = random.Random(derive_seed(seed, "calibration_selection"))
    shuffle_rng = random.Random(derive_seed(seed, "calibration_shuffle"))
    result = []
    selected = [selection_rng.randrange(len(sequences)) for _ in range(requests)]
    for method in BASELINE_METHODS:
        for sample_id in range(requests):
            if method == "markov_alpha1":
                seq, status = sample_markov(counts, markov_rng)
            else:
                seq, status = sequences[selected[sample_id]], "ok"
                if method == "shuffle_train":
                    chars = list(seq)
                    shuffle_rng.shuffle(chars)
                    seq = "".join(chars)
            result.append({"method": method, "seed": str(seed), "sample_id": str(sample_id),
                           "sequence": seq, "status": status})
    return result


def write_samples(path, rows):
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def read_samples(paths, expected_requests, expected_groups=None):
    groups = collections.defaultdict(list)
    seen = set()
    for path in paths:
        reject_test_path(path)
        with Path(path).open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            validate_headers(reader.fieldnames)
            if not set(OUTPUT_FIELDS).issubset(reader.fieldnames or []):
                raise ValueError("samples missing required schema: " + str(path))
            for row in reader:
                if None in row or any(row.get(field) is None for field in OUTPUT_FIELDS):
                    raise ValueError("ragged sample row")
                if not all(row[field] for field in ("method", "seed", "sample_id", "status")):
                    raise ValueError("sample metadata must be nonempty; empty sequence is allowed")
                key = (row["method"], row["seed"], row["sample_id"])
                if key in seen:
                    raise ValueError("duplicate sample key: " + repr(key))
                seen.add(key)
                groups[key[:2]].append(row)
    if not groups:
        raise ValueError("no sample rows")
    if expected_groups is not None and set(groups) != set(expected_groups):
        raise ValueError("method/seed groups mismatch; missing=" + repr(sorted(set(expected_groups) - set(groups))) +
                         "; unexpected=" + repr(sorted(set(groups) - set(expected_groups))))
    for key, rows in groups.items():
        if len(rows) != expected_requests:
            raise ValueError("request budget mismatch for " + repr(key) + ": " + str(len(rows)))
    return dict(sorted(groups.items()))


def global_identity(a, b):
    """Exact copy of data_audit/audit_data.py reference rule (not imported)."""
    prev = [(-2 * j, 0, -j) for j in range(len(b) + 1)]
    for i, aa in enumerate(a, 1):
        curr = [(-2 * i, 0, -i)]
        for j, bb in enumerate(b, 1):
            match = int(aa == bb)
            diag = (prev[j - 1][0] + (2 if match else -1),
                    prev[j - 1][1] + match, prev[j - 1][2] - 1)
            deletion = (prev[j][0] - 2, prev[j][1], prev[j][2] - 1)
            insertion = (curr[j - 1][0] - 2, curr[j - 1][1], curr[j - 1][2] - 1)
            curr.append(max(diag, deletion, insertion))
        prev = curr
    score, matches, negative_columns = prev[-1]
    columns = -negative_columns
    return {"score": score, "matches": matches, "aligned_columns": columns,
            "identity": matches / columns if columns else 1.0,
            "edge_at_70_percent": 10 * matches >= 7 * columns}


def nearest_train(sequence, train, train_ids):
    # Every global alignment has >= max(lengths) columns and <= min(lengths)
    # matches. Pruning cannot remove a strictly better identity.
    ordered = sorted(enumerate(train), key=lambda item: (
        -min(len(sequence), len(item[1])) / max(len(sequence), len(item[1])), item[1], item[0]))
    best = None
    aligned = 0
    for index, seq in ordered:
        upper = min(len(sequence), len(seq)) / max(len(sequence), len(seq))
        if best is not None and upper <= best["identity"]:
            break
        result = global_identity(sequence, seq)
        aligned += 1
        if best is None or result["identity"] > best["identity"]:
            best = dict(result, train_id=train_ids[index], train_sequence=seq)
    return dict(best, evaluated_pairs=aligned, total_train_sequences=len(train))


def entropy(counts):
    total = sum(counts)
    return -sum((n / total) * math.log2(n / total) for n in counts if n) if total else None


def ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def summarize_numbers(values):
    if not values:
        return {"n": 0, "mean": None, "min": None, "max": None, "median": None}
    ordered = sorted(values)
    n = len(ordered)
    return {"n": n, "mean": sum(values) / n, "min": ordered[0], "max": ordered[-1],
            "median": (ordered[(n - 1) // 2] + ordered[n // 2]) / 2}


def feature_vector(seq):
    """421 fixed features, no fitted scales; each of three blocks weight 1/3."""
    counts = collections.Counter(seq)
    pairs = collections.Counter(seq[i:i + 2] for i in range(len(seq) - 1))
    scale = math.sqrt(3)
    return [len(seq) / 48 / scale] + [counts[a] / len(seq) / scale for a in AA] + [
        pairs[a + b] / max(1, len(seq) - 1) / scale for a in AA for b in AA]


def sequence_summary(sequences):
    """Only call on nonempty canonical sequences; no hidden per-row filtering."""
    aa = collections.Counter("".join(sequences))
    pairs = collections.Counter(seq[i:i + 2] for seq in sequences for i in range(len(seq) - 1))
    seq_counts = collections.Counter(sequences)
    n = len(sequences)
    if n:
        vectors = [feature_vector(seq) for seq in sequences]
        mean_vector = [sum(v[i] for v in vectors) / n for i in range(421)]
    else:
        mean_vector = None
    return {
        "n": n, "unique_count": len(seq_counts), "unique_fraction": ratio(len(seq_counts), n),
        "duplicate_fraction": ratio(n - len(seq_counts), n),
        "top_mode_fraction": ratio(max(seq_counts.values(), default=0), n),
        "pair_collision_probability": ratio(sum(c * (c - 1) for c in seq_counts.values()), n * (n - 1)),
        "sequence_entropy_bits": entropy(seq_counts.values()), "pooled_aa_entropy_bits": entropy(aa.values()),
        "pooled_dipeptide_entropy_bits": entropy(pairs.values()),
        "mean_per_sequence_aa_entropy_bits": ratio(sum(entropy(collections.Counter(seq).values()) for seq in sequences), n),
        "length": summarize_numbers([len(seq) for seq in sequences]),
        "length_counts": dict(sorted(collections.Counter(map(len, sequences)).items())),
        "aa_counts": {a: aa[a] for a in AA},
        "aa_frequency": {a: ratio(aa[a], sum(aa.values())) for a in AA},
        "dipeptide_counts": {a + b: pairs[a + b] for a in AA for b in AA},
        "dipeptide_frequency": {a + b: ratio(pairs[a + b], sum(pairs.values())) for a in AA for b in AA},
        "fixed_feature_mean": mean_vector,
    }


def js_divergence(p, q):
    if any(value is None for value in list(p.values()) + list(q.values())):
        return None
    result = 0.0
    for key in p:
        mean = (p[key] + q[key]) / 2
        if p[key]:
            result += p[key] * math.log2(p[key] / mean) / 2
        if q[key]:
            result += q[key] * math.log2(q[key] / mean) / 2
    return result


def length_ks(p, q):
    if not p or not q:
        return None
    p_count, q_count = collections.Counter(p), collections.Counter(q)
    p_sum = q_sum = result = 0.0
    for value in sorted(set(p) | set(q)):
        p_sum += p_count[value] / len(p)
        q_sum += q_count[value] / len(q)
        result = max(result, abs(p_sum - q_sum))
    return result


def compare_summaries(sample, reference):
    result = {}
    for feature in ("aa", "dipeptide"):
        p, q = sample[feature + "_frequency"], reference[feature + "_frequency"]
        result[feature + "_jensen_shannon_bits"] = js_divergence(p, q)
        result[feature + "_l1_distance"] = sum(abs(p[k] - q[k]) for k in p) if all(
            value is not None for value in list(p.values()) + list(q.values())) else None
    p, q = sample["fixed_feature_mean"], reference["fixed_feature_mean"]
    result["mmd2_linear_biased_fixed_features"] = sum((a - b) ** 2 for a, b in zip(p, q)) if p is not None and q is not None else None
    p_lengths = [int(k) for k, count in sample["length_counts"].items() for _ in range(count)]
    q_lengths = [int(k) for k, count in reference["length_counts"].items() for _ in range(count)]
    result["length_ks_statistic_no_pvalue"] = length_ks(p_lengths, q_lengths)
    return result


def evaluate_group(rows, train, train_ids, validation, cache=None):
    cache = {} if cache is None else cache
    train_set = set(train)
    details = []
    for row in rows:
        seq = row["sequence"]
        alphabet = not (set(seq) - AA_SET)  # Empty alphabet is true; nonempty separately required.
        valid1 = alphabet and 1 <= len(seq) <= 48
        valid5 = alphabet and 5 <= len(seq) <= 48
        if valid1 and seq not in cache:
            cache[seq] = nearest_train(seq, train, train_ids)
        detail = {key: row[key] for key in OUTPUT_FIELDS}
        detail.update(length=len(seq), empty=seq == "", invalid_characters="".join(sorted(set(seq) - AA_SET)),
            alphabet_only=alphabet, valid_1_48=valid1, valid_5_48=valid5,
            status_ok=row["status"] == "ok", status_ok_and_valid_1_48=row["status"] == "ok" and valid1,
            exact_copy_train=seq in train_set, nearest_train=cache[seq] if valid1 else None,
            nearest_unavailable_reason=None if valid1 else "outside canonical 1..48 sequence cohort")
        details.append(detail)
    valid = [row["sequence"] for row in details if row["valid_1_48"]]
    valid5 = [row["sequence"] for row in details if row["valid_5_48"]]
    all_counts = collections.Counter(row["sequence"] for row in rows)
    n = len(rows)
    summary = {"requests": n, "status_counts": dict(sorted(collections.Counter(row["status"] for row in rows).items())),
        "empty_count": sum(row["empty"] for row in details),
        "invalid_alphabet_count": sum(not row["alphabet_only"] for row in details),
        "length_over_48_count": sum(row["length"] > 48 for row in details),
        "all_output_length": summarize_numbers([row["length"] for row in details]),
        "all_output_unique_count_including_empty_and_failed": len(all_counts),
        "all_output_unique_fraction_including_empty_and_failed": ratio(len(all_counts), n),
        "valid_1_48_count": len(valid), "valid_1_48_fraction_of_requests": ratio(len(valid), n),
        "valid_5_48_count": len(valid5), "valid_5_48_fraction_of_requests": ratio(len(valid5), n),
        "status_ok_fraction_of_requests": ratio(sum(row["status_ok"] for row in details), n),
        "status_ok_and_valid_1_48_fraction_of_requests": ratio(sum(row["status_ok_and_valid_1_48"] for row in details), n),
        "exact_copy_train_count_all_requests": sum(row["exact_copy_train"] for row in details),
        "exact_copy_train_fraction_of_requests": ratio(sum(row["exact_copy_train"] for row in details), n),
        "exact_copy_train_fraction_of_valid_1_48": ratio(sum(row["exact_copy_train"] for row in details), len(valid)),
        "nearest_train_identity_valid_1_48": summarize_numbers([row["nearest_train"]["identity"] for row in details if row["valid_1_48"]]),
        "nearest_train_at_least_70_percent_count_valid_1_48": sum(row["nearest_train"]["edge_at_70_percent"] for row in details if row["valid_1_48"]),
        "nearest_train_at_least_70_percent_fraction_of_valid_1_48": ratio(sum(row["nearest_train"]["edge_at_70_percent"] for row in details if row["valid_1_48"]), len(valid)),
        "valid_1_48": sequence_summary(valid), "valid_5_48": sequence_summary(valid5),
        "denominator_warning": "Conditional distribution/diversity metrics use named valid cohort; rates above retain every requested output, including empty and failed. Status is never silently filtered.",
    }
    for name, reference in (("train", train), ("validation", validation)):
        summary["valid_1_48_vs_" + name] = compare_summaries(summary["valid_1_48"], sequence_summary(reference))
    return summary, details


def command_baselines(args):
    start = time.perf_counter()
    train, _ = read_dataset(args.train, "train", args.sequence_column)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=False)
    fit_start = time.perf_counter()
    counts = fit_markov(train)
    fit_seconds = time.perf_counter() - fit_start
    seeds = [int(seed) for seed in args.seeds.split(",")]
    if len(set(seeds)) != len(seeds):
        raise ValueError("duplicate seed")
    rows = []
    generation_times = {}
    for seed in seeds:
        generation_start = time.perf_counter()
        rows.extend(generate_baselines(train, seed, args.requests, counts))
        generation_times[str(seed)] = time.perf_counter() - generation_start
    samples_path = out / "samples.csv"
    write_samples(samples_path, rows)
    write_json(out / "markov_counts.json", counts)
    write_json(out / "run.json", {"created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "command": sys.argv, "policy": POLICY, "source": record(__file__), "train_input": record(args.train),
        "train_sequences": len(train), "seeds": seeds, "requests_per_method_seed": args.requests,
        "methods": BASELINE_METHODS, "rows": len(rows), "samples": record(samples_path),
        "markov_model": record(out / "markov_counts.json"), "markov_fit_seconds": fit_seconds,
        "all_three_baseline_generation_seconds_by_seed": generation_times,
        "wall_seconds": time.perf_counter() - start, "python": sys.version,
        "compute_note": "Markov fit is closed form; not equal training compute to 200 gradient updates. Generation requests are equal per method/seed."})
    print(json.dumps({"output_dir": str(out), "rows": len(rows), "markov_fit_seconds": fit_seconds}))


def command_evaluate(args):
    start = time.perf_counter()
    train, train_ids = read_dataset(args.train, "train", args.sequence_column)
    validation, _ = read_dataset(args.validation, "validation", args.sequence_column)
    if set(train) & set(validation):
        raise ValueError("train/validation exact-sequence overlap")
    methods = args.expected_methods.split(",")
    seeds = [str(int(seed)) for seed in args.expected_seeds.split(",")]
    if not all(methods) or len(set(methods)) != len(methods) or len(set(seeds)) != len(seeds):
        raise ValueError("expected methods/seeds must be nonempty and unique")
    expected_groups = [(method, seed) for method in methods for seed in seeds]
    groups = read_samples(args.samples, args.expected_requests, expected_groups)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=False)
    cache = {}
    summaries, details = [], []
    for (method, seed), rows in groups.items():
        summary, item_details = evaluate_group(rows, train, train_ids, validation, cache)
        summaries.append(dict(method=method, seed=seed, **summary))
        details.extend(item_details)
    write_json(out / "metrics.json", {"scope": "validation pilot; no test read", "policy": POLICY, "groups": summaries,
        "train_reference": sequence_summary(train), "validation_reference": sequence_summary(validation)})
    write_json(out / "sample_audit.json", details)
    flat_fields = OUTPUT_FIELDS + ["length", "empty", "invalid_characters", "valid_1_48", "valid_5_48",
        "status_ok", "exact_copy_train", "nearest_train_id", "nearest_train_identity", "nearest_train_matches", "nearest_train_columns"]
    with (out / "sample_audit.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=flat_fields)
        writer.writeheader()
        for row in details:
            flat = {field: row[field] for field in flat_fields if field in row}
            nearest = row["nearest_train"]
            flat.update(nearest_train_id=nearest["train_id"] if nearest else "",
                nearest_train_identity=nearest["identity"] if nearest else "", nearest_train_matches=nearest["matches"] if nearest else "",
                nearest_train_columns=nearest["aligned_columns"] if nearest else "")
            writer.writerow(flat)
    write_json(out / "run.json", {"created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "command": sys.argv, "source": record(__file__), "train_input": record(args.train),
        "validation_input": record(args.validation), "sample_inputs": [record(path) for path in args.samples],
        "outputs": [record(out / name) for name in ("metrics.json", "sample_audit.json", "sample_audit.csv")],
        "expected_requests_per_group": args.expected_requests, "groups": len(groups), "rows_retained": len(details),
        "expected_groups": expected_groups,
        "unique_valid_sequences_aligned": len(cache), "wall_seconds": time.perf_counter() - start,
        "python": sys.version, "test_contents_read": False,
        "guard_limit": "Path and row-role checks prevent ordinary accidental test use; they are not a security boundary against deliberately relabelled data. Input hashes record exactly what was read."})
    print(json.dumps({"output_dir": str(out), "groups": len(groups), "rows_retained": len(details), "wall_seconds": time.perf_counter() - start}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    baseline = subparsers.add_parser("baselines")
    baseline.add_argument("--train", required=True)
    baseline.add_argument("--sequence-column", default="sequence")
    baseline.add_argument("--seeds", required=True)
    baseline.add_argument("--requests", required=True, type=int)
    baseline.add_argument("--output-dir", required=True)
    baseline.set_defaults(func=command_baselines)
    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--train", required=True)
    evaluate.add_argument("--validation", required=True)
    evaluate.add_argument("--sequence-column", default="sequence")
    evaluate.add_argument("--samples", required=True, nargs="+")
    evaluate.add_argument("--expected-requests", required=True, type=int)
    evaluate.add_argument("--expected-methods", required=True, help="comma-separated required methods; entire missing groups are rejected")
    evaluate.add_argument("--expected-seeds", required=True, help="comma-separated integer seeds; every method/seed combination is required")
    evaluate.add_argument("--output-dir", required=True)
    evaluate.set_defaults(func=command_evaluate)
    args = parser.parse_args()
    if getattr(args, "requests", getattr(args, "expected_requests", 0)) <= 0:
        parser.error("request budget must be positive")
    args.func(args)


if __name__ == "__main__":
    main()
