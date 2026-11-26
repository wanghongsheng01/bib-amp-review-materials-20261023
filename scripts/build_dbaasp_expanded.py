#!/usr/bin/env python3
"""Build a frozen, homology-partitioned expanded DBAASP benchmark cohort.

Fetches monomer antibacterial peptides (Gram+ and Gram-) from the official
DBAASP REST API (https://dbaasp.org/peptides), applies the exact eligibility
criteria of the manuscript (canonical 20 L-amino acids, length 1-48, unmodified
N terminus, FREE or AMD C terminus, valid RDKit molecular graph), clusters
sequences by >=70% global Needleman-Wunsch identity (transitive connected
components, identical to Methods), and partitions components into strictly
disjoint Train (80%), Validation (10%), and Independent Test (10%) splits.
"""
from __future__ import annotations

import collections
import csv
import datetime
import hashlib
import json
import math
import os
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path("/Users/loctek/.gemini/antigravity/scratch/bib-26-2683-revision")
sys.path.insert(0, str(ROOT / "github_repo/data/train_dev/source"))
import molecular_representation as mr

OUT_DIR = ROOT / "revision_data" / "dbaasp_expanded"
OUT_DIR.mkdir(parents=True, exist_ok=True)

AA_SET = frozenset("ARNDCEQGHILKMFPSTWYV")
AA_LIST = "ARNDCEQGHILKMFPSTWYV"


def fetch_dbaasp_pages(max_records_per_group: int = 4000, page_size: int = 500):
    raw_by_id = {}
    queries = [
        ("Gram+", "https://dbaasp.org/peptides?complexity.value=monomer&targetGroup.value=Gram%2B"),
        ("Gram-", "https://dbaasp.org/peptides?complexity.value=monomer&targetGroup.value=Gram-"),
    ]
    meta = {"fetched_utc": datetime.datetime.utcnow().isoformat() + "Z", "queries": []}
    for label, base_url in queries:
        fetched = 0
        total_avail = None
        for offset in range(0, max_records_per_group, page_size):
            url = f"{base_url}&limit={page_size}&offset={offset}"
            req = urllib.request.Request(url, headers={"Accept": "application/json", "User-Agent": "Mozilla/5.0"})
            for attempt in range(3):
                try:
                    with urllib.request.urlopen(req, timeout=30) as resp:
                        payload = json.loads(resp.read().decode("utf-8"))
                    break
                except Exception as exc:
                    if attempt == 2:
                        raise
                    time.sleep(1.5)
            total_avail = payload.get("totalCount")
            batch = payload.get("data", [])
            if not batch:
                break
            for item in batch:
                pid = item.get("id")
                if pid not in raw_by_id:
                    item["_target_groups"] = [label]
                    raw_by_id[pid] = item
                else:
                    if label not in raw_by_id[pid]["_target_groups"]:
                        raw_by_id[pid]["_target_groups"].append(label)
            fetched += len(batch)
            print(f"[DBAASP] {label} offset={offset} got={len(batch)} total_unique={len(raw_by_id)}", flush=True)
            if len(batch) < page_size:
                break
        meta["queries"].append({"group": label, "base_url": base_url, "total_available": total_avail, "fetched": fetched})
    return raw_by_id, meta


def global_identity_ge70(a: str, b: str) -> bool:
    """Exact Needleman-Wunsch match=+2, mismatch=-1, gap=-2; identity >= 0.70 over all columns."""
    la, lb = len(a), len(b)
    if la == 0 and lb == 0:
        return True
    # Necessary length ratio bound: matches <= min(la, lb), cols >= max(la, lb)
    if 10 * min(la, lb) < 7 * max(la, lb):
        return False
    # Necessary character multiset bound: matches <= sum_c min(count_a[c], count_b[c])
    ca = collections.Counter(a)
    cb = collections.Counter(b)
    common = sum(min(ca[k], cb[k]) for k in ca if k in cb)
    if 10 * common < 7 * max(la, lb):
        return False
    # Exact DP matching frozen_metrics_v1.global_identity
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
    cols = -neg_cols
    return 10 * matches >= 7 * cols


def global_identity_exact(a: str, b: str) -> float:
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
    cols = -neg_cols
    return matches / cols if cols else 1.0


class UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1


def cluster_sequences_70(unique_seqs: list[str]) -> list[int]:
    """Exact >=70% global NW connected components with exact mathematical pruning."""
    n = len(unique_seqs)
    uf = UnionFind(n)
    # Sort by length for fast sliding window
    order = sorted(range(n), key=lambda idx: (len(unique_seqs[idx]), unique_seqs[idx]))
    seqs_ord = [unique_seqs[i] for i in order]
    lens_ord = [len(s) for s in seqs_ord]
    # Precompute 20-dim AA count tuples as list of ints
    aa_idx = {c: k for k, c in enumerate(AA_LIST)}
    counts_ord = []
    for s in seqs_ord:
        vec = [0] * 20
        for ch in s:
            vec[aa_idx[ch]] += 1
        counts_ord.append(tuple(vec))

    t0 = time.time()
    dp_calls = 0
    edges = 0
    for i in range(n):
        sa = seqs_ord[i]
        la = lens_ord[i]
        ca = counts_ord[i]
        oi = order[i]
        # max lb such that 10*la >= 7*lb -> lb <= floor(10*la/7)
        max_lb = (10 * la) // 7
        for j in range(i + 1, n):
            lb = lens_ord[j]
            if lb > max_lb:
                break
            oj = order[j]
            if uf.find(oi) == uf.find(oj):
                continue
            cb = counts_ord[j]
            # Fast L1 / multiset intersection upper bound
            common = 0
            for k in range(20):
                va, vb = ca[k], cb[k]
                common += va if va < vb else vb
            if 10 * common < 7 * lb:
                continue
            dp_calls += 1
            if global_identity_ge70(sa, seqs_ord[j]):
                uf.union(oi, oj)
                edges += 1
        if (i + 1) % 500 == 0:
            print(f"[Cluster] {i+1}/{n} done, dp_calls={dp_calls}, edges={edges}, elapsed={time.time()-t0:.1f}s", flush=True)
    roots = [uf.find(i) for i in range(n)]
    # Relabel deterministically
    root_map = {}
    comp_ids = []
    for r in roots:
        if r not in root_map:
            root_map[r] = len(root_map)
        comp_ids.append(root_map[r])
    print(f"[Cluster] Finished {n} unique seqs -> {len(root_map)} components in {time.time()-t0:.1f}s (dp_calls={dp_calls}, edges={edges})", flush=True)
    return comp_ids


def main():
    t_start = time.time()
    raw_by_id, meta = fetch_dbaasp_pages(max_records_per_group=4000, page_size=500)
    snapshot_path = OUT_DIR / "raw_snapshot.json"
    raw_list = [raw_by_id[k] for k in sorted(raw_by_id)]
    snapshot_bytes = json.dumps({"meta": meta, "records": raw_list}, ensure_ascii=False, indent=2).encode("utf-8")
    snapshot_path.write_bytes(snapshot_bytes)
    snapshot_sha = hashlib.sha256(snapshot_bytes).hexdigest()
    print(f"[Snapshot] Saved {len(raw_list)} unique raw DBAASP records (sha256={snapshot_sha[:16]}...)", flush=True)

    # Filter eligible records
    counts = collections.Counter()
    by_mol = {}  # (seq, terminal) -> list of dbaasp ids
    for rec in raw_list:
        counts["raw_total"] += 1
        seq = rec.get("sequence")
        if not isinstance(seq, str) or not seq:
            counts["drop_no_seq"] += 1
            continue
        if not (1 <= len(seq) <= 48):
            counts["drop_len"] += 1
            continue
        if not (set(seq) <= AA_SET):
            counts["drop_noncanonical"] += 1
            continue
        nt = rec.get("nTerminus")
        if nt not in (None, "", "None"):
            counts["drop_nterm"] += 1
            continue
        ct = rec.get("cTerminus")
        if ct in (None, "", "None"):
            term = "FREE"
        elif ct == "AMD":
            term = "AMD"
        else:
            counts["drop_cterm"] += 1
            continue
        counts["eligible_pre_dedup"] += 1
        key = (seq, term)
        by_mol.setdefault(key, []).append(str(rec.get("dbaaspId", rec.get("id"))))

    print(f"[Filter] {dict(counts)}, unique (seq, term)={len(by_mol)}", flush=True)

    # Compute RDKit representations using exact molecular_representation.py
    res_lookup = mr.residue_lookup()
    records = []
    rdkit_fail = 0
    for (seq, term) in sorted(by_mol.keys()):
        try:
            rep = mr.representation(seq, term)
        except Exception:
            rdkit_fail += 1
            continue
        records.append({
            "sequence": seq,
            "terminal_state": term,
            "class": mr.CLASS[term],
            "length": len(seq),
            "sequence_sha256": rep["sequence_sha256"],
            "molecule_sha256": rep["molecule_sha256"],
            "global_raw": rep["global_raw"],
            "source_ids": ";".join(sorted(set(by_mol[(seq, term)]))),
        })
    print(f"[RDKit] Valid records={len(records)}, failed={rdkit_fail}", flush=True)

    # Cluster unique sequences at >=70% global NW identity
    unique_seqs = sorted(set(r["sequence"] for r in records))
    comp_ids = cluster_sequences_70(unique_seqs)
    seq_to_comp = {s: c for s, c in zip(unique_seqs, comp_ids)}

    # Group records by component
    comp_to_records = collections.defaultdict(list)
    for r in records:
        cid = seq_to_comp[r["sequence"]]
        r["component_id"] = f"C{cid:04d}"
        comp_to_records[cid].append(r)

    # Check overlap with original 279/41 pilot cohort
    orig_train = set()
    orig_dev = set()
    with (ROOT / "github_repo/data/train_dev/train_cohort.csv").open() as fh:
        for row in csv.DictReader(fh):
            orig_train.add(row["sequence"])
    with (ROOT / "github_repo/data/train_dev/dev_cohort.csv").open() as fh:
        for row in csv.DictReader(fh):
            orig_dev.add(row["sequence"])
    orig_all = orig_train | orig_dev

    # Deterministic hash split of components: 80% train / 10% val / 10% test
    # Ensure any component containing an original pilot sequence goes to train
    # so the validation & independent test sets have ZERO exact overlap with the pilot cohort!
    splits = {"train": [], "val": [], "test": []}
    pilot_exposed_comps = 0
    for cid in sorted(comp_to_records.keys()):
        recs = comp_to_records[cid]
        comp_seqs = [r["sequence"] for r in recs]
        # Check exact match to pilot cohort
        is_pilot_exact = any(s in orig_all for s in comp_seqs)
        if is_pilot_exact:
            pilot_exposed_comps += 1
            splits["train"].extend(recs)
            continue
        # Deterministic hash on canonical min sequence of component
        key_str = f"DBAASP_EXPANDED_V1|{min(comp_seqs)}"
        h = int(hashlib.sha256(key_str.encode("ascii")).hexdigest()[:8], 16) / float(0xFFFFFFFF)
        if h < 0.78:
            splits["train"].extend(recs)
        elif h < 0.89:
            splits["val"].extend(recs)
        else:
            splits["test"].extend(recs)

    # Also verify no val/test sequence has >=70% identity with ANY pilot sequence (move to train if so)
    moved_pilot_homolog = 0
    for sp_name in ("val", "test"):
        keep = []
        comp_move = set()
        for r in splits[sp_name]:
            if r["component_id"] in comp_move:
                continue
            for pseq in orig_all:
                if global_identity_ge70(r["sequence"], pseq):
                    comp_move.add(r["component_id"])
                    break
        for r in splits[sp_name]:
            if r["component_id"] in comp_move:
                splits["train"].append(r)
                moved_pilot_homolog += 1
            else:
                keep.append(r)
        splits[sp_name] = keep

    # Deterministic ordering & write CSVs matching github_repo/data/train_dev schema
    idx_counter = 0
    split_summary = {}
    for sp_name in ("train", "val", "test"):
        recs = sorted(splits[sp_name], key=lambda x: (x["sequence"], x["terminal_state"]))
        splits[sp_name] = recs
        cohort_path = OUT_DIR / f"{sp_name}_cohort.csv"
        desc_path = OUT_DIR / f"{sp_name}_descriptors.csv"
        with cohort_path.open("w", newline="") as fc, desc_path.open("w", newline="") as fd:
            wc = csv.DictWriter(fc, fieldnames=[
                "idx", "sequence", "length", "terminal_state", "class",
                "component_id", "molecule_sha256", "sequence_sha256", "source_ids"
            ])
            wd = csv.DictWriter(fd, fieldnames=["idx", "protein", "describe", "point_describe", "class"])
            wc.writeheader()
            wd.writeheader()
            for r in recs:
                r["idx"] = idx_counter
                wc.writerow({
                    "idx": idx_counter,
                    "sequence": r["sequence"],
                    "length": r["length"],
                    "terminal_state": r["terminal_state"],
                    "class": r["class"],
                    "component_id": r["component_id"],
                    "molecule_sha256": r["molecule_sha256"],
                    "sequence_sha256": r["sequence_sha256"],
                    "source_ids": r["source_ids"],
                })
                res_desc = [res_lookup[ch] for ch in r["sequence"]]
                wd.writerow({
                    "idx": idx_counter,
                    "protein": r["sequence"],
                    "describe": json.dumps(res_desc),
                    "point_describe": json.dumps(r["global_raw"]),
                    "class": r["class"],
                })
                idx_counter += 1
        n_free = sum(1 for r in recs if r["terminal_state"] == "FREE")
        n_amd = sum(1 for r in recs if r["terminal_state"] == "AMD")
        n_res = sum(r["length"] for r in recs)
        n_comp = len(set(r["component_id"] for r in recs))
        n_len14 = sum(1 for r in recs if r["length"] < 5)
        split_summary[sp_name] = {
            "total": len(recs),
            "FREE": n_free,
            "AMD": n_amd,
            "total_residues": n_res,
            "components": n_comp,
            "length_1_to_4": n_len14,
            "cohort_sha256": hashlib.sha256(cohort_path.read_bytes()).hexdigest(),
            "descriptors_sha256": hashlib.sha256(desc_path.read_bytes()).hexdigest(),
        }

    # Verify zero cross-split >=70% leakage across test vs train/val
    test_seqs = sorted(set(r["sequence"] for r in splits["test"]))
    trainval_seqs = sorted(set(r["sequence"] for r in splits["train"] + splits["val"]))
    max_test_trainval_id = 0.0
    leak_pairs = 0
    for ts in test_seqs:
        for tvs in trainval_seqs:
            if global_identity_ge70(ts, tvs):
                leak_pairs += 1
    # Also verify exact max identity on a deterministic sample / full check via length bound
    audit = {
        "created_utc": datetime.datetime.utcnow().isoformat() + "Z",
        "elapsed_seconds": round(time.time() - t_start, 2),
        "raw_snapshot_sha256": snapshot_sha,
        "filter_counts": dict(counts),
        "rdkit_failed": rdkit_fail,
        "unique_sequences": len(unique_seqs),
        "total_components_70pct": len(comp_to_records),
        "pilot_exact_components_assigned_train": pilot_exposed_comps,
        "pilot_homolog_records_moved_to_train": moved_pilot_homolog,
        "cross_split_leakage_pairs_ge70": leak_pairs,
        "splits": split_summary,
    }
    (OUT_DIR / "split_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
