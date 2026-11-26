#!/usr/bin/env python3
"""Offline verifier for BIB-26-2683 Major Revision (v8-revision-2026-11-26); stdlib only."""
import hashlib, json, subprocess, sys
from pathlib import Path

R = Path(__file__).resolve().parent

def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()

# 1. Run full v7+v8 manifest integrity and v7 baseline verification
v7 = subprocess.run([sys.executable, str(R / "verify_all.py")], cwd=R, check=True, text=True, capture_output=True)
v7_res = json.loads(v7.stdout)

# 2. Verify expanded DBAASP benchmark hashes & zero cross-split >=70% pairs
audit = json.loads((R / "revision_r1/data/dbaasp_expanded/split_audit.json").read_text())
assert audit["splits"]["train"]["total"] == 3120
assert audit["splits"]["val"]["total"] == 223
assert audit["splits"]["test"]["total"] == 222
assert audit["cross_split_leakage_pairs_ge70"] == 0
for sp in ("train", "val", "test"):
    assert sha256(R / f"revision_r1/data/dbaasp_expanded/{sp}_cohort.csv") == audit["splits"][sp]["cohort_sha256"]
    assert sha256(R / f"revision_r1/data/dbaasp_expanded/{sp}_descriptors.csv") == audit["splits"][sp]["descriptors_sha256"]

# 3. Verify master summary and wet-lab deep audit
ms = json.loads((R / "revision_r1/results/master_summary.json").read_text())
wl = json.loads((R / "revision_r1/results/wetlab_deep_audit.json").read_text())
assert len(wl["od600_threshold_sensitivity"]) == 6
assert wl["od600_threshold_sensitivity"][1]["concordance_of_108"] == 105
assert wl["od600_threshold_sensitivity"][1]["saureus_le_31_25"] == 6
assert len(wl["peptide_profiles"]) == 27

out = {
    "release": "v8-revision-2026-11-26",
    "v7_baseline": v7_res,
    "r1_dbaasp_counts": {k: audit["splits"][k]["total"] for k in ("train", "val", "test")},
    "r1_cross_split_ge70_pairs": audit["cross_split_leakage_pairs_ge70"],
    "r1_od600_primary_concordance": "105/108",
    "r1_saureus_hits_le31_25": "6/27",
    "status": "PASS",
}
print(json.dumps(out, indent=2))
