#!/usr/bin/env python3
"""Re-verify SZL-MiniEmbed.

1. sha256 the shipped vectors.npz + vocab.json against TRAINING_RECEIPT.json (refuse on mismatch).
2. Deterministically regenerate the embeddings via scripts/forge.py from the same corpus and
   compare the re-measured intrinsic nearest-neighbour lists to the receipt: report the mean
   Jaccard overlap of the top-k neighbour sets (tolerance: mean overlap >= 0.90 across lib versions),
   plus the SVD explained-variance delta (±0.02).
Run from repo root: python scripts/eval.py"""
import hashlib, json, subprocess, sys, tempfile, os, shutil
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
receipt = json.load(open(f"{root}/TRAINING_RECEIPT.json"))

ok = True
for fname, key in [("vectors.npz", "vectors.npz"), ("vocab.json", "vocab.json")]:
    got = hashlib.sha256(open(f"{root}/{fname}", "rb").read()).hexdigest()
    want = receipt["model"]["files"][key]
    m = got == want
    ok = ok and m
    print(f"{fname} sha256 {'MATCHES receipt' if m else 'MISMATCH — refuse'}: {got[:16]}…")
if not ok:
    sys.exit(1)

with tempfile.TemporaryDirectory() as td:
    os.makedirs(f"{td}/scripts", exist_ok=True)
    shutil.copy(f"{root}/scripts/forge.py", f"{td}/scripts/forge.py")
    # ship an in-repo corpus/ copy if present so eval works from a fresh clone
    if os.path.isdir(f"{root}/corpus"):
        shutil.copytree(f"{root}/corpus", f"{td}/corpus")
    out = subprocess.run([sys.executable, f"{td}/scripts/forge.py"],
                         capture_output=True, text=True, cwd=td)
    print(out.stdout[-400:] if out.returncode == 0 else out.stderr[-800:])
    if out.returncode:
        sys.exit(1)
    re_receipt = json.load(open(f"{td}/TRAINING_RECEIPT.json"))
    a = receipt["metrics_MEASURED"]["intrinsic_nearest_neighbors"]
    b = re_receipt["metrics_MEASURED"]["intrinsic_nearest_neighbors"]
    overlaps = []
    for term in a:
        if term in b:
            sa = {w for w, _ in a[term]}
            sb = {w for w, _ in b[term]}
            overlaps.append(len(sa & sb) / max(1, len(sa | sb)))
    mean_overlap = sum(overlaps) / max(1, len(overlaps))
    ev_delta = abs(re_receipt["metrics_MEASURED"]["svd_explained_variance_ratio"]
                   - receipt["metrics_MEASURED"]["svd_explained_variance_ratio"])
    print(f"neighbour-set mean Jaccard overlap vs receipt: {mean_overlap:.4f} "
          f"({'OK ≥0.90' if mean_overlap >= 0.90 else 'FAIL'})")
    print(f"SVD explained-variance delta vs receipt: {ev_delta:.4f} "
          f"({'OK ≤0.02' if ev_delta <= 0.02 else 'FAIL'})")
    sys.exit(0 if (mean_overlap >= 0.90 and ev_delta <= 0.02) else 1)
