"""Export the GBDT's uncertain band as text pairs for a cross-encoder re-ranker (trained on a Kaggle GPU).

tr  = train folds 0-2, OOF p in band (+ a small sample of confident pairs)  -> fine-tune the cross-encoder
cal = train folds 3-4, OOF p in band                                          -> fit the GBDT+CE blend, measure the gain
te  = every test shard, GBDT p in band                                        -> re-scored, blended, re-decided locally
Text = name_norm | addr_norm (normalised + transliterated, so Indic names are Latin). Local map keeps te pid -> pair.
"""
from __future__ import annotations

import glob
import os
import sys

import numpy as np
import pandas as pd

W = "D:/er_data/work_dev"
OUT = "D:/er_data/ce_data"
CAP_TE = 5_000_000
os.makedirs(OUT, exist_ok=True)


def texts(split):
    r = pd.read_parquet(os.path.join(W, split, "records.parquet"), columns=["name_norm", "addr_norm"])
    return (r.name_norm.fillna("").astype(str) + " | " + r.addr_norm.fillna("").astype(str)).to_numpy(object)


def band(p, lo):
    return (p >= lo) & (p <= 1 - lo)


# ---- test band size -> choose threshold so the test set stays <= CAP_TE pairs
splits = sorted(os.path.basename(d) for d in glob.glob(os.path.join(W, "test@*#*"))
                if os.path.exists(os.path.join(d, "scores_gbdt.parquet")))
ps = {s: pd.read_parquet(os.path.join(W, s, "scores_gbdt.parquet"), columns=["i1", "i2", "p"]) for s in splits}
lo = None
for cand in (0.003, 0.01, 0.02, 0.04):
    n = sum(int(band(d.p.to_numpy(), cand).sum()) for d in ps.values())
    print(f"test band [{cand}, {1 - cand}]: {n:,} pairs", flush=True)
    if n <= CAP_TE:
        lo = cand
        break
lo = lo or 0.04
print("using band lo =", lo, flush=True)

# ---- train (OOF) pairs
oof = pd.read_parquet(os.path.join(W, "train", "oof.parquet"), columns=["i1", "i2", "y", "fold", "p"])
tt = texts("train")
m = band(oof.p.to_numpy(), lo)
rng = np.random.default_rng(0)
f012 = oof.fold.isin([0, 1, 2]).to_numpy()
easy = (~m) & f012 & (rng.random(len(oof)) < 150_000 / max(1, ((~m) & f012).sum()))
tr = oof[(m & f012) | easy]
cal = oof[m & ~f012]
print(f"train band: tr {len(tr):,} (pos {int(tr.y.sum()):,}), cal {len(cal):,} (pos {int(cal.y.sum()):,})", flush=True)
for name, d in (("tr", tr), ("cal", cal)):
    pd.DataFrame({"pid": np.arange(len(d), dtype=np.int64), "a": tt[d.i1.to_numpy()], "b": tt[d.i2.to_numpy()],
                  "y": d.y.to_numpy(np.float32), "p": d.p.to_numpy(np.float32)}).to_parquet(
        os.path.join(OUT, f"{name}.parquet"), compression="zstd", index=False)
cal[["i1", "i2", "y", "fold", "p"]].reset_index(drop=True).to_parquet(os.path.join(W, "train", "ce_cal_map.parquet"))
del oof, tt

# ---- test band pairs (one file per shard) + local pid map
base = 0
maps = []
for s in splits:
    d = ps[s]
    mm = band(d.p.to_numpy(), lo)
    d = d[mm].reset_index(drop=True)
    tx = texts(s)
    pid = np.arange(base, base + len(d), dtype=np.int64)
    pd.DataFrame({"pid": pid, "a": tx[d.i1.to_numpy()], "b": tx[d.i2.to_numpy()], "p": d.p.to_numpy(np.float32)}).to_parquet(
        os.path.join(OUT, f"te_{s.replace('@', '_').replace('#', '_')}.parquet"), compression="zstd", index=False)
    maps.append(pd.DataFrame({"pid": pid, "split": s, "i1": d.i1.to_numpy(), "i2": d.i2.to_numpy()}))
    base += len(d)
    print(f"{s}: {len(d):,} band pairs", flush=True)
pd.concat(maps, ignore_index=True).to_parquet(os.path.join(W, "ce_te_map.parquet"))
print("TOTAL test band pairs:", base, "| files in", OUT, flush=True)
for f in sorted(os.listdir(OUT)):
    print(f, round(os.path.getsize(os.path.join(OUT, f)) / 1e6, 1), "MB")
