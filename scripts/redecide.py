"""Re-run only the decision layer on stored shard scores (no features, no models): sweep or apply the unseen-country
bias (`decision_unseen_bias`) and compare per-country volume with the training prior and an optional reference file.

Each predict step leaves `work/<split>/scores_gbdt.parquet` (i1, i2, p = shift-corrected, p_raw) and
`artifacts/decision.json`, so a new bias costs seconds per shard instead of a full predict.

usage:
  sweep : python scripts/redecide.py --config configs/local_final_tuned.yaml --splits "test@france*" --sweep -0.7 -0.5 -0.3 0 0.3
          [--reference D:/er_data/ref/other_v7_0972_matching_results.tsv]
  apply : python scripts/redecide.py --config configs/local_final_tuned.yaml --splits "test@france*" --apply -0.3
          (rewrites output/<split>/matching_results{,_nocorr}.tsv; then run `python -m ber.pipeline finalize`)
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from ber import decision as Dm  # noqa: E402
from ber import io as bio  # noqa: E402

TRAIN_PRIOR = {"empty": 0.0558, "links": 3.46}


def load_split(cfg, split):
    wd = os.path.join(cfg["work_dir"], split)
    rec = pd.read_parquet(os.path.join(wd, "records.parquet"), columns=["entity_id", "src", "country_norm"])
    sc = pd.read_parquet(os.path.join(wd, "scores_gbdt.parquet"), columns=["i1", "i2", "p", "p_raw"])
    return rec, sc


def decide_split(rec, sc, dcfg, seen, bias, col):
    eid = rec["entity_id"].to_numpy()
    pairs = sc[["i1", "i2", col]].rename(columns={"i1": "s1", "i2": "cand", col: "p"})
    pairs["_unseen"] = ~np.isin(rec["country_norm"].to_numpy()[sc.i1.to_numpy()], list(seen))
    pred_int = Dm.decide(pairs, {**dcfg, "unseen_bias": float(bias)}, "p")
    return {eid[a]: [eid[b] for b in bs] for a, bs in pred_int.items()}


def volume(pred, s1_ids):
    n = np.array([len(pred.get(s, [])) for s in s1_ids])
    return {"S1": len(n), "empty": float((n == 0).mean()), "links": float(n.mean())}


def read_reference(path):
    out = {}
    with open(path, encoding="utf-8") as f:
        next(f)
        for line in f:
            s1, _, rest = line.rstrip("\n").partition("\t")
            out[s1] = [x for x in rest.split(",") if x]
    return out


def agreement(pred, ref, s1_ids):
    """pair-level overlap with a reference submission: share of our links they also have, and vice versa."""
    ours = {(s, c) for s in s1_ids for c in pred.get(s, [])}
    theirs = {(s, c) for s in s1_ids for c in ref.get(s, [])}
    inter = len(ours & theirs)
    return {"ours_in_ref": inter / max(len(ours), 1), "ref_in_ours": inter / max(len(theirs), 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--splits", default="test@france*", help="glob over work_dir split folders")
    ap.add_argument("--sweep", type=float, nargs="*", default=None)
    ap.add_argument("--apply", type=float, default=None)
    ap.add_argument("--reference", default=None, help="another matching_results.tsv to compare volume/agreement with")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config, encoding="utf-8"))
    with open(os.path.join(cfg["artifacts_dir"], "decision.json")) as f:
        dcfg = json.load(f)
    with open(os.path.join(cfg["artifacts_dir"], "train_countries.json")) as f:
        seen = set(json.load(f))
    splits = sorted(os.path.basename(d) for d in glob.glob(os.path.join(cfg["work_dir"], a.splits))
                    if os.path.exists(os.path.join(d, "scores_gbdt.parquet")))
    if not splits:
        sys.exit(f"no split under {cfg['work_dir']} matching {a.splits} has scores_gbdt.parquet")
    print("decision:", dcfg, "| seen countries:", sorted(seen), "| splits:", splits)
    ref = read_reference(a.reference) if a.reference else None
    data = {s: load_split(cfg, s) for s in splits}
    s1_by_split = {s: r.entity_id[r.src == 1].tolist() for s, (r, _) in data.items()}
    s1_all = [x for s in splits for x in s1_by_split[s]]

    if a.sweep is not None:
        rows = []
        for b in a.sweep:
            for col in ("p", "p_raw"):
                pred = {}
                for s in splits:
                    pred.update(decide_split(*data[s], dcfg, seen, b, col))
                row = {"bias": b, "score": col, **volume(pred, s1_all)}
                if ref is not None:
                    row.update(agreement(pred, ref, s1_all))
                rows.append(row)
        if ref is not None:
            rows.append({"bias": "reference", "score": "-", **volume(ref, s1_all)})
        rows.append({"bias": "train prior", "score": "-", "S1": "-", **TRAIN_PRIOR})
        print(pd.DataFrame(rows).to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    if a.apply is not None:
        for s in splits:
            out_dir = os.path.join(cfg["out_dir"], s)
            vol = {}
            for col, name in (("p", "matching_results.tsv"), ("p_raw", "matching_results_nocorr.tsv")):
                pred = decide_split(*data[s], dcfg, seen, a.apply, col)
                bio.write_matching(os.path.join(out_dir, name), s1_by_split[s], pred)
                vol[col] = volume(pred, s1_by_split[s])
            print(f"{s}: rewrote matching_results with unseen_bias {a.apply:+.2f} ->", vol)
        print("now run: cd src && python -m ber.pipeline finalize --config ../" + a.config.replace("\\", "/"))


if __name__ == "__main__":
    main()
