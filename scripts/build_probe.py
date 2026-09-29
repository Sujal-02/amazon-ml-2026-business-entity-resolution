"""Build a leaderboard PROBE file: per country, take the links from our predicted shards ('ours'), from a reference
submission ('ref'), or nothing ('empty'). Macro F0.5 is additive over S1, so probes isolate per-country scores:

  LB(ours FR + ref US/IN) - LB(ref)           = w_fr * (F_fr(ours) - F_fr(ref))
  LB(ours FR + ref US/IN) - LB(FR empty + ref) = w_fr * (F_fr(ours) - s_fr)        (w_fr ~ 0.1498, s_fr ~ 0.056)

'ours' merges output/<split>/matching_results[_nocorr].tsv of every predicted shard of that country and resolves
cross-shard duplicates (no-state candidates sit in several state shards) by the highest calibrated p, as finalize does.
Probes are measurements only; the final submission is always the full pipeline output (pipeline finalize).

usage: python scripts/build_probe.py --out D:/er_data/work_dev/output/probes/P1.tsv --france ours --us ref --india ref
          [--ref D:/er_data/ref/other_v7_0972_matching_results.tsv] [--variant nocorr]
"""
from __future__ import annotations

import argparse
import glob
import os

import pandas as pd
import yaml

NL, TAB = chr(10), chr(9)


def read_lists(path):
    out = {}
    with open(path, encoding="utf-8") as f:
        next(f)
        for line in f:
            s1, _, rest = line.rstrip(NL).partition(TAB)
            out[s1] = [x for x in rest.split(",") if x]
    return out


def ours(cfg, country, variant):
    fname = "matching_results_nocorr.tsv" if variant == "nocorr" else "matching_results.tsv"
    dirs = sorted(d for d in glob.glob(os.path.join(cfg["out_dir"], f"test@{country}*"))
                  if os.path.exists(os.path.join(d, fname)))
    if not dirs:
        raise SystemExit(f"no predicted shard for {country} under {cfg['out_dir']}")
    match = {}
    for d in dirs:
        for s1, ids in read_lists(os.path.join(d, fname)).items():
            if ids:
                match.setdefault(s1, []).extend(ids)
    mp = pd.DataFrame([(a, b) for a, bs in match.items() for b in bs], columns=["s1_id", "cand_id"])
    dup = mp[mp.cand_id.duplicated(keep=False)]
    dropped = 0
    if len(dup):
        parts = []
        for d in dirs:
            sp = os.path.join(cfg["work_dir"], os.path.basename(d), "scores_gbdt.parquet")
            parts.append(pd.read_parquet(sp, columns=["s1_id", "cand_id", "p"]).merge(dup, on=["s1_id", "cand_id"]))
        best = pd.concat(parts).sort_values("p", ascending=False).drop_duplicates("cand_id")
        owner = dict(zip(best.cand_id, best.s1_id))
        for s1 in list(match):
            keep = [x for x in match[s1] if owner.get(x, s1) == s1]
            dropped += len(match[s1]) - len(keep)
            match[s1] = keep
    print(f"ours[{country}]: {len(dirs)} shards {[os.path.basename(d) for d in dirs]}, cross-shard duplicates dropped {dropped}")
    return match, len(dirs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "configs", "local_final.yaml"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--ref", default="D:/er_data/ref/other_v7_0972_matching_results.tsv")
    ap.add_argument("--test-dir", default="D:/er_data/student_resource/dataset/test")
    ap.add_argument("--variant", default="main", choices=["main", "nocorr"])
    ap.add_argument("--expect-shards", type=int, nargs=3, default=[3, 3, 4], metavar=("FR", "US", "IN"),
                    help="refuse 'ours' for a country unless this many shards are predicted")
    for c in ("france", "us", "india"):
        ap.add_argument(f"--{c}", choices=["ours", "ref", "empty"], required=True)
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config, encoding="utf-8"))
    s1 = pd.read_csv(os.path.join(a.test_dir, "test_source1.tsv"), sep=TAB, usecols=["entity_id", "country"], dtype=str,
                     keep_default_na=False)
    ref = read_lists(a.ref)
    src = {"france": a.france, "us": a.us, "india": a.india}
    expect = dict(zip(("france", "us", "india"), a.expect_shards))
    got = {}
    for c, how in src.items():
        if how == "ours":
            got[c], n = ours(cfg, c, a.variant)
            if n < expect[c]:
                raise SystemExit(f"only {n}/{expect[c]} {c} shards predicted: rows of the missing shards would be empty")
    cmap = {"France": "france", "US": "us", "India": "india"}
    rows, stats = [], {}
    for sid, ctry in zip(s1.entity_id, s1.country):
        c = cmap.get(ctry, ctry.lower())
        how = src.get(c, "ref")
        ids = got[c].get(sid, []) if how == "ours" else ref.get(sid, []) if how == "ref" else []
        rows.append(sid + TAB + ",".join(ids))
        st = stats.setdefault(ctry, [0, 0, 0])
        st[0] += 1; st[1] += len(ids); st[2] += (len(ids) == 0)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", encoding="utf-8", newline="") as f:
        f.write("source1_entity_id" + TAB + "matched_entity_ids" + NL + NL.join(rows) + NL)
    for ctry, (n, l, e) in sorted(stats.items()):
        print(f"{ctry:8s} S1 {n:8d}  links/S1 {l / n:.3f}  empty {e / n:.4f}  source={src.get(cmap.get(ctry, ''), 'ref')}")
    print("wrote", a.out, len(rows), "rows")


if __name__ == "__main__":
    main()
