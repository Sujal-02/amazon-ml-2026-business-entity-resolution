"""Label-free comparison of our submission with a reference submission (e.g. the team's other pipeline, LB 0.972).

Per country: links/S1, empty rate, pair overlap, and for each disagreement class (ours-only, ref-only) whether the pair
is in our candidate set and where our calibrated probability sits. Prints a few French disagreements for eyeballing.

usage: python scripts/compare_ref.py --config configs/local_final_tuned.yaml
          --matching D:/er_data/work_dev/output/final/matching_results.tsv
          --reference D:/er_data/ref/other_v7_0972_matching_results.tsv
          --test-dir D:/er_data/student_resource/dataset/test [--show 12]
"""
from __future__ import annotations

import argparse
import glob
import os

import numpy as np
import pandas as pd
import yaml


def read_lists(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        next(f)
        for line in f:
            s1, _, rest = line.rstrip("\n").partition("\t")
            rows.extend((s1, c) for c in rest.split(",") if c)
    return pd.DataFrame(rows, columns=["s1", "cand"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--matching", required=True)
    ap.add_argument("--reference", required=True)
    ap.add_argument("--test-dir", required=True)
    ap.add_argument("--show", type=int, default=12)
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config, encoding="utf-8"))

    s1 = pd.read_csv(os.path.join(a.test_dir, "test_source1.tsv"), sep="\t", dtype=str, keep_default_na=False)
    country = dict(zip(s1.entity_id, s1.country))
    ours, ref = read_lists(a.matching), read_lists(a.reference)
    ours["o"], ref["r"] = 1, 1
    m = ours.merge(ref, on=["s1", "cand"], how="outer").fillna(0)
    m["country"] = m.s1.map(country)

    # our calibrated probability (shift-corrected p and p_raw) for every pair either submission links
    need = m[["s1", "cand"]]
    got = []
    for d in sorted(glob.glob(os.path.join(cfg["work_dir"], "test@*#*"))):
        sp = os.path.join(d, "scores_gbdt.parquet")
        if not os.path.exists(sp):
            continue
        rec = pd.read_parquet(os.path.join(d, "records.parquet"), columns=["entity_id"])
        eid = rec.entity_id.to_numpy()
        sc = pd.read_parquet(sp, columns=["i1", "i2", "p", "p_raw"])
        sc = pd.DataFrame({"s1": eid[sc.i1.to_numpy()], "cand": eid[sc.i2.to_numpy()], "p": sc.p.to_numpy(),
                           "p_raw": sc.p_raw.to_numpy()})
        got.append(sc.merge(need, on=["s1", "cand"], how="inner"))
    if got:
        # a candidate without a parsed state can sit in several state sub-shards: keep its max score
        sc = pd.concat(got).groupby(["s1", "cand"], as_index=False)[["p", "p_raw"]].max()
        m = m.merge(sc, on=["s1", "cand"], how="left")
    else:
        m["p"] = m["p_raw"] = np.nan
    m["cls"] = np.select([(m.o == 1) & (m.r == 1), m.o == 1], ["both", "ours_only"], "ref_only")

    rows = []
    for c, g in m.groupby("country"):
        n1 = sum(1 for v in country.values() if v == c)
        o_s1 = g[g.o == 1].s1.nunique()
        r_s1 = g[g.r == 1].s1.nunique()
        rows.append({"country": c, "S1": n1, "our_links": g.o.sum() / n1, "ref_links": g.r.sum() / n1,
                     "our_empty": 1 - o_s1 / n1, "ref_empty": 1 - r_s1 / n1,
                     "both": int((g.cls == "both").sum()), "ours_only": int((g.cls == "ours_only").sum()),
                     "ref_only": int((g.cls == "ref_only").sum()),
                     "ref_only_in_cands": float(g[g.cls == "ref_only"].p.notna().mean()) if (g.cls == "ref_only").any() else np.nan})
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    print("\nour p (shift-corrected) quantiles by class x country  [q10 q25 q50 q75 q90]:")
    for (c, k), g in m[m.p.notna()].groupby(["country", "cls"]):
        q = np.quantile(g.p, [0.1, 0.25, 0.5, 0.75, 0.9])
        print(f"  {c:8s} {k:9s} n={len(g):8d}  " + " ".join(f"{v:.3f}" for v in q))

    if a.show:
        names = {}
        for s in ("test_source1.tsv", "test_source2.tsv", "test_source3.tsv"):
            p = os.path.join(a.test_dir, s)
            if os.path.exists(p):
                t = pd.read_csv(p, sep="\t", dtype=str, keep_default_na=False)
                names.update(zip(t.entity_id, t.business_name + " | " + t.business_address))
        fr = m[m.country == "France"]
        for k in ("ref_only", "ours_only"):
            g = fr[fr.cls == k].sample(min(a.show, int((fr.cls == k).sum())), random_state=0)
            print(f"\n--- France {k} (sample) ---")
            for r in g.itertuples():
                print(f"p={r.p if r.p == r.p else float('nan'):.3f}  {names.get(r.s1, '?')}\n           ~ {names.get(r.cand, '?')}")


if __name__ == "__main__":
    main()
