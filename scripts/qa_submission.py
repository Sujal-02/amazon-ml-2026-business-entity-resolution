"""Label-free QA of a submission: per-country link statistics vs the training prior.

The training ground truth has 5.6% singletons and 3.46 matches per S1 (US 3.459, India 3.465), identical across
countries -- the generator does not depend on the country. Test countries (incl. the unseen France) should look
similar after accounting for our recall (~0.95): a country far below the others signals under-matching
(e.g. miscalibration on an unseen country), far above signals false merges.

usage: python scripts/qa_submission.py --matching <matching_results.tsv> --candidates <candidate_pairs.tsv>
                                      --test-dir D:/er_data/student_resource/dataset/test [--scores-glob "..."]
"""
from __future__ import annotations

import argparse
import collections
import csv
import glob
import os

import numpy as np
import pandas as pd

TRAIN_PRIOR = {"singleton_frac": 0.0558, "mean_links": 3.46}


def read_lists(path, col):
    out = {}
    with open(path, encoding="utf-8") as f:
        next(f)
        for line in f:
            s1, _, rest = line.rstrip("\n").partition("\t")
            out[s1] = [x for x in rest.split(",") if x]
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--matching", required=True)
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--test-dir", required=True)
    ap.add_argument("--scores-glob", default=None, help="e.g. D:/er_data/work_dev/test@*/scores_gbdt.parquet")
    a = ap.parse_args(argv)
    s1 = pd.read_csv(os.path.join(a.test_dir, "test_source1.tsv"), sep="\t", dtype=str, keep_default_na=False,
                     quoting=csv.QUOTE_NONE, encoding="utf-8-sig", usecols=["entity_id", "country"])
    country = dict(zip(s1.entity_id, s1.country.str.strip()))
    m = read_lists(a.matching, "matched_entity_ids")
    c = read_lists(a.candidates, "candidate_entity_ids")
    rows = []
    for ctry in sorted(set(country.values())):
        ids = [s for s, k in country.items() if k == ctry]
        n = np.array([len(m.get(s, [])) for s in ids])
        nc = np.array([len(c.get(s, [])) for s in ids])
        src = collections.Counter(x[:2] for s in ids for x in m.get(s, []))
        rows.append({"country": ctry, "n_s1": len(ids), "empty_frac": round(float((n == 0).mean()), 4),
                     "mean_links": round(float(n.mean()), 3), "mean_links_nonempty": round(float(n[n > 0].mean()), 3) if (n > 0).any() else 0,
                     "p50/p90/max": f"{int(np.median(n))}/{int(np.quantile(n, .9))}/{int(n.max())}",
                     "mean_candidates": round(float(nc.mean()), 1), "S2:S3": f"{src.get('S2', 0)}:{src.get('S3', 0)}"})
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    print(f"train prior: singleton_frac {TRAIN_PRIOR['singleton_frac']}, mean matches/S1 {TRAIN_PRIOR['mean_links']} "
          "(expect empty_frac ~0.06-0.10 and mean_links ~3.1-3.4 given ~0.95 recall)")
    allm = [x for v in m.values() for x in v]
    dup = len(allm) - len(set(allm))
    print(f"total S1 rows {len(m)} (expected {len(country)}), matched ids {len(allm)}, ids matched to >1 S1: {dup}")
    if a.scores_glob:
        parts = [pd.read_parquet(p, columns=["s1_id", "p"]) for p in glob.glob(a.scores_glob)]
        if parts:
            sc = pd.concat(parts, ignore_index=True)
            sc["country"] = sc.s1_id.map(country)
            bins = [0, 0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99, 1.0001]
            h = sc.groupby("country")["p"].apply(lambda s: np.round(np.histogram(s, bins)[0] / len(s), 4))
            print("score histogram per country, bins", bins)
            for k, v in h.items():
                print(f"  {k:8s} {v}")


if __name__ == "__main__":
    main()
