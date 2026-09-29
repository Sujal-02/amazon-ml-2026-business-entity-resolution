"""Fill the {{PLACEHOLDER}} fields of Documentation.md from the run reports (run after `pipeline finalize`).

usage: python scripts/fill_doc.py --config configs/local_final_tuned.yaml [--lb 0.97x] [--date "27 Sep 2026"]
                                  [--doc Documentation.md] [--test-dir D:/er_data/student_resource/dataset/test]
Placeholders without a source are left untouched and listed at the end.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re

import numpy as np
import pandas as pd
import yaml


def jload(path):
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return None


def f4(x):
    return f"{x:.4f}" if isinstance(x, (int, float)) and x == x else "n/a"


def count_lists(path):
    n_s1 = n_pairs = 0
    with open(path, encoding="utf-8") as f:
        next(f)
        for line in f:
            n_s1 += 1
            rest = line.rstrip("\n").partition("\t")[2]
            n_pairs += len([x for x in rest.split(",") if x])
    return n_s1, n_pairs


def oracle_ceiling(work):
    """macro F0.5 of a perfect matcher restricted to the train candidate set (P = 1, R = found / true)."""
    gt = jload(os.path.join(work, "train", "gt.json"))
    cp = os.path.join(work, "train", "cands.parquet")
    if gt is None or not os.path.exists(cp):
        return None
    rec = pd.read_parquet(os.path.join(work, "train", "records.parquet"), columns=["entity_id", "src"])
    eid = rec.entity_id.to_numpy()
    c = pd.read_parquet(cp, columns=["i1", "i2"])
    found = pd.DataFrame({"s1": eid[c.i1.to_numpy()], "cand": eid[c.i2.to_numpy()]})
    truth = pd.DataFrame([(s, x) for s, xs in gt.items() for x in xs], columns=["s1", "cand"])
    hit = truth.merge(found, on=["s1", "cand"], how="inner").groupby("s1").size()
    n_true = truth.groupby("s1").size()
    s1_all = eid[rec.src.to_numpy() == 1]
    r = (hit.reindex(n_true.index).fillna(0) / n_true)
    f = (1.25 * r / (0.25 + r)).where(r > 0, 0.0)          # F0.5 with P = 1
    scores = pd.Series(1.0, index=pd.Index(s1_all))           # singletons: empty prediction scores 1
    scores.loc[f.index.intersection(scores.index)] = f
    return float(scores.mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--doc", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Documentation.md"))
    ap.add_argument("--test-dir", default="D:/er_data/student_resource/dataset/test")
    ap.add_argument("--lb", default=None)
    ap.add_argument("--date", default=dt.date.today().strftime("%d %b %Y"))
    ap.add_argument("--no-oracle", action="store_true")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config, encoding="utf-8"))
    work, art = cfg["work_dir"], cfg["artifacts_dir"]
    final_dir = cfg.get("finalize", {}).get("out_dir", os.path.join(cfg["out_dir"], "final"))

    v = {"SUBMISSION_DATE": a.date}
    rep = jload(os.path.join(work, "train", "oof_report_p.json"))
    if rep:
        bc = rep.get("by_country", {})
        v.update(OOF_MACRO_F05=f4(rep["macro_f05"]), OOF_F05_SINGLETONS=f4(rep.get("f05_singletons")),
                 OOF_F05_NONSINGLETONS=f4(rep.get("f05_nonsingletons")), OOF_MICRO_PRECISION=f4(rep.get("micro_precision")),
                 OOF_MICRO_RECALL=f4(rep.get("micro_recall")), OOF_F05_US=f4(bc.get("us")), OOF_F05_INDIA=f4(bc.get("india")),
                 VALIDATION_SAMPLE=f"{rep['n_s1']:,} S1 (whole-state geographic sample of train)")
    for k in ("lgb", "xgb", "cat"):
        r2 = jload(os.path.join(work, "train", f"oof_report_p2_{k}.json"))
        v[f"OOF_F05_P2_{k.upper()}"] = f4(r2["macro_f05"]) if r2 else "n/a"
    dec = jload(os.path.join(art, "decision.json"))
    if dec:
        dec = {**{k: x for k, x in dec.items() if k != "score"}, "unseen_bias": cfg.get("decision_unseen_bias")}
        v["DECISION_CFG"] = "`" + json.dumps(dec) + "`"
    st = jload(os.path.join(work, "train", "stack_report.json"))
    v["STACK_MACRO_F05"] = f4(st["macro_f05"]) if st else "not used (GBDT-only submission)"
    v["GBDT_ONLY_SAME_HOLDOUT"] = f4(st.get("gbdt_only")) if st and "gbdt_only" in st else "n/a"
    br = jload(os.path.join(work, "train", "blocking_report.json"))
    if br:
        v.update(BLOCKING_RECALL=f4(br["pair_recall"]), ENTITY_FULL_RECALL=f4(br["entity_full_recall"]),
                 AVG_CANDIDATES=f"{br['avg_candidates']:.1f}",
                 N_CANDIDATE_PAIRS_TRAIN=f"{int(round(br['avg_candidates'] * br['n_s1'])):,}")
    if not a.no_oracle:
        oc = oracle_ceiling(work)
        if oc is not None:
            v["ORACLE_F05_CEILING"] = f4(oc)
    cand_path = os.path.join(final_dir, "candidate_pairs.tsv")
    if os.path.exists(cand_path):
        n_s1, n_pairs = count_lists(cand_path)
        v.update(N_CANDIDATE_PAIRS_TEST=f"{n_pairs:,}", AVG_CANDIDATES_TEST=f"{n_pairs / max(n_s1, 1):.1f}")
        cnt = {s: pd.read_csv(os.path.join(a.test_dir, f"test_source{s}.tsv"), sep="\t", usecols=["country"], dtype=str)
               .country.value_counts() for s in (1, 2, 3)}
        cart = sum(int(cnt[1][c]) * int(cnt[2].get(c, 0) + cnt[3].get(c, 0)) for c in cnt[1].index)
        v["REDUCTION_RATIO_TEST"] = f"{100 * (1 - n_pairs / cart):.4f}% (of {cart / 1e9:.2f}B pairs)"
    log_path = os.path.join(work, "logs", ".done_finalize.log")
    if os.path.exists(log_path):
        txt = open(log_path, encoding="utf-8", errors="replace").read()
        v["VALIDATOR_STATUS"] = "PASS" if re.search(r"\bPASS", txt) else "see `.done_finalize.log`"
    for step, key in (("train_stack", "DEV_TRAIN_WALLCLOCK"), ("predict", "DEV_PREDICT_WALLCLOCK")):
        dl = os.path.join(work, "logs", "driver.log")
        if os.path.exists(dl):
            lines = open(dl, encoding="utf-8").read().splitlines()
            ts = [(ln[1:9], "START" in ln) for ln in lines if f".done_{step}" in ln and ("START" in ln or "DONE" in ln)]
            if ts and not ts[-1][1]:
                t0 = next(t for t, s in reversed(ts) if s)
                t1 = ts[-1][0]
                sec = (dt.datetime.strptime(t1, "%H:%M:%S") - dt.datetime.strptime(t0, "%H:%M:%S")).seconds
                v[key] = f"{sec / 3600:.1f} h" if step == "train_stack" else f"{sec / 60:.0f} min (last shard)"
    if a.lb:
        v["PUBLIC_LB_F05"] = a.lb

    doc = open(a.doc, encoding="utf-8").read()
    for k, val in v.items():
        doc = doc.replace("{{" + k + "}}", str(val))
    open(a.doc, "w", encoding="utf-8").write(doc)
    left = sorted(set(re.findall(r"\{\{([A-Z0-9_]+)\}\}", doc)))
    print("filled:", {k: v[k] for k in sorted(v)})
    print("still open:", left or "none")


if __name__ == "__main__":
    main()
