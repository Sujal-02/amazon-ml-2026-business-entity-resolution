"""Decision-level, label-free prior-shift correction for pairs whose CANDIDATE has no parsed state (mostly empty addresses).

Why: prepare.geo_sample kept unmatched no-state candidates at only 8% (matched ones at 100%), so in train an address-less
same-name candidate is usually a TRUE match; in every test state-shard the country's whole no-state pool is present
(pairs with such a candidate: train 10% vs US test 27-30%). Same Bayes correction as ber/shift.py on one extra stratum:
odds *= lam, lam = neg_train / neg_test per S1 (positives per S1 taken from train). France is skipped (its true matches
often lack the region label, so the equal-positives assumption fails there).

Reads work/<split>/scores_gbdt.parquet + records, writes <out_dir>_<tag>/<split>/matching_results.tsv (+ copied
candidate_pairs.tsv) for each strength, then runs `pipeline finalize` (official validator) per variant.
usage: python scripts/nostate_shift.py --config configs/local_final_tuned.yaml --variants B:1.0 C:0.5
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import shutil
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "..", "src")
sys.path.insert(0, SRC)
from ber import decision as Dm  # noqa: E402
from ber import io as bio  # noqa: E402

NL, TAB = chr(10), chr(9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--variants", nargs="+", default=["B:1.0"], help="tag:strength (lam**strength)")
    ap.add_argument("--lam-min", type=float, default=0.15)
    ap.add_argument("--skip", nargs="*", default=["france"])
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config, encoding="utf-8"))
    W, ART, OUT = cfg["work_dir"], cfg["artifacts_dir"], cfg["out_dir"]
    dcfg = {k: v for k, v in json.load(open(os.path.join(ART, "decision.json"))).items() if k != "score"}
    seen = set(json.load(open(os.path.join(ART, "train_countries.json"))))
    ub = float(cfg.get("decision_unseen_bias", -0.3))
    variants = [(v.split(":")[0], float(v.split(":")[1])) for v in a.variants]
    t0 = time.time()

    # ---- train stratum rates per country (pairs / S1 with a no-state candidate, and true ones among them)
    rec = pd.read_parquet(os.path.join(W, "train", "records.parquet"), columns=["src", "country_norm", "addr_state"])
    emp = rec["addr_state"].fillna("").to_numpy(object) == ""
    ctry = rec["country_norm"].to_numpy(object)
    n_s1 = rec[rec.src == 1].groupby("country_norm").size()
    n, pos = collections.Counter(), collections.Counter()
    for p in sorted(glob.glob(os.path.join(W, "train", "feats", "part-*.parquet"))):
        f = pd.read_parquet(p, columns=["i1", "i2", "y"])
        m = emp[f.i2.to_numpy()]
        c = ctry[f.i1.to_numpy()]
        y = f.y.to_numpy() == 1
        for cc in np.unique(c):
            sel = (c == cc) & m
            n[cc] += int(sel.sum()); pos[cc] += int((sel & y).sum())
    rates = {cc: (n[cc] / n_s1[cc], pos[cc] / n_s1[cc]) for cc in n_s1.index}
    print("train no-state stratum (pairs/S1, true pairs/S1):", {k: (round(v[0], 3), round(v[1], 3)) for k, v in rates.items()},
          flush=True)

    splits = sorted(os.path.basename(d) for d in glob.glob(os.path.join(W, "test@*#*"))
                    if os.path.exists(os.path.join(d, "scores_gbdt.parquet")))
    print("splits:", splits, flush=True)
    for split in splits:
        ts = time.time()
        r = pd.read_parquet(os.path.join(W, split, "records.parquet"), columns=["entity_id", "src", "country_norm", "addr_state"])
        eid = r.entity_id.to_numpy()
        remp = r["addr_state"].fillna("").to_numpy(object) == ""
        s1m = r.src.to_numpy() == 1
        country = str(r.country_norm[s1m].iloc[0])
        s1_all = r.entity_id[s1m].tolist()
        sc = pd.read_parquet(os.path.join(W, split, "scores_gbdt.parquet"), columns=["i1", "i2", "p"])
        i1, i2 = sc.i1.to_numpy(), sc.i2.to_numpy()
        m = remp[i2]
        n_test = m.sum() / max(1, s1m.sum())
        if country in a.skip or country not in rates:
            lam = 1.0
        else:
            rn, rp = rates[country]
            lam = float(np.clip((rn - rp) / max(n_test - rp, 1e-9), a.lam_min, 1.0))
        p0 = np.clip(sc.p.to_numpy(np.float64), 1e-7, 1 - 1e-7)
        unseen = ~np.isin(r.country_norm.to_numpy()[i1], list(seen))
        # before-stats from the existing (submission A) shard output
        cand_emp = dict(zip(eid[~s1m], remp[~s1m]))
        old = bio_read(os.path.join(OUT, split, "matching_results.tsv"))
        ol = [c for v in old.values() for c in v]
        msg = [f"{split}: country={country} nostate pairs/S1 test={n_test:.2f} lam={lam:.3f} | A: links {len(ol)} "
               f"empty {sum(1 for s in s1_all if not old.get(s)) / len(s1_all):.4f} nostate-link share "
               f"{np.mean([cand_emp.get(c, False) for c in ol]) if ol else 0:.4f}"]
        for tag, strength in variants:
            lv = lam ** strength
            pv = p0.copy()
            if lv < 1.0:
                o = pv[m] / (1 - pv[m]) * lv
                pv[m] = o / (1 + o)
            pairs = pd.DataFrame({"s1": i1, "cand": i2, "p": pv, "_unseen": unseen})
            pred_int = Dm.decide(pairs, {**dcfg, "unseen_bias": ub}, "p")
            pred = {eid[x]: [eid[b] for b in bs] for x, bs in pred_int.items()}
            od = os.path.join(OUT + "_" + tag, split)
            os.makedirs(od, exist_ok=True)
            bio.write_matching(os.path.join(od, "matching_results.tsv"), s1_all, pred)
            shutil.copyfile(os.path.join(OUT, split, "candidate_pairs.tsv"), os.path.join(od, "candidate_pairs.tsv"))
            nl = [c for v in pred.values() for c in v]
            msg.append(f"   {tag}(lam^{strength}={lv:.3f}): links {len(nl)} empty "
                       f"{sum(1 for s in s1_all if not pred.get(s)) / len(s1_all):.4f} nostate-link share "
                       f"{np.mean([cand_emp.get(c, False) for c in nl]) if nl else 0:.4f}")
        print(NL.join(msg) + f"   ({time.time() - ts:.0f}s)", flush=True)

    # ---- finalize each variant (global exclusivity + official validator) into <out>_<tag>/final
    for tag, _ in variants:
        vc = dict(cfg)
        vc["out_dir"] = OUT + "_" + tag
        vc["finalize"] = {**cfg.get("finalize", {}), "out_dir": OUT + "_" + tag + "/final"}
        vpath = os.path.join(HERE, "..", "configs", f"local_final_{tag}.yaml")
        yaml.safe_dump(vc, open(vpath, "w", encoding="utf-8"), sort_keys=False)
        rr = subprocess.run([sys.executable, "-m", "ber.pipeline", "finalize", "--config", os.path.abspath(vpath)],
                            cwd=os.path.abspath(SRC), capture_output=True, text=True)
        tail = [ln for ln in (rr.stdout + rr.stderr).splitlines() if "finalize:" in ln or "PASS" in ln or "FAIL" in ln
                or "validator" in ln or "rows" in ln]
        print(f"=== finalize {tag} (exit {rr.returncode}):" + NL + NL.join(tail[-12:]), flush=True)
    print(f"ALL DONE in {time.time() - t0:.0f}s", flush=True)


def bio_read(path):
    out = {}
    with open(path, encoding="utf-8") as f:
        next(f)
        for line in f:
            s1, _, rest = line.rstrip(NL).partition(TAB)
            out[s1] = [x for x in rest.split(",") if x]
    return out


if __name__ == "__main__":
    main()
