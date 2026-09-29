"""Blend the Kaggle cross-encoder (CE) with the GBDT on the uncertain band.

--eval : cross-fitted logistic blend  y ~ w1*logit(p_gbdt) + w2*ce + b  on held-out train folds 3-4 (CE trained on 0-2),
         then macro F0.5 on folds 3-4 S1s: baseline p vs blended p (same tuned decision) -> writes blend.json
--apply: blended p on every test shard's band (shift correction kept as an additive log-lambda), re-decide,
         write <out_dir>_CE/<split>/, run pipeline finalize (official validator).
"""
from __future__ import annotations

import argparse
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
from ber import metrics as M  # noqa: E402
from ber.features import fold_of  # noqa: E402


def lg(p):
    p = np.clip(np.asarray(p, np.float64), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def sig(z):
    return 1 / (1 + np.exp(-z))


def fit_lr(X, y):
    from sklearn.linear_model import LogisticRegression
    m = LogisticRegression(C=100.0, max_iter=500).fit(X, y)
    return m.coef_[0].tolist(), float(m.intercept_[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ce-dir", default="D:/er_data/ce_out")
    ap.add_argument("--mode", choices=["eval", "apply"], required=True)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--no-finalize", action="store_true")
    a = ap.parse_args()
    global DIRS
    DIRS = a.ce_dir.split(",")
    cfg = yaml.safe_load(open(a.config, encoding="utf-8"))
    W, ART, OUT = cfg["work_dir"], cfg["artifacts_dir"], cfg["out_dir"]
    dcfg = {k: v for k, v in json.load(open(os.path.join(ART, "decision.json"))).items() if k != "score"}
    t0 = time.time()

    if a.mode == "eval":
        cm = pd.read_parquet(os.path.join(W, "train", "ce_cal_map.parquet"))
        cols = []
        for j, dd in enumerate(DIRS):
            ce = pd.read_parquet(os.path.join(dd, "cal_ce.parquet")).set_index("pid").ce
            cm[f"ce{j}"] = ce.reindex(np.arange(len(cm))).to_numpy(); cols.append(f"ce{j}")
        X = np.c_[lg(cm.p), cm[cols].to_numpy(np.float64)]
        y = cm.y.to_numpy()
        pb = np.empty(len(cm))
        for f_fit, f_app in ((3, 4), (4, 3)):
            w, b = fit_lr(X[cm.fold.to_numpy() == f_fit], y[cm.fold.to_numpy() == f_fit])
            sel = cm.fold.to_numpy() == f_app
            pb[sel] = sig(X[sel] @ np.array(w) + b)
        w_all, b_all = fit_lr(X, y)
        ll = lambda p: float(-np.mean(y * np.log(np.clip(p, 1e-7, 1)) + (1 - y) * np.log(np.clip(1 - p, 1e-7, 1))))
        print(f"band pairs {len(cm):,} | logloss GBDT {ll(cm.p.to_numpy()):.4f}  blend {ll(pb):.4f} | "
              f"blend weights (all) w={np.round(w_all, 3)} b={b_all:.3f}", flush=True)
        cm["pb"] = pb
        oof = pd.read_parquet(os.path.join(W, "train", "oof.parquet"), columns=["i1", "i2", "s1_id", "cand_id", "fold", "p"])
        sub = oof[oof.fold.isin([3, 4])].merge(cm[["i1", "i2", "pb"]], on=["i1", "i2"], how="left")
        sub["pn"] = np.where(sub.pb.notna(), sub.pb, sub.p)
        gt = json.load(open(os.path.join(W, "train", "gt.json")))
        gids = np.array(list(gt), dtype=object)
        s1_ids = gids[np.isin(fold_of(gids, 5), [3, 4])].tolist()
        res = {}
        for col in ("p", "pn"):
            pairs = sub[["s1_id", "cand_id", col]].rename(columns={"s1_id": "s1", "cand_id": "cand", col: "p"})
            pred = Dm.decide(pairs, dcfg, "p")
            res[col] = M.macro_f05(pred, gt, s1_ids)
        print(f"macro F0.5 on folds 3-4 ({len(s1_ids):,} S1): GBDT {res['p']:.5f}  ->  GBDT+CE {res['pn']:.5f}  "
              f"(gain {res['pn'] - res['p']:+.5f})  [{time.time() - t0:.0f}s]", flush=True)
        json.dump({"w": w_all, "b": b_all, "gain": res["pn"] - res["p"], "base": res["p"], "blend": res["pn"]},
                  open(os.path.join(DIRS[0], "blend.json"), "w"))
        return

    bl = json.load(open(os.path.join(DIRS[0], "blend.json")))
    w, b = np.array(bl["w"]), bl["b"]
    seen = set(json.load(open(os.path.join(ART, "train_countries.json"))))
    ub = float(cfg.get("decision_unseen_bias", -0.3))
    tmap = pd.read_parquet(os.path.join(W, "ce_te_map.parquet"))
    for j, dd in enumerate(DIRS):
        tce = pd.read_parquet(os.path.join(dd, "te_ce.parquet")).rename(columns={"ce": f"ce{j}"})
        tmap = tmap.merge(tce, on="pid", how="inner")
    cols = [f"ce{j}" for j in range(len(DIRS))]
    print(f"test CE scores: {len(tmap):,}", flush=True)
    OUTC = OUT + "_CE" + (a.tag or str(len(DIRS)))
    for split, g in tmap.groupby("split"):
        r = pd.read_parquet(os.path.join(W, split, "records.parquet"), columns=["entity_id", "src", "country_norm"])
        eid = r.entity_id.to_numpy()
        s1_all = r.entity_id[r.src == 1].tolist()
        sc = pd.read_parquet(os.path.join(W, split, "scores_gbdt.parquet"), columns=["i1", "i2", "p", "p_raw"])
        sc = sc.merge(g[["i1", "i2"] + cols], on=["i1", "i2"], how="left")
        has = sc[cols[0]].notna().to_numpy()
        z = w[0] * lg(sc.p_raw) + sc[cols].fillna(0).to_numpy(np.float64) @ w[1:] + b + (lg(sc.p) - lg(sc.p_raw))
        pn = np.where(has, sig(z), sc.p.to_numpy(np.float64))
        unseen = ~np.isin(r.country_norm.to_numpy()[sc.i1.to_numpy()], list(seen))
        pairs = pd.DataFrame({"s1": sc.i1.to_numpy(), "cand": sc.i2.to_numpy(), "p": pn, "_unseen": unseen})
        pi = Dm.decide(pairs, {**dcfg, "unseen_bias": ub}, "p")
        pred = {eid[x]: [eid[c] for c in cs] for x, cs in pi.items()}
        od = os.path.join(OUTC, split)
        os.makedirs(od, exist_ok=True)
        bio.write_matching(os.path.join(od, "matching_results.tsv"), s1_all, pred)
        shutil.copyfile(os.path.join(OUT, split, "candidate_pairs.tsv"), os.path.join(od, "candidate_pairs.tsv"))
        print(f"{split}: band {int(has.sum()):,}  links {sum(len(v) for v in pred.values()):,}  "
              f"empty {sum(1 for s in s1_all if not pred.get(s)) / len(s1_all):.4f}", flush=True)
    if a.no_finalize:
        print("ALL DONE (no finalize)", flush=True); return
    vc = dict(cfg)
    vc["out_dir"] = OUTC
    vc["finalize"] = {**cfg.get("finalize", {}), "out_dir": OUTC + "/final"}
    vpath = os.path.join(HERE, "..", "configs", f"local_final_CE{a.tag or len(DIRS)}.yaml")
    yaml.safe_dump(vc, open(vpath, "w", encoding="utf-8"), sort_keys=False)
    rr = subprocess.run([sys.executable, "-m", "ber.pipeline", "finalize", "--config", os.path.abspath(vpath)],
                        cwd=os.path.abspath(SRC), capture_output=True, text=True)
    tail = [ln for ln in (rr.stdout + rr.stderr).splitlines() if "finalize:" in ln or "PASS" in ln or "FAIL" in ln or "rows" in ln]
    print(f"=== finalize CE (exit {rr.returncode})\n" + "\n".join(tail[-10:]), flush=True)
    print("ALL DONE", round(time.time() - t0), "s", flush=True)


if __name__ == "__main__":
    main()
