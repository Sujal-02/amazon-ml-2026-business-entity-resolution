"""Leave-one-country-out calibration of the unseen-country decision bias (proxy for France).

Train the GBDT stack (stage-1 LGB -> ctx/support -> stage-2 LGB) on SOURCE-country S1 only (5-fold OOF inside the
source), isotonic + decision tuned on the source OOF, then score the TARGET country with the source-only models and
report the target macro F0.5 for a grid of extra logit biases (the `decision_unseen_bias` applied to unseen
countries at test time). Also reports the in-country OOF score of the production run for reference.

usage: python scripts/loco.py --config configs/local_final_tuned.yaml --source us --target india
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from ber import decision as D  # noqa: E402
from ber import features as F  # noqa: E402
from ber import metrics as M  # noqa: E402
from ber import store  # noqa: E402
from ber.models import gbdt as G  # noqa: E402
from ber.pipeline import load_cfg  # noqa: E402

log = logging.getLogger("loco")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--source", default="us")
    ap.add_argument("--target", default="india")
    ap.add_argument("--rounds", type=int, default=800)
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = load_cfg(a.config)
    cfg = {**cfg, "gbdt_params": {"lgb": {**cfg.get("gbdt_params", {}).get("lgb", {}), "num_boost_round": a.rounds}}}
    wd = os.path.join(cfg["work_dir"], "train")
    rec = pd.read_parquet(os.path.join(wd, "records.parquet"), columns=["entity_id", "src", "country_norm", "addr_norm", "name_core"])
    ctry = rec["country_norm"].to_numpy()
    df = store.load_all(cfg, "train")
    df["_c"] = ctry[df.i1.to_numpy()]
    gt = json.load(open(os.path.join(wd, "gt.json")))
    s1c = dict(zip(rec.entity_id[rec.src == 1], rec.country_norm[rec.src == 1]))
    src = df[df._c == a.source].reset_index(drop=True)
    tgt = df[df._c == a.target].reset_index(drop=True)
    del df
    cols = G.feature_cols(src)
    log.info("source %s: %d pairs, target %s: %d pairs, %d features", a.source, len(src), a.target, len(tgt), len(cols))
    # stage 1: OOF within source + full source model for the target
    src["p1"] = G.oof_fit(src, cols, "lgb", {**cfg, "artifacts_dir": os.path.join(cfg["artifacts_dir"], "loco")}, "loco_s1")
    import pickle

    m1 = pickle.load(open(os.path.join(cfg["artifacts_dir"], "loco", "model_loco_s1.pkl"), "rb"))
    tgt["p1"] = G._predict("lgb", m1["model"], tgt[m1["cols"]])
    src = G.add_stage2_inputs(src, rec, "p1")
    tgt = G.add_stage2_inputs(tgt, rec, "p1")
    c2 = [c for c in G.feature_cols(src) if c != "p1"] + ["p1"]
    src["p2"] = G.oof_fit(src, c2, "lgb", {**cfg, "artifacts_dir": os.path.join(cfg["artifacts_dir"], "loco")}, "loco_s2")
    m2 = pickle.load(open(os.path.join(cfg["artifacts_dir"], "loco", "model_loco_s2.pkl"), "rb"))
    tgt["p2"] = G._predict("lgb", m2["model"], tgt[m2["cols"]])
    from sklearn.isotonic import IsotonicRegression

    iso = IsotonicRegression(out_of_bounds="clip").fit(src["p2"], src["y"])
    src["p"] = iso.predict(src["p2"]); tgt["p"] = iso.predict(tgt["p2"])
    s_ids = [s for s in gt if s1c.get(s) == a.source]
    t_ids = [s for s in gt if s1c.get(s) == a.target]
    sp = src[["s1_id", "cand_id", "p"]].rename(columns={"s1_id": "s1", "cand_id": "cand"})
    best = D.tune_decision(sp, gt, s_ids, "p")
    log.info("decision tuned on %s OOF: %s", a.source, best)
    tp = tgt[["s1_id", "cand_id", "p"]].rename(columns={"s1_id": "s1", "cand_id": "cand"})
    tp["_unseen"] = True
    res = {}
    for b in (0.0, -0.3, -0.5, -0.7, -1.0, -1.5, -2.0, 0.3):
        pred = D.decide(tp, {**best, "unseen_bias": b}, "p")
        r = M.report(pred, gt, t_ids)
        res[b] = {k: round(r[k], 5) for k in ("macro_f05", "f05_singletons", "f05_nonsingletons", "micro_precision", "micro_recall")}
        log.info("target %s unseen_bias %+.1f -> %s", a.target, b, res[b])
    ref = None
    oofp = os.path.join(wd, "oof.parquet")
    if os.path.exists(oofp):
        o = pd.read_parquet(oofp, columns=["s1_id", "cand_id", "p"]).rename(columns={"s1_id": "s1", "cand_id": "cand"})
        dcfg = json.load(open(os.path.join(cfg["artifacts_dir"], "decision.json")))
        ref = M.macro_f05(D.decide(o, dcfg, "p"), gt, t_ids)
    out = {"source": a.source, "target": a.target, "decision": best, "by_bias": res, "target_in_country_oof_ref": ref,
           "best_bias": max(res, key=lambda b: res[b]["macro_f05"])}
    json.dump(out, open(os.path.join(cfg["artifacts_dir"], f"loco_{a.source}_to_{a.target}.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
