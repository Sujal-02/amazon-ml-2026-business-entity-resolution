"""Two-stage GBDT matcher with out-of-fold stacking, calibration and macro-F0.5 decision tuning.

stage 1 : LightGBM on pair + blocking features                          -> p1 (OOF on train)
stage 2 : {LightGBM, XGBoost, CatBoost} on pair + ctx(p1) + support(p1)  -> p2_m (OOF on train)
          (+ any external score columns listed in cfg.stack.extra_scores, e.g. cross-encoder OOF logits)
blend   : mean of logits over stage-2 models (weights tunable) -> isotonic calibration on OOF
decide  : exclusivity + expected-F0.5 per S1, config grid-searched on OOF macro F0.5 over ALL train S1
Folds are S1-level (features.fold_of) and identical to the ones used for the out-of-fold target encoding.
"""
from __future__ import annotations

import json
import logging
import os
import pickle
import time
from typing import Dict, List

import numpy as np
import pandas as pd

from .. import decision as Dm
from .. import features as F
from .. import metrics as M

log = logging.getLogger(__name__)

NON_FEATURES = {"i1", "i2", "y", "fold", "s1_id", "cand_id", "blk_prescore", "s1_hash"}


def _wp(cfg, split, name):
    d = os.path.join(cfg["work_dir"], split); os.makedirs(d, exist_ok=True); return os.path.join(d, name)


def _ap(cfg, name):
    os.makedirs(cfg["artifacts_dir"], exist_ok=True); return os.path.join(cfg["artifacts_dir"], name)


#: columns produced by models trained on deep_train_folds (inflated on those folds) -> only the holdout stacker uses them
DEEP_PREFIXES = ("cos_dense", "blk_sim_dense", "blk_rank_dense", "ce_", "llm_")


def feature_cols(df: pd.DataFrame, extra_exclude=(), allow_deep: bool = False) -> List[str]:
    ex = NON_FEATURES | set(extra_exclude)
    return [c for c in df.columns if c not in ex and df[c].dtype.kind in "fiub"
            and (allow_deep or not c.startswith(DEEP_PREFIXES))]


def config_excluded(cfg: dict, columns) -> set:
    """config `exclude_features`: exact column names, or prefixes ending in '*' (e.g. 'blk_rank_*'), dropped from both
    stages (e.g. pool-size-dependent features that shift between the train sample and the test shards). Predict needs
    no change: every saved model carries its own column list."""
    out = set()
    for pat in cfg.get("exclude_features") or []:
        out |= {c for c in columns if (c.startswith(pat[:-1]) if pat.endswith("*") else c == pat)}
    return out


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# ----------------------------------------------------------------------------------------------------------------
# learners
# ----------------------------------------------------------------------------------------------------------------
def _fit(kind: str, X: pd.DataFrame, y: np.ndarray, Xv: pd.DataFrame, yv: np.ndarray, params: dict):
    if kind == "lgb":
        import lightgbm as lgb

        p = dict(objective="binary", learning_rate=0.05, num_leaves=127, min_data_in_leaf=100, feature_fraction=0.8,
                 bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, max_bin=255, verbose=-1, n_jobs=-1, seed=0)
        p.update(params.get("lgb", {}))
        if "num_threads" in p:
            p.pop("n_jobs", None)
        rounds = p.pop("num_boost_round", 3000)
        m = lgb.train(p, lgb.Dataset(X, y), rounds, valid_sets=[lgb.Dataset(Xv, yv)],
                      callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)])
        return m
    if kind == "xgb":
        import xgboost as xgb

        p = dict(objective="binary:logistic", eval_metric="logloss", tree_method="hist", max_depth=8, eta=0.05,
                 subsample=0.8, colsample_bytree=0.8, min_child_weight=5, reg_lambda=1.0, nthread=-1, seed=0)
        try:
            import torch

            if torch.cuda.is_available():
                p["device"] = "cuda"
        except Exception:
            pass
        p.update(params.get("xgb", {}))
        rounds = p.pop("num_boost_round", 3000)
        dtr = xgb.DMatrix(X, y); dv = xgb.DMatrix(Xv, yv)
        return xgb.train(p, dtr, rounds, evals=[(dv, "v")], early_stopping_rounds=100, verbose_eval=False)
    if kind == "cat":
        from catboost import CatBoostClassifier

        p = dict(iterations=3000, learning_rate=0.08, depth=8, l2_leaf_reg=3, loss_function="Logloss",
                 random_seed=0, verbose=0, od_type="Iter", od_wait=100, thread_count=-1)
        try:
            import torch

            if torch.cuda.is_available():
                p["task_type"] = "GPU"
        except Exception:
            pass
        p.update(params.get("cat", {}))
        m = CatBoostClassifier(**p)
        m.fit(X, y, eval_set=(Xv, yv), use_best_model=True)
        return m
    raise ValueError(kind)


def _predict(kind: str, m, X: pd.DataFrame) -> np.ndarray:
    if kind == "lgb":
        return m.predict(X, num_iteration=m.best_iteration)
    if kind == "xgb":
        import xgboost as xgb

        return m.predict(xgb.DMatrix(X), iteration_range=(0, m.best_iteration + 1))
    if kind == "cat":
        return m.predict_proba(X)[:, 1]
    raise ValueError(kind)


def _downsample(df: pd.DataFrame, cfg: dict, seed: int = 0) -> np.ndarray:
    """Training-fold mask: all positives, hard negatives (top-ranked by prescore), random share of the rest."""
    ds = cfg.get("neg_downsample", {})
    rng = np.random.default_rng(seed)
    pos = df["y"].to_numpy() == 1
    rank = df.groupby("i1")["blk_prescore"].rank(ascending=False, method="first").to_numpy()
    hard = rank <= ds.get("keep_top", 15)
    rnd = rng.random(len(df)) < ds.get("rest_frac", 0.3)
    m = pos | hard | rnd
    frac = cfg.get("gbdt_train_s1_frac", 1.0)  # fit on a hashed subset of S1 entities (OOF still covers all rows)
    if frac < 1.0:
        m &= F.s1_bucket(df, 1000) < int(frac * 1000)
    return m


def oof_fit(df: pd.DataFrame, cols: List[str], kind: str, cfg: dict, tag: str) -> np.ndarray:
    """K-fold OOF predictions over ALL rows of df; also fits and saves a full-data model."""
    k = int(df["fold"].max()) + 1
    oof = np.zeros(len(df), np.float32)
    params = cfg.get("gbdt_params", {})
    for fo in range(k):
        t0 = time.time()
        va = (df["fold"] == fo).to_numpy()
        tr = ~va & _downsample(df, cfg, fo)
        # early-stopping set = 20% of the training folds' S1 (the held-out fold stays untouched)
        es = tr & (F.s1_bucket(df, 10) >= 8)
        fit_m = tr & ~es
        m = _fit(kind, df.loc[fit_m, cols], df.loc[fit_m, "y"].to_numpy(), df.loc[es, cols], df.loc[es, "y"].to_numpy(), params)
        oof[va] = _predict(kind, m, df.loc[va, cols])
        log.info("[%s] fold %d done (%.0fs)", tag, fo, time.time() - t0)
    # full model for inference
    tr = _downsample(df, cfg, 99)
    es = tr & (F.s1_bucket(df, 10) >= 9)
    m = _fit(kind, df.loc[tr & ~es, cols], df.loc[tr & ~es, "y"].to_numpy(), df.loc[es, cols], df.loc[es, "y"].to_numpy(), params)
    with open(_ap(cfg, f"model_{tag}.pkl"), "wb") as f:
        pickle.dump({"kind": kind, "model": m, "cols": cols}, f)
    return oof


def blend_weights(L: np.ndarray, y: np.ndarray, step: float = 0.1) -> np.ndarray:
    """Simplex grid search of logit-blend weights minimising OOF logloss. L: (n_models, n_rows) logits."""
    k = L.shape[0]
    if k == 1:
        return np.ones(1)
    rng = np.random.default_rng(0)
    idx = rng.choice(L.shape[1], min(L.shape[1], 3_000_000), replace=False)
    Ls, ys = L[:, idx], y[idx].astype(np.float64)
    grid = [w for w in _simplex(k, int(round(1 / step)))]
    best, bw = np.inf, np.full(k, 1.0 / k)
    for w in grid:
        z = np.tensordot(np.asarray(w), Ls, axes=1)
        ll = np.mean(np.logaddexp(0, z) - ys * z)
        if ll < best:
            best, bw = ll, np.asarray(w, dtype=np.float64)
    return bw


def _simplex(k: int, n: int):
    """All weight vectors (c_1..c_k)/n with non-negative integer c summing to n."""
    def comp(k, n):
        if k == 1:
            yield (n,)
            return
        for i in range(n + 1):
            for rest in comp(k - 1, n - i):
                yield (i,) + rest
    for c in comp(k, n):
        yield tuple(x / n for x in c)


def add_stage2_inputs(df: pd.DataFrame, rec: pd.DataFrame, score_col: str) -> pd.DataFrame:
    ctx = F.context_features(df, score_col, "ctx")
    sup = F.support_features(df, rec, score_col, prefix="sup")
    return pd.concat([df, ctx, sup], axis=1)


# ----------------------------------------------------------------------------------------------------------------
def train_stack(cfg: dict) -> dict:
    from .. import store

    t0 = time.time()
    df = store.load_all(cfg, "train")
    rec = pd.read_parquet(_wp(cfg, "train", "records.parquet"), columns=["entity_id", "src", "country_norm", "addr_norm", "name_core"])
    with open(_wp(cfg, "train", "gt.json")) as f:
        gt = json.load(f)
    s1_ids = list(gt.keys())
    with open(_ap(cfg, "train_countries.json"), "w") as f:  # for the unseen-country precision guard at predict time
        json.dump(sorted(set(rec.loc[rec.src == 1, "country_norm"])), f)
    extra = cfg.get("stack", {}).get("extra_scores", [])
    for e in extra:  # e.g. cross-encoder OOF logits written by models/cross_encoder.py
        path = _wp(cfg, "train", f"score_{e}.parquet")
        if os.path.exists(path):
            s = pd.read_parquet(path)
            df = df.merge(s, on=["i1", "i2"], how="left")
    base_cols = feature_cols(df, extra_exclude=config_excluded(cfg, df.columns))
    log.info("train pairs=%d pos=%d S1=%d features=%d", len(df), int(df.y.sum()), len(s1_ids), len(base_cols))
    if cfg.get("exclude_features"):
        log.info("exclude_features %s -> dropped %s", cfg["exclude_features"], sorted(config_excluded(cfg, df.columns)))
    df["p1"] = oof_fit(df, base_cols, "lgb", cfg, "stage1")
    log.info("stage1 OOF logloss-ish check: mean p1 on pos=%.3f neg=%.3f", df.p1[df.y == 1].mean(), df.p1[df.y == 0].mean())
    df = add_stage2_inputs(df, rec, "p1")
    s2_cols = feature_cols(df, extra_exclude=config_excluded(cfg, df.columns))
    s2_cols = [c for c in s2_cols if c != "p1"] + ["p1"]
    models = cfg.get("stack", {}).get("stage2_models", ["lgb", "xgb", "cat"])
    logits = []
    for kind in models:
        p = oof_fit(df, s2_cols, kind, cfg, f"stage2_{kind}")
        df[f"p2_{kind}"] = p
        logits.append(logit(p))
        _eval(cfg, df, gt, s1_ids, rec, f"p2_{kind}", tune=False)
    weights = blend_weights(np.vstack(logits), df["y"].to_numpy())
    log.info("stage-2 blend weights (OOF logloss-optimal): %s", dict(zip(models, np.round(weights, 2))))
    df["p_blend"] = sigmoid(np.tensordot(weights, np.vstack(logits), axes=1)).astype(np.float32)
    from sklearn.isotonic import IsotonicRegression

    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(df["p_blend"], df["y"])
    df["p"] = iso.predict(df["p_blend"]).astype(np.float32)
    with open(_ap(cfg, "calibrator.pkl"), "wb") as f:
        pickle.dump({"iso": iso, "models": models, "weights": weights.tolist(), "s2_cols": s2_cols, "base_cols": base_cols}, f)
    res = _eval(cfg, df, gt, s1_ids, rec, "p", tune=True)
    df[["i1", "i2", "s1_id", "cand_id", "y", "fold", "p1", "p_blend", "p"] + [f"p2_{k}" for k in models]].to_parquet(
        _wp(cfg, "train", "oof.parquet"))
    log.info("train_stack done in %.0fs: %s", time.time() - t0, res)
    return res


def _eval(cfg, df, gt, s1_ids, rec, col, tune=True):
    pairs = df[["s1_id", "cand_id", col]].rename(columns={"s1_id": "s1", "cand_id": "cand", col: "p"})
    country = dict(zip(rec.entity_id[rec.src == 1], rec.country_norm[rec.src == 1]))
    if tune:
        best = Dm.tune_decision(pairs, gt, s1_ids, "p")
        with open(_ap(cfg, "decision.json"), "w") as f:
            json.dump(best, f, indent=1)
        pred = Dm.decide(pairs, best, "p")
    else:
        best = {"exclusive": True, "mode": "expected_f"}
        pred = Dm.decide(pairs, best, "p")
    rep = M.report(pred, gt, s1_ids, country)
    rep["decision"] = best
    log.info("OOF %s: %s", col, json.dumps(rep))
    with open(_wp(cfg, "train", f"oof_report_{col}.json"), "w") as f:
        json.dump(rep, f, indent=1)
    return rep


# ----------------------------------------------------------------------------------------------------------------
def _load_model(cfg, tag):
    with open(_ap(cfg, f"model_{tag}.pkl"), "rb") as f:
        return pickle.load(f)


def predict_stack(cfg: dict, split: str = "test") -> None:
    """Memory-bounded inference over feature parts:
    pass 1 (per part) stage-1 p1 -> global slim (i1,i2,p1) -> candidate-side competition features (global) ->
    pass 2 (per part) support features + stage-2 models -> blend -> isotonic -> decision on integer ids."""
    from .. import io as bio
    from .. import store

    t0 = time.time()
    rec = pd.read_parquet(_wp(cfg, split, "records.parquet"), columns=["entity_id", "src", "addr_norm", "name_core", "country_norm"])
    eid = rec["entity_id"].to_numpy()
    with open(_ap(cfg, "calibrator.pkl"), "rb") as f:
        cal = pickle.load(f)
    m1 = _load_model(cfg, "stage1")
    m2 = {k: _load_model(cfg, f"stage2_{k}") for k in cal["models"]}
    extra = [e for e in cfg.get("stack", {}).get("extra_scores", []) if os.path.exists(_wp(cfg, split, f"score_{e}.parquet"))]

    def with_extra(part):
        for e in extra:
            part = part.merge(pd.read_parquet(_wp(cfg, split, f"score_{e}.parquet")), on=["i1", "i2"], how="left")
        return part

    slim = []
    for k, part in store.iter_parts(cfg, split):
        part = with_extra(part)
        slim.append(pd.DataFrame({"i1": part.i1.to_numpy(), "i2": part.i2.to_numpy(),
                                  "p1": _predict(m1["kind"], m1["model"], part[m1["cols"]]).astype(np.float32)}))
        log.info("predict %s: stage1 part %d (%.0fs)", split, k, time.time() - t0)
    slim = pd.concat(slim, ignore_index=True)
    ctx = F.context_features(slim, "p1", "ctx")
    outs, off = [], 0
    for k, part in store.iter_parts(cfg, split):
        part = with_extra(part)
        n = len(part)
        part["p1"] = slim["p1"].to_numpy()[off:off + n]
        c = ctx.iloc[off:off + n].reset_index(drop=True)
        off += n
        sup = F.support_features(part[["i1", "i2", "p1"]], rec, "p1", prefix="sup")
        X = pd.concat([part.reset_index(drop=True), c, sup.reset_index(drop=True)], axis=1)
        o = pd.DataFrame({"i1": X.i1.to_numpy(), "i2": X.i2.to_numpy(), "p1": X.p1.to_numpy()})
        for kk, m in m2.items():
            o[f"p2_{kk}"] = _predict(m["kind"], m["model"], X[m["cols"]]).astype(np.float32)
        w = np.asarray(cal.get("weights", [1.0 / len(m2)] * len(m2)))
        o["p_blend"] = sigmoid(np.tensordot(w, np.vstack([logit(o[f"p2_{kk}"].to_numpy()) for kk in cal["models"]]), axes=1)).astype(np.float32)
        o["p"] = cal["iso"].predict(o["p_blend"]).astype(np.float32)
        outs.append(o)
        log.info("predict %s: stage2 part %d (%.0fs)", split, k, time.time() - t0)
        del X, part, sup, c
    sc = pd.concat(outs, ignore_index=True)
    sc["s1_id"] = eid[sc.i1.to_numpy()]
    sc["cand_id"] = eid[sc.i2.to_numpy()]
    # label-free train->test prior-shift correction on near-twin strata (see ber/shift.py); p_raw keeps the original
    sc["p_raw"] = sc["p"]
    sp = _ap(cfg, "strata_train.json")
    if cfg.get("shift_correction", True) and os.path.exists(sp):
        from .. import shift as SH

        with open(sp) as f:
            tr = json.load(f)
        st = store.load_all(cfg, split, columns=SH.STRATA_COLS)
        if len(st) == len(sc) and np.array_equal(st["i1"].to_numpy(), sc["i1"].to_numpy()):
            sid = SH.strata(st)
            n_s1 = max(1, int((rec.src == 1).sum()))
            test_counts = pd.DataFrame({"s": sid[sid > 0]}).groupby("s").size().to_frame("n") / n_s1
            c_major = pd.Series(rec.loc[rec.src == 1, "country_norm"]).mode()
            country = str(c_major.iloc[0]) if len(c_major) else ""
            trc = tr.get(country) or tr["_pooled"]
            train_counts = pd.DataFrame.from_dict(trc, orient="index")
            train_counts.index = train_counts.index.astype(int)
            lam = SH.fit_lambdas(train_counts, test_counts, floor=float(cfg.get("shift_floor", 0.1)),
                                 lam_min=float(cfg.get("shift_lam_min", 0.02)))
            sc["p"] = SH.apply(sc["p_raw"].to_numpy(), sid, lam)
            summ = {SH.describe(s): round(v, 3) for s, v in sorted(lam.items()) if abs(v - 1) > 0.05}
            tot_tr = float(train_counts["n"].sum()); tot_te = float(test_counts["n"].sum())
            log.info("shift %s (%s): near-twin pairs/S1 train %.3f test %.3f; lambdas != 1: %s", split, country if country in tr else "pooled",
                     tot_tr, tot_te, summ)
            with open(_wp(cfg, split, "shift_lambda.json"), "w") as f:
                json.dump({"country": country, "train_per_s1": tot_tr, "test_per_s1": tot_te,
                           "lambda": {str(k): v for k, v in lam.items()}}, f, indent=1)
        else:
            log.warning("shift %s: strata rows not aligned with scores -> no correction", split)
    sc.to_parquet(_wp(cfg, split, "scores_gbdt.parquet"))
    with open(_ap(cfg, "decision.json")) as f:
        dcfg = json.load(f)
    pairs = sc[["i1", "i2", "p"]].rename(columns={"i1": "s1", "i2": "cand"})
    tc_path = _ap(cfg, "train_countries.json")
    if os.path.exists(tc_path):
        with open(tc_path) as f:
            seen = set(json.load(f))
        unseen = ~np.isin(rec["country_norm"].to_numpy()[sc.i1.to_numpy()], list(seen))
        pairs["_unseen"] = unseen
        dcfg = {**dcfg, "unseen_bias": float(cfg.get("decision_unseen_bias", -0.7))}
        log.info("predict %s: %d pairs of unseen-country S1 -> unseen_bias %.2f", split, int(unseen.sum()), dcfg["unseen_bias"])
    pred_int = Dm.decide(pairs, dcfg, "p")
    pred = {eid[a]: [eid[b] for b in bs] for a, bs in pred_int.items()}
    cand_int = sc.groupby("i1")["i2"].agg(list).to_dict()
    cands = {eid[a]: [eid[b] for b in bs] for a, bs in cand_int.items()}
    s1_all = rec.entity_id[rec.src == 1].tolist()
    out_dir = os.path.join(cfg["out_dir"], split)  # per split/shard; merged by `pipeline finalize`
    bio.write_matching(os.path.join(out_dir, "matching_results.tsv"), s1_all, pred)
    bio.write_candidates(os.path.join(out_dir, "candidate_pairs.tsv"), s1_all, cands)
    # uncorrected variant (same decision, p_raw) for A/B comparison; finalize merges it with finalize.variant: nocorr
    alt = pairs.assign(p=sc["p_raw"].to_numpy())
    alt_int = Dm.decide(alt, dcfg, "p")
    bio.write_matching(os.path.join(out_dir, "matching_results_nocorr.tsv"), s1_all,
                       {eid[a]: [eid[b] for b in bs] for a, bs in alt_int.items()})
    n_nonempty = sum(1 for s in s1_all if pred.get(s))
    log.info("predict %s: %d S1, %d non-empty, %d matched ids, in %.0fs", split, len(s1_all), n_nonempty,
             sum(len(v) for v in pred.values()), time.time() - t0)
