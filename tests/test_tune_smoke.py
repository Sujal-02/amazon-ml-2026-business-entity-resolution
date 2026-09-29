"""Smoke tests for ber.models.tune on a synthetic work dir (no real data, seconds on CPU)."""
import json
import os
import sys
import zlib

import numpy as np
import pandas as pd
import pytest
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

pytest.importorskip("optuna")
pytest.importorskip("lightgbm")

from ber import features as F  # noqa: E402
from ber.models import gbdt as G  # noqa: E402
from ber.models import tune as T  # noqa: E402

WORDS = ["alpha", "beta", "gamma", "delta", "omega", "sigma", "kappa", "lotus", "cedar", "maple", "river", "stone"]


def _sig(x):
    return 1 / (1 + np.exp(-x))


def make_synthetic(root, n_s1=260, seed=0, with_oof=True, n_no_pairs=5):
    """work_dir/train/{feats,oof,records}.parquet + gt.json with the same columns/semantics as the pipeline."""
    rng = np.random.default_rng(seed)
    tr = os.path.join(root, "train")
    os.makedirs(tr, exist_ok=True)
    s1_ids = [f"S1-{i:05d}" for i in range(n_s1)]
    rec = []
    for i, s in enumerate(s1_ids):
        nm = " ".join(rng.choice(WORDS, 2, replace=False)) + f" {i}"
        rec.append((s, 1, nm, f"{rng.integers(1, 999)} {rng.choice(WORDS)} street", "us" if i % 3 else "in"))
    gt, owner, cid = {}, [], 0
    n_match = rng.choice([0, 1, 2, 3], size=n_s1, p=[0.15, 0.45, 0.25, 0.15])
    for i, m in enumerate(n_match):
        ids = []
        for _ in range(m):
            src = 2 + cid % 2
            e = f"S{src}-{cid:06d}"; cid += 1
            nm = rec[i][2] if rng.random() < 0.85 else "lyraverasyn"   # name replaced by a pseudo-word
            ad = rec[i][3] if rng.random() < 0.85 else ""              # empty address
            rec.append((e, src, nm, ad, rec[i][4])); owner.append(i); ids.append(e)
        gt[s1_ids[i]] = ids
    for _ in range(n_s1):  # unmatched distractors: an S1 name + inserted category word
        src = 2 + cid % 2
        e = f"S{src}-{cid:06d}"; cid += 1
        b = int(rng.integers(n_s1))
        rec.append((e, src, rec[b][2] + " motors", rec[b][3], rec[b][4])); owner.append(-1)
    rec = pd.DataFrame(rec, columns=["entity_id", "src", "name_core", "addr_norm", "country_norm"])
    rec.insert(0, "idx", np.arange(len(rec), dtype=np.int64))
    owner = np.asarray(owner)
    cand_rows = np.arange(n_s1, len(rec))
    rows = []
    for i in range(n_s1 - n_no_pairs):  # the last S1 have no candidates at all (still in gt)
        own = cand_rows[owner == i]
        own = own[rng.random(len(own)) < 0.9]  # blocking misses
        for c in np.unique(np.r_[own, rng.choice(cand_rows, size=7, replace=False)]):
            rows.append((i, int(c)))
    df = pd.DataFrame(rows, columns=["i1", "i2"]).astype(np.int64)
    y = (owner[df.i2.to_numpy() - n_s1] == df.i1.to_numpy()).astype(np.int8)
    n = len(df)
    nz = lambda s=1.0: rng.normal(0, s, n)  # noqa: E731
    df["blk_prescore"] = (0.4 + 0.3 * y + nz(0.2)).astype(np.float32)
    df["blk_rank_lsa_name_f"] = df.groupby("i1")["blk_prescore"].rank(ascending=False, method="first").astype(np.int16)
    df["blk_key_name"] = ((y == 1) & (rng.random(n) < 0.6)).astype(np.int8)
    crc = np.array([zlib.crc32(s1_ids[i].encode()) for i in df.i1], dtype=np.int64)
    df["s1_hash"] = (crc % 1000).astype(np.int16)
    df["y"] = y
    df["fold"] = (crc % 5).astype(np.int8)
    df["nm_core_tset"] = np.clip(0.55 + 0.3 * y + nz(0.15), 0, 1).astype(np.float32)
    df["ad_tset"] = np.where(rng.random(n) < 0.1, np.nan, np.clip(0.5 + 0.3 * y + nz(0.2), 0, 1)).astype(np.float32)
    df["nm_extra_c"] = np.clip(0.3 - 0.2 * y + nz(0.2), 0, None).astype(np.float32)
    df["cos_dense_x"] = rng.random(n).astype(np.float32)  # deep prefix -> must never become a GBDT feature
    eid = rec.entity_id.to_numpy()
    df["s1_id"] = eid[df.i1.to_numpy()]
    df["cand_id"] = eid[df.i2.to_numpy()]
    df.to_parquet(os.path.join(tr, "feats.parquet"), index=False)
    rec.to_parquet(os.path.join(tr, "records.parquet"), index=False)
    with open(os.path.join(tr, "gt.json"), "w") as f:
        json.dump(gt, f)
    if with_oof:
        o = df[["i1", "i2", "s1_id", "cand_id", "y", "fold"]].copy()
        o["p1"] = _sig(2.5 * (2 * y - 1) + nz(1.5)).astype(np.float32)
        o["p_blend"] = _sig(3.0 * (2 * y - 1) + nz(1.2)).astype(np.float32)
        o["p"] = o["p_blend"]
        o.sample(frac=1.0, random_state=0).to_parquet(os.path.join(tr, "oof.parquet"), index=False)  # order-free
    return df


def _cfg(root):
    return {"work_dir": str(root), "artifacts_dir": os.path.join(str(root), "artifacts"), "n_folds": 5, "seed": 0,
            "neg_downsample": {"keep_top": 5, "rest_frac": 0.5},
            "gbdt_params": {"lgb": {"num_leaves": 15, "min_data_in_leaf": 10, "num_boost_round": 300},
                            "xgb": {"max_depth": 4, "num_boost_round": 300}},
            "tune": {"space": {"lgb": {"min_data_in_leaf": {"low": 5, "high": 40}, "max_bin": 63, "lambda_l1": None},
                               "xgb": {"min_child_weight": {"low": 0.5, "high": 5.0}}}}}


def test_tune_lgb_two_trials(tmp_path):
    root = tmp_path / "work"
    make_synthetic(str(root))
    before = sorted(os.listdir(root / "train"))
    cfg = _cfg(root)
    rep = T.run(cfg, kind="lgb", trials=2, time_budget_min=5, s1_frac=0.6, max_pairs=None, eval_folds=2, threads=1,
                max_rounds=80, seed=0)
    assert sorted(os.listdir(root / "train")) == before  # work_dir is read-only for the tuner
    art = root / "artifacts"
    for fn in ("best_params_lgb.yaml", "tune_report_lgb.json", "tune_trials_lgb.csv"):
        assert (art / fn).exists(), fn
    with open(art / "best_params_lgb.yaml") as f:
        bp = yaml.safe_load(f)
    # exact structure of cfg['gbdt_params']['lgb']: a flat mapping of scalar learner params
    assert isinstance(bp, dict) and all(isinstance(v, (int, float, str, bool)) for v in bp.values())
    assert set(T.SPACES["lgb"]) - {"max_bin", "lambda_l1"} <= set(bp)
    assert bp["max_bin"] == 63 and "lambda_l1" not in bp            # fixed / removed through cfg.tune.space
    assert 5 <= bp["min_data_in_leaf"] <= 40 and isinstance(bp["num_leaves"], int)
    assert bp["num_boost_round"] == 300                               # production round cap kept
    assert not {"n_jobs", "verbose", "nthread"} & set(bp)             # runtime keys never leak into the yaml
    assert rep["trials"]["run"] == 2
    info = rep["info"]
    assert info["stage1_source"] == "oof" and info["ctx_scope"] == "closure" and info["s1_hash_threshold"] == 600
    assert "cos_dense_x" not in json.dumps(rep["top_trials"])
    assert rep["best"]["logloss"] <= rep["baseline"]["logloss"] + 0.05 and rep["best"]["logloss"] < 0.4
    fc = rep["final_check"]
    for k in ("best", "baseline", "prod_blend"):
        assert 0.0 <= fc[k]["macro_f05"] <= 1.0
    assert fc["n_s1"] > 0 and fc["outside_competitors"]
    # the yaml plugs straight into gbdt._fit exactly like cfg['gbdt_params']['lgb']
    cfg2 = T.apply_best_params(cfg, kinds=["lgb"])
    assert cfg2["gbdt_params"]["lgb"] == {**cfg["gbdt_params"]["lgb"], **bp}
    rng = np.random.default_rng(1)
    X = pd.DataFrame(rng.random((300, 3)), columns=list("abc")).astype(np.float32)
    yy = (X.a > 0.5).astype(int).to_numpy()
    m = G._fit("lgb", X[:200], yy[:200], X[200:], yy[200:],
               {"lgb": {**cfg2["gbdt_params"]["lgb"], "num_boost_round": 20, "n_jobs": 1}})
    assert G._predict("lgb", m, X).shape == (300,)


def test_closure_context_matches_full_data(tmp_path):
    """Stage-2 inputs on the S1 subsample == gbdt.add_stage2_inputs on the FULL data, restricted to the subsample."""
    root = tmp_path / "work"
    make_synthetic(str(root))
    cfg = _cfg(root)
    df, cols, ex = T.load_frame(cfg, stage=2, s1_frac=0.5, max_pairs=None, ctx_scope="full")
    assert ex["ctx_scope"] == "closure"
    tr = root / "train"
    full = pd.read_parquet(tr / "feats.parquet")
    full = full.merge(pd.read_parquet(tr / "oof.parquet", columns=["i1", "i2", "p1"]), on=["i1", "i2"], how="left")
    ref = G.add_stage2_inputs(full, pd.read_parquet(tr / "records.parquet"), "p1")
    ref_cols = [c for c in G.feature_cols(ref) if c != "p1"] + ["p1"]
    assert cols == ref_cols and "cos_dense_x" not in cols
    ref = ref[F.s1_bucket(ref, 1000) < ex["thr"]]
    m = df[["i1", "i2"] + cols].merge(ref[["i1", "i2"] + cols], on=["i1", "i2"], suffixes=("", "_ref"))
    assert len(m) == len(df) == len(ref)
    for c in cols:
        np.testing.assert_allclose(m[c].to_numpy(np.float64), m[c + "_ref"].to_numpy(np.float64), rtol=1e-6,
                                   atol=1e-7, equal_nan=True, err_msg=c)
    # the lean 'sub' scope is an approximation: candidate-side competition only sees sampled S1
    df2, cols2, ex2 = T.load_frame(cfg, stage=2, s1_frac=0.5, max_pairs=None, ctx_scope="sub")
    assert ex2["ctx_scope"] == "subsample" and cols2 == cols


def test_refit_fallback_xgb_and_stage1(tmp_path):
    pytest.importorskip("xgboost")
    root = tmp_path / "work"
    make_synthetic(str(root), with_oof=False)
    cfg = _cfg(root)
    rep = T.run(cfg, kind="xgb", trials=2, time_budget_min=5, s1_frac=1.0, max_pairs=None, eval_folds=1, threads=1,
                max_rounds=60)
    assert rep["info"]["stage1_source"] == "refit" and "prod_blend" not in rep["final_check"]
    with open(root / "artifacts" / "best_params_xgb.yaml") as f:
        bp = yaml.safe_load(f)
    assert bp["num_boost_round"] == 300 and bp["max_depth"] >= 4 and "nthread" not in bp
    assert 0.5 <= bp["min_child_weight"] <= 5.0
    rep1 = T.run(cfg, kind="lgb", stage=1, trials=2, time_budget_min=5, s1_frac=1.0, max_pairs=None, eval_folds=1,
                 threads=1, max_rounds=60, baseline=False)
    assert (root / "artifacts" / "best_params_stage1_lgb.yaml").exists()
    assert rep1["final_check"] == {} and rep1["info"]["n_features"] == 5
