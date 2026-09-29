"""Holdout stacker: combines GBDT OOF scores with deep-model scores (cross-encoders, LLM, dense-retrieval cosines).

Deep models are trained on S1 folds `deep_train_folds`; this stacker is fit ONLY on the remaining folds (default 3,4),
with an inner 2-way split for honest OOF evaluation, then refit on all holdout rows for test inference.
Inputs per pair: logit(p_blend), logit(p2_*), ce_*/llm_* logits (NaN when the pair was outside the scoring band),
dense cosines, plus competition features recomputed on the blended score, plus the strongest raw pair features.
Output: calibrated p_final -> decision layer tuned on the holdout S1 set (exact macro F0.5 incl. singletons).

  python -m ber.models.stack train   --config configs/aws.yaml
  python -m ber.models.stack predict --config configs/aws.yaml
"""
from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import pickle

import numpy as np
import pandas as pd

from .. import decision as Dm
from .. import features as F
from .. import metrics as M
from .gbdt import _fit, _predict, logit

log = logging.getLogger(__name__)

RAW_KEEP = ["nm_core_tset", "nm_idf_jac", "nm_max_extra_c", "nm_extra_harm_max", "nm_extra_unknown",
            "ad_num_first_eq", "ad_house_eq", "ad_tset", "ad_empty_c", "blk_n_channels"]


def _wp(cfg, split, name):
    return os.path.join(cfg["work_dir"], split, name)


S1_GROUPED = ("_rank_s1", "_gap_to_best_s1", "_margin_next_c", "_n_c_for_s1", "_n_strong_s1", "_sum_s1")


def _assemble(cfg, split, expected_scores=None):
    """GBDT scores + selected raw feats + deep scores. Candidate-side competition is computed over ALL S1 of the split
    (same regime as test) BEFORE the train holdout filter; post-filter competition features are S1-grouped only."""
    import pyarrow.parquet as pq

    base = "oof.parquet" if split == "train" else "scores_gbdt.parquet"
    df = pd.read_parquet(_wp(cfg, split, base))
    from .. import store

    have = set(store.columns(cfg, split))
    keep = ["i1", "i2"] + [c for c in have if c in RAW_KEEP or c.startswith(("cos_dense", "blk_n_dense"))]
    df = df.merge(store.load_all(cfg, split, columns=keep), on=["i1", "i2"], how="left")
    paths = sorted(glob.glob(_wp(cfg, split, "score_*.parquet")))
    if expected_scores is not None:  # test must provide exactly the deep scores the stacker was trained with
        names = {os.path.basename(p)[len("score_"):-len(".parquet")] for p in paths}
        missing = sorted(set(expected_scores) - names)
        if missing:
            raise RuntimeError(f"missing deep score files for {split}: {missing} (a deep job failed?)")
    for path in paths:
        df = df.merge(pd.read_parquet(path), on=["i1", "i2"], how="left")
    df["lg_blend"] = logit(df["p_blend"].to_numpy())
    df = pd.concat([df, F.context_features(df[["i1", "i2", "p_blend"]], "p_blend", "gctx")], axis=1)
    if split == "train":
        deep_folds = set(cfg.get("deep_train_folds", [0, 1, 2]))
        inf = df["fold"].isin(deep_folds).to_numpy()
        # best calibrated GBDT score among competing S1s of the deep-train folds (needed for honest exclusivity)
        oth = df.loc[inf].groupby("i2")["p"].max()
        df = df[~inf].reset_index(drop=True)
        df["oth_fold_best_p"] = df["i2"].map(oth).fillna(0.0).astype(np.float32)
    p2 = [c for c in df.columns if c.startswith("p2_")]
    for c in p2:
        df[f"lg_{c}"] = logit(df[c].to_numpy())
    deep = [c for c in df.columns if c.startswith(("ce_", "llm_"))]
    # cross-encoders skip only confident negatives (p below their band) -> very negative logit;
    # LLMs score the middle band only -> outside it fall back to the GBDT blend logit (both tails)
    for c in deep:
        df[c] = df[c].fillna(df["lg_blend"]) if c.startswith("llm_") else df[c].fillna(-12.0)
    df["deep_mean"] = df[deep].mean(axis=1) if deep else np.nan
    df["combo"] = df[["lg_blend"] + deep].mean(axis=1)
    ctx = F.context_features(df.assign(_s=1 / (1 + np.exp(-df["combo"]))), "_s", "sctx")
    ctx = ctx[[c for c in ctx.columns if c.endswith(S1_GROUPED)]]
    return pd.concat([df, ctx], axis=1)


def _cols(df):
    return [c for c in df.columns if c.startswith(("lg_", "ce_", "llm_", "sctx_", "gctx_", "cos_dense", "blk_n_dense"))
            or c in RAW_KEEP or c in ("deep_mean", "combo")]


def _deep_score_names(cfg, split="train"):
    return sorted(os.path.basename(p)[len("score_"):-len(".parquet")] for p in glob.glob(_wp(cfg, split, "score_*.parquet")))


def train(cfg):
    df = _assemble(cfg, "train")
    with open(_wp(cfg, "train", "gt.json")) as f:
        gt = json.load(f)
    hold = set(range(cfg.get("n_folds", 5))) - set(cfg.get("deep_train_folds", [0, 1, 2]))
    fo = F.fold_of(list(gt.keys()), cfg.get("n_folds", 5))
    s1_ids = [s for s, k in zip(gt.keys(), fo) if k in hold]
    cols = _cols(df)
    log.info("stacker: %d holdout pairs, %d S1, cols=%s", len(df), len(s1_ids), cols)
    sid = df["s1_id"].to_numpy()
    inner = F.fold_of(sid, 2)
    es_bucket = F.fold_of(np.char.add(sid.astype(str), "|es"), 5)  # independent of the fold hash -> 20% ES, 80% fit
    oof = np.zeros(len(df), np.float32)
    params = cfg.get("stack_params", {"lgb": {"num_leaves": 31, "learning_rate": 0.03, "min_data_in_leaf": 200}})
    models = []
    for k in (0, 1):
        va = inner == k
        tr = ~va
        es = tr & (es_bucket == 0)
        m = _fit("lgb", df.loc[tr & ~es, cols], df.loc[tr & ~es, "y"].to_numpy(), df.loc[es, cols], df.loc[es, "y"].to_numpy(), params)
        models.append(m)
        oof[va] = _predict("lgb", m, df.loc[va, cols])
    from sklearn.isotonic import IsotonicRegression

    iso = IsotonicRegression(out_of_bounds="clip").fit(oof, df["y"])
    df["p_final"] = iso.predict(oof)
    # exclusivity must also see competitors from the deep-train folds (scored by the calibrated GBDT)
    beaten = (df["oth_fold_best_p"] > df["p_final"]).to_numpy()
    pairs = df.loc[~beaten, ["s1_id", "cand_id", "p_final"]].rename(columns={"s1_id": "s1", "cand_id": "cand", "p_final": "p"})
    best = Dm.tune_decision(pairs, gt, s1_ids, "p")
    pred = Dm.decide(pairs, best, "p")
    rec = pd.read_parquet(_wp(cfg, "train", "records.parquet"), columns=["entity_id", "src", "country_norm"])
    country = dict(zip(rec.entity_id[rec.src == 1], rec.country_norm[rec.src == 1]))
    rep = M.report(pred, gt, s1_ids, country)
    # reference: GBDT-only score on the same holdout S1 set (same exclusivity treatment)
    beaten_g = (df["oth_fold_best_p"] > df["p"]).to_numpy()
    ref = df.loc[~beaten_g, ["s1_id", "cand_id", "p"]].rename(columns={"s1_id": "s1", "cand_id": "cand"})
    rep["gbdt_only_same_holdout"] = M.macro_f05(Dm.decide(ref, Dm.tune_decision(ref, gt, s1_ids, "p"), "p"), gt, s1_ids)
    rep["decision"] = best
    rep["deep_scores"] = _deep_score_names(cfg, "train")
    log.info("STACK holdout report: %s", json.dumps(rep))
    with open(os.path.join(cfg["artifacts_dir"], "stacker.pkl"), "wb") as f:
        # the two inner models are used at test time: exactly the score distribution iso + decision were fit on
        pickle.dump({"models": models, "cols": cols, "iso": iso, "decision": best, "deep_scores": rep["deep_scores"]}, f)
    with open(_wp(cfg, "train", "stack_report.json"), "w") as f:
        json.dump(rep, f, indent=1)
    return rep


def predict(cfg, split="test"):
    from .. import io as bio

    with open(os.path.join(cfg["artifacts_dir"], "stacker.pkl"), "rb") as f:
        st = pickle.load(f)
    df = _assemble(cfg, split, expected_scores=st.get("deep_scores", []))
    missing = [c for c in st["cols"] if c not in df.columns]
    if missing:
        raise RuntimeError(f"stacker columns missing at predict time: {missing}")
    X = df[st["cols"]]
    df["p_final"] = np.mean([st["iso"].predict(_predict("lgb", m, X)) for m in st["models"]], axis=0)
    pairs = df[["s1_id", "cand_id", "p_final"]].rename(columns={"s1_id": "s1", "cand_id": "cand", "p_final": "p"})
    pred = Dm.decide(pairs, st["decision"], "p")
    rec = pd.read_parquet(_wp(cfg, split, "records.parquet"), columns=["entity_id", "src"])
    s1_all = rec.entity_id[rec.src == 1].tolist()
    from .. import store

    feats = store.load_all(cfg, split, columns=["s1_id", "cand_id"])
    cands = feats.groupby("s1_id")["cand_id"].agg(list).to_dict()
    out_dir = os.path.join(cfg["out_dir"], "stack_" + split)  # per split/shard; merged by `pipeline finalize`
    bio.write_matching(os.path.join(out_dir, "matching_results.tsv"), s1_all, pred)
    bio.write_candidates(os.path.join(out_dir, "candidate_pairs.tsv"), s1_all, cands)
    log.info("stack predict %s: %d S1 non-empty of %d", split, sum(1 for s in s1_all if pred.get(s)), len(s1_all))


def main(argv=None):
    from ..pipeline import load_cfg

    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["train", "predict"])
    ap.add_argument("--config", required=True)
    ap.add_argument("--split", default="test")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = load_cfg(a.config)
    train(cfg) if a.cmd == "train" else predict(cfg, a.split)


if __name__ == "__main__":
    main()
