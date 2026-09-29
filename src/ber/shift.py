"""Train -> test prior-shift handling (label-free on the test side).

Fact (measured by another team's pipeline, confirmed on our data by `strata_report`): the test generator injects many
more "near-twin" distractors than train -- a record with the S1 name/street whose house number is shifted by a value
drawn from {1,2,3,4,5,7,9,11,13,21}, often with an appended branch word -- ~0.48 per S1 in test vs ~0.1 in train.
A model calibrated on train priors over-accepts them (their holdout 0.99 -> leaderboard 0.95).

Correction = Bayes prior (label-shift) adjustment per stratum s (same street & similar name & house diff d, x an
extra-name-token flag), estimated WITHOUT test labels:
    pos_rate_s  = positives per S1 in stratum s on TRAIN (assumed unchanged in test: the noise channel of true
                  matches is the same; only the distractor injection changed)
    neg_train_s = negatives per S1 in s on TRAIN
    neg_test_s  = max(total_test_s - pos_rate_s, floor * total_test_s)   (total_test_s = test candidate pairs per S1)
    odds_test = odds_train * (neg_train_s / neg_test_s)    -> lambda_s, clipped to [lam_min, 1.25]
Strata where test density ~ train density (real typos: shifts 6, 8, 10 ...) get lambda ~ 1 automatically.
"""
from __future__ import annotations

import json
import logging
import os
from typing import Dict, Optional

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

STRATA_COLS = ["i1", "ad_house_absdiff_log", "ad_street_tset", "ad_tset", "nm_core_tset", "nm_n_extra_c"]
MAX_D = 40


def strata(df: pd.DataFrame, street_min: float = 0.9, name_min: float = 0.8) -> np.ndarray:
    """Stratum id per pair: 0 = not a near-twin; else 1 + 2*(d-1) + extra_flag for house diff d in 1..MAX_D."""
    d = np.rint(np.expm1(df["ad_house_absdiff_log"].to_numpy(np.float64)))
    street = df["ad_street_tset"].fillna(df["ad_tset"]).to_numpy(np.float64)
    ok = (d >= 1) & (d <= MAX_D) & (street >= street_min) & (df["nm_core_tset"].to_numpy(np.float64) >= name_min)
    extra = (df["nm_n_extra_c"].fillna(0).to_numpy() > 0).astype(np.int64)
    s = np.zeros(len(df), np.int64)
    dd = np.nan_to_num(d, nan=0).astype(np.int64)
    s[ok] = 1 + 2 * (dd[ok] - 1) + extra[ok]
    return s


def describe(sid: int) -> str:
    if sid == 0:
        return "none"
    d, e = (sid - 1) // 2 + 1, (sid - 1) % 2
    return f"d={d}{'+extra' if e else ''}"


def stratum_counts(df: pd.DataFrame, n_s1: int, y: Optional[np.ndarray] = None) -> pd.DataFrame:
    s = strata(df)
    out = pd.DataFrame({"s": s})
    if y is not None:
        out["y"] = y
    g = out[out.s > 0].groupby("s")
    res = pd.DataFrame({"n": g.size() / n_s1})
    if y is not None:
        res["pos"] = g["y"].sum() / n_s1
        res["neg"] = res["n"] - res["pos"]
    return res


def fit_lambdas(train_counts: pd.DataFrame, test_counts: pd.DataFrame, floor: float = 0.1, lam_min: float = 0.02,
                min_train_neg: float = 1e-5) -> Dict[int, float]:
    lam = {}
    for s in test_counts.index:
        tot = float(test_counts.loc[s, "n"])
        if s in train_counts.index:
            pos, neg = float(train_counts.loc[s, "pos"]), float(train_counts.loc[s, "neg"])
        else:
            pos, neg = 0.0, 0.0
        neg_test = max(tot - pos, floor * tot)
        if neg < min_train_neg or neg_test <= 0:
            continue
        lam[int(s)] = float(np.clip(neg / neg_test, lam_min, 1.25))
    return lam


def apply(p: np.ndarray, sid: np.ndarray, lam: Dict[int, float]) -> np.ndarray:
    if not lam:
        return p
    f = np.ones(len(p))
    for s, v in lam.items():
        f[sid == s] = v
    q = np.clip(p.astype(np.float64), 1e-7, 1 - 1e-7)
    o = q / (1 - q) * f
    return (o / (1 + o)).astype(np.float32)


def load(cfg: dict) -> Optional[dict]:
    p = os.path.join(cfg["artifacts_dir"], "shift_lambda.json")
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)
