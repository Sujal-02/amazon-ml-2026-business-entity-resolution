"""Decision layer: turn calibrated pair probabilities into per-S1 match lists that maximise macro F0.5.

Two structural facts from the training data drive this module:
  1. every S2/S3 record belongs to AT MOST ONE S1 entity (0 of 7.6M matched ids are shared), so a candidate
     should only ever be emitted for the S1 that owns it (exclusivity / bipartite constraint);
  2. the metric is F0.5 per S1, macro-averaged, with singletons scoring 1 only for an empty prediction, so the
     choice "how many of the ranked candidates to emit" is a per-entity expected-utility decision.

`select_expected_f` solves (2) exactly under an independence assumption: for each S1, candidates are sorted by
probability and the prefix length k maximising E[F0.5] is chosen, where TP ~ PoissonBinomial(p_1..p_k) and
FN ~ PoissonBinomial(p_{k+1}..p_n) (+ an optional expected count of matches lost by blocking).
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

try:
    from numba import njit
except Exception:  # numba optional; pure-python fallback is slower but identical
    def njit(*a, **k):
        def deco(f):
            return f
        return deco if not (a and callable(a[0])) else a[0]


@njit(cache=True)
def _poisson_binomial(p):
    n = p.shape[0]
    dist = np.zeros(n + 1)
    dist[0] = 1.0
    for i in range(n):
        pi = p[i]
        for j in range(i + 1, 0, -1):
            dist[j] = dist[j] * (1.0 - pi) + dist[j - 1] * pi
        dist[0] = dist[0] * (1.0 - pi)
    return dist


@njit(cache=True)
def _best_prefix(p_sorted, beta2, missing_mass, max_k):
    """Return (best_k, best_expected_f) for one entity. p_sorted is descending."""
    n = p_sorted.shape[0]
    # k = 0: F = 1 iff there is no true match at all
    prob_none = 1.0
    for i in range(n):
        prob_none *= (1.0 - p_sorted[i])
    prob_none *= np.exp(-missing_mass)
    best_k, best_f = 0, prob_none
    kmax = min(n, max_k)
    for k in range(1, kmax + 1):
        tp = _poisson_binomial(p_sorted[:k])
        fn = _poisson_binomial(p_sorted[k:])
        # approximate lost-by-blocking matches as a Poisson(missing_mass) added to FN (truncate at 3)
        if missing_mass > 0:
            extra = np.zeros(4)
            e = np.exp(-missing_mass)
            fact = 1.0
            for j in range(4):
                if j > 0:
                    fact *= j
                extra[j] = e * missing_mass ** j / fact
            fn2 = np.zeros(fn.shape[0] + 3)
            for a in range(fn.shape[0]):
                for j in range(4):
                    fn2[a + j] += fn[a] * extra[j]
            fn = fn2
        ef = 0.0
        for a in range(1, k + 1):
            if tp[a] == 0.0:
                continue
            for b in range(fn.shape[0]):
                if fn[b] == 0.0:
                    continue
                # F_beta = (1+b2) TP / (|pred| + b2 |true|),  |pred| = k, |true| = a + b
                ef += tp[a] * fn[b] * (1.0 + beta2) * a / (k + beta2 * (a + b))
        if ef > best_f:
            best_f, best_k = ef, k
    return best_k, best_f


@njit(cache=True)
def _select_all(offsets, probs, beta2, missing_mass, max_k):
    n_groups = offsets.shape[0] - 1
    ks = np.zeros(n_groups, dtype=np.int64)
    efs = np.zeros(n_groups)
    for g in range(n_groups):
        s, e = offsets[g], offsets[g + 1]
        k, f = _best_prefix(probs[s:e], beta2, missing_mass, max_k)
        ks[g] = k
        efs[g] = f
    return ks, efs


def enforce_exclusivity(pairs: pd.DataFrame, score_col: str = "p", margin: float = 0.0) -> pd.DataFrame:
    """Keep each candidate only for its highest-scoring S1 (optionally require a margin over the runner-up)."""
    df = pairs.sort_values(["cand", score_col], ascending=[True, False])
    first = ~df["cand"].duplicated(keep="first")
    if margin > 0:
        second = df.groupby("cand")[score_col].transform(lambda s: s.iloc[1] if len(s) > 1 else 0.0)
        first &= (df[score_col] - second) >= margin
    return df[first]


def select_expected_f(pairs: pd.DataFrame, score_col: str = "p", beta: float = 0.5,
                      missing_mass: float = 0.0, max_k: int = 16, min_p: float = 1e-4) -> Dict[str, List[str]]:
    """Per-S1 expected-F0.5-optimal prefix selection. `pairs` needs columns s1, cand, <score_col>."""
    df = pairs[pairs[score_col] >= min_p].sort_values(["s1", score_col], ascending=[True, False])
    if df.empty:
        return {}
    s1 = df["s1"].to_numpy()
    starts = np.flatnonzero(np.r_[True, s1[1:] != s1[:-1]])
    offsets = np.r_[starts, len(df)].astype(np.int64)
    probs = np.clip(df[score_col].to_numpy(dtype=np.float64), 0.0, 1.0)
    ks, _ = _select_all(offsets, probs, beta * beta, float(missing_mass), int(max_k))
    cands = df["cand"].to_numpy()
    out: Dict[str, List[str]] = {}
    for g in np.flatnonzero(ks > 0):
        s = offsets[g]
        out[s1[s]] = list(cands[s:s + ks[g]])
    return out


def select_threshold(pairs: pd.DataFrame, thr: float, score_col: str = "p") -> Dict[str, List[str]]:
    df = pairs[pairs[score_col] >= thr]
    return df.groupby("s1")["cand"].agg(list).to_dict()


def decide(pairs: pd.DataFrame, cfg: dict, score_col: str = "p") -> Dict[str, List[str]]:
    """Full decision layer driven by a small config dict (tuned on validation).

    If `pairs` has a boolean column `_unseen` (S1 of a country never seen in training, e.g. France), those rows get
    an extra logit shift cfg['unseen_bias'] (<0 = more conservative): thresholds tuned on US/India cannot be
    validated there, and a false merge costs more than a missed link under F0.5."""
    df = pairs
    ub = cfg.get("unseen_bias", 0.0)
    if ub and "_unseen" in df.columns and df["_unseen"].any():
        p = np.clip(df[score_col].to_numpy(np.float64), 1e-7, 1 - 1e-7)
        lg = np.log(p / (1 - p)) + np.where(df["_unseen"].to_numpy(), ub, 0.0)
        df = df.assign(**{score_col: 1 / (1 + np.exp(-lg))})
    if cfg.get("exclusive", True):
        df = enforce_exclusivity(df, score_col, cfg.get("excl_margin", 0.0))
    if cfg.get("mode", "expected_f") == "expected_f":
        if cfg.get("temperature", 1.0) != 1.0:
            p = np.clip(df[score_col].to_numpy(), 1e-7, 1 - 1e-7)
            logit = np.log(p / (1 - p)) / cfg["temperature"] + cfg.get("bias", 0.0)
            df = df.assign(**{score_col: 1 / (1 + np.exp(-logit))})
        return select_expected_f(df, score_col, missing_mass=cfg.get("missing_mass", 0.0))
    return select_threshold(df, cfg.get("threshold", 0.5), score_col)


def tune_decision(pairs: pd.DataFrame, gt: Dict[str, List[str]], s1_ids, score_col: str = "p",
                  country: Dict[str, str] | None = None) -> dict:
    """Grid-search the decision config on a validation fold; returns best cfg + score."""
    from .metrics import macro_f05

    best = {"score": -1}
    grid = []
    for excl in (True, False):
        for thr in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95):
            grid.append({"exclusive": excl, "mode": "threshold", "threshold": thr})
        for temp in (0.7, 1.0, 1.4, 2.0):
            for bias in (-2.0, -1.5, -1.0, -0.5, 0.0, 0.5):
                for mm in (0.0, 0.05, 0.1):
                    grid.append({"exclusive": excl, "mode": "expected_f", "temperature": temp, "bias": bias, "missing_mass": mm})
    # exclusivity does not depend on temperature/bias (monotone per pair): compute it once per flag.
    # pairs below 1e-3 are never selected under any grid point -> drop them up front (big speed-up)
    base = pairs[pairs[score_col] >= 1e-3]
    pre = {True: enforce_exclusivity(base, score_col), False: base}
    s1_ids = list(s1_ids)
    for cfg in grid:
        pred = decide(pre[cfg["exclusive"]], {**cfg, "exclusive": False}, score_col)
        sc = macro_f05(pred, gt, s1_ids)
        if sc > best["score"]:
            best = {"score": sc, **cfg}
    return best
