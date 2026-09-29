"""Challenge metric: F_0.5 per Source-1 entity, macro-averaged (singletons included)."""
from __future__ import annotations

from typing import Dict, Iterable, Set

import numpy as np

BETA2 = 0.25  # beta = 0.5


def f05(pred: Set[str], true: Set[str]) -> float:
    if not true:
        return 1.0 if not pred else 0.0
    if not pred:
        return 0.0
    tp = len(pred & true)
    if tp == 0:
        return 0.0
    p = tp / len(pred)
    r = tp / len(true)
    return (1 + BETA2) * p * r / (BETA2 * p + r)


def macro_f05(pred: Dict[str, Iterable[str]], gt: Dict[str, Iterable[str]], s1_ids: Iterable[str] | None = None) -> float:
    ids = list(s1_ids) if s1_ids is not None else list(gt.keys())
    if not ids:
        return float("nan")
    return float(np.mean([f05(set(pred.get(s, ())), set(gt.get(s, ()))) for s in ids]))


def report(pred: Dict[str, Iterable[str]], gt: Dict[str, Iterable[str]], s1_ids=None, country: Dict[str, str] | None = None) -> dict:
    """Macro F0.5 plus diagnostic breakdowns (singleton vs matched, per country, micro P/R)."""
    ids = list(s1_ids) if s1_ids is not None else list(gt.keys())
    scores, tp, npred, ntrue = [], 0, 0, 0
    sing, nons = [], []
    by_c: Dict[str, list] = {}
    for s in ids:
        p, t = set(pred.get(s, ())), set(gt.get(s, ()))
        sc = f05(p, t)
        scores.append(sc)
        (sing if not t else nons).append(sc)
        tp += len(p & t); npred += len(p); ntrue += len(t)
        if country is not None:
            by_c.setdefault(country.get(s, "?"), []).append(sc)
    # precision-oriented error KPIs + where the macro-F0.5 points are lost (sums of 1-F per error type / n_s1)
    n = max(1, len(ids))
    k = {"singleton_fp": 0, "missed_all": 0, "wrong_only": 0, "extra_fp": 0, "partial_recall": 0}
    loss = {key: 0.0 for key in k}
    for s, sc in zip(ids, scores):
        p, t = set(pred.get(s, ())), set(gt.get(s, ()))
        if not t:
            if p:
                k["singleton_fp"] += 1; loss["singleton_fp"] += 1 - sc
            continue
        hit = len(p & t)
        if not p:
            k["missed_all"] += 1; loss["missed_all"] += 1 - sc
        elif hit == 0:
            k["wrong_only"] += 1; loss["wrong_only"] += 1 - sc
        elif len(p) > hit:
            k["extra_fp"] += 1; loss["extra_fp"] += 1 - sc
        elif hit < len(t):
            k["partial_recall"] += 1; loss["partial_recall"] += 1 - sc
    out = {
        "macro_f05": float(np.mean(scores)) if scores else float("nan"),
        "n_s1": len(ids),
        "singleton_frac": len(sing) / max(1, len(ids)),
        "f05_singletons": float(np.mean(sing)) if sing else float("nan"),
        "f05_nonsingletons": float(np.mean(nons)) if nons else float("nan"),
        "micro_precision": tp / max(1, npred),
        "micro_recall": tp / max(1, ntrue),
        "error_counts": k,
        "macro_loss_by_error": {key: v / n for key, v in loss.items()},
    }
    if country is not None:
        out["by_country"] = {c: float(np.mean(v)) for c, v in sorted(by_c.items())}
    return out


def candidate_recall(cands: Dict[str, Iterable[str]], gt: Dict[str, Iterable[str]], s1_ids=None) -> dict:
    """Blocking quality: pair recall, per-entity full-recall rate, avg candidates per S1."""
    ids = list(s1_ids) if s1_ids is not None else list(gt.keys())
    hit = tot = full = ncand = 0
    for s in ids:
        c, t = set(cands.get(s, ())), set(gt.get(s, ()))
        ncand += len(c)
        hit += len(c & t); tot += len(t)
        full += int(t <= c)
    return {"pair_recall": hit / max(1, tot), "entity_full_recall": full / max(1, len(ids)),
            "avg_candidates": ncand / max(1, len(ids)), "n_s1": len(ids)}
