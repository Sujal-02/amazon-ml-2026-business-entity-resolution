"""Optuna hyperparameter search for the GBDT learners of models/gbdt.py (stage 2 by default), CPU/laptop friendly.

What a trial sees is exactly what `gbdt.train_stack` feeds a stage-2 learner, restricted to a hashed subsample of S1:
  * feats.parquet (+ cfg.stack.extra_scores files), feature set/order of `gbdt.feature_cols` with p1 moved last;
  * the stage-1 OOF score p1 taken from oof.parquet (fallback: re-fitted OOF on the subsample);
  * context + support features on p1 (`gbdt.add_stage2_inputs`); candidate-side competition features are computed
    on the *closure* of the sampled candidates (every pair touching a sampled candidate), so their values are
    identical to the full-data ones although only ~s1_frac of the pairs are loaded;
  * S1-grouped folds from the 'fold' column, negative downsampling `gbdt._downsample`, the early-stopping split of
    `gbdt.oof_fit`, and the learners themselves via `gbdt._fit` / `gbdt._predict`.

S1 subsample = S1 whose s1_hash (crc32 % 1000) is below a threshold -> the same buckets `gbdt_train_s1_frac` keeps.
Objective  : OOF logloss over ALL candidate pairs of the evaluated folds (what OOF prediction later scores), with
             fold-level median pruning and a wall-clock budget.
Final check: macro F0.5 (decision.tune_decision, singletons included) of the best trial vs the current config
             ("baseline") vs the production blend `p` in oof.parquet, on the same S1 set and exclusivity treatment.
Outputs    : <out_dir>/best_params_<kind>.yaml   flat dict, the exact structure of cfg['gbdt_params'][kind]
             <out_dir>/tune_report_<kind>.json   baseline/best logloss, final-check reports, importances, timing
             <out_dir>/tune_trials_<kind>.csv    every trial
             (stage 1: *_stage1_<kind>.*).  work_dir is only read, never written.

  python -m ber.models.tune --config configs/local_dev.yaml --kind lgb --trials 40 --time-budget-min 60
  python -m ber.models.tune --config configs/local_dev.yaml --kind xgb --trials 25 --time-budget-min 60 --threads 2

Optional config block (CLI flags win):
  tune: {s1_frac: 0.25, max_pairs: 1500000, eval_folds: 3, threads: 2, max_rounds: 2000,
         space: {lgb: {max_bin: 255, min_data_in_leaf: {low: 50, high: 5000, log: true}, lambda_l1: null}}}
  (space entry: scalar = fixed value, null = do not tune (keep config/default), dict = range or {choices: [...]}).
"""
from __future__ import annotations

import argparse
import copy
import gc
import json
import logging
import math
import os
import sys
import time
import warnings
import zlib
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from .. import decision as Dm
from .. import features as F
from .. import metrics as M
from . import gbdt as G

log = logging.getLogger(__name__)

KINDS = ("lgb", "xgb", "cat")
ROUNDS_KEY = {"lgb": "num_boost_round", "xgb": "num_boost_round", "cat": "iterations"}
DEFAULT_ROUNDS = 3000  # the cap gbdt._fit uses for all three learners when the config gives none
#: count-like regularisers that grow with training-set size (only rescaled with --scale-count-params)
COUNT_PARAMS = {"lgb": ("min_data_in_leaf", "min_sum_hessian_in_leaf"), "xgb": ("min_child_weight",), "cat": ()}
META = ("i1", "i2", "y", "fold", "blk_prescore", "s1_hash", "s1_id", "cand_id")

# name -> ("float"|"int", low, high, log) | ("cat", choices).  Names are the canonical ones gbdt._fit uses, so the
# written params override its defaults without alias clashes.
SPACES: Dict[str, Dict[str, tuple]] = {
    "lgb": {
        "learning_rate": ("float", 0.03, 0.2, True),
        "num_leaves": ("int", 15, 511, True),
        "min_data_in_leaf": ("int", 20, 2000, True),
        "min_sum_hessian_in_leaf": ("float", 1e-3, 10.0, True),
        "feature_fraction": ("float", 0.4, 1.0, False),
        "bagging_fraction": ("float", 0.5, 1.0, False),
        "lambda_l1": ("float", 1e-4, 10.0, True),
        "lambda_l2": ("float", 1e-3, 50.0, True),
        "min_gain_to_split": ("float", 0.0, 1.0, False),
        "max_bin": ("cat", [63, 127, 255]),
    },
    "xgb": {
        "eta": ("float", 0.03, 0.2, True),
        "max_depth": ("int", 4, 12, False),
        "min_child_weight": ("float", 0.5, 200.0, True),
        "subsample": ("float", 0.5, 1.0, False),
        "colsample_bytree": ("float", 0.4, 1.0, False),
        "reg_lambda": ("float", 1e-3, 50.0, True),
        "reg_alpha": ("float", 1e-4, 10.0, True),
        "gamma": ("float", 0.0, 2.0, False),
        "max_bin": ("cat", [128, 256]),
    },
    "cat": {
        "learning_rate": ("float", 0.03, 0.3, True),
        "depth": ("int", 4, 10, False),
        "l2_leaf_reg": ("float", 0.5, 30.0, True),
        "random_strength": ("float", 0.05, 10.0, True),
        "bagging_temperature": ("float", 0.0, 2.0, False),
        "border_count": ("cat", [64, 128, 254]),
    },
}


def runtime_params(kind: str, threads: int) -> dict:
    """Thread/logging keys used during the search only (never written to the yaml)."""
    if kind == "lgb":
        return {"n_jobs": int(threads), "verbose": -1}
    if kind == "xgb":
        return {"nthread": int(threads)}
    return {"thread_count": int(threads), "allow_writing_files": False, "verbose": 0}


def params_filename(kind: str, stage: int = 2) -> str:
    return f"best_params_{kind}.yaml" if stage == 2 else f"best_params_stage1_{kind}.yaml"


def _tag(kind: str, stage: int) -> str:
    return kind if stage == 2 else f"stage1_{kind}"


# ----------------------------------------------------------------------------------------------------------------
# search space
# ----------------------------------------------------------------------------------------------------------------
def build_space(kind: str, overrides: Optional[dict] = None):
    """Default space for `kind` with cfg['tune']['space'][kind] overrides -> (space, fixed_params)."""
    space = dict(SPACES[kind])
    fixed: dict = {}
    for name, v in (overrides or {}).items():
        if v is None:
            space.pop(name, None)
        elif isinstance(v, dict):
            if "choices" in v:
                space[name] = ("cat", list(v["choices"]))
                continue
            old = space.get(name)
            num_old = old is not None and old[0] != "cat"
            typ = v.get("type") or (old[0] if num_old else
                                    ("int" if isinstance(v["low"], int) and isinstance(v["high"], int) else "float"))
            space[name] = (typ, v["low"], v["high"], bool(v.get("log", old[3] if num_old else False)))
        else:
            space.pop(name, None)
            fixed[name] = v
    return space, fixed


def suggest(trial, space: dict) -> dict:
    out = {}
    for name, spec in space.items():
        if spec[0] == "cat":
            out[name] = trial.suggest_categorical(name, list(spec[1]))
        elif spec[0] == "int":
            out[name] = trial.suggest_int(name, int(spec[1]), int(spec[2]), log=bool(spec[3]))
        else:
            out[name] = trial.suggest_float(name, float(spec[1]), float(spec[2]), log=bool(spec[3]))
    return out


# ----------------------------------------------------------------------------------------------------------------
# small utilities
# ----------------------------------------------------------------------------------------------------------------
def _wp(cfg, split, name):  # read-only twin of gbdt._wp (never creates directories inside work_dir)
    return os.path.join(cfg["work_dir"], split, name)


def _rss_gb() -> Dict[str, float]:
    """Current / peak resident memory of this process in GB (psutil, else Win32 / POSIX fallbacks)."""
    try:
        import psutil  # type: ignore

        return {"rss_gb": round(psutil.Process().memory_info().rss / 2 ** 30, 2)}
    except Exception:
        pass
    try:
        if sys.platform == "win32":
            import ctypes
            from ctypes import wintypes

            class PMC(ctypes.Structure):
                _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                            ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                            ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]

            c = PMC(); c.cb = ctypes.sizeof(PMC)
            k32 = ctypes.WinDLL("kernel32", use_last_error=True)
            k32.GetCurrentProcess.restype = wintypes.HANDLE
            psapi = ctypes.WinDLL("psapi", use_last_error=True)
            psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(PMC), wintypes.DWORD]
            if psapi.GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(c), c.cb):
                return {"rss_gb": round(c.WorkingSetSize / 2 ** 30, 2), "peak_gb": round(c.PeakWorkingSetSize / 2 ** 30, 2)}
        else:
            import resource

            return {"peak_gb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20, 2)}
    except Exception:
        pass
    return {}


def _logloss(y: np.ndarray, p: np.ndarray) -> float:
    y = np.asarray(y, np.float64)
    p = np.clip(np.asarray(p, np.float64), 1e-7, 1 - 1e-7)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log1p(-p)))


def _best_iter(kind: str, m) -> int:
    try:
        return int(m.get_best_iteration()) if kind == "cat" else int(m.best_iteration)
    except Exception:
        return -1


def _py(v):
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if isinstance(v, (int, np.integer)):
        return int(v)
    if isinstance(v, (float, np.floating)):
        return float(f"{float(v):.6g}")
    return v


def _scan(path: str, columns: Sequence[str], row_filter=None, float32: Sequence[str] = (),
          batch_size: int = 131_072) -> pd.DataFrame:
    """Bounded-memory parquet read: batch by batch (no dataset-scanner read-ahead, which multiplies peak RAM),
    rows kept where row_filter(batch) -> bool array is True, optional float32 cast done in arrow."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(path, buffer_size=1 << 22)
    cols = list(columns)
    schema = pa.schema([pf.schema_arrow.field(c) for c in cols])
    parts = []
    for b in pf.iter_batches(batch_size=batch_size, columns=cols):
        if row_filter is not None:
            b = b.filter(pa.array(np.asarray(row_filter(b), dtype=bool)))
        if b.num_rows:
            parts.append(b)
    t = pa.Table.from_batches(parts, schema=schema) if parts else schema.empty_table()
    del parts
    f32 = set(float32)
    if f32:
        sch = pa.schema([pa.field(f.name, pa.float32()) if f.name in f32 and f.type != pa.float32() else f
                         for f in t.schema])
        t = t.cast(sch, safe=False)
    return t.to_pandas(self_destruct=True, split_blocks=True)


def _part_paths(cfg, split):
    from .. import store

    ks = store.parts(cfg, split)
    base = os.path.join(cfg["work_dir"], split)
    return [(os.path.join(base, "feats", f"part-{k:05d}.parquet"), os.path.join(base, "freq", f"part-{k:05d}.parquet"))
            for k in ks]


def _store_schema(cfg, split):
    """(column names, total rows, empty pandas frame with dtypes) over feats+freq parts (legacy single file fallback)."""
    import pyarrow.parquet as pq

    pp = _part_paths(cfg, split)
    if not pp:
        pf = pq.ParquetFile(_wp(cfg, split, "feats.parquet"))
        return pf.schema_arrow.names, pf.metadata.num_rows, pf.schema_arrow.empty_table().to_pandas()
    fp, qp = pp[0]
    fe = pq.ParquetFile(fp).schema_arrow.empty_table().to_pandas()
    if os.path.exists(qp):
        qe = pq.ParquetFile(qp).schema_arrow.empty_table().to_pandas()
        fe = pd.concat([fe, qe[[c for c in qe.columns if c not in fe.columns]]], axis=1)
    n = sum(pq.ParquetFile(f).metadata.num_rows for f, _ in pp)
    return list(fe.columns), n, fe


def _scan_store(cfg, split, columns, row_filter=None, float32=()):
    """_scan over the chunked store: per part, read the feats columns, apply row_filter, add the freq columns of the
    same rows, cast float32 columns; concatenate parts (legacy single feats.parquet fallback)."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    pp = _part_paths(cfg, split)
    if not pp:
        return _scan(_wp(cfg, split, "feats.parquet"), columns, row_filter, float32)
    cols = list(columns)
    f32 = set(float32)
    frames = []
    for fp, qp in pp:
        fn = pq.ParquetFile(fp).schema_arrow.names
        qn = pq.ParquetFile(qp).schema_arrow.names if os.path.exists(qp) else []
        fcols = [c for c in cols if c in fn]
        qcols = [c for c in cols if c in qn and c not in fn]
        need_filter = [c for c in ("s1_hash", "s1_id") if c in fn and c not in fcols]
        t = pq.read_table(fp, columns=fcols + need_filter)
        mask = None
        if row_filter is not None:
            mask = pa.array(np.asarray(row_filter(t), dtype=bool))
            t = t.filter(mask)
        if need_filter:
            t = t.drop(need_filter)
        if qcols:
            q = pq.read_table(qp, columns=qcols)
            if mask is not None:
                q = q.filter(mask)
            for c in qcols:
                t = t.append_column(c, q.column(c))
        if t.num_rows:
            d = t.to_pandas(self_destruct=True, split_blocks=True)
            for c in d.columns:
                if c in f32 and d[c].dtype != np.float32:
                    d[c] = d[c].astype(np.float32)
            frames.append(d[[c for c in cols if c in d.columns]])
        del t
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=cols)


def _isin(col: str, values: np.ndarray):
    """row_filter for _scan: batch[col] in values (values sorted unique)."""
    def f(b):
        v = b.column(col).to_numpy()
        pos = np.minimum(np.searchsorted(values, v), len(values) - 1)
        return values[pos] == v
    return f


def _aligned(keys: pd.DataFrame, part: pd.DataFrame, cols: Sequence[str]) -> Dict[str, np.ndarray]:
    """Left-join `part` (keyed by i1,i2) onto the row order of `keys` -> {col: array}."""
    al = keys.merge(part, on=["i1", "i2"], how="left", validate="one_to_one")
    return {c: al[c].to_numpy() for c in cols}


def _add_cols(df: pd.DataFrame, new: Dict[str, np.ndarray]) -> pd.DataFrame:
    """Append columns in one concat (no fragmented-frame inserts)."""
    if not new:
        return df
    return pd.concat([df, pd.DataFrame(new, index=df.index)], axis=1)


def _sample_threshold(n_rows: int, s1_frac: float, max_pairs) -> int:
    frac = float(s1_frac)
    if max_pairs:
        frac = min(frac, float(max_pairs) / max(1, n_rows))
    return int(min(1000, max(1, math.floor(frac * 1000 + 1e-9))))


def _compact_records(cfg, i2: np.ndarray):
    """addr_norm/name_core of the needed candidate rows only -> (records frame in compact order, compact i2)."""
    import pyarrow.parquet as pq

    rp = _wp(cfg, "train", "records.parquet")
    need = np.unique(i2)
    names = pq.ParquetFile(rp).schema_arrow.names
    if "idx" in names:  # idx == row position (prepare.py)
        r = _scan(rp, ["idx", "addr_norm", "name_core"], _isin("idx", need))
        r = r.sort_values("idx").reset_index(drop=True)
        if len(r) != len(need) or not np.array_equal(r["idx"].to_numpy(), need):
            raise RuntimeError("records.parquet idx does not cover the candidate rows of feats.parquet")
        r = r[["addr_norm", "name_core"]]
    else:
        r = pd.read_parquet(rp, columns=["addr_norm", "name_core"]).iloc[need].reset_index(drop=True)
    return r, np.searchsorted(need, i2)


# ----------------------------------------------------------------------------------------------------------------
# data assembly
# ----------------------------------------------------------------------------------------------------------------
def load_frame(cfg: dict, stage: int = 2, s1_frac: float = 0.25, max_pairs=1_500_000, ctx_scope: str = "full",
               stage1_source: str = "auto", threads: int = 2, max_rounds: int = 2000):
    """Tuning frame for the S1 subsample -> (df, cols, extras).

    df holds META + the learner's feature columns (float32); extras holds arrays that must NOT be features
    (production score `p` of the same rows, the candidate-closure of oof.parquet for exclusivity, bookkeeping)."""
    import pyarrow.parquet as pq

    t0 = time.time()
    names, n_total, empty = _store_schema(cfg, "train")
    missing = [c for c in ("i1", "i2", "y", "fold", "blk_prescore") if c not in names]
    if missing:
        raise RuntimeError(f"train features lack {missing} (train split features required)")
    if "s1_hash" not in names and "s1_id" not in names:
        raise RuntimeError("train features need s1_hash or s1_id for S1-level sampling")
    feat = G.feature_cols(empty)
    want = set(feat) | set(META)
    read = [c for c in names if c in want]
    thr = _sample_threshold(n_total, s1_frac, max_pairs)
    if thr >= 1000:
        filt = None
    elif "s1_hash" in names:
        filt = lambda b: b.column("s1_hash").to_numpy() < thr  # noqa: E731
    else:  # old feats without s1_hash
        filt = lambda b: F.fold_of(b.column("s1_id").to_pylist(), 1000) < thr  # noqa: E731
    df = _scan_store(cfg, "train", read, filt, float32=feat)
    if df.empty:
        raise RuntimeError(f"S1 subsample is empty (threshold {thr}/1000)")
    log.info("tune: loaded %d / %d pairs (s1_hash < %d/1000), %d feature cols, %.0fs %s",
             len(df), n_total, thr, len(feat), time.time() - t0, _rss_gb())
    sub_i1 = np.unique(df["i1"].to_numpy())
    keys = df[["i1", "i2"]].copy()
    new: Dict[str, np.ndarray] = {}
    for e in cfg.get("stack", {}).get("extra_scores", []):  # same merge as train_stack (before base_cols)
        path = _wp(cfg, "train", f"score_{e}.parquet")
        if os.path.exists(path):
            s = _scan(path, pq.ParquetFile(path).schema_arrow.names, _isin("i1", sub_i1))
            new.update(_aligned(keys, s, [c for c in s.columns if c not in ("i1", "i2")]))
    df = _add_cols(df, new)
    extras = {"thr": thr, "n_pairs_total": int(n_total), "stage1_source": None, "ctx_scope": None,
              "p_prod": None, "clo": None}
    if stage == 1:
        cols = G.feature_cols(df)
        return df, cols, extras

    # ---------------- stage-2 inputs: p1 + context + support features ----------------
    op = _wp(cfg, "train", "oof.parquet")
    src = stage1_source
    onames = pq.ParquetFile(op).schema_arrow.names if os.path.exists(op) else []
    if src == "auto":
        src = "oof" if "p1" in onames else "refit"
    rec_c, i2c = _compact_records(cfg, df["i2"].to_numpy())
    log.info("tune: %d candidate records loaded (%.0fs)", len(rec_c), time.time() - t0)
    closure = None
    if src == "oof":
        if "p1" not in onames:
            raise RuntimeError(f"{op} missing or without p1 (run the train step first, or --stage1-source refit)")
        ocols = ["i1", "i2", "p1"] + (["p"] if "p" in onames else [])
        if thr >= 1000:
            clo = _scan(op, ocols)
        elif ctx_scope == "full":  # every pair touching a sampled candidate -> exact candidate-side competition
            clo = _scan(op, ocols, _isin("i2", np.unique(df["i2"].to_numpy())))
        else:
            clo = _scan(op, ocols, _isin("i1", sub_i1))
        in_sub = np.isin(clo["i1"].to_numpy(), sub_i1)
        al = _aligned(keys, clo.loc[in_sub, ocols], ocols[2:])
        if np.isnan(al["p1"]).any():
            raise RuntimeError(f"oof.parquet does not cover {int(np.isnan(al['p1']).sum())} sampled pairs (stale oof?)")
        p1 = al["p1"].astype(np.float32)
        extras["p_prod"] = al["p"].astype(np.float32) if "p" in al else None
        if "p" in ocols:
            extras["clo"] = {"i1": clo["i1"].to_numpy(), "i2": clo["i2"].to_numpy(), "p": clo["p"].to_numpy(np.float32)}
        if thr < 1000 and ctx_scope == "full":
            closure = clo
        del al, clo
        log.info("tune: p1 from oof.parquet (closure %s rows, %.0fs)", len(closure) if closure is not None else "-",
                 time.time() - t0)
    else:
        log.warning("tune: no stage-1 OOF (oof.parquet) -> re-fitting stage-1 OOF on the subsample; candidate-side "
                    "context features then only see sampled S1 (approximation)")
        p1 = _refit_stage1(df, G.feature_cols(df), cfg, threads, max_rounds)
        log.info("tune: stage-1 OOF re-fitted (%.0fs)", time.time() - t0)
    extras["stage1_source"] = src
    light = pd.DataFrame({"i1": df["i1"].to_numpy(), "i2": i2c, "p1": p1}, index=df.index)
    new = {}
    if closure is not None:
        extras["ctx_scope"] = "closure"
        ctx = F.context_features(closure[["i1", "i2", "p1"]], "p1", "ctx")
        in_sub = np.isin(closure["i1"].to_numpy(), sub_i1)
        part = pd.concat([closure.loc[in_sub, ["i1", "i2"]], ctx.loc[in_sub]], axis=1)
        new.update(_aligned(keys, part, list(ctx.columns)))
        sup = F.support_features(light, rec_c, "p1", prefix="sup")
        new.update({c: sup[c].to_numpy() for c in sup.columns})
        del ctx, part, sup
    else:
        extras["ctx_scope"] = "full" if thr >= 1000 else "subsample"
        s2 = G.add_stage2_inputs(light, rec_c, "p1")
        new.update({c: s2[c].to_numpy() for c in s2.columns if c not in light.columns})
        del s2
    new["p1"] = p1
    df = _add_cols(df, new)
    del closure, light, rec_c, new, keys
    gc.collect()
    cols = [c for c in G.feature_cols(df) if c != "p1"] + ["p1"]
    log.info("tune: stage-2 frame ready: %d pairs x %d cols (p1 from %s, context on %s) %.0fs %s", len(df), len(cols),
             src, extras["ctx_scope"], time.time() - t0, _rss_gb())
    return df, cols, extras


def _refit_stage1(df, cols, cfg, threads, max_rounds) -> np.ndarray:
    """Stage-1 OOF p1 on the subsample, mirroring gbdt.oof_fit (no model file is written)."""
    base = dict(cfg.get("gbdt_params", {}).get("lgb", {}))
    base["num_boost_round"] = min(int(base.get("num_boost_round", DEFAULT_ROUNDS)), int(max_rounds))
    base.update(runtime_params("lgb", threads))
    y = df["y"].to_numpy()
    fold = df["fold"].to_numpy()
    es_b = F.s1_bucket(df, 10) >= 8
    oof = np.zeros(len(df), np.float32)
    for fo in range(int(fold.max()) + 1):
        va = fold == fo
        if not va.any():
            continue
        tr = ~va & G._downsample(df, cfg, fo)
        es = tr & es_b
        m = G._fit("lgb", df.loc[tr & ~es, cols], y[tr & ~es], df.loc[es, cols], y[es], {"lgb": base})
        oof[va] = G._predict("lgb", m, df.loc[va, cols])
    return oof


# ----------------------------------------------------------------------------------------------------------------
# cross-validation core (mirrors gbdt.oof_fit fold by fold)
# ----------------------------------------------------------------------------------------------------------------
class FoldCV:
    def __init__(self, df: pd.DataFrame, cols: List[str], cfg: dict, kind: str, eval_folds: Sequence[int], runtime: dict):
        self.df, self.cols, self.kind, self.runtime = df, cols, kind, runtime
        self.y = df["y"].to_numpy()
        fold = df["fold"].to_numpy()
        es_b = F.s1_bucket(df, 10) >= 8
        self.folds = []
        for fo in eval_folds:
            va = fold == fo
            if not va.any():
                continue
            tr = ~va & G._downsample(df, cfg, fo)
            es = tr & es_b
            self.folds.append((int(fo), va, tr & ~es, es))
        if not self.folds:
            raise RuntimeError(f"no rows in eval folds {list(eval_folds)}")
        self.eval_mask = np.any([f[1] for f in self.folds], axis=0)
        self.fit_rows = [int(f[2].sum()) for f in self.folds]

    def run(self, params: dict, trial=None, stop=None) -> dict:
        import optuna

        kp = {self.kind: {**params, **self.runtime}}
        oof = np.full(len(self.df), np.nan, np.float32)
        tot = n = 0.0
        lls, iters = [], []
        for step, (fo, va, fit, es) in enumerate(self.folds):
            if trial is not None and stop is not None and stop():
                trial.study.stop()
                raise optuna.TrialPruned("time budget exhausted")
            m = G._fit(self.kind, self.df.loc[fit, self.cols], self.y[fit], self.df.loc[es, self.cols], self.y[es], kp)
            p = np.asarray(G._predict(self.kind, m, self.df.loc[va, self.cols]), np.float32)
            iters.append(_best_iter(self.kind, m))
            del m
            oof[va] = p
            ll = _logloss(self.y[va], p)
            lls.append(ll)
            tot += ll * va.sum(); n += va.sum()
            if trial is not None:
                trial.report(tot / n, step)
                if trial.should_prune():
                    raise optuna.TrialPruned()
        return {"logloss": float(tot / n), "fold_logloss": [float(x) for x in lls], "best_iter": iters, "oof": oof}


# ----------------------------------------------------------------------------------------------------------------
# final macro-F0.5 check
# ----------------------------------------------------------------------------------------------------------------
def final_check(cfg: dict, df: pd.DataFrame, extras: dict, cv: FoldCV, scores: Dict[str, np.ndarray],
                min_p: float = 1e-3) -> dict:
    """Isotonic-calibrate each OOF score on the evaluated rows, then decision.tune_decision on the same S1 set.

    S1 set = gt S1 (incl. singletons / S1 without candidates) of the sample buckets and evaluated folds. Exclusivity
    also sees competing S1 outside that set through the production score p of oof.parquet (as models/stack.py does)."""
    from sklearn.isotonic import IsotonicRegression

    if "s1_id" not in df or "cand_id" not in df:
        log.warning("tune: feats.parquet has no s1_id/cand_id -> final check skipped")
        return {}
    with open(_wp(cfg, "train", "gt.json")) as f:
        gt = json.load(f)
    k = int(cfg.get("n_folds", int(df["fold"].max()) + 1))
    ev_folds = {f[0] for f in cv.folds}
    thr = extras["thr"]
    s1_eval = []
    for s in gt:
        h = zlib.crc32(s.encode())
        if h % 1000 < thr and h % k in ev_folds:
            s1_eval.append(s)
    ev = cv.eval_mask
    y = df["y"].to_numpy()[ev]
    i1 = df["i1"].to_numpy()[ev]
    i2 = df["i2"].to_numpy()[ev]
    s1 = df["s1_id"].to_numpy()[ev]
    cand = df["cand_id"].to_numpy()[ev]
    oth = np.zeros(len(i1), np.float32)
    clo = extras.get("clo")
    if clo is not None:
        outside = ~np.isin(clo["i1"], np.unique(i1))
        if outside.any():
            best_other = pd.Series(clo["p"][outside]).groupby(clo["i2"][outside]).max()
            oth = pd.Series(i2).map(best_other).fillna(0.0).to_numpy(np.float32)
    out = {"n_s1": len(s1_eval), "n_pairs": int(ev.sum()), "min_p": min_p,
           "outside_competitors": clo is not None}
    for name, sc in scores.items():
        if sc is None:
            continue
        t0 = time.time()
        raw = np.asarray(sc, np.float64)[ev]
        if name == "prod_blend":
            p = raw  # already isotonic-calibrated OOF of the production blend
        else:
            p = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(raw, y).predict(raw)
        keep = (p >= min_p) & ~(oth > p)
        pairs = pd.DataFrame({"s1": s1[keep], "cand": cand[keep], "p": p[keep]})
        dec = Dm.tune_decision(pairs, gt, s1_eval, "p")
        rep = M.report(Dm.decide(pairs, dec, "p"), gt, s1_eval)
        rep["decision"] = {kk: _py(v) for kk, v in dec.items()}
        rep["logloss_raw"] = _logloss(y, raw)
        out[name] = rep
        log.info("tune final check [%s]: macro F0.5 = %.5f (singletons %.4f, matched %.4f, P %.4f R %.4f) %.0fs",
                 name, rep["macro_f05"], rep["f05_singletons"], rep["f05_nonsingletons"], rep["micro_precision"],
                 rep["micro_recall"], time.time() - t0)
    if "best" in out and "baseline" in out:
        out["delta_macro_f05_best_vs_baseline"] = out["best"]["macro_f05"] - out["baseline"]["macro_f05"]
    return out


# ----------------------------------------------------------------------------------------------------------------
# output
# ----------------------------------------------------------------------------------------------------------------
def write_params_yaml(path: str, params: dict, header: Sequence[str] = ()) -> None:
    import yaml

    body = yaml.safe_dump({k: _py(v) for k, v in params.items()}, sort_keys=False, default_flow_style=False)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for line in header:
            f.write(f"# {line}\n")
        f.write(body)
    os.replace(tmp, path)


def apply_best_params(cfg: dict, kinds: Sequence[str] = KINDS, stage: int = 2, params_dir: Optional[str] = None) -> dict:
    """Copy of cfg with gbdt_params[kind] (stage 1: gbdt_params_stage1[kind]) overridden by the tuned yaml files."""
    import yaml

    out = copy.deepcopy(cfg)
    d = params_dir or cfg["artifacts_dir"]
    key = "gbdt_params" if stage == 2 else "gbdt_params_stage1"
    for k in kinds:
        p = os.path.join(d, params_filename(k, stage))
        if os.path.exists(p):
            with open(p, encoding="utf-8") as f:
                bp = yaml.safe_load(f) or {}
            out.setdefault(key, {})
            out[key][k] = {**(out[key].get(k) or {}), **bp}
            log.info("apply_best_params: %s.%s <- %s", key, k, p)
    return out


def _final_params(base: dict, fixed: dict, tuned: dict, kind: str, scale: float = 1.0) -> dict:
    out = {**base, **fixed, **tuned}
    rk = ROUNDS_KEY[kind]
    out[rk] = int(base.get(rk, DEFAULT_ROUNDS))  # production cap; early stopping picks the length
    if scale != 1.0:
        for c in COUNT_PARAMS[kind]:
            if c in out:
                out[c] = int(round(out[c] * scale)) if isinstance(out[c], (int, np.integer)) else float(out[c] * scale)
    return out


# ----------------------------------------------------------------------------------------------------------------
# driver
# ----------------------------------------------------------------------------------------------------------------
def run(cfg: dict, kind: str = "lgb", trials: int = 40, time_budget_min: float = 60.0, stage: int = 2,
        s1_frac: Optional[float] = None, max_pairs=None, eval_folds=None, threads: Optional[int] = None,
        max_rounds: Optional[int] = None, seed: Optional[int] = None, ctx_scope: str = "full",
        stage1_source: str = "auto", out_dir: Optional[str] = None, storage: Optional[str] = None,
        study_name: Optional[str] = None, baseline: bool = True, final: bool = True,
        scale_count_params: bool = False, final_min_p: float = 1e-3) -> dict:
    import optuna

    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}")
    if stage not in (1, 2):
        raise ValueError("stage must be 1 or 2")
    if stage == 1 and kind != "lgb":
        log.warning("gbdt.train_stack always uses LightGBM for stage 1; tuning stage-1 %s is informational only", kind)
    tc = cfg.get("tune", {}) or {}
    s1_frac = float(s1_frac if s1_frac is not None else tc.get("s1_frac", 0.25))
    max_pairs = max_pairs if max_pairs is not None else tc.get("max_pairs", 1_500_000)
    eval_folds = eval_folds if eval_folds is not None else tc.get("eval_folds", 3)
    threads = int(threads if threads is not None else tc.get("threads", 2))
    max_rounds = int(max_rounds if max_rounds is not None else tc.get("max_rounds", 2000))
    seed = int(seed if seed is not None else cfg.get("seed", 0))
    try:  # parquet decoding shares the box with other jobs -> same thread budget as the learners
        import pyarrow as pa

        pa.set_cpu_count(max(1, threads))
    except Exception:
        pass
    out_dir = out_dir or cfg["artifacts_dir"]
    os.makedirs(out_dir, exist_ok=True)
    tag = _tag(kind, stage)
    t_start = time.time()
    deadline = t_start + 60.0 * float(time_budget_min)

    df, cols, extras = load_frame(cfg, stage, s1_frac, max_pairs, ctx_scope, stage1_source, threads, max_rounds)
    k = int(df["fold"].max()) + 1
    folds = list(range(min(int(eval_folds), k))) if isinstance(eval_folds, (int, np.integer)) else [int(f) for f in eval_folds]
    cv = FoldCV(df, cols, cfg, kind, folds, runtime_params(kind, threads))
    base = dict((cfg.get("gbdt_params", {}) or {}).get(kind) or {})  # stage 1 of train_stack also reads gbdt_params.lgb
    space, fixed = build_space(kind, (tc.get("space", {}) or {}).get(kind))
    rk = ROUNDS_KEY[kind]
    cap = min(int(base.get(rk, DEFAULT_ROUNDS)), max_rounds)
    info = {"kind": kind, "stage": stage, "n_pairs_total": extras["n_pairs_total"], "n_pairs": len(df),
            "n_s1": int(df["i1"].nunique()), "s1_hash_threshold": extras["thr"], "s1_frac_effective": extras["thr"] / 1000,
            "n_features": len(cols), "eval_folds": [f[0] for f in cv.folds], "n_eval_pairs": int(cv.eval_mask.sum()),
            "fit_rows_per_fold": cv.fit_rows, "threads": threads, "round_cap_during_search": cap,
            "stage1_source": extras["stage1_source"], "ctx_scope": extras["ctx_scope"], "load_secs": round(time.time() - t_start, 1),
            "memory_after_load": _rss_gb()}
    log.info("tune %s: %s", tag, json.dumps(info))

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    warnings.filterwarnings("ignore", category=optuna.exceptions.ExperimentalWarning)
    if storage == "auto":
        storage = "sqlite:///" + os.path.join(out_dir, f"tune_{tag}.db").replace("\\", "/")
    study_name = study_name or f"ber_{tag}_thr{extras['thr']}_f{''.join(map(str, info['eval_folds']))}"
    sampler = optuna.samplers.TPESampler(seed=seed, multivariate=True, n_startup_trials=max(2, min(10, trials // 4)))
    pruner = optuna.pruners.MedianPruner(n_startup_trials=max(2, min(5, trials // 5)), n_warmup_steps=0)
    study = optuna.create_study(direction="minimize", sampler=sampler, pruner=pruner, storage=storage,
                                study_name=study_name, load_if_exists=storage is not None)

    # ---------------- baseline = current config, exactly as gbdt.train_stack would fit it ----------------
    base_res = None
    if baseline:
        if "baseline" in study.user_attrs:
            base_res = dict(study.user_attrs["baseline"])
        else:
            t0 = time.time()
            r = cv.run({**base, rk: cap})
            base_res = {"logloss": r["logloss"], "fold_logloss": r["fold_logloss"], "best_iter": r["best_iter"],
                        "secs": round(time.time() - t0, 1)}
            base_res["oof"] = r["oof"]
            study.set_user_attr("baseline", {kk: v for kk, v in base_res.items() if kk != "oof"})
            log.info("tune %s baseline (current config): logloss %.6f iters %s (%.0fs)", tag, r["logloss"], r["best_iter"],
                     base_res["secs"])

    ppath = os.path.join(out_dir, params_filename(kind, stage))
    state = {"best_ll": math.inf, "number": None, "oof": None}
    n_rows_ratio = extras["n_pairs_total"] / max(1, len(df))

    def header(value, number):
        return [f"tuned by ber.models.tune ({tag}, stage {stage}) {time.strftime('%Y-%m-%d %H:%M')}",
                f"OOF logloss {value:.6f} (trial {number}); baseline {base_res['logloss']:.6f}" if base_res else
                f"OOF logloss {value:.6f} (trial {number})",
                f"subsample: {len(df)} pairs / {info['n_s1']} S1 (s1_hash < {extras['thr']}/1000), folds {info['eval_folds']}",
                f"paste under gbdt_params.{kind} (or use ber.models.tune.apply_best_params)"]

    def stop():
        done = sum(t.state == optuna.trial.TrialState.COMPLETE for t in study.get_trials(deepcopy=False))
        return time.time() > deadline and done >= 1

    def objective(trial):
        t0 = time.time()
        tuned = suggest(trial, space)
        r = cv.run({**base, **fixed, **tuned, rk: cap}, trial, stop)
        trial.set_user_attr("fold_logloss", r["fold_logloss"])
        trial.set_user_attr("best_iter", r["best_iter"])
        trial.set_user_attr("secs", round(time.time() - t0, 1))
        if r["logloss"] < state["best_ll"]:
            state.update(best_ll=r["logloss"], number=trial.number, oof=r["oof"])
            write_params_yaml(ppath, _final_params(base, fixed, tuned, kind), header(r["logloss"], trial.number))
        log.info("tune %s trial %d: logloss %.6f (best %.6f) iters %s %.0fs %s", tag, trial.number, r["logloss"],
                 state["best_ll"], r["best_iter"], time.time() - t0, {kk: _py(v) for kk, v in tuned.items()})
        return r["logloss"]

    remaining = max(1.0, deadline - time.time())
    try:
        study.optimize(objective, n_trials=int(trials), timeout=remaining, catch=(Exception,), gc_after_trial=True)
    except KeyboardInterrupt:
        log.warning("tune: interrupted -> finalising with the completed trials")
    all_trials = study.get_trials(deepcopy=False)
    complete = [t for t in all_trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not complete:
        failed = [t for t in all_trials if t.state == optuna.trial.TrialState.FAIL]
        raise RuntimeError(f"no trial completed ({len(failed)} failed, {len(all_trials)} total) -- see warnings above")
    best = study.best_trial
    tuned_best = {kk: v for kk, v in best.params.items() if kk in space}
    scale = n_rows_ratio if scale_count_params else 1.0
    best_params = _final_params(base, fixed, tuned_best, kind, scale)
    write_params_yaml(ppath, best_params, header(best.value, best.number))
    search_secs = time.time() - t_start

    # ---------------- final check ----------------
    fin = {}
    if final and stage == 2:
        best_oof = state["oof"] if state["number"] == best.number else None
        if best_oof is None:  # best trial comes from a resumed study -> re-run its CV once
            best_oof = cv.run({**base, **fixed, **tuned_best, rk: cap})["oof"]
        if base_res is not None and base_res.get("oof") is None:
            base_res["oof"] = cv.run({**base, rk: cap})["oof"]
        fin = final_check(cfg, df, extras, cv, {"best": best_oof, "baseline": base_res["oof"] if base_res else None,
                                               "prod_blend": extras.get("p_prod")}, min_p=final_min_p)
    imp = {}
    try:
        if len(complete) >= 3:
            imp = {kk: round(float(v), 4) for kk, v in optuna.importance.get_param_importances(study).items()}
    except Exception as e:  # importance needs enough distinct completed trials
        log.info("tune: param importances unavailable (%s)", e)
    hit_cap = max(best.user_attrs.get("best_iter", [0]) or [0]) >= 0.95 * cap
    states = pd.Series([t.state.name for t in all_trials]).value_counts().to_dict()
    top = sorted(complete, key=lambda t: t.value)[:10]
    report = {
        "info": info,
        "trials": {"requested": int(trials), "run": len(all_trials), **{kk.lower(): int(v) for kk, v in states.items()}},
        "baseline": {kk: v for kk, v in (base_res or {}).items() if kk != "oof"} or None,
        "baseline_params": {kk: _py(v) for kk, v in {**base, rk: int(base.get(rk, DEFAULT_ROUNDS))}.items()},
        "best": {"number": best.number, "logloss": best.value, "fold_logloss": best.user_attrs.get("fold_logloss"),
                 "best_iter": best.user_attrs.get("best_iter"), "hit_round_cap": bool(hit_cap)},
        "delta_logloss_best_vs_baseline": (best.value - base_res["logloss"]) if base_res else None,
        "best_params": {kk: _py(v) for kk, v in best_params.items()},
        "count_params_scaled_by": scale,
        "fixed_params": {kk: _py(v) for kk, v in fixed.items()},
        "space": {kk: list(v) if v[0] != "cat" else ["cat", list(v[1])] for kk, v in space.items()},
        "param_importances": imp,
        "top_trials": [{"number": t.number, "logloss": t.value, "params": {kk: _py(v) for kk, v in t.params.items()},
                        "best_iter": t.user_attrs.get("best_iter")} for t in top],
        "final_check": fin,
        "search_secs": round(search_secs, 1),
        "total_secs": round(time.time() - t_start, 1),
        "memory_end": _rss_gb(),
        "files": {"params": ppath},
    }
    rpath = os.path.join(out_dir, f"tune_report_{tag}.json")
    tpath = os.path.join(out_dir, f"tune_trials_{tag}.csv")
    report["files"].update(report=rpath, trials=tpath)
    try:
        study.trials_dataframe().to_csv(tpath, index=False)
    except Exception as e:
        log.warning("tune: could not write trials csv (%s)", e)
    with open(rpath, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=1, default=lambda o: _py(o) if isinstance(o, np.generic) else str(o))
    if hit_cap:
        log.warning("tune: best trial reached %d%% of the round cap during search (%d) -- raise --max-rounds / the "
                    "production %s if the tuned learning rate is low", 95, cap, rk)
    log.info("tune %s done: best logloss %.6f (trial %d, baseline %s) -> %s  [%.0fs, %s]", tag, best.value, best.number,
             f"{base_res['logloss']:.6f}" if base_res else "n/a", ppath, time.time() - t_start, _rss_gb())
    return report


def load_cfg(path: str) -> dict:
    """Same semantics as pipeline.load_cfg, without importing the pipeline (faiss, blocking, ...) into the tuner."""
    import yaml

    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    for k in ("work_dir", "data_dir", "out_dir", "artifacts_dir"):
        if k in cfg:
            cfg[k] = os.path.expanduser(os.path.expandvars(cfg[k]))
    return cfg


def main(argv=None):
    ap = argparse.ArgumentParser(description="Optuna search for the stage-2 GBDT of ber.models.gbdt")
    ap.add_argument("--config", required=True)
    ap.add_argument("--kind", default="lgb", choices=KINDS)
    ap.add_argument("--stage", type=int, default=2, choices=(1, 2), help="2 = stage-2 learner (default), 1 = stage-1 LGB")
    ap.add_argument("--trials", type=int, default=40)
    ap.add_argument("--time-budget-min", type=float, default=60.0, help="wall-clock budget of load + search")
    ap.add_argument("--s1-frac", type=float, default=None, help="S1 subsample fraction (default cfg.tune or 0.25)")
    ap.add_argument("--max-pairs", type=int, default=None, help="cap on loaded pairs; lowers s1-frac (default 1.5M)")
    ap.add_argument("--eval-folds", type=int, default=None, help="evaluate the first N folds (default 3)")
    ap.add_argument("--threads", type=int, default=None, help="learner threads (default 2)")
    ap.add_argument("--max-rounds", type=int, default=None, help="boosting-round cap during the search (default 2000)")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--ctx-scope", default="full", choices=("full", "sub"),
                    help="full: exact candidate-side context via the candidate closure; sub: leaner approximation")
    ap.add_argument("--stage1-source", default="auto", choices=("auto", "oof", "refit"))
    ap.add_argument("--out-dir", default=None, help="default: cfg artifacts_dir")
    ap.add_argument("--storage", default=None, help="optuna storage URL, or 'auto' (sqlite in out-dir, resumable)")
    ap.add_argument("--study-name", default=None)
    ap.add_argument("--no-baseline", action="store_true")
    ap.add_argument("--no-final-check", action="store_true")
    ap.add_argument("--scale-count-params", action="store_true",
                    help="multiply min_data_in_leaf/min_sum_hessian/min_child_weight by full/subsample pair ratio")
    ap.add_argument("--final-min-p", type=float, default=1e-3)
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = load_cfg(a.config)
    rep = run(cfg, kind=a.kind, trials=a.trials, time_budget_min=a.time_budget_min, stage=a.stage, s1_frac=a.s1_frac,
              max_pairs=a.max_pairs, eval_folds=a.eval_folds, threads=a.threads, max_rounds=a.max_rounds, seed=a.seed,
              ctx_scope=a.ctx_scope, stage1_source=a.stage1_source, out_dir=a.out_dir, storage=a.storage,
              study_name=a.study_name, baseline=not a.no_baseline, final=not a.no_final_check,
              scale_count_params=a.scale_count_params, final_min_p=a.final_min_p)
    fc = rep.get("final_check") or {}
    print(json.dumps({"best_logloss": rep["best"]["logloss"], "baseline_logloss": (rep["baseline"] or {}).get("logloss"),
                      "macro_f05": {kk: fc[kk]["macro_f05"] for kk in ("best", "baseline", "prod_blend") if kk in fc},
                      "params": rep["files"]["params"], "report": rep["files"]["report"]}, indent=1))


if __name__ == "__main__":
    main()
