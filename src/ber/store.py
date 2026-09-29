"""Chunked feature storage: work/<split>/feats/part-XXXXX.parquet (+ freq/part-XXXXX.parquet, same row ranges).

Pairs are stored sorted by i1, and every part holds whole S1 groups, so anything grouped by S1 can be computed per part;
anything grouped by candidate (i2) must be computed on a slim global table. Keeps a 16GB laptop within memory on the
full test split (~70-90M pairs).
"""
from __future__ import annotations

import glob
import os
import shutil
from typing import Iterator, List, Optional, Sequence

import numpy as np
import pandas as pd


def _dir(cfg: dict, split: str, kind: str) -> str:
    return os.path.join(cfg["work_dir"], split, kind)


def reset(cfg: dict, split: str) -> None:
    for kind in ("feats", "freq"):
        d = _dir(cfg, split, kind)
        if os.path.isdir(d):
            shutil.rmtree(d)
        os.makedirs(d, exist_ok=True)
    legacy = os.path.join(cfg["work_dir"], split, "feats.parquet")
    if os.path.exists(legacy):
        os.remove(legacy)


def write_part(cfg: dict, split: str, kind: str, k: int, df: pd.DataFrame) -> None:
    df.to_parquet(os.path.join(_dir(cfg, split, kind), f"part-{k:05d}.parquet"), index=False)


def parts(cfg: dict, split: str) -> List[int]:
    fs = sorted(glob.glob(os.path.join(_dir(cfg, split, "feats"), "part-*.parquet")))
    return [int(os.path.basename(f)[5:10]) for f in fs]


def split_points(i1: np.ndarray, chunk_pairs: int) -> List[tuple]:
    """Row ranges [a, b) of ~chunk_pairs rows that never cut an S1 group (i1 must be sorted)."""
    n = len(i1)
    if n == 0:
        return []
    starts = np.flatnonzero(np.r_[True, i1[1:] != i1[:-1]])
    out, a = [], 0
    while a < n:
        target = a + chunk_pairs
        if target >= n:
            out.append((a, n))
            break
        j = np.searchsorted(starts, target)
        b = int(starts[j]) if j < len(starts) else n
        if b <= a:
            b = n
        out.append((a, b))
        a = b
    return out


def _schema(path: str) -> List[str]:
    import pyarrow.parquet as pq

    return pq.read_schema(path).names


def read_part(cfg: dict, split: str, k: int, columns: Optional[Sequence[str]] = None) -> pd.DataFrame:
    fp = os.path.join(_dir(cfg, split, "feats"), f"part-{k:05d}.parquet")
    qp = os.path.join(_dir(cfg, split, "freq"), f"part-{k:05d}.parquet")
    fcols = _schema(fp)
    qcols = _schema(qp) if os.path.exists(qp) else []
    want_f = [c for c in fcols if columns is None or c in columns]
    want_q = [c for c in qcols if (columns is None or c in columns) and c not in fcols]
    df = pd.read_parquet(fp, columns=want_f)
    if want_q:
        q = pd.read_parquet(qp, columns=want_q)
        df = pd.concat([df.reset_index(drop=True), q.reset_index(drop=True)], axis=1)
    return df


def iter_parts(cfg: dict, split: str, columns: Optional[Sequence[str]] = None) -> Iterator[tuple]:
    for k in parts(cfg, split):
        yield k, read_part(cfg, split, k, columns)


def load_all(cfg: dict, split: str, columns: Optional[Sequence[str]] = None,
             row_filter=None) -> pd.DataFrame:
    """Concatenate all parts (optionally only some columns / rows). Falls back to a legacy single feats.parquet."""
    legacy = os.path.join(cfg["work_dir"], split, "feats.parquet")
    ks = parts(cfg, split)
    if not ks and os.path.exists(legacy):
        df = pd.read_parquet(legacy, columns=list(columns) if columns is not None else None)
        return df if row_filter is None else df[row_filter(df)].reset_index(drop=True)
    frames = []
    for k in ks:
        d = read_part(cfg, split, k, columns)
        if row_filter is not None:
            d = d[row_filter(d)]
        frames.append(d)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def columns(cfg: dict, split: str) -> List[str]:
    ks = parts(cfg, split)
    if not ks:
        legacy = os.path.join(cfg["work_dir"], split, "feats.parquet")
        return _schema(legacy) if os.path.exists(legacy) else []
    fp = os.path.join(_dir(cfg, split, "feats"), f"part-{ks[0]:05d}.parquet")
    qp = os.path.join(_dir(cfg, split, "freq"), f"part-{ks[0]:05d}.parquet")
    return _schema(fp) + ([c for c in _schema(qp)] if os.path.exists(qp) else [])
