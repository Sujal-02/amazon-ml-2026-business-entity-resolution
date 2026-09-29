"""Multi-channel, bidirectional candidate generation (blocking).

Channels (all run per country value -- countries are treated as an open set of labels; train showed 100% country
agreement between matched records):
  * lsa   : char n-gram TF-IDF -> TruncatedSVD dense vectors for name / address / combined text, FAISS kNN.
  * dense : (optional) fine-tuned multilingual bi-encoder embeddings (see models/biencoder.py), FAISS kNN.
  * keys  : exact composite keys (house number + street, name key, concatenated name, skeleton) with block caps.
Directions:
  * cand -> S1 (top k_rev): each S2/S3 record has at most one true S1, so its nearest S1 records are the natural
    candidate set (this direction also yields "reverse rank" features used by the matcher);
  * S1 -> cand (top k_fwd): recall for S1 entities with many noisy duplicates.
The union is capped per S1 by a cheap pre-score; the final set is written to candidate_pairs.tsv.
"""
from __future__ import annotations

import gc
import logging
import os
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

try:
    import faiss  # type: ignore
except Exception:  # pragma: no cover
    faiss = None


# ----------------------------------------------------------------------------------------------------------------
# FAISS helpers
# ----------------------------------------------------------------------------------------------------------------
def _n_gpus() -> int:
    try:
        return faiss.get_num_gpus() if faiss is not None else 0
    except Exception:
        return 0


def knn(index_vecs: np.ndarray, query_vecs: np.ndarray, k: int, index_type: str = "auto",
        batch: int = 65536, hnsw_m: int = 32, ef_search: int = 128) -> tuple:
    """Inner-product kNN (vectors must be L2-normalised). Returns (sims float16, idx int32)."""
    assert faiss is not None, "faiss is required for dense blocking"
    n, d = index_vecs.shape
    k = min(k, n)
    if k <= 0 or len(query_vecs) == 0:
        return np.zeros((len(query_vecs), 0), np.float16), np.zeros((len(query_vecs), 0), np.int32)
    xb = np.ascontiguousarray(index_vecs, dtype=np.float32)
    if index_type == "auto":
        # flat for small sets / GPU; IVF for large CPU sets (HNSW build on 4-5M vectors takes ~45 min per channel)
        index_type = "flat" if (_n_gpus() > 0 or n <= 200_000) else "ivf"
    if index_type == "ivf":
        nlist = int(min(16384, max(256, 4 * np.sqrt(n))))
        quant = faiss.IndexFlatIP(d)
        index = faiss.IndexIVFFlat(quant, d, nlist, faiss.METRIC_INNER_PRODUCT)
        rng = np.random.default_rng(0)
        tr = xb[rng.choice(n, min(n, 64 * nlist), replace=False)]
        index.train(tr)
        index.add(xb)
        index.nprobe = int(min(nlist, max(32, nlist // 64)))
    elif index_type == "flat":
        index = faiss.IndexFlatIP(d)
        if _n_gpus() > 0:
            co = faiss.GpuMultipleClonerOptions()
            co.useFloat16 = True
            index = faiss.index_cpu_to_all_gpus(index, co)
        index.add(xb)
    elif index_type == "hnsw":
        index = faiss.IndexHNSWFlat(d, hnsw_m, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = 80
        index.add(xb)
        index.hnsw.efSearch = max(ef_search, 2 * k)
    else:
        raise ValueError(index_type)
    D = np.empty((len(query_vecs), k), np.float16)
    I = np.empty((len(query_vecs), k), np.int32)
    for s in range(0, len(query_vecs), batch):
        q = np.ascontiguousarray(query_vecs[s:s + batch], dtype=np.float32)
        d_, i_ = index.search(q, k)
        D[s:s + batch] = d_
        I[s:s + batch] = i_
    del index
    gc.collect()
    return D, I


def l2n(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32, copy=False)
    nrm = np.linalg.norm(x, axis=1, keepdims=True)
    nrm[nrm == 0] = 1.0
    return x / nrm


# ----------------------------------------------------------------------------------------------------------------
# LSA (char TF-IDF + SVD) vectors
# ----------------------------------------------------------------------------------------------------------------
class LSAEncoder:
    """Char n-gram TF-IDF projected to a dense space; fit on a sample, applied in chunks (memory-bounded)."""

    def __init__(self, dim: int = 128, ngram=(2, 4), max_features: int = 300_000, fit_sample: int = 1_500_000,
                 svd_sample: int = 600_000, seed: int = 0):
        self.dim, self.ngram, self.max_features = dim, ngram, max_features
        self.fit_sample, self.svd_sample, self.seed = fit_sample, svd_sample, seed
        self.vec = None
        self.comp = None

    def fit(self, texts: Sequence[str]) -> "LSAEncoder":
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer

        rng = np.random.default_rng(self.seed)
        texts = np.asarray(texts, dtype=object)
        sample = texts[rng.choice(len(texts), min(self.fit_sample, len(texts)), replace=False)]
        self.vec = TfidfVectorizer(analyzer="char_wb", ngram_range=self.ngram, min_df=3, max_features=self.max_features,
                                   sublinear_tf=True, dtype=np.float32)
        X = self.vec.fit_transform(sample)
        sub = X[rng.choice(X.shape[0], min(self.svd_sample, X.shape[0]), replace=False)]
        svd = TruncatedSVD(n_components=self.dim, algorithm="randomized", n_iter=4, random_state=self.seed)
        svd.fit(sub)
        self.comp = svd.components_.T.astype(np.float32)  # (n_features, dim)
        return self

    def transform(self, texts: Sequence[str], chunk: int = 200_000, n_jobs: int = 1) -> np.ndarray:
        texts = list(texts)
        spans = [(s, min(s + chunk, len(texts))) for s in range(0, len(texts), chunk)]
        out = np.empty((len(texts), self.dim), np.float32)
        if n_jobs > 1 and len(spans) > 1:
            from joblib import Parallel, delayed

            parts = Parallel(n_jobs=n_jobs, backend="loky")(
                delayed(_lsa_chunk)(self.vec, self.comp, texts[a:b]) for a, b in spans)
            for (a, b), p in zip(spans, parts):
                out[a:b] = p
        else:
            for a, b in spans:
                out[a:b] = _lsa_chunk(self.vec, self.comp, texts[a:b])
        return out


def _lsa_chunk(vec, comp, texts):
    return l2n(np.asarray(vec.transform(texts) @ comp))


# ----------------------------------------------------------------------------------------------------------------
# Key blocking
# ----------------------------------------------------------------------------------------------------------------
def _h64(values) -> np.ndarray:
    """Vectorised 64-bit hash of string values (collisions negligible at our scale)."""
    return pd.util.hash_array(np.asarray(values, dtype=object)).view(np.int64)


def _explode_tokens(texts) -> tuple:
    """(record position, token) for every whitespace token of every text; empty texts yield nothing."""
    s = pd.Series(np.asarray(texts, dtype=object)).str.split().explode()
    s = s[s.notna() & (s != "")]
    return s.index.to_numpy(np.int64), s.to_numpy(dtype=object)


def _group_cross(ga: np.ndarray, gb: np.ndarray) -> tuple:
    """Index pairs (ia, ib) of the per-group cross product of two SORTED int group-label arrays."""
    ua, sa, na = np.unique(ga, return_index=True, return_counts=True)
    ub, sb, nb = np.unique(gb, return_index=True, return_counts=True)
    _, xa, xb = np.intersect1d(ua, ub, assume_unique=True, return_indices=True)
    return _expand(sa[xa], na[xa], sb[xb], nb[xb])


def _expand(sa, na, sb, nb) -> tuple:
    npair = na.astype(np.int64) * nb
    tot = int(npair.sum())
    if tot == 0:
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    g = np.repeat(np.arange(len(npair)), npair)
    t = np.arange(tot, dtype=np.int64) - np.repeat(np.cumsum(npair) - npair, npair)
    nbg = nb[g].astype(np.int64)
    return sa[g] + t // nbg, sb[g] + t % nbg


def record_keys(df: pd.DataFrame, spec) -> tuple:
    """(global idx, int64 key) rows of one key channel. spec = column name (the whole non-empty value is the key) or
    a list of columns (every combination of one whitespace token per column is a key, e.g. [name_key, addr_nums]
    -> 'name token x address number'). Keys always include the country (countries are an open set of labels)."""
    idx = df["idx"].to_numpy(np.int64)
    country = _h64(df["country_norm"].to_numpy())
    if isinstance(spec, str):
        v = df[spec].to_numpy(dtype=object)
        ok = v != ""
        return idx[ok], _h64(v[ok]) * np.int64(1000003) + country[ok]
    pos, h = None, None
    for j, col in enumerate(spec):
        tp, tt = _explode_tokens(df[col].to_numpy(dtype=object))
        th = _h64(tt) * np.int64(1000003 + 2 * j)
        if pos is None:
            pos, h = tp, country[tp] + th
            continue
        o = np.argsort(pos, kind="stable"); pos, h = pos[o], h[o]
        ia, ib = _group_cross(pos, tp)  # tp is sorted (explode keeps record order)
        pos, h = pos[ia], h[ia] * np.int64(7919) + th[ib]
    if pos is None or len(pos) == 0:
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    r, k = idx[pos], h
    o = np.lexsort((r, k))
    r, k = r[o], k[o]
    keep = np.r_[True, (k[1:] != k[:-1]) | (r[1:] != r[:-1])]  # one row per (record, key)
    return r[keep], k[keep]


def join_keys(ka: np.ndarray, ra: np.ndarray, kb: np.ndarray, rb: np.ndarray, max_a: int, max_b: int) -> tuple:
    """All (ra, rb) with equal key, restricted to blocks with |A| <= max_a and |B| <= max_b (large blocks carry
    no information and explode the pair count)."""
    oa = np.argsort(ka, kind="stable"); ka, ra = ka[oa], ra[oa]
    ob = np.argsort(kb, kind="stable"); kb, rb = kb[ob], rb[ob]
    ua, sa, na = np.unique(ka, return_index=True, return_counts=True)
    ub, sb, nb = np.unique(kb, return_index=True, return_counts=True)
    _, xa, xb = np.intersect1d(ua, ub, assume_unique=True, return_indices=True)
    ok = (na[xa] <= max_a) & (nb[xb] <= max_b)
    xa, xb = xa[ok], xb[ok]
    pa, pb = _expand(sa[xa], na[xa], sb[xb], nb[xb])
    return ra[pa], rb[pb]


def key_pairs(s1: pd.DataFrame, cand: pd.DataFrame, key_cols: Dict[str, object], max_block: int = 50,
              block_ratio: int = 8, max_block_by_chan: Optional[Dict[str, int]] = None) -> Dict[str, tuple]:
    """Key joins. key_cols: {channel: column | [columns]} (see record_keys). A block is used when it has at most
    max_block S1 records (per-channel override in max_block_by_chan) and at most max_block*block_ratio candidates.
    Returns {channel: (i1, i2)} (global idx arrays)."""
    out = {}
    mbc = max_block_by_chan or {}
    for chan, spec in key_cols.items():
        mb = int(mbc.get(chan, max_block))
        ra, ka = record_keys(s1, spec)
        rb, kb = record_keys(cand, spec)
        i1, i2 = join_keys(ka, ra, kb, rb, mb, mb * block_ratio)
        out[chan] = (i1, i2)
        log.info("key channel %s: %d pairs", chan, len(i1))
    return out


# ----------------------------------------------------------------------------------------------------------------
# Orchestration
# ----------------------------------------------------------------------------------------------------------------
def dense_pairs(vec1: np.ndarray, vec2: np.ndarray, s1_idx: np.ndarray, c_idx: np.ndarray, k_fwd: int, k_rev: int,
                chan: str, index_type: str = "auto") -> pd.DataFrame:
    """Bidirectional kNN inside one country block. vec1/vec2 are row-aligned with s1_idx/c_idx (global indices)."""
    frames = []
    if k_fwd > 0 and len(vec2) and len(vec1):
        D, I = knn(vec2, vec1, k_fwd, index_type)
        frames.append(pd.DataFrame({"i1": np.repeat(s1_idx, I.shape[1]), "i2": c_idx[I.ravel()],
                                    "sim": D.ravel().astype(np.float32), "rank": np.tile(np.arange(I.shape[1]), len(I)),
                                    "dir": "f"}))
    if k_rev > 0 and len(vec2) and len(vec1):
        D, I = knn(vec1, vec2, k_rev, index_type)
        frames.append(pd.DataFrame({"i1": s1_idx[I.ravel()], "i2": np.repeat(c_idx, I.shape[1]),
                                    "sim": D.ravel().astype(np.float32), "rank": np.tile(np.arange(I.shape[1]), len(I)),
                                    "dir": "r"}))
    if not frames:
        return pd.DataFrame(columns=["i1", "i2", "sim", "rank", "dir", "chan"])
    df = pd.concat(frames, ignore_index=True)
    df = df[df.i2 >= 0]
    df["chan"] = chan
    return df


def generate_candidates(s1: pd.DataFrame, cand: pd.DataFrame, vectors: Dict[str, tuple], cfg: dict,
                        label_fn=None, model_path: Optional[str] = None) -> pd.DataFrame:
    """
    s1 / cand: normalised frames with columns idx (global int), country_norm, key columns (+ PRESCORE_FLAGS).
    vectors: {channel: (vec_s1, vec_cand)} row-aligned with s1 / cand.
    label_fn(i1, i2) -> 0/1 (TRAIN only): fit the pre-score model (saved to model_path); otherwise the model at
    model_path is applied (see merge_channels).
    Returns the selected unique (i1, i2) pairs with per-channel best rank/sim columns, key flags, cos_* of every
    vector channel and the pre-score.
    """
    k_fwd, k_rev = cfg.get("k_fwd", 30), cfg.get("k_rev", 8)
    hits: Dict[str, List[tuple]] = {}
    for country in sorted(set(s1.country_norm) | set(cand.country_norm)):
        m1 = (s1.country_norm == country).to_numpy()
        m2 = (cand.country_norm == country).to_numpy()
        if not m1.any() or not m2.any():
            continue
        log.info("blocking country=%s  n1=%d n2=%d", country, m1.sum(), m2.sum())
        g1, g2 = s1.idx.to_numpy()[m1], cand.idx.to_numpy()[m2]
        for chan, (v1, v2) in vectors.items():
            kf = cfg.get(f"k_fwd_{chan}", k_fwd)
            kr = cfg.get(f"k_rev_{chan}", k_rev)
            it = cfg.get("index_type", "auto")
            if kf > 0:
                D, I = knn(v2[m2], v1[m1], kf, it)
                ok = I.ravel() >= 0
                hits.setdefault(f"{chan}_f", []).append((np.repeat(g1, I.shape[1])[ok], g2[I.ravel()[ok]],
                                                         D.ravel()[ok], np.tile(np.arange(I.shape[1]), len(I))[ok]))
            if kr > 0:
                D, I = knn(v1[m1], v2[m2], kr, it)
                ok = I.ravel() >= 0
                hits.setdefault(f"{chan}_r", []).append((g1[I.ravel()[ok]], np.repeat(g2, I.shape[1])[ok],
                                                         D.ravel()[ok], np.tile(np.arange(I.shape[1]), len(I))[ok]))
            log.info("  channel %s done", chan)
    keys = key_pairs(s1, cand, cfg.get("key_cols", {}), cfg.get("max_block", 50), cfg.get("block_ratio", 8),
                     cfg.get("key_max_block"))
    for chan, (a, b) in keys.items():
        z = np.zeros(len(a), np.float32)
        hits.setdefault(f"key_{chan}", []).append((a, b, z + 1, z.astype(np.int16)))
    del keys
    n_max = int(max(s1.idx.max(), cand.idx.max())) + 1
    # global idx -> row of the vectors / flags (s1 and cand rows are disjoint idx sets)
    pos = np.full(n_max, -1, np.int64)
    pos[s1.idx.to_numpy()] = np.arange(len(s1))
    pos[cand.idx.to_numpy()] = np.arange(len(cand))
    flags = {}
    for col in PRESCORE_FLAGS:
        if col in s1.columns and col in cand.columns:
            f = np.zeros(n_max, np.int8)
            f[s1.idx.to_numpy()] = s1[col].to_numpy().astype(np.int8)
            f[cand.idx.to_numpy()] = cand[col].to_numpy().astype(np.int8)
            flags[col] = f
    return merge_channels(hits, n_max, cfg, vectors=vectors, pos=pos, flags=flags, n_s1=len(s1),
                          label_fn=label_fn, model_path=model_path)


# record flags handed to the pre-scorer (both sides), e.g. an empty candidate address makes cos_addr meaningless
PRESCORE_FLAGS = ("addr_empty", "name_had_indic", "name_is_domain")


def _group_max(g: np.ndarray, v: np.ndarray, n: int) -> np.ndarray:
    o = np.full(n, -np.inf, np.float32)
    np.maximum.at(o, g, v)
    return o


def group_rank(g: np.ndarray, score: np.ndarray) -> np.ndarray:
    """0-based rank of each row inside its group g by descending score (ties -> input order)."""
    order = np.lexsort((-score, g))
    gs = g[order]
    start = np.r_[0, np.flatnonzero(gs[1:] != gs[:-1]) + 1]
    r = np.arange(len(gs)) - np.repeat(start, np.diff(np.r_[start, len(gs)]))
    out = np.empty(len(g), np.int32)
    out[order] = r
    return out


def merge_channels(hits: Dict[str, List[tuple]], n_max: int, cfg: dict, vectors: Optional[Dict[str, tuple]] = None,
                   pos: Optional[np.ndarray] = None, flags: Optional[Dict[str, np.ndarray]] = None,
                   n_s1: Optional[int] = None, label_fn=None, model_path: Optional[str] = None) -> pd.DataFrame:
    """One row per unique (i1, i2): max sim / min rank per channel-direction, key flags, the cosine of EVERY vector
    channel (so key-only pairs are scored like the others), a pre-score and the candidate selection.

    Pre-score: a small LightGBM on blocking-only features (see prescore_features) fitted on TRAIN labels
    (cfg select.model; cross-fitted so train pairs get out-of-fold scores) and applied unchanged to test; without a
    model the heuristic max-cosine + 0.05*keys is used. Selection (cfg select): a global budget of avg_per_s1 * n_s1
    pairs taken by descending pre-score, plus every candidate's top min_per_cand S1s and every S1's top min_per_s1.
    Without cfg select the legacy per-S1 cap (max_cand_per_s1 + always_keep_rev_rank) is applied.

    Memory-lean: pairs are int64 keys i1*n_max+i2, merged via np.unique + searchsorted (no pivot, no string cols)."""
    cat = {n: tuple(np.concatenate([p[j] for p in parts]) for j in range(4)) for n, parts in hits.items() if parts}
    hits.clear()
    if not cat:
        return pd.DataFrame(columns=["i1", "i2", "blk_prescore", "blk_n_channels"])
    uniq = np.unique(np.concatenate([a[0].astype(np.int64) * n_max + a[1].astype(np.int64) for a in cat.values()]))
    out = {"i1": (uniq // n_max).astype(np.int64), "i2": (uniq % n_max).astype(np.int64)}
    nch = np.zeros(len(uniq), np.int16)
    nkey = np.zeros(len(uniq), np.int8)
    pre = np.zeros(len(uniq), np.float32)
    min_rev_rank = np.full(len(uniq), 999, np.int16)
    for n in sorted(cat):
        i1, i2, sim, rank = cat.pop(n)
        p = np.searchsorted(uniq, i1.astype(np.int64) * n_max + i2.astype(np.int64))
        del i1, i2
        if n.startswith("key_"):
            col = np.zeros(len(uniq), np.int8)
            col[p] = 1
            out[f"blk_{n}"] = col
            nkey += col
            nch += col
            continue
        s = np.full(len(uniq), -np.inf, np.float32)
        np.maximum.at(s, p, sim.astype(np.float32))
        r = np.full(len(uniq), 999, np.int16)
        np.minimum.at(r, p, rank.astype(np.int16))
        hit = np.isfinite(s)
        nch += hit
        out[f"blk_rank_{n}"] = r
        pre = np.maximum(pre, np.where(hit, s, 0))
        s[~hit] = np.nan
        out[f"blk_sim_{n}"] = s
        if n.endswith("_r"):
            min_rev_rank = np.minimum(min_rev_rank, r)
    del uniq
    out["blk_n_channels"] = nch
    if vectors and pos is not None:
        # every channel's cosine for every pair (features must not depend on which channel retrieved the pair)
        p1, p2 = pos[out["i1"]], pos[out["i2"]]
        for ch, (v1, v2) in vectors.items():
            cos = np.empty(len(p1), np.float32)
            for s in range(0, len(p1), 1_000_000):
                cos[s:s + 1_000_000] = np.einsum("ij,ij->i", np.asarray(v1[p1[s:s + 1_000_000]], np.float32),
                                                 np.asarray(v2[p2[s:s + 1_000_000]], np.float32))
            out[f"cos_{ch}"] = cos
            pre = np.maximum(pre, cos)
        del p1, p2
    df = pd.DataFrame(out)
    del out
    sel = cfg.get("select")
    if not sel:
        df["blk_prescore"] = pre + 0.05 * nkey
        # legacy: cap per S1, but always keep pairs where the S1 is among the candidate's top reverse neighbours
        cap = cfg.get("max_cand_per_s1", 60)
        order = np.lexsort((-df["blk_prescore"].to_numpy(), df["i1"].to_numpy()))
        df = df.iloc[order].reset_index(drop=True)
        rnk = df.groupby("i1").cumcount().to_numpy()
        keep = (rnk < cap) | (min_rev_rank[order] < cfg.get("always_keep_rev_rank", 3))
        log.info("merge_channels: %d unique pairs -> %d after per-S1 cap %d", len(df), int(keep.sum()), cap)
        return df[keep].reset_index(drop=True)
    del min_rev_rank
    X = prescore_features(df, list(vectors or {}), flags or {}, n_max)
    score = None
    if label_fn is not None and sel.get("fit", True):
        score = fit_prescore(X, label_fn(df["i1"].to_numpy(), df["i2"].to_numpy()), df["i1"].to_numpy(), sel,
                             model_path)
    elif model_path and os.path.exists(model_path):
        score = predict_prescore(X, model_path)
    else:
        log.warning("merge_channels: no pre-score model (%s) -> heuristic max-cosine pre-score", model_path)
    if score is None:
        score = pre + 0.05 * nkey
    del X, pre
    df["blk_prescore"] = score.astype(np.float32)
    keep = select_mask(df["i1"].to_numpy(), df["i2"].to_numpy(), df["blk_prescore"].to_numpy(), sel,
                       n_s1 or df["i1"].nunique())
    log.info("merge_channels: %d unique pairs -> %d selected (%.1f / S1)", len(df), int(keep.sum()),
             keep.sum() / max(1, n_s1 or df["i1"].nunique()))
    return df[keep].reset_index(drop=True)


def prescore_features(df: pd.DataFrame, vec_chans: Sequence[str], flags: Dict[str, np.ndarray], n_max: int) -> Dict[str, np.ndarray]:
    """Blocking-only features for the pre-scorer, computed over the full (pre-selection) union: per-channel ranks /
    key flags, every cosine and its margin to the best cosine of the same candidate / S1, the within-candidate and
    within-S1 rank, candidate-list sizes and record flags. All label-free and country-agnostic."""
    i1, i2 = df["i1"].to_numpy(), df["i2"].to_numpy()
    X: Dict[str, np.ndarray] = {}
    for c in df.columns:
        if c.startswith(("blk_rank_", "blk_key_")) or c == "blk_n_channels":
            X[c] = df[c].to_numpy()
    for ch in vec_chans:
        c = df[f"cos_{ch}"].to_numpy()
        X[f"cos_{ch}"] = c
        X[f"pre_dc_{ch}"] = (c - _group_max(i2, c, n_max)[i2]).astype(np.float16)
        X[f"pre_ds_{ch}"] = (c - _group_max(i1, c, n_max)[i1]).astype(np.float16)
        X[f"pre_rc_{ch}"] = np.minimum(group_rank(i2, c), 999).astype(np.int16)
        X[f"pre_rs_{ch}"] = np.minimum(group_rank(i1, c), 999).astype(np.int16)
    X["pre_nc"] = np.minimum(np.bincount(i2, minlength=n_max)[i2], 30000).astype(np.int16)
    X["pre_ns"] = np.minimum(np.bincount(i1, minlength=n_max)[i1], 30000).astype(np.int16)
    for name, f in flags.items():
        X[f"pre_{name}_1"] = f[i1]
        X[f"pre_{name}_2"] = f[i2]
    return X


def _matrix(X: Dict[str, np.ndarray], cols: Sequence[str], rows) -> np.ndarray:
    n = len(next(iter(X.values())))
    return np.column_stack([np.asarray(X[c][rows], np.float32) if c in X else np.full(n, np.nan, np.float32)[rows]
                            for c in cols])


def _lgb_params(sel: dict) -> dict:
    p = dict(objective="binary", learning_rate=0.1, num_leaves=31, min_data_in_leaf=200, feature_fraction=0.9,
             bagging_fraction=0.8, bagging_freq=1, num_threads=int(sel.get("n_jobs", 2)), verbose=-1, seed=0)
    p.update(sel.get("lgb_params", {}))
    return p


def fit_prescore(X: Dict[str, np.ndarray], y: np.ndarray, i1: np.ndarray, sel: dict, model_path: Optional[str]) -> np.ndarray:
    """Cross-fitted pre-score on TRAIN: 2 folds by S1 -> out-of-fold scores for the train pairs (so train candidates
    are selected exactly as test ones will be), plus a model on all pairs saved to model_path for test.
    Negatives are subsampled (neg_frac, re-weighted) to keep the fit small."""
    import lightgbm as lgb

    cols = sorted(X)
    rounds = int(sel.get("rounds", 150))
    neg_frac = float(sel.get("neg_frac", 0.15))
    rng = np.random.default_rng(int(sel.get("seed", 0)))
    y = np.asarray(y).astype(bool)
    sub = y | (rng.random(len(y)) < neg_frac)
    fold = (np.random.default_rng(12345).random(int(i1.max()) + 1) < 0.5)[i1]
    w_all = np.where(y, 1.0, 1.0 / neg_frac).astype(np.float32)
    params = _lgb_params(sel)
    score = np.empty(len(y), np.float32)
    for f in (False, True):
        tr = np.flatnonzero(sub & (fold != f))
        m = lgb.train(params, lgb.Dataset(_matrix(X, cols, tr), y[tr].astype(np.float32), weight=w_all[tr],
                                          feature_name=cols), rounds)
        te = np.flatnonzero(fold == f)
        for s in range(0, len(te), 2_000_000):
            score[te[s:s + 2_000_000]] = m.predict(_matrix(X, cols, te[s:s + 2_000_000]))
        del m
    tr = np.flatnonzero(sub)
    m = lgb.train(params, lgb.Dataset(_matrix(X, cols, tr), y[tr].astype(np.float32), weight=w_all[tr],
                                      feature_name=cols), rounds)
    imp = sorted(zip(m.feature_importance("gain"), cols), reverse=True)[:10]
    log.info("prescore fitted on %d pairs (%d pos); top features %s", len(tr), int(y.sum()),
             [(c, int(g)) for g, c in imp])
    if model_path:
        os.makedirs(os.path.dirname(model_path) or ".", exist_ok=True)
        m.save_model(model_path)
    return score


def predict_prescore(X: Dict[str, np.ndarray], model_path: str) -> np.ndarray:
    import lightgbm as lgb

    m = lgb.Booster(model_file=model_path)
    cols = m.feature_name()
    miss = [c for c in cols if c not in X]
    if miss:
        log.warning("prescore model expects missing features %s (config changed since fit?) -> NaN", miss)
    n = len(next(iter(X.values())))
    score = np.empty(n, np.float32)
    for s in range(0, n, 2_000_000):
        score[s:s + 2_000_000] = m.predict(_matrix(X, cols, np.arange(s, min(n, s + 2_000_000))))
    return score


def select_mask(i1: np.ndarray, i2: np.ndarray, score: np.ndarray, sel: dict, n_s1: int) -> np.ndarray:
    """Global-budget selection: every candidate's top `min_per_cand` S1s and every S1's top `min_per_s1`
    candidates, then the highest pre-scores until avg_per_s1 * n_s1 pairs (no S1 above max_per_s1 via the budget).
    Each S2/S3 record matches at most one S1, so a per-candidate floor protects recall cheaply; the global budget
    spends the rest where the pre-score says matches are likely (label-free at test time)."""
    rc = group_rank(i2, score)
    rs = group_rank(i1, score)
    forced = (rc < int(sel.get("min_per_cand", 1))) | (rs < int(sel.get("min_per_s1", 0)))
    budget = int(float(sel.get("avg_per_s1", 55)) * n_s1) - int(forced.sum())
    free = ~forced
    if sel.get("max_per_s1"):
        free &= rs < int(sel["max_per_s1"])
    if sel.get("min_prob") is not None:
        free &= score >= float(sel["min_prob"])
    keep = forced.copy()
    if budget > 0 and free.any():
        cand_idx = np.flatnonzero(free)
        if budget < len(cand_idx):
            top = np.argpartition(-score[cand_idx], budget - 1)[:budget]
            cand_idx = cand_idx[top]
        keep[cand_idx] = True
    return keep
