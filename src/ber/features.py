"""Pairwise + contextual features for (S1, candidate) pairs.

Groups:
  * name  : rapidfuzz battery on several views (norm / core / key / sorted / concat / skeleton), max over DBA parts,
            IDF-weighted token overlap (IDF computed per split & country on the split's own corpus -> works for France),
            name-rarity mass (a shared generic name "chennai services" is weak evidence, a shared rare name is strong),
            insertion evidence (candidate = S1 name + 1-2 inserted tokens: position, S1 doc-frequency of the inserted
            token -- typos ~0, category words high --, generic-word flag, and an OOF "extra token harm" encoding learned
            on near-duplicate pairs only: the generator builds distractors by inserting category words into an S1 name),
            legal-form agreement, domain/handle comparisons against the concatenated name, script/translit flags.
  * addr  : house-number equality / numeric-set Jaccard / numeric edit distance, street & locality similarity,
            locality similarity with SOURCE-SPECIFIC admin tokens removed (label-free; e.g. French regions only in S1,
            departements only in S2/S3), state agreement (NaN when a side has no recognised state; train applies
            state-dropout so NaN is not learned as "empty address"), unit / PO-box flags, empty-address flags.
  * block : channel sims / ranks from blocking (+ per-channel cosine recomputed for every pair).
  * freq  : same-name counts in S1 / candidates (chains & generic names), score-free address competition ranks.
  * ctx   : within-S1 and within-candidate rank / margin / count features on a base score (stage 2, OOF scores).
No feature uses the country label itself (open-set countries).
"""
from __future__ import annotations

import math
import os
import zlib
from collections import Counter
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler, Levenshtein

from .normalize import GENERIC_WORDS, skeleton_str


# ----------------------------------------------------------------------------------------------------------------
# Per-split, per-country label-free statistics
# ----------------------------------------------------------------------------------------------------------------
class IDF:
    """IDF over name_core+addr_norm, S1 name-token doc-freq, and source-specific locality tokens, per country value."""

    def __init__(self, docs_by_country: Dict[str, Iterable[str]], min_df: int = 2):
        self.tab: Dict[str, Dict[str, float]] = {}
        self.default: Dict[str, float] = {}
        self.s1df: Dict[str, Dict[str, float]] = {}
        self.x1: Dict[str, set] = {}
        self.x2: Dict[str, set] = {}
        for c, docs in docs_by_country.items():
            df = Counter()
            n = 0
            for d in docs:
                n += 1
                df.update(set(d.split()))
            self.tab[c] = {t: math.log((n + 1) / (v + 1)) + 1.0 for t, v in df.items() if v >= min_df}
            self.default[c] = math.log((n + 1) / 2) + 1.0

    @classmethod
    def from_records(cls, rec: pd.DataFrame, min_df: int = 2, ratio: float = 20.0, min_tok_df: int = 30) -> "IDF":
        docs = {c: (g["name_core"] + " " + g["addr_norm"]).tolist() for c, g in rec.groupby("country_norm")}
        obj = cls(docs, min_df)
        for c, g in rec.groupby("country_norm"):
            is1 = (g["src"] == 1).to_numpy()
            names1 = g["name_core"].to_numpy()[is1]
            d = Counter(t for s in names1 for t in set(s.split()))
            obj.s1df[c] = {t: math.log1p(v) for t, v in d.items()}
            loc = g["addr_locality"].to_numpy()
            d1 = Counter(t for s in loc[is1] for t in set(s.split()))
            d2 = Counter(t for s in loc[~is1] for t in set(s.split()))
            n1, n2 = max(1, int(is1.sum())), max(1, int((~is1).sum()))
            both = d1.keys() | d2.keys()
            obj.x2[c] = {t for t in both if d1[t] + d2[t] >= min_tok_df and ((d2[t] + 1) / n2) / ((d1[t] + 1) / n1) > ratio}
            obj.x1[c] = {t for t in both if d1[t] + d2[t] >= min_tok_df and ((d1[t] + 1) / n1) / ((d2[t] + 1) / n2) > ratio}
        return obj

    def get(self, country: str):
        return self.tab.get(country, {}), self.default.get(country, 10.0)

    def ctx(self, country: str) -> dict:
        return {"idf": self.tab.get(country, {}), "dflt": self.default.get(country, 10.0),
                "s1df": self.s1df.get(country, {}), "x1": self.x1.get(country, set()), "x2": self.x2.get(country, set())}


def _idf_overlap(a: str, b: str, idf: dict, dflt: float):
    """Returns jac, cov_a, extra_a, extra_b, cov_b, (max_extra_a, max_extra_b), (shared, mass_a, mass_b)."""
    ta, tb = set(a.split()), set(b.split())
    if not ta and not tb:
        return np.nan, np.nan, 0.0, 0.0, np.nan, (0.0, 0.0), (0.0, 0.0, 0.0)
    wa = {t: idf.get(t, dflt) for t in ta}
    wb = {t: idf.get(t, dflt) for t in tb}
    inter = ta & tb
    wi = sum(wa[t] for t in inter)
    sa, sb = sum(wa.values()), sum(wb.values())
    jac = wi / (sa + sb - wi) if (sa + sb - wi) > 0 else np.nan
    cov_a = wi / sa if sa else np.nan
    cov_b = wi / sb if sb else np.nan
    max_extra_b = max((wb[t] for t in tb - ta), default=0.0)
    max_extra_a = max((wa[t] for t in ta - tb), default=0.0)
    return jac, cov_a, sa - wi, sb - wi, cov_b, (max_extra_a, max_extra_b), (wi, sa, sb)


def _soft_overlap(a: str, b: str, idf: dict, dflt: float, thr: float = 0.85):
    """Soft-TFIDF: IDF-weighted share of tokens of a that have a JW>=thr partner in b (typo tolerant)."""
    ta, tb = a.split(), b.split()
    if not ta or not tb:
        return np.nan
    tot = hit = 0.0
    for t in set(ta):
        w = idf.get(t, dflt)
        tot += w
        best = max(JaroWinkler.normalized_similarity(t, u) for u in tb)
        if best >= thr:
            hit += w * best
    return hit / tot if tot else np.nan


def _monge_elkan(a: str, b: str):
    ta, tb = a.split(), b.split()
    if not ta or not tb:
        return np.nan
    return sum(max(JaroWinkler.normalized_similarity(t, u) for u in tb) for t in ta) / len(ta)


def _num_feats(na: str, nb: str):
    A, B = na.split(), nb.split()
    if not A or not B:
        return np.nan, np.nan, np.nan, np.nan
    sa, sb = set(A), set(B)
    jac = len(sa & sb) / len(sa | sb)
    first_eq = float(A[0] == B[0])
    first_lev = Levenshtein.normalized_similarity(A[0], B[0])  # typo-tolerant (975 vs 976, 11/6 vs 1/6)
    any_first = float(A[0] in sb or B[0] in sa)
    return jac, first_eq, first_lev, any_first


def _str_battery(prefix: str, a: str, b: str, out: dict, full: bool = True):
    if not a or not b:
        for k in ("ratio", "tset", "tsort", "partial", "jw", "lev") if full else ("ratio", "tset", "jw"):
            out[f"{prefix}_{k}"] = np.nan
        return
    out[f"{prefix}_ratio"] = fuzz.ratio(a, b) / 100.0
    out[f"{prefix}_tset"] = fuzz.token_set_ratio(a, b) / 100.0
    out[f"{prefix}_jw"] = JaroWinkler.normalized_similarity(a, b)
    if full:
        out[f"{prefix}_tsort"] = fuzz.token_sort_ratio(a, b) / 100.0
        out[f"{prefix}_partial"] = fuzz.partial_ratio(a, b) / 100.0
        out[f"{prefix}_lev"] = Levenshtein.normalized_similarity(a, b)


def _drop(s: str, bad: set) -> str:
    return " ".join(t for t in s.split() if t not in bad) if bad else s


def pair_features(r1: dict, r2: dict, cx: dict) -> dict:
    """r1 = S1 normalised record, r2 = candidate normalised record; cx = IDF.ctx(country)."""
    idf, dflt, s1df = cx["idf"], cx["dflt"], cx["s1df"]
    f: dict = {}
    # legal/honorific-only names ('La & Co') have an empty core: fall back to the full normalised name
    n1 = r1["name_core"] or r1["name_norm"]
    n2 = r2["name_core"] or r2["name_norm"]
    # ---------------- name
    _str_battery("nm_core", n1, n2, f)
    _str_battery("nm_norm", r1["name_norm"], r2["name_norm"], f, full=False)
    _str_battery("nm_key", r1["name_key"], r2["name_key"], f, full=False)
    f["nm_sorted_eq"] = float(bool(n1) and r1["name_sorted"] == r2["name_sorted"])
    f["nm_core_eq"] = float(bool(n1) and n1 == n2)
    c1, c2 = r1["name_concat"], r2["name_concat"]
    f["nm_concat_ratio"] = fuzz.ratio(c1, c2) / 100.0 if c1 and c2 else np.nan
    f["nm_concat_partial"] = fuzz.partial_ratio(c1, c2) / 100.0 if c1 and c2 else np.nan
    full1 = r1["name_norm"].replace(" ", "")
    f["nm_concat_full_partial"] = fuzz.partial_ratio(full1, c2) / 100.0 if full1 and c2 else np.nan
    f["nm_prefix_share"] = (len(_common_prefix(c1, c2)) / max(1, min(len(c1), len(c2)))) if c1 and c2 else np.nan
    sk1, sk2 = skeleton_str(n1), skeleton_str(n2)
    f["nm_skel_tset"] = fuzz.token_set_ratio(sk1, sk2) / 100.0 if sk1 and sk2 else np.nan
    f["nm_skel_ratio"] = fuzz.ratio(sk1, sk2) / 100.0 if sk1 and sk2 else np.nan
    f["nm_monge_elkan"] = _monge_elkan(n2, n1)
    f["nm_soft_c_in_s1"] = _soft_overlap(n2, n1, idf, dflt)
    f["nm_soft_s1_in_c"] = _soft_overlap(n1, n2, idf, dflt)
    jac, cov1, ex1, ex2, cov2, (mx1, mx2), (wi, sa, sb) = _idf_overlap(n1, n2, idf, dflt)
    f.update(nm_idf_jac=jac, nm_idf_cov_s1=cov1, nm_idf_cov_c=cov2, nm_extra_s1=ex1, nm_extra_c=ex2,
             nm_max_extra_s1=mx1, nm_max_extra_c=mx2, nm_idf_shared=wi, nm_idf_s1=sa, nm_idf_c=sb)
    toks1, toks2 = n1.split(), n2.split()
    t1, t2 = set(toks1), set(toks2)
    f["nm_idf_min_s1"] = min((idf.get(t, dflt) for t in t1), default=np.nan)
    extra_c = t2 - t1
    extra_s1 = t1 - t2
    f["nm_n_extra_c"] = len(extra_c)
    f["nm_n_extra_s1"] = len(extra_s1)
    f["nm_ntok_s1"] = len(t1)
    f["nm_ntok_c"] = len(t2)
    # insertion pattern: candidate = S1 name + inserted token(s)
    f["nm_is_superset"] = float(bool(t1) and t1 < t2)
    ex_list = [t for t in toks2 if t not in t1]
    f["nm_extra_pos"] = toks2.index(ex_list[0]) / max(1, len(toks2) - 1) if ex_list else np.nan
    f["nm_extra_s1df_max"] = max((s1df.get(t, 0.0) for t in extra_c), default=np.nan)
    f["nm_extra_generic"] = float(any(t in GENERIC_WORDS for t in extra_c)) if extra_c else np.nan
    # best over DBA parts (alias usually carries the true name)
    parts = [p for p in r2["name_parts"].split("|") if p] or [n2]
    f["nm_parts_best_tset"] = max(fuzz.token_set_ratio(n1, p) for p in parts) / 100.0 if n1 else np.nan
    f["nm_parts_best_ratio"] = max(fuzz.ratio(n1, p) for p in parts) / 100.0 if n1 else np.nan
    f["nm_has_dba_c"] = float(r2["name_has_dba"])
    f["nm_is_domain_c"] = float(r2["name_is_domain"])
    f["nm_had_indic_c"] = float(r2["name_had_indic"])
    l1, l2 = set(r1["name_legal"].split()), set(r2["name_legal"].split())
    f["nm_legal_eq"] = float(l1 == l2) if (l1 or l2) else np.nan
    f["nm_legal_jac"] = len(l1 & l2) / len(l1 | l2) if (l1 | l2) else np.nan
    f["nm_legal_conflict"] = float(bool(l1) and bool(l2) and not (l1 & l2))
    f["nm_len_ratio"] = min(len(n1), len(n2)) / max(1, max(len(n1), len(n2)))
    # ---------------- address
    a1, a2 = r1["addr_norm"], r2["addr_norm"]
    f["ad_empty_c"] = float(r2["addr_empty"])
    _str_battery("ad", a1, a2, f)
    jac, cov1, ex1, ex2, cov2, (_, mx2), _ = _idf_overlap(a1, a2, idf, dflt)
    f.update(ad_idf_jac=jac, ad_idf_cov_s1=cov1, ad_idf_cov_c=cov2, ad_extra_s1=ex1, ad_extra_c=ex2, ad_max_extra_c=mx2)
    # source-specific admin tokens removed (French region in S1 vs departement in S2/S3, generator-only tokens)
    a1x, a2x = _drop(a1, cx["x1"]), _drop(a2, cx["x2"])
    f["ad_tset_x"] = fuzz.token_set_ratio(a1x, a2x) / 100.0 if (a1x and a2x) else np.nan
    nj, feq, flev, fany = _num_feats(r1["addr_nums"], r2["addr_nums"])
    f.update(ad_num_jac=nj, ad_num_first_eq=feq, ad_num_first_lev=flev, ad_num_first_any=fany)
    h1, h2 = r1["addr_house"], r2["addr_house"]
    f["ad_house_eq"] = float(h1 == h2) if (h1 and h2) else np.nan
    f["ad_house_lev"] = Levenshtein.normalized_similarity(h1, h2) if (h1 and h2) else np.nan
    f["ad_house_missing_c"] = float(not h2)
    # house-number arithmetic: distractors sit at NEIGHBOURING numbers (1133 vs 1135: small diff, same parity) while
    # the noise process makes DIGIT edits (2482 -> 482, 975 -> 976 as a typo) -> let the model separate the two
    n1, n2 = _lead_int(h1), _lead_int(h2)
    if n1 is not None and n2 is not None:
        d = abs(n1 - n2)
        f["ad_house_absdiff_log"] = math.log1p(d)
        f["ad_house_reldiff"] = d / max(1, max(n1, n2))
        f["ad_house_same_parity"] = float(n1 % 2 == n2 % 2)
        s1n, s2n = str(n1), str(n2)
        f["ad_house_len_diff"] = float(abs(len(s1n) - len(s2n)))
        f["ad_house_digit_edits"] = float(Levenshtein.distance(s1n, s2n))
        f["ad_house_is_substr"] = float(s1n in s2n or s2n in s1n)
    else:
        for k in ("ad_house_absdiff_log", "ad_house_reldiff", "ad_house_same_parity", "ad_house_len_diff",
                  "ad_house_digit_edits", "ad_house_is_substr"):
            f[k] = np.nan
    f["c_src"] = float(r2.get("src", 0))
    s1s, s2s = r1["addr_street"], r2["addr_street"]
    f["ad_street_tset"] = fuzz.token_set_ratio(s1s, s2s) / 100.0 if (s1s and s2s) else np.nan
    f["ad_street_ratio"] = fuzz.ratio(s1s, s2s) / 100.0 if (s1s and s2s) else np.nan
    lo1, lo2 = r1["addr_locality"], r2["addr_locality"]
    f["ad_loc_tset"] = fuzz.token_set_ratio(lo1, lo2) / 100.0 if (lo1 and lo2) else np.nan
    f["ad_loc_partial"] = fuzz.partial_ratio(lo1, lo2) / 100.0 if (lo1 and lo2) else np.nan
    f["ad_loc_best_tok"] = _best_token_sim(lo1, lo2)
    lo1x, lo2x = _drop(lo1, cx["x1"]), _drop(lo2, cx["x2"])
    f["ad_loc_tset_x"] = fuzz.token_set_ratio(lo1x, lo2x) / 100.0 if (lo1x and lo2x) else np.nan
    st1, st2 = r1["addr_state"], r2["addr_state"]
    f["ad_state_eq"] = float(st1 == st2) if (st1 and st2) else np.nan  # NaN when unknown (e.g. unseen countries)
    f["ad_pobox_c"] = float(bool(r2["addr_pobox"]))
    f["ad_unit_c"] = float(bool(r2["addr_unit"]))
    f["ad_unit_tset"] = fuzz.token_set_ratio(r1["addr_unit"], r2["addr_unit"]) / 100.0 if (r1["addr_unit"] and r2["addr_unit"]) else np.nan
    f["ad_nsegs_c"] = r2["addr_n_segs"]
    f["x_name_in_addr"] = len(t2 & set(a1.split())) / max(1, len(t2))
    return f


_LEAD_INT_RE = __import__("re").compile(r"\d+")


def _lead_int(s: str):
    if not s:
        return None
    m = _LEAD_INT_RE.search(s)
    if not m or len(m.group(0)) > 9:
        return None
    return int(m.group(0))


def _common_prefix(a: str, b: str) -> str:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return a[:i]


def _best_token_sim(a: str, b: str):
    ta, tb = a.split(), b.split()
    if not ta or not tb:
        return np.nan
    return max(JaroWinkler.normalized_similarity(x, y) for x in ta for y in tb)


# ----------------------------------------------------------------------------------------------------------------
# Batch computation (each task ships only the records its pairs touch -> bounded memory on spawn and fork)
# ----------------------------------------------------------------------------------------------------------------
REC_FIELDS = ["name_norm", "name_core", "name_key", "name_sorted", "name_concat", "name_legal", "name_parts",
              "name_is_domain", "name_has_dba", "name_had_indic", "addr_norm", "addr_street", "addr_house",
              "addr_nums", "addr_locality", "addr_state", "addr_unit", "addr_pobox", "addr_n_segs", "addr_empty",
              "country_norm", "src"]

_W: dict = {}


def _init_worker(idf_obj):
    _W["idf"] = idf_obj
    _W["cx"] = {}


def _work(task):
    j0, loc1, loc2, cols = task
    recs = [dict(zip(REC_FIELDS, v)) for v in zip(*(cols[c] for c in REC_FIELDS))]
    names, out = None, None
    cache = _W["cx"]
    for j, (a, b) in enumerate(zip(loc1, loc2)):
        c = recs[a]["country_norm"]
        cx = cache.get(c)
        if cx is None:
            cx = cache[c] = _W["idf"].ctx(c)
        f = pair_features(recs[a], recs[b], cx)
        if out is None:
            names = list(f)
            out = np.empty((len(loc1), len(names)), np.float32)
        out[j] = [f[k] for k in names]
    return j0, names, out


def _bounded_imap(pool, fn, tasks, max_inflight: int):
    """Like imap_unordered, but at most `max_inflight` task payloads exist at once (Pool.imap eagerly materialises
    every task of the iterable -> several GB for a 4M-pair part on a 16GB laptop)."""
    from collections import deque

    pending = deque()
    for t in tasks:
        pending.append(pool.apply_async(fn, (t,)))
        while len(pending) >= max_inflight:
            yield pending.popleft().get()
    while pending:
        yield pending.popleft().get()


def _tasks(i1, i2, rec_cols, chunk):
    for s in range(0, len(i1), chunk):
        a, b = i1[s:s + chunk], i2[s:s + chunk]
        u, inv = np.unique(np.r_[a, b], return_inverse=True)
        yield s, inv[:len(a)], inv[len(a):], {c: [rec_cols[c][x] for x in u] for c in REC_FIELDS}


def compute_pair_features(pairs: pd.DataFrame, rec: pd.DataFrame, idf_obj: IDF, extra_harm=None,
                          n_jobs: int = 1, chunk: int = 50_000) -> pd.DataFrame:
    """pairs: columns i1, i2 (row positions into `rec`). Returns float32 frame aligned with `pairs`."""
    rec_cols = {c: rec[c].to_numpy(dtype=object) for c in REC_FIELDS}
    i1, i2 = pairs["i1"].to_numpy(), pairs["i2"].to_numpy()
    res: Optional[np.ndarray] = None
    names: Optional[List[str]] = None
    try:
        if n_jobs <= 1:
            _init_worker(idf_obj)
            it = map(_work, _tasks(i1, i2, rec_cols, chunk))
            results = it
            pool = None
        else:
            from multiprocessing import get_context

            ctx = get_context("spawn" if os.name == "nt" else "fork")
            pool = ctx.Pool(n_jobs, initializer=_init_worker, initargs=(idf_obj,))
            results = _bounded_imap(pool, _work, _tasks(i1, i2, rec_cols, chunk), 2 * n_jobs)
        for j0, nm, out in results:
            if out is None:
                continue
            if res is None:
                names = nm
                res = np.empty((len(pairs), len(names)), np.float32)
            res[j0:j0 + len(out)] = out
        if pool is not None:
            pool.close(); pool.join()
    finally:
        _W.clear()
    if res is None:
        return pd.DataFrame(index=pairs.index)
    return pd.DataFrame(res, columns=names, index=pairs.index)


# ----------------------------------------------------------------------------------------------------------------
# Label-free frequency / competition features (vectorised)
# ----------------------------------------------------------------------------------------------------------------
FREQ_SRC = ["ad_tset", "ad_num_first_eq", "ad_street_tset", "ad_house_eq", "nm_core_tset"]


def frequency_features(pairs: pd.DataFrame, rec: pd.DataFrame, feats: pd.DataFrame) -> pd.DataFrame:
    """Same-name counts (chains, generic names) + score-free address competition among S1s sharing a candidate."""
    key = (rec["country_norm"] + "\x1f" + rec["name_core"]).to_numpy(object)
    s1m = rec["src"].to_numpy() == 1
    fs1 = pd.Series(key[s1m]).value_counts()
    fc = pd.Series(key[~s1m]).value_counts()
    i1, i2 = pairs["i1"].to_numpy(), pairs["i2"].to_numpy()
    out = pd.DataFrame(index=pairs.index)
    out["nm_s1_samename_n"] = pd.Series(key[i1]).map(fs1).fillna(0).to_numpy(np.float32)
    out["nm_c_name_in_s1_n"] = pd.Series(key[i2]).map(fs1).fillna(0).to_numpy(np.float32)
    out["nm_c_samename_n"] = pd.Series(key[i2]).map(fc).fillna(0).to_numpy(np.float32)
    for c in ("ad_tset", "ad_num_first_eq", "ad_street_tset", "ad_house_eq", "nm_core_tset"):
        if c not in feats:
            continue
        v = feats[c].fillna(-1.0)
        g = v.groupby(pairs["i2"])
        out[f"{c}_rank_c"] = g.rank(ascending=False, method="min").astype(np.float32)
        out[f"{c}_gap_c"] = (g.transform("max") - v).astype(np.float32)
        g1 = v.groupby(pairs["i1"])
        out[f"{c}_rank_s1"] = g1.rank(ascending=False, method="min").astype(np.float32)
    return out


# ----------------------------------------------------------------------------------------------------------------
# Folds and target encodings
# ----------------------------------------------------------------------------------------------------------------
def fold_of(entity_ids, k: int = 5) -> np.ndarray:
    """Deterministic S1-level fold id shared by every OOF computation (target encodings, GBDT CV, stacking)."""
    return np.fromiter((zlib.crc32(e.encode()) % k for e in entity_ids), dtype=np.int16)


def s1_bucket(df: pd.DataFrame, k: int) -> np.ndarray:
    """fold_of(s1_id, k) via the precomputed s1_hash (= crc32 % 1000) column; exact because k divides 1000."""
    if "s1_hash" in df.columns:
        assert 1000 % k == 0, f"bucket count {k} must divide 1000"
        return (df["s1_hash"].to_numpy() % k).astype(np.int16)
    return fold_of(df["s1_id"].to_numpy(), k)


def _near_dup_extras(core: List[str], i1: np.ndarray, i2: np.ndarray):
    """(row, token) for pairs where cand name = S1 name + 1-2 inserted tokens (the distractor construction)."""
    rows, toks = [], []
    for j, (a, b) in enumerate(zip(i1, i2)):
        s1t = set(core[a].split())
        if not s1t:
            continue
        ct = set(core[b].split())
        ex = ct - s1t
        if 1 <= len(ex) <= 2 and s1t <= ct:
            rows.extend([j] * len(ex)); toks.extend(ex)
    return np.asarray(rows, np.int64), toks


def _harm_from_counts(cnt, pos, base, min_count, prior_weight):
    lo = lambda p: np.log(np.maximum(p, 1e-4) / np.maximum(1 - p, 1e-4))
    p = (pos + prior_weight * base) / (cnt + prior_weight)
    return np.where(cnt >= min_count, lo(base) - lo(p), np.nan)  # >0 = harmful (less likely a match than average)


def oof_extra_harm(pairs: pd.DataFrame, rec: pd.DataFrame, k: int, min_count: int = 30, prior_weight: float = 20.0):
    """Single-pass OOF target encoding of inserted tokens (near-dup pairs only).

    Returns (features frame aligned with pairs, test-time table = mean over fold tables of tokens known in a majority)."""
    core = rec["name_core"].tolist()
    rows, toks = _near_dup_extras(core, pairs["i1"].to_numpy(), pairs["i2"].to_numpy())
    N = len(pairs)
    mx = np.full(N, np.nan); sm = np.zeros(N); unk = np.zeros(N)
    if len(rows) == 0:
        return pd.DataFrame({"nm_extra_harm_max": mx, "nm_extra_harm_sum": sm, "nm_extra_unknown": unk}, index=pairs.index).astype(np.float32), {}
    cat = pd.Categorical(toks)
    codes = cat.codes.astype(np.int64)
    V = len(cat.categories)
    y = pairs["y"].to_numpy(np.float64); fo = pairs["fold"].to_numpy(np.int64)
    ry, rf = y[rows], fo[rows]
    cnt = np.zeros((k, V)); pos = np.zeros((k, V))
    np.add.at(cnt, (rf, codes), 1.0); np.add.at(pos, (rf, codes), ry)
    tables = []
    mxa = np.full(N, -np.inf)
    for f in range(k):
        other = rf != f
        base = ry[other].mean() if other.any() else 0.5
        h = _harm_from_counts(cnt.sum(0) - cnt[f], pos.sum(0) - pos[f], base, min_count, prior_weight)
        tables.append(h)
        sel = rf == f
        r, hv = rows[sel], h[codes[sel]]
        kn = ~np.isnan(hv)
        np.maximum.at(mxa, r[kn], hv[kn]); np.add.at(sm, r[kn], hv[kn]); np.add.at(unk, r[~kn], 1.0)
    mx = np.where(np.isfinite(mxa), mxa, np.nan)
    T = np.vstack(tables)
    known = (~np.isnan(T)).sum(0) > k // 2
    test_tab = {cat.categories[v]: float(np.nanmean(T[:, v])) for v in np.flatnonzero(known)}
    feats = pd.DataFrame({"nm_extra_harm_max": mx, "nm_extra_harm_sum": sm, "nm_extra_unknown": unk}, index=pairs.index)
    return feats.astype(np.float32), test_tab


def extra_harm_features(pairs: pd.DataFrame, rec: pd.DataFrame, harm: Dict[str, float]) -> pd.DataFrame:
    core = rec["name_core"].tolist()
    rows, toks = _near_dup_extras(core, pairs["i1"].to_numpy(), pairs["i2"].to_numpy())
    N = len(pairs)
    mxa = np.full(N, -np.inf); sm = np.zeros(N); unk = np.zeros(N)
    for r, t in zip(rows, toks):
        h = harm.get(t)
        if h is None:
            unk[r] += 1
        else:
            mxa[r] = max(mxa[r], h); sm[r] += h
    mx = np.where(np.isfinite(mxa), mxa, np.nan)
    return pd.DataFrame({"nm_extra_harm_max": mx, "nm_extra_harm_sum": sm, "nm_extra_unknown": unk}, index=pairs.index).astype(np.float32)


# ----------------------------------------------------------------------------------------------------------------
# Context features (need a base score per pair; use OOF scores during training)
# ----------------------------------------------------------------------------------------------------------------
def support_features(df: pd.DataFrame, rec: pd.DataFrame, score: str, top: int = 3, min_score: float = 0.5,
                     min_row_score: float = 0.01, prefix: str = "sup") -> pd.DataFrame:
    """Cluster support: similarity of each candidate to the S1's OTHER confident candidates (S2<->S3, S2<->S2).

    Duplicates of one entity inside S2/S3 corroborate each other (e.g. a record whose name was replaced by a random
    word but whose address equals the address of the other confident members). Rows below min_row_score are skipped
    (the decision layer never emits them)."""
    from rapidfuzz.process import cdist

    addr = rec["addr_norm"].to_numpy(object)
    name = rec["name_core"].to_numpy(object)
    d = df[["i1", "i2", score]].sort_values(["i1", score], ascending=[True, False])
    i1 = d["i1"].to_numpy(); i2 = d["i2"].to_numpy(); sc = d[score].to_numpy()
    starts = np.flatnonzero(np.r_[True, i1[1:] != i1[:-1]])
    ends = np.r_[starts[1:], len(d)]
    s_addr = np.full(len(d), np.nan, np.float32)
    s_name = np.full(len(d), np.nan, np.float32)
    s_n = np.zeros(len(d), np.float32)
    for s, e in zip(starts, ends):
        strong = [j for j in range(s, min(e, s + top + 1)) if sc[j] >= min_score]
        if not strong:
            continue
        rows = np.arange(s, e)[sc[s:e] >= min_row_score]
        if not len(rows):
            continue
        strong_arr = np.asarray(strong)
        for arr, out in ((addr, s_addr), (name, s_name)):
            A = [arr[i2[j]] for j in rows]
            S = [arr[i2[k]] for k in strong]
            M = cdist(A, S, scorer=fuzz.token_set_ratio, dtype=np.float32) / 100.0
            M[rows[:, None] == strong_arr[None, :]] = np.nan          # exclude self
            M[[not x for x in A], :] = np.nan                         # empty -> unknown, not 0
            M[:, [not x for x in S]] = np.nan
            with np.errstate(all="ignore"):
                ok = ~np.all(np.isnan(M), axis=1)
                out[rows[ok]] = np.nanmax(M[ok], axis=1)
        s_n[rows] = np.array([len(strong) - int(j in strong) for j in rows], np.float32)
    out = pd.DataFrame({f"{prefix}_addr_max": s_addr, f"{prefix}_name_max": s_name, f"{prefix}_n_strong": s_n}, index=d.index)
    return out.reindex(df.index)


def _second_best_by(key: np.ndarray, score: np.ndarray) -> np.ndarray:
    o = np.lexsort((-score, key))
    ks, ss = key[o], score[o]
    start = np.r_[True, ks[1:] != ks[:-1]]
    first = np.flatnonzero(start)
    cnt = np.diff(np.r_[first, len(ss)])
    sec = np.where(cnt > 1, ss[np.minimum(first + 1, len(ss) - 1)], -1.0)
    out = np.empty(len(key), np.float64)
    out[o] = sec[np.cumsum(start) - 1]
    return out


def context_features(df: pd.DataFrame, score: str, prefix: str = "ctx") -> pd.DataFrame:
    """Within-S1 and within-candidate competition features on column `score` (fully vectorised)."""
    out = pd.DataFrame(index=df.index)
    s = df[score].to_numpy(np.float64)
    g1 = df.groupby("i1")[score]
    g2 = df.groupby("i2")[score]
    out[f"{prefix}_rank_s1"] = g1.rank(ascending=False, method="first").astype(np.float32)
    out[f"{prefix}_rank_c"] = g2.rank(ascending=False, method="first").astype(np.float32)
    mx1 = g1.transform("max").to_numpy()
    mx2 = g2.transform("max").to_numpy()
    out[f"{prefix}_gap_to_best_s1"] = (mx1 - s).astype(np.float32)
    out[f"{prefix}_gap_to_best_c"] = (mx2 - s).astype(np.float32)
    # margin over the best OTHER S1 competing for this candidate (exclusivity signal)
    second2 = _second_best_by(df["i2"].to_numpy(), s)
    other_best = np.where(s >= mx2, second2, mx2)
    out[f"{prefix}_margin_other_s1"] = (s - other_best).astype(np.float32)
    second1 = _second_best_by(df["i1"].to_numpy(), s)
    out[f"{prefix}_margin_next_c"] = np.where(s >= mx1, s - second1, s - mx1).astype(np.float32)
    out[f"{prefix}_n_c_for_s1"] = g1.transform("size").astype(np.float32)
    out[f"{prefix}_n_s1_for_c"] = g2.transform("size").astype(np.float32)
    strong = pd.Series(s > 0.5, index=df.index)
    out[f"{prefix}_n_strong_s1"] = strong.groupby(df["i1"]).transform("sum").astype(np.float32)
    out[f"{prefix}_sum_s1"] = g1.transform("sum").astype(np.float32)
    out[f"{prefix}_mutual_best"] = ((out[f"{prefix}_rank_c"] == 1) & (out[f"{prefix}_rank_s1"] <= out[f"{prefix}_n_strong_s1"].clip(lower=1))).astype(np.float32)
    return out
