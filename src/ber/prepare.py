"""Stage 0: load a split, learn/apply transliteration, normalise every record, optional geographic dev sampling."""
from __future__ import annotations

import logging
import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from . import io as bio
from . import normalize as N
from . import translit

log = logging.getLogger(__name__)


def _norm_chunk(args):
    names, addrs, countries, tdict, extra_lookup = args
    translit.install(tdict)
    for c, m in (extra_lookup or {}).items():  # learned admin lookup for open-set countries (spawn-safe: per task)
        N.STATE_LOOKUP[c] = m
    return [N.normalize_record(n, a, c) for n, a, c in zip(names, addrs, countries)]


def normalize_frame(df: pd.DataFrame, tdict: dict, n_jobs: int = 1, chunk: int = 50_000, extra_lookup=None) -> pd.DataFrame:
    names = df["business_name"].tolist(); addrs = df["business_address"].tolist(); cs = df["country"].tolist()
    tasks = [(names[s:s + chunk], addrs[s:s + chunk], cs[s:s + chunk], tdict, extra_lookup) for s in range(0, len(df), chunk)]
    if n_jobs <= 1:
        res = [_norm_chunk(t) for t in tasks]
    else:
        from multiprocessing import get_context

        with get_context("spawn" if os.name == "nt" else "fork").Pool(n_jobs) as pool:
            res = pool.map(_norm_chunk, tasks, chunksize=1)
    out = pd.DataFrame([r for part in res for r in part])
    out.index = df.index
    return out


def load_split(data_dir: str, split: str, cache_dir: Optional[str] = None) -> Dict[int, pd.DataFrame]:
    out = {}
    for k in (1, 2, 3):
        pq = os.path.join(cache_dir, f"{split}_s{k}.parquet") if cache_dir else None
        if pq and os.path.exists(pq):
            out[k] = pd.read_parquet(pq)
        else:
            out[k] = bio.read_source(os.path.join(data_dir, split, f"{split}_source{k}.tsv"))
            if pq:
                os.makedirs(cache_dir, exist_ok=True)
                out[k].to_parquet(pq)
    return out


def learn_translit_from_train(src: Dict[int, pd.DataFrame], gt: Dict[str, List[str]], s1_ids=None) -> dict:
    s1 = src[1] if s1_ids is None else src[1][src[1].entity_id.isin(set(s1_ids))]
    c = pd.concat([src[2], src[3]])
    nat = c[c.business_name.str.contains("[ऀ-ൿ]", regex=True) | c.business_address.str.contains("[ऀ-ൿ]", regex=True)]
    natset = set(nat.entity_id)
    s1set = set(s1.entity_id)
    pairs = [(a, b) for a, bs in gt.items() if a in s1set for b in bs if b in natset]
    log.info("translit: %d native candidate records, %d aligned pairs", len(nat), len(pairs))
    return translit.learn(dict(zip(s1.entity_id, s1.business_name)), dict(zip(s1.entity_id, s1.business_address)),
                          dict(zip(nat.entity_id, nat.business_name)), dict(zip(nat.entity_id, nat.business_address)), pairs)


def extend_translit_oov(tdict: dict, src: Dict[int, pd.DataFrame]) -> dict:
    """Resolve native tokens of THIS split that the train dictionary does not know (closed-vocabulary mining)."""
    table = dict(tdict.get("tokens", {})); table.update(tdict.get("segments", {}))
    allnames = pd.concat([src[k].business_name for k in (1, 2, 3)])
    vocab = translit.latin_vocab(allnames)
    native = allnames[allnames.str.contains("[ऀ-ൿ]", regex=True)]
    oov = translit.resolve_oov(native, table, vocab)
    d = dict(tdict); d["oov"] = oov
    log.info("translit: resolved %d OOV native tokens for this split", len(oov))
    return d


def fast_state(addr: str, country: str) -> str:
    """Cheap state parse for sampling only: exact comma-segment lookup (no transliteration / full normalisation)."""
    lk = N.STATE_LOOKUP.get((country or "").strip().lower())
    if not lk:
        return ""
    for seg in reversed(addr.split(",")):
        st = lk.get(seg.strip().lower())
        if st:
            return st
    return ""


def _seg(s: str) -> str:
    import unicodedata

    s = unicodedata.normalize("NFKD", s.strip().lower())
    return "".join(ch for ch in s if not unicodedata.combining(ch)).replace("-", " ").strip(" .")


def region_labeler(s1: pd.DataFrame, cands: Optional[pd.DataFrame] = None, min_count: int = 20, share: float = 0.9):
    """Label-free admin-region labeller for countries WITHOUT a state lookup (open set, e.g. France).

    * S1 region = the most globally frequent comma segment of its address (a region spans several cities, so it is
      more frequent than any of its cities; works whatever the segment order);
    * city -> region learned from the other (non-numeric) S1 segments; département/other -> region learned from
      candidate segments that co-occur with an already-mapped city (both only when >= `share` consistent).
    Returns label(addr) -> region ('' when unknown)."""
    import collections

    segs1 = [[_seg(x) for x in a.split(",") if x.strip()] for a in s1.business_address]
    freq = collections.Counter(x for ss in segs1 for x in set(ss) if x and not any(ch.isdigit() for ch in x))
    regions = []
    for ss in segs1:
        cand = [x for x in ss if x in freq]
        regions.append(max(cand, key=lambda x: freq[x]) if cand else "")
    votes = collections.defaultdict(collections.Counter)
    for ss, r in zip(segs1, regions):
        for x in ss:
            if x and x != r and not any(ch.isdigit() for ch in x):
                votes[x][r] += 1
    known_regions = {r for r, n in collections.Counter(regions).items() if r and n >= min_count}

    def dominant(v):
        return {k: c.most_common(1)[0][0] for k, c in v.items()
                if sum(c.values()) >= min_count and c.most_common(1)[0][1] / sum(c.values()) >= share}

    city = {k: r for k, r in dominant(votes).items() if r in known_regions and k not in known_regions}
    if cands is not None and len(cands):
        v2 = collections.defaultdict(collections.Counter)
        for a in cands.business_address.to_numpy()[: 2_000_000]:
            ss = [_seg(x) for x in a.split(",") if x.strip()]
            rs = [city[x] for x in ss if x in city]
            if rs:
                for x in ss:
                    if x not in city and x not in known_regions and not any(ch.isdigit() for ch in x):
                        v2[x][rs[0]] += 1
        city.update({k: r for k, r in dominant(v2).items() if r in known_regions})
    log.info("region labeller: %d regions, %d city/dept tokens mapped", len(known_regions), len(city))

    def label(addr: str) -> str:
        ss = [_seg(x) for x in addr.split(",") if x.strip()]
        for x in ss:
            if x in known_regions:
                return x
        for x in ss:
            if x in city:
                return city[x]
        return ""

    return label


def learn_admin_lookup(src: Dict[int, pd.DataFrame], min_count: int = 20, share: float = 0.9,
                       last_share: float = 0.7) -> Dict[str, str]:
    """Label-free admin-region lookup for an open-set country WITHOUT a built-in state list (e.g. France), so that
    normalisation treats regions/départements exactly like US/India states (stripped from the address, compared as
    `addr_state`). Regions come from the region labeller (most frequent S1 segment); départements = candidate LAST
    segments that are not S1 cities and co-occur (>= share) with cities of a single region.
    Returns {normalised segment surface form (as produced inside norm_addr): region code}."""
    import collections

    s1, cands = src[1], pd.concat([src[2], src[3]], ignore_index=True)
    segs1 = [[_seg(x) for x in a.split(",") if x.strip()] for a in s1.business_address]
    freq = collections.Counter(x for ss in segs1 for x in set(ss) if x and not any(ch.isdigit() for ch in x))
    regions = [max((x for x in ss if x in freq), key=lambda x: freq[x], default="") for ss in segs1]
    reg_n = collections.Counter(regions)
    known = {r for r, n in reg_n.items() if r and n >= min_count}
    votes = collections.defaultdict(collections.Counter)
    for ss, r in zip(segs1, regions):
        for x in ss:
            if x and x != r and not any(ch.isdigit() for ch in x):
                votes[x][r] += 1
    city = {k: c.most_common(1)[0][0] for k, c in votes.items()
            if sum(c.values()) >= min_count and c.most_common(1)[0][1] / sum(c.values()) >= share and c.most_common(1)[0][0] in known}
    v2, last_n, tot_n = collections.defaultdict(collections.Counter), collections.Counter(), collections.Counter()
    for a in cands.business_address.to_numpy()[: 2_000_000]:
        ss = [_seg(x) for x in a.split(",") if x.strip()]
        if not ss:
            continue
        for j, x in enumerate(ss):
            tot_n[x] += 1
            if j == len(ss) - 1:
                last_n[x] += 1
        rs = [city[x] for x in ss if x in city]
        if rs:
            for x in ss:
                if x not in city and x not in known and not any(ch.isdigit() for ch in x):
                    v2[x][rs[0]] += 1
    depts = {k: c.most_common(1)[0][0] for k, c in v2.items()
             if sum(c.values()) >= min_count and c.most_common(1)[0][1] / sum(c.values()) >= share
             and last_n[k] / max(1, tot_n[k]) >= last_share}
    admin = {**{r: r for r in known}, **depts}
    # register the exact surface forms norm_addr produces for a segment (fold_ascii + space collapse, hyphens kept)
    out = {}
    for form, code in admin.items():
        for variant in {form, form.replace(" ", "-")}:
            out[N.SPACE_RE.sub(" ", N.fold_ascii(variant)).strip(" .-")] = code
    log.info("admin lookup: %d regions, %d départements/other admin units -> %d surface forms (e.g. %s)", len(known),
             len(depts), len(out), sorted(depts.items())[:6])
    return out


def _labels(d: pd.DataFrame, country: str, labeler) -> np.ndarray:
    if labeler is None:
        return np.array([fast_state(a, c) for a, c in zip(d.business_address, d.country)], dtype=object)
    return np.array([labeler(a) for a in d.business_address], dtype=object)


def state_groups(s1: pd.DataFrame, n_groups: int, labeler=None, weight_labels=None) -> List[set]:
    """Deterministic balanced bin-packing of states into n_groups (largest load first, greedy argmin-load).

    `weight_labels`: state labels of another record set (e.g. S2+S3 candidates) to balance on instead of S1 count.
    Candidate volume per state can be very uneven relative to S1 count (e.g. India: one state carries 22.8
    candidates/S1 vs another's 10.4), and candidate count is what drives blocking memory (kNN + key joins are
    O(n2) per shard) -- balancing on S1 count alone can leave one shard with 70%+ more candidates than another
    at the same n_groups. Falls back to S1 count when omitted (e.g. no candidate data available yet)."""
    states = pd.Series(_labels(s1, "", labeler)).value_counts()
    states = states[states.index != ""]  # a state-less S1 (rare parsing miss) can't be grouped by state; it is
    # simply absent from every group's S1 set and gets an empty prediction at finalize (like any other singleton).
    # It must never be treated as an assignable "state": candidates with no parsed state are already added to EVERY
    # group unconditionally (state_shard's `| (stc == "")`), so weighting "" would double-count that whole pool as
    # one state's load and could isolate it into its own degenerate group.
    weight = pd.Series(_labels_to_counts(weight_labels)) if weight_labels is not None else states
    weight = weight.reindex(states.index).fillna(0)
    groups = [set() for _ in range(n_groups)]
    load = [0] * n_groups
    for state in weight.sort_values(ascending=False).index:
        j = int(np.argmin(load))
        groups[j].add(state)
        load[j] += int(weight[state])
    log.info("state groups (%s load): %s", "candidate" if weight_labels is not None else "S1", load)
    return groups


def _labels_to_counts(labels) -> Dict[str, int]:
    return dict(pd.Series(labels).value_counts())


def _india_rebalance_groups_6_7(country: str, n_groups: int, groups: List[set]) -> None:
    """Targeted fix (Sun 27 Sep, post-crash): candidate-weighted packing (see state_groups) does not account for
    cross-state `links` expansion, so group 6 ended up at 3.6M candidates (an/ar/ch/hr/py each independently pull in
    a DIFFERENT huge linked cluster -- Maharashtra/Delhi/Tamil Nadu) while groups 0-5 sat at 1.6-2.2M. Groups 0-5
    were ALREADY BUILT (features done) under the original packing, so the general algorithm cannot change without
    orphaning or duplicating their states across shards -- this only touches groups 6 and 7 (not yet built),
    replacing them with the brute-force-optimal 2-way split of the SAME 13 states (verified: both landed at ~2.6M,
    vs the un-rebalanced 3.6M/1.6M and the original crash at 2.73M; see scratchpad/fix_group67.py for the search).
    Guarded: only fires when groups 6/7 still hold exactly the 13 states this was computed for, so a config/sample
    change can't silently misapply a stale override -- it falls through to the (unbalanced but correct) computed
    groups instead, logging a warning."""
    if country != "india" or n_groups != 8 or len(groups) != 8:
        return
    want = {"an", "ar", "ch", "ga", "hr", "la", "mz", "py", "uk", "wb"} | {"kl", "od", "tg"}
    if groups[6] | groups[7] != want or groups[6] & groups[7]:
        log.warning("India group 6/7 rebalance override skipped: state set changed since it was computed (got %s / %s)",
                   sorted(groups[6]), sorted(groups[7]))
        return
    groups[6] = {"an", "ar", "ch", "ga", "kl", "la", "mz", "od", "uk"}
    groups[7] = {"hr", "py", "tg", "wb"}
    log.info("India group 6/7 rebalanced (post-crash fix): %s / %s", sorted(groups[6]), sorted(groups[7]))


def state_shard(src: Dict[int, pd.DataFrame], k: int, n_groups: int, links: Optional[Dict[str, List[str]]] = None) -> Dict[int, pd.DataFrame]:
    """Sub-shard k of n_groups of ONE country (same construction as the dev geo-sample the models were trained on):
    S1 whose state is in group k; candidates whose parsed state is in group k OR who have no parsed state (those are
    duplicated across sub-shards; `finalize` restores global exclusivity by score). Countries without a state
    lookup (open set, e.g. France) cannot be split -> n_groups must be 1 for them.

    Groups are balanced on CANDIDATE count (see state_groups), not S1 count: a state's S1 and candidate volume are
    not proportional, and candidate volume is what bounds blocking memory."""
    country = str(src[1]["country"].iloc[0]).strip().lower() if len(src[1]) else ""
    labeler = None
    cand_all = pd.concat([src[2], src[3]], ignore_index=True)
    if not N.STATE_LOOKUP.get(country):  # open-set country without a state list -> learned region labeller
        labeler = region_labeler(src[1], cand_all)
    cand_labels = _labels(cand_all, country, labeler)
    groups = state_groups(src[1], n_groups, labeler, weight_labels=cand_labels)
    _india_rebalance_groups_6_7(country, n_groups, groups)  # targeted post-crash fix, see function docstring
    g = groups[k]
    # linked states (learned from train pairs: e.g. Telangana<->Andhra Pradesh, WA<->DC, Delhi NCR): candidates parsed
    # to a state linked with this group's states are included too (cross-shard duplicates resolved in finalize)
    gc = set(g)
    if links and labeler is None:
        for st in list(g):
            gc.update(links.get(f"{country}|{st}", []))
    cand_labels_by_key = {2: cand_labels[:len(src[2])], 3: cand_labels[len(src[2]):]}  # cand_all = concat([2, 3])
    out = {}
    for key, d in src.items():
        stc = _labels(d, country, labeler) if key == 1 else cand_labels_by_key[key]
        if key == 1:
            m = np.isin(stc, list(g))
        else:
            m = np.isin(stc, list(gc)) | (stc == "")
        out[key] = d[m].reset_index(drop=True)
    log.info("state shard %d/%d: S1 %d, S2 %d, S3 %d (states %d, +%d linked for candidates)", k, n_groups, len(out[1]),
             len(out[2]), len(out[3]), len(g), len(gc) - len(g))
    return out


def geo_sample(src: Dict[int, pd.DataFrame], gt: Optional[Dict[str, List[str]]], frac: float, seed: int = 0) -> Dict[int, pd.DataFrame]:
    """Dev sampling that preserves LOCAL confuser density: keep whole states (parsed from the S1 address).

    S1 rows are kept if their state is selected; S2/S3 rows are kept if they match a kept S1 or (unmatched) if their own
    parsed state is selected / or with probability `frac` when no state can be parsed."""
    rng = np.random.default_rng(seed)
    s1 = src[1]
    st = [fast_state(a, c) for a, c in zip(s1.business_address, s1.country)]
    s1 = s1.assign(_st=st)
    keep_states = set()
    for c, g in s1.groupby("country"):
        states = g["_st"].value_counts()
        states = states[states.index != ""]
        order = rng.permutation(len(states))
        tot, acc = len(g), 0
        for i in order:
            if acc >= frac * tot:
                break
            keep_states.add((c, states.index[i])); acc += states.iloc[i]
    k1 = s1[[(c, s) in keep_states for c, s in zip(s1.country, s1["_st"])]].drop(columns="_st")
    kept_s1 = set(k1.entity_id)
    matched_to_kept = set()
    matched_any = set()
    if gt is not None:
        for a, bs in gt.items():
            matched_any.update(bs)
            if a in kept_s1:
                matched_to_kept.update(bs)
    out = {1: k1}
    for k in (2, 3):
        d = src[k]
        in_gt = d.entity_id.isin(matched_to_kept).to_numpy()
        unmatched = ~d.entity_id.isin(matched_any).to_numpy() if gt is not None else np.ones(len(d), bool)
        stc = np.array([fast_state(a, c) for a, c in zip(d.business_address, d.country)], dtype=object)
        sel_state = np.array([(c, s) in keep_states for c, s in zip(d.country, stc)])
        nostate = stc == ""
        rnd = rng.random(len(d)) < frac
        keep = in_gt | (unmatched & (sel_state | (nostate & rnd)))
        out[k] = d[keep]
    log.info("geo sample: S1 %d, S2 %d, S3 %d (states=%d)", len(out[1]), len(out[2]), len(out[3]), len(keep_states))
    return out


def build_records(src: Dict[int, pd.DataFrame], tdict: dict, n_jobs: int = 1, extra_lookup=None) -> pd.DataFrame:
    frames = []
    for k in (1, 2, 3):
        d = src[k].reset_index(drop=True)
        nrm = normalize_frame(d, tdict, n_jobs, extra_lookup=extra_lookup)
        nrm.insert(0, "entity_id", d["entity_id"].values)
        nrm.insert(1, "src", np.int8(k))
        nrm["raw_name"] = d["business_name"].values  # raw text for transformer matchers
        nrm["raw_addr"] = d["business_address"].values
        frames.append(nrm)
    rec = pd.concat(frames, ignore_index=True)
    rec.insert(0, "idx", np.arange(len(rec), dtype=np.int64))
    return rec
