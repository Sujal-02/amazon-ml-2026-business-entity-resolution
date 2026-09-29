"""End-to-end pipeline CLI.

  python -m ber.pipeline prepare  --config configs/default.yaml --split train
  python -m ber.pipeline block    --config ... --split train
  python -m ber.pipeline features --config ... --split train
  python -m ber.pipeline train    --config ...                    # GBDT stack (+ optional cross-encoder scores)
  python -m ber.pipeline prepare/block/features --split test
  python -m ber.pipeline predict  --config ... --split test       # writes output/matching_results.tsv + candidate_pairs.tsv
  python -m ber.pipeline all      --config ...                    # everything above in order

Work files live in cfg.work_dir/<split>/ (parquet). The candidate set scored by the matcher is exactly what is written
to candidate_pairs.tsv.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import time
from typing import Dict, List

import numpy as np
import pandas as pd
import yaml

from . import blocking as B
from . import features as F
from . import io as bio
from . import metrics as M
from . import prepare as P
from . import translit
from .normalize import skeleton
from . import normalize as normalize_mod

log = logging.getLogger("ber")


# ----------------------------------------------------------------------------------------------------------------
def load_cfg(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    for k in ("work_dir", "data_dir", "out_dir", "artifacts_dir"):
        if k in cfg:
            cfg[k] = os.path.expanduser(os.path.expandvars(cfg[k]))
    return cfg


def base_split(split: str) -> str:
    """'test@france' -> 'test' (country shards run as independent splits: blocking is per country anyway)."""
    return split.partition("@")[0]


def shard_country(split: str) -> str:
    """'test@india#2of4' -> 'india'"""
    return split.partition("@")[2].partition("#")[0].strip().lower()


def shard_group(split: str):
    """'test@india#2of4' -> (2, 4) (0-based group index of n state groups); None when not sub-sharded."""
    g = split.partition("#")[2]
    if not g:
        return None
    k, _, n = g.partition("of")
    return int(k), int(n)


def wpath(cfg, split, name):
    d = os.path.join(cfg["work_dir"], split)
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, name)


def apath(cfg, name):
    os.makedirs(cfg["artifacts_dir"], exist_ok=True)
    return os.path.join(cfg["artifacts_dir"], name)


# ----------------------------------------------------------------------------------------------------------------
# text views for embedding channels and key columns
# ----------------------------------------------------------------------------------------------------------------
def text_views(rec: pd.DataFrame) -> Dict[str, List[str]]:
    name = rec["name_core"].where(rec["name_core"] != "", rec["name_norm"])
    addr = (rec["addr_house"] + " " + rec["addr_street"] + " " + rec["addr_locality"]).str.strip()
    return {
        "name": name.tolist(),
        "addr": addr.tolist(),
        "both": (name + " | " + addr).tolist(),
    }


def _street_head(street: str) -> str:
    """First street token that is not a road type / affix ('rue de la paix' -> 'paix' after stop-word removal)."""
    for t in street.split():
        if t not in normalize_mod.ROAD_TYPES:
            return t
    return ""


def add_key_columns(rec: pd.DataFrame) -> pd.DataFrame:
    street_first = rec["addr_street"].str.split().str[0].fillna("")
    # open-set countries (no built-in state list, e.g. France) put the road type FIRST ('rue X'): skip road-type
    # tokens there, otherwise the house+street key degenerates to 'house|rue' (15% usable blocks in France vs 90% US).
    # Countries seen in training keep the original definition so train/test keys stay identical.
    open_set = ~rec["country_norm"].isin(list(normalize_mod._builtin_countries()))
    if open_set.any():
        street_first = street_first.where(~open_set, rec["addr_street"].map(_street_head))
    loc_first = rec["addr_locality"].str.split().str[0].fillna("")
    house = rec["addr_house"].str.replace(r"[^0-9a-z]", "", regex=True)
    rec["key_house_street"] = np.where((house != "") & (street_first != ""), house + "|" + street_first.map(skeleton), "")
    rec["key_house_loc"] = np.where((house != "") & (loc_first != ""), house + "|" + loc_first.map(skeleton), "")
    rec["key_name"] = rec["name_key"].map(lambda s: " ".join(sorted(set(s.split()))))
    rec["key_concat"] = np.where(rec["name_concat"].str.len() >= 6, rec["name_concat"], "")
    rec["key_skel"] = rec["key_name"].map(lambda s: " ".join(sorted(skeleton(t) for t in s.split())))
    return rec


KEY_COLS = {"house_street": "key_house_street", "house_loc": "key_house_loc", "name": "key_name",
            "concat": "key_concat", "skel": "key_skel"}


# ----------------------------------------------------------------------------------------------------------------
# steps
# ----------------------------------------------------------------------------------------------------------------
def step_prepare(cfg, split):
    t0 = time.time()
    src = P.load_split(cfg["data_dir"], base_split(split), cfg.get("cache_dir"))
    if shard_country(split):
        src = {k: d[d["country"].str.strip().str.lower() == shard_country(split)].reset_index(drop=True) for k, d in src.items()}
        log.info("shard %s: S1 %d, S2 %d, S3 %d", split, len(src[1]), len(src[2]), len(src[3]))
        if shard_group(split):
            lp = apath(cfg, "state_links.json")
            links = json.load(open(lp)) if os.path.exists(lp) else None
            if links is None:
                log.warning("no state_links.json -> cross-state true pairs (e.g. TG/AP) may be lost by state sharding")
            src = P.state_shard(src, *shard_group(split), links=links)
    gt = bio.read_ground_truth(os.path.join(cfg["data_dir"], "train", "train_ground_truth.tsv")) if base_split(split) == "train" else None
    tpath = apath(cfg, "translit.json")
    if cfg.get("force_translit", False) or not os.path.exists(tpath):
        if gt is None:
            gt = bio.read_ground_truth(os.path.join(cfg["data_dir"], "train", "train_ground_truth.tsv"))
            tsrc = P.load_split(cfg["data_dir"], "train", cfg.get("cache_dir"))
        else:
            tsrc = src
        tdict = P.learn_translit_from_train(tsrc, gt)
        translit.save(tdict, tpath)
    tdict = translit.load(tpath)
    frac = cfg.get("sample", {}).get(split, cfg.get("sample", {}).get(base_split(split), 1.0))
    if frac < 1.0:
        src = P.geo_sample(src, gt, frac, cfg.get("seed", 0))
    tdict = P.extend_translit_oov(tdict, src)  # closed-vocabulary OOV resolution on the (possibly sampled) split
    extra_lookup = None
    country = shard_country(split)
    if country and cfg.get("learn_admin_lookup", True) and not normalize_mod.STATE_LOOKUP.get(country):
        # open-set country without a state list (France): learn region/département lookup label-free from this split
        # so admin units are stripped from addresses and compared as addr_state exactly like US/India states
        extra_lookup = {country: P.learn_admin_lookup(src)}
    rec = P.build_records(src, tdict, cfg.get("n_jobs", 1), extra_lookup=extra_lookup)
    rec = add_key_columns(rec)
    rec.to_parquet(wpath(cfg, split, "records.parquet"))
    if gt is not None:
        keep = set(rec.entity_id[rec.src == 1])
        with open(wpath(cfg, split, "gt.json"), "w") as f:
            json.dump({k: v for k, v in gt.items() if k in keep}, f)
    log.info("prepare %s: %d records in %.0fs", split, len(rec), time.time() - t0)


def step_block(cfg, split):
    t0 = time.time()
    rec = pd.read_parquet(wpath(cfg, split, "records.parquet"))
    bcfg = cfg.get("blocking", {})
    s1 = rec[rec.src == 1].reset_index(drop=True)
    cand = rec[rec.src != 1].reset_index(drop=True)
    views1, views2 = text_views(s1), text_views(cand)
    vectors = {}
    lsa_cfg = bcfg.get("lsa", {})
    import pickle

    for view in lsa_cfg.get("views", ["name", "addr", "both"]):
        # state sub-shards of one country share ONE encoder (fitted on the first sub-shard's corpus): consistent vectors
        # across sub-shards and ~15 min saved per extra sub-shard
        cache = apath(cfg, f"lsa_{base_split(split)}_{shard_country(split)}_{view}.pkl") if shard_group(split) else None
        if cache and os.path.exists(cache):
            with open(cache, "rb") as f:
                enc = pickle.load(f)
            log.info("lsa view %s: reusing country encoder %s", view, os.path.basename(cache))
        else:
            enc = B.LSAEncoder(dim=lsa_cfg.get("dim", 128), ngram=tuple(lsa_cfg.get("ngram", [2, 4])),
                               max_features=lsa_cfg.get("max_features", 300000), seed=cfg.get("seed", 0))
            enc.fit(views1[view] + views2[view])  # unsupervised, on this split's own corpus (transductive, no labels)
            if cache:
                with open(cache, "wb") as f:
                    pickle.dump(enc, f)
        nj = cfg.get("n_jobs", 1)
        vectors[f"lsa_{view}"] = (enc.transform(views1[view], n_jobs=nj), enc.transform(views2[view], n_jobs=nj))
        log.info("lsa view %s encoded (%.0fs)", view, time.time() - t0)
    dense_dir = bcfg.get("dense_model")
    if dense_dir and os.path.exists(dense_dir):
        from .models.biencoder import encode, record_text

        # identical text format to biencoder training
        vectors["dense"] = (encode(dense_dir, record_text(s1)), encode(dense_dir, record_text(cand)))
        log.info("dense channel encoded (%.0fs)", time.time() - t0)
    kc = bcfg.get("key_cols", KEY_COLS)
    # pre-score model: fitted (cross-fitted) on TRAIN labels, applied unchanged to test shards
    label_fn = None
    if base_split(split) == "train" and os.path.exists(wpath(cfg, split, "gt.json")):
        with open(wpath(cfg, split, "gt.json")) as f:
            gt = json.load(f)
        n_max = int(rec["idx"].max()) + 1
        posmap = dict(zip(rec["entity_id"], rec["idx"]))
        tkey = np.sort(np.fromiter((posmap[a] * n_max + posmap[b] for a, bs in gt.items() if a in posmap
                                    for b in bs if b in posmap), dtype=np.int64))

        def label_fn(i1, i2):
            return np.isin(i1.astype(np.int64) * n_max + i2.astype(np.int64), tkey, assume_unique=False)
    model_path = apath(cfg, "prescore.txt") if bcfg.get("select") else None
    # merge_channels also adds cos_<channel> of EVERY vector channel for every pair
    cands = B.generate_candidates(s1, cand, vectors, {**bcfg, "key_cols": kc}, label_fn=label_fn, model_path=model_path)
    del vectors
    cands.to_parquet(wpath(cfg, split, "cands.parquet"))
    if os.path.exists(wpath(cfg, split, "gt.json")):
        report_blocking(cfg, split, cands, rec)
    log.info("block %s: %d pairs (%.1f / S1) in %.0fs", split, len(cands), len(cands) / max(1, len(s1)), time.time() - t0)


def label_pairs(cands: pd.DataFrame, rec: pd.DataFrame, gt: Dict[str, List[str]]) -> np.ndarray:
    eid = rec["entity_id"].to_numpy()
    truth = {(a, b) for a, bs in gt.items() for b in bs}
    return np.fromiter(((eid[a], eid[b]) in truth for a, b in zip(cands.i1.to_numpy(), cands.i2.to_numpy())),
                       dtype=np.int8, count=len(cands))


def report_blocking(cfg, split, cands, rec):
    with open(wpath(cfg, split, "gt.json")) as f:
        gt = json.load(f)
    eid = rec["entity_id"].to_numpy()
    cmap: Dict[str, list] = {}
    for a, b in zip(eid[cands.i1.to_numpy()], eid[cands.i2.to_numpy()]):
        cmap.setdefault(a, []).append(b)
    r = M.candidate_recall(cmap, gt)
    log.info("BLOCKING %s: %s", split, r)
    with open(wpath(cfg, split, "blocking_report.json"), "w") as f:
        json.dump(r, f, indent=1)


def step_features(cfg, split):
    t0 = time.time()
    rec = pd.read_parquet(wpath(cfg, split, "records.parquet"), columns=["entity_id"] + F.REC_FIELDS)
    cands = pd.read_parquet(wpath(cfg, split, "cands.parquet")).reset_index(drop=True)
    idf = F.IDF.from_records(rec)  # label-free, this split's own corpus
    log.info("features %s: IDF/s1df/source-specific tokens ready (%.0fs); x1=%s x2=%s", split, time.time() - t0,
             {c: len(v) for c, v in idf.x1.items()}, {c: len(v) for c, v in idf.x2.items()})
    hp = apath(cfg, "extra_harm.json")
    k = cfg.get("n_folds", 5)
    eid = rec["entity_id"].to_numpy()
    import zlib

    rec_hash = np.fromiter((zlib.crc32(e.encode()) % 1000 for e in eid), dtype=np.int16, count=len(eid))
    cands["s1_hash"] = rec_hash[cands.i1.to_numpy()]
    if base_split(split) == "train":
        with open(wpath(cfg, split, "gt.json")) as f:
            gt = json.load(f)
        cands["y"] = label_pairs(cands, rec, gt)
        # ONE fold definition everywhere (biencoder / cross-encoder / stacker use features.fold_of too)
        cands["fold"] = F.fold_of(eid, k)[cands.i1.to_numpy()].astype(np.int8)
        # out-of-fold target encoding of inserted tokens (no label leakage); test uses the fold-mean table
        harm_feats, harm_tab = F.oof_extra_harm(cands, rec, k)
        with open(hp, "w") as f:
            json.dump(harm_tab, f)
    else:
        with open(hp) as f:
            harm_feats = F.extra_harm_features(cands, rec, json.load(f))
    log.info("features %s: harm features done (%.0fs)", split, time.time() - t0)
    # chunked: parts hold whole S1 groups (cands are sorted by i1) -> bounded memory on 8-16GB boxes
    from . import store

    order = np.argsort(cands["i1"].to_numpy(), kind="stable")
    if not np.array_equal(order, np.arange(len(cands))):
        cands = cands.iloc[order].reset_index(drop=True)
        harm_feats = harm_feats.iloc[order].reset_index(drop=True)
    store.reset(cfg, split)
    ranges = store.split_points(cands["i1"].to_numpy(), cfg.get("features_chunk_pairs", 6_000_000))
    slim = []
    sd = float(cfg.get("state_dropout", 0.25))
    q = int(round(1 / sd)) if sd > 0 else 0
    for k_part, (a, b) in enumerate(ranges):
        c = cands.iloc[a:b].reset_index(drop=True)
        feats = F.compute_pair_features(c[["i1", "i2"]], rec, idf, None, n_jobs=cfg.get("n_jobs", 1))
        out = pd.concat([c, feats, harm_feats.iloc[a:b].reset_index(drop=True)], axis=1)
        if base_split(split) == "train" and q:
            # teach the model that "state unknown" also happens with full addresses (unseen countries such as France)
            aug = (out["s1_hash"].to_numpy() // 10 % q) == 0
            out.loc[aug, "ad_state_eq"] = np.nan
        out["s1_id"] = eid[out.i1.to_numpy()]
        out["cand_id"] = eid[out.i2.to_numpy()]
        store.write_part(cfg, split, "feats", k_part, out)
        slim.append(feats[[x for x in F.FREQ_SRC if x in feats.columns]])
        log.info("features %s: part %d/%d rows [%d,%d) done (%.0fs)", split, k_part + 1, len(ranges), a, b, time.time() - t0)
        del feats, out, c
    # candidate-grouped (cross-part) label-free features on a slim global table, written with the same row ranges
    freq = F.frequency_features(cands[["i1", "i2"]], rec, pd.concat(slim, ignore_index=True))
    for k_part, (a, b) in enumerate(ranges):
        store.write_part(cfg, split, "freq", k_part, freq.iloc[a:b].reset_index(drop=True))
    log.info("features %s: %d pairs in %d parts in %.0fs", split, len(cands), len(ranges), time.time() - t0)


def step_train(cfg):
    from .models import gbdt

    gbdt.train_stack(cfg)


NL, TAB = chr(10), chr(9)


def step_finalize(cfg):
    """Merge per-shard outputs (cfg.finalize.shards: list of dirs holding matching_results.tsv + candidate_pairs.tsv)
    into ONE submission in test_source1.tsv order, then run the official validator."""
    import subprocess
    import sys

    import glob

    fin = cfg.get("finalize", {})
    shard_dirs = fin.get("shards") or sorted(d for d in glob.glob(os.path.join(cfg["out_dir"], "test@*")) if os.path.isdir(d))
    test_dir = fin.get("test_dir", os.path.join(cfg["data_dir"], "test"))
    s1_all = bio.read_source(os.path.join(test_dir, "test_source1.tsv"))["entity_id"].tolist()
    match, cand = {}, {}
    for d in shard_dirs:
        mname = "matching_results_nocorr.tsv" if fin.get("variant") == "nocorr" else "matching_results.tsv"
        for fname, dest in ((mname, match), ("candidate_pairs.tsv", cand)):
            with open(os.path.join(d, fname), encoding="utf-8") as f:
                next(f)
                for line in f:
                    s1, _, rest = line.rstrip(NL).partition(TAB)
                    ids = [x for x in rest.split(",") if x]
                    if ids:
                        dest.setdefault(s1, []).extend(ids)
        log.info("finalize: merged shard %s (%d S1 with matches so far)", d, len(match))
    for s1, ids in match.items():  # a final match must be one of the candidates the model scored
        cs = set(cand.get(s1, []))
        match[s1] = [x for x in ids if x in cs]
    # global exclusivity: candidates without a parsed state are present in several state sub-shards -> keep only the
    # S1 with the highest calibrated probability (each S2/S3 record belongs to at most one S1)
    mp = pd.DataFrame([(a, b) for a, bs in match.items() for b in bs], columns=["s1_id", "cand_id"])
    parts = []
    for d in shard_dirs:
        sp = os.path.join(cfg["work_dir"], os.path.basename(d), "scores_gbdt.parquet")
        if os.path.exists(sp):
            parts.append(pd.read_parquet(sp, columns=["s1_id", "cand_id", "p"]).merge(mp, on=["s1_id", "cand_id"]))
    owner = {}
    if parts:
        best = pd.concat(parts, ignore_index=True).sort_values("p", ascending=False).drop_duplicates("cand_id")
        owner = dict(zip(best.cand_id, zip(best.s1_id, best.p)))
    dropped = 0
    for s1 in list(match):
        keep = [x for x in match[s1] if x not in owner or owner[x][0] == s1]
        dropped += len(match[s1]) - len(keep)
        match[s1] = keep
    log.info("finalize: global exclusivity dropped %d duplicate cross-shard links", dropped)
    out = fin.get("out_dir", cfg["out_dir"])
    bio.write_matching(os.path.join(out, "matching_results.tsv"), s1_all, match)
    bio.write_candidates(os.path.join(out, "candidate_pairs.tsv"), s1_all, cand)
    n_ne = sum(1 for s in s1_all if match.get(s))
    log.info("finalize: %d S1 rows, %d non-empty (%.1f%%), %d matched ids", len(s1_all), n_ne, 100 * n_ne / max(1, len(s1_all)),
             sum(len(v) for v in match.values()))
    validator = fin.get("validator", os.path.join(os.path.dirname(cfg["data_dir"]), "utils", "validate_submission.py"))
    if os.path.exists(validator):
        r = subprocess.run([sys.executable, validator, "--matching", os.path.join(out, "matching_results.tsv"),
                            "--candidate", os.path.join(out, "candidate_pairs.tsv"), "--test-dir", test_dir, "--check-ids"],
                           capture_output=True, text=True)
        log.info("validator exit=%d %s%s", r.returncode, r.stdout[-3000:], r.stderr[-2000:])


def step_predict(cfg, split="test"):
    from .models import gbdt

    gbdt.predict_stack(cfg, split)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["prepare", "block", "features", "train", "predict", "finalize", "all"])
    ap.add_argument("--config", required=True)
    ap.add_argument("--split", default="train")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = load_cfg(a.config)
    if a.step == "all":
        for sp in ("train", "test"):
            step_prepare(cfg, sp); step_block(cfg, sp); step_features(cfg, sp)
        step_train(cfg); step_predict(cfg, "test")
        return
    {"prepare": lambda: step_prepare(cfg, a.split), "block": lambda: step_block(cfg, a.split),
     "features": lambda: step_features(cfg, a.split), "train": lambda: step_train(cfg),
     "predict": lambda: step_predict(cfg, a.split), "finalize": lambda: step_finalize(cfg)}[a.step]()


if __name__ == "__main__":
    main()
