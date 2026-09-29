"""Compliant adaptation to countries that are present in TEST but absent from TRAIN (e.g. France, 15% of test S1).

Nothing here uses external data. The statistics come from three places:
  (a) the TRAIN gold pairs, which give the noise channel S1 -> S2/S3 and the distractor construction;
  (b) the UNLABELLED test records of the unseen country, which give label-free marginals such as surface forms, the
      admin-segment replacement and rates. This is the same transductive use as the per-split IDF in features.py;
  (c) the pipeline's own test scores, which give pseudo-labels.
The generic French street-type / legal-form tables are the same kind of linguistic rules as normalize.py.

  1. unseen countries : unseen_countries() = set(test countries) - set(train countries), case-insensitive.
  2. positives        : NoiseChannel.fit(train gold pairs) detects the ops that turn each S1 into its S2/S3 record.
                        Name ops: token shuffle / drop / duplicate, generic-word append, legal-form
                        swap / drop / add / move-to-front / bracket, honorifics, leetspeak, char typos, accents, case,
                        junk prefix, phone suffix, DBA / t-a / fka alias with a pseudo-word, domain name.com,
                        "| www.x.com", handle @name / #name, acronym, pseudo-word replacement.
                        Address ops: segment reorder, case, street-type abbreviation, admin (state) keep / drop /
                        re-form, city drop / replace / typo / suffix, unit drop, house number zero-pad / typo / range /
                        suffix / "#" / prefix (H.no, N deg), null token, PO box, duplicated segment, empty address.
                        Per pair it stores one op signature for the name and one for the address. Sampling a whole
                        (name, address) signature pair keeps the correlations between ops. It also stores the vocabularies
                        (pseudo-words, connectors, added words, junk prefixes, typo alphabet, accent map) and the
                        (#S2, #S3) cluster-size table, which includes singletons.
                        Locale.fit(unseen-country S1, unseen-country S2/S3), label-free, learns: street-type and
                        legal-form surface distributions (Rue / R. / RUE, Avenue / Av., Boulevard / Bd, SAS / S.A.S.),
                        '&' surfaces (& / + / et), mined abbreviations (Freres -> Frs), house-number prefixes (N deg),
                        zero-padding, the S1 admin segments (regions, found by co-occurrence degree) and what S2/S3 put
                        in their place (keep / departement found by city co-occurrence / drop), empty-address and
                        upper-case rates, accent add/strip rates, candidate-only name tokens and their role (appended
                        generic word vs distractor insert, from the address agreement of near-duplicates), TLDs and
                        PO-box templates. Synthesizer applies a sampled train signature to an S1 record using the
                        locale's surfaces and rates, producing an S2-like record.
  3. negatives        : DistractorModel.fit(train records, gt) learns how unmatched distractors are built: category
                        word inserted into an existing S1 name, or a token replaced; insert position; address relation.
                        It then generates the same kind of records for the unseen country. The synthetic split also
                        contains the S1-S1 nearest neighbours of every seed S1 (S1 is deduplicated, so these are
                        guaranteed non-matches). Every synthetic record of S1 b is a hard negative for the neighbours
                        of b, and the pipeline's own blocking retrieves them exactly as it would on test.
  4. pseudo-labels    : select_pseudo_labels() on work_dir/test/scores_gbdt.parquet. Positives are unseen-country pairs
                        with p >= 0.98 that are the mutual best under exclusivity and beat the runner-up S1 by >= 0.3.
                        Negatives have p <= 0.01. Only complete S1 groups are kept, so every row of an S1 is labelled.
  5. emission         : build_synth_split() writes work_dir/<adapt.synth_split>/{records.parquet, gt.json} with the
                        pipeline's exact records schema (prepare.build_records + pipeline.add_key_columns). The real
                        pipeline block + features steps then run on it. label_split() adds y / fold, which gives exactly
                        the schema of work_dir/train/feats.parquet. pseudo_label_split() writes the same schema from
                        test rows.

INTEGRATION (models/gbdt.py is unchanged; cfg block `adapt:`, see DEFAULTS):
  python -m ber.adapt synth       --config C   # channel + locale + synthetic split + block + features + labels
  python -m ber.adapt pseudo      --config C   # optional, after a first `pipeline predict` (needs scores_gbdt.parquet)
  python -m ber.adapt materialize --config C   # work_dir/adapt_aug/train/{feats,records}.parquet + gt.json =
                                               #   train rows + synthetic rows (+ pseudo rows), indices offset
  python -m ber.adapt train       --config C   # gbdt.train_stack({**cfg, work_dir: work_dir/adapt_aug}); artifacts
                                               #   go to the usual artifacts_dir; calibration + decision re-fit on
                                               #   REAL train S1 only (adapt.recalibrate_real)
  python -m ber.pipeline predict  --config C --split test                  # unchanged
In-process alternative (2-line hook in gbdt.train_stack right after df / rec / gt are loaded):
  if cfg.get("adapt", {}).get("enabled"):
      from .. import adapt; df, rec, gt = adapt.augment_train(cfg, df, rec, gt)
Validation (read-only on the parquet cache): python -m ber.adapt validate --pq D:/er_data/pq --out <scratch>
"""
from __future__ import annotations

import argparse
import collections
import json
import logging
import math
import os
import re
import time
import zlib
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler, Levenshtein

from . import normalize as N

log = logging.getLogger(__name__)

DEFAULTS = {
    "enabled": True,
    "countries": "auto",          # or an explicit list, e.g. [france]
    "synth_split": "adapt_synth",
    "pseudo_split": "adapt_pseudo",
    "aug_root": "adapt_aug",       # relative to work_dir
    "n_seeds": 8000,               # seed S1 per unseen country (their S1 kNN neighbours are added)
    "k_neighbours": 4,             # S1-S1 neighbours per seed (name view and name+address view)
    "channel_s1": 60000,           # train S1 sampled to learn the noise channel
    "locale_cands": 400000,        # unlabelled unseen-country S2/S3 records used for the label-free marginals
    "distractor_scale": 1.0,       # x the train rate of unmatched distractors per S1
    "pseudo": True,
    "pseudo_p_pos": 0.98, "pseudo_margin": 0.3, "pseudo_p_neg": 0.01, "pseudo_max_s1": 50000,
    "tune_on_aug": False,          # decision tuning on real train S1 only (recommended)
    "recalibrate_real": True,
    "seed": 0,
}

AUG_PREFIX_SYNTH = "A:"
AUG_PREFIX_PSEUDO = "P:"


def acfg(cfg: dict) -> dict:
    out = dict(DEFAULTS)
    out.update((cfg or {}).get("adapt", {}) or {})
    return out


# ================================================================================================================
# 1. unseen countries
# ================================================================================================================
def ckey(c) -> str:
    return (c or "").strip().lower()


def unseen_countries(train_countries: Iterable[str], test_countries: Iterable[str]) -> List[str]:
    """Countries (normalised like country_norm) present in test but not in train. Open set: nothing is hard-coded."""
    tr = {ckey(c) for c in train_countries if ckey(c)}
    return sorted({ckey(c) for c in test_countries if ckey(c)} - tr)


def countries_in_cache(cache_dir: str, split: str) -> set:
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    vals = set()
    for k in (1, 2, 3):
        p = os.path.join(cache_dir, f"{split}_s{k}.parquet")
        if os.path.exists(p):
            vals |= {ckey(v) for v in pc.unique(pq.read_table(p, columns=["country"])["country"]).to_pylist()}
    return vals


def unseen_from_cache(cache_dir: str) -> List[str]:
    return unseen_countries(countries_in_cache(cache_dir, "train"), countries_in_cache(cache_dir, "test"))


# ================================================================================================================
# small text helpers
# ================================================================================================================
INDIC_RE = N.INDIC_RE
_ACCENT_RE = re.compile(r"[À-ÖØ-öø-ÿĀ-ſ]")
_ALPHA2_RE = re.compile(r"[A-Za-zÀ-ÿ]{2,}")
_DIGIT_RE = re.compile(r"\d")
CONNECTOR_RE = re.compile(
    r"(?i)(?<![\w/.])(d\s*/\s*b\s*/\s*a|d\.b\.a\.?|dba|doing business as|t\s*/\s*a|trading as|a\s*/\s*k\s*/\s*a|aka|"
    r"also known as|formerly known as|formerly|f\s*/\s*k\s*/\s*a|fka|nee|née)(?![\w/])(\s*:)?")
_TLD = r"(co\.in|org\.in|net\.in|com|net|org|in|co|biz|info|fr|us|io)"
DOMAIN_RE = re.compile(r"(?i)^(www\.)?([^\s|@#]+?)\." + _TLD + r"$")
SEPURL_RE = re.compile(r"(?i)^(.*\S)\s*\|\s*(www\.)?([\w-]+)\." + _TLD + r"\s*$")
PHONE_TAIL_RE = re.compile(r"\s*([-,|])?\s*(\+?\d[\d\s-]{8,}\d)\s*$")
HANDLE_RE = re.compile(r"^([@#])(\S+)$")
JUNK_RE = re.compile(r"^(\*{2,}|-{2,}|\.{2,}|>{2,}|<{2,}|~{2,}|={2,}|_{2,}|\+{2,})\s*")
JSUF_RE = re.compile(r"\s*([!+&]+)$")
_LEAD_CH = "([{<"
_TRAIL_CH = ")]}>,;:!?"
LEGAL_KEYS = set(N.LEGAL_MAP) | {"pvt", "ltd", "private", "limited"}
HON_KEYS = set(N.HONORIFICS) | {"dr", "shri", "sri", "smt", "mr", "mrs", "ms", "m/s", "messrs", "the", "ets"}
AMP_KEYS = {"&", "+", "and", "et"}
STOP_KEYS = set(N.STOP_NAME) - AMP_KEYS
LEET_FWD = {"o": "0", "l": "1", "i": "1", "e": "3", "a": "4", "s": "5", "t": "7"}
ACCENT_DEFAULT = {"a": "áàâä", "e": "éèêë", "i": "íîï", "o": "óôö", "u": "úùûü", "c": "ç", "n": "ñ", "y": "ÿ"}
NUM_WORDS = {"first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth", "tenth"}
UNIT_RE = re.compile(r"(?i)^\s*(unit|apt|apartment|suite|ste|fl|floor|flr|bldg|building|room|rm|shop|office|flat|"
                     r"appartement|app|batiment|bat|etage|escalier|porte)\b")
POBOX_ANY_RE = re.compile(r"(?i)\b(p\.?\s*o\.?\s*box|post office box|pmb|p\.m\.b\.?|b\.?\s*p\.?|cs|cedex)\s*#?\s*\d+")
STREET_TYPES = {"st", "rd", "ave", "dr", "ln", "ct", "cir", "pl", "blvd", "hwy", "pkwy", "trl", "ter", "sq", "way", "expy",
                "fwy", "plz", "aly", "rue", "all", "imp", "chem", "rte", "quai", "crs", "fbg", "res", "pass", "marg", "sec",
                "colony", "nagar", "layout", "cross", "main", "xing", "cv", "loop", "path", "walk", "row", "run", "pt"}


def fold(s: str) -> str:
    """lower-cased ASCII fold (accents / ligatures folded, Indic romanised) -- identical to the pipeline's fold."""
    return N.fold_ascii(s or "")


def _capwords(s: str) -> str:
    return " ".join(w[:1].upper() + w[1:].lower() if w else w for w in s.split(" "))


def _case_of(s2: str, s1: str) -> str:
    l2 = [c for c in s2 if c.isalpha()]
    l1 = [c for c in s1 if c.isalpha()]
    if not l2:
        return ""
    if all(c.isupper() for c in l2) and not all(c.isupper() for c in l1):
        return "U"
    if all(c.islower() for c in l2) and not all(c.islower() for c in l1):
        return "L"
    return ""


def apply_case(s: str, c: str) -> str:
    if c == "U":
        return s.upper()
    if c == "L":
        return s.lower()
    if c == "T":
        return _capwords(s)
    return s


def _weighted(rng: np.random.Generator, items: Sequence, weights: Sequence[float]):
    w = np.asarray(weights, dtype=np.float64)
    if not len(items) or w.sum() <= 0:
        return None
    return items[int(rng.choice(len(items), p=w / w.sum()))]


def _counter_pick(rng, c) -> Optional[str]:
    """Sample from a Counter / dict / [[item, count], ...] list."""
    if not c:
        return None
    if isinstance(c, dict):
        items, w = list(c.keys()), list(c.values())
    else:
        items, w = [x[0] for x in c], [x[1] for x in c]
    return _weighted(rng, items, w)


def _top(counter: collections.Counter, n: int) -> List[list]:
    return [[k, int(v)] for k, v in counter.most_common(n)]


# ================================================================================================================
# 2a. name tokens + alignment
# ================================================================================================================
class Tok:
    __slots__ = ("s", "lead", "trail", "key", "cls")

    def __init__(self, s, lead="", trail="", key=None, cls=None):
        self.s, self.lead, self.trail = s, lead, trail
        self.key = _tkey(s) if key is None else key
        self.cls = _tcls(self.key) if cls is None else cls

    def render(self) -> str:
        return self.lead + self.s + self.trail

    def copy(self) -> "Tok":
        return Tok(self.s, self.lead, self.trail, self.key, self.cls)


def _tkey(s: str) -> str:
    k = fold(s).replace(".", "")
    return re.sub(r"[^a-z0-9&+/']", "", k)


def _tcls(k: str) -> str:
    if k in AMP_KEYS:
        return "amp"
    if k in LEGAL_KEYS:
        return "legal"
    if k in HON_KEYS:
        return "hon"
    if k in STOP_KEYS:
        return "stop"
    return "core"


def tokenize_name(s: str) -> List[Tok]:
    out = []
    for ch in (s or "").split():
        lead = trail = ""
        while ch and ch[0] in _LEAD_CH:
            lead += ch[0]; ch = ch[1:]
        while ch and ch[-1] in _TRAIL_CH:
            trail = ch[-1] + trail; ch = ch[:-1]
        if not ch:
            if out:
                out[-1].trail += lead + trail
            continue
        out.append(Tok(ch, lead, trail))
    return out


def render(toks: List[Tok]) -> str:
    return " ".join(t.render() for t in toks if t.render())


def _tok_sim(a: Tok, b: Tok) -> float:
    if a.cls == "legal" and b.cls == "legal":
        return 1.0 if N.LEGAL_MAP.get(a.key, a.key) == N.LEGAL_MAP.get(b.key, b.key) else 0.86
    if a.cls == "amp" and b.cls == "amp":
        return 1.0
    if not a.key or not b.key:
        return 0.0
    kb = N._deleet(b.key)
    if a.key == kb or a.key == b.key:
        return 1.0
    return JaroWinkler.normalized_similarity(a.key, kb)


def align_tokens(t1: List[Tok], t2: List[Tok], thr: float = 0.84, thr_low: float = 0.62) -> Dict[int, int]:
    """Greedy best-first token alignment. Returns {index in t2: index in t1}."""
    cand = []
    for i, a in enumerate(t1):
        for j, b in enumerate(t2):
            s = _tok_sim(a, b)
            if s >= thr_low:
                cand.append((s, -abs(i - j), i, j))
    cand.sort(reverse=True)
    u1, u2, m = set(), set(), {}
    for s, _, i, j in cand:
        if i in u1 or j in u2:
            continue
        if s < thr and not (t1[i].cls == "core" and t2[j].cls == "core" and t1[i].key[:1] == N._deleet(t2[j].key)[:1]
                            and len(t1[i].key) >= 4):
            continue
        m[j] = i; u1.add(i); u2.add(j)
    return m


def _bracket(t: Tok) -> str:
    """'[]' / '()' when the token is wrapped in brackets, else ''."""
    if t.lead.endswith(("(", "[")) and t.trail.startswith((")", "]")):
        return t.lead[-1] + t.trail[0]
    return ""


def _edit_types(a: str, b: str) -> List[str]:
    ops = Levenshtein.editops(a, b)
    out, skip = [], set()
    ops = list(ops)
    for k, op in enumerate(ops):
        if k in skip:
            continue
        if (op.tag == "replace" and k + 1 < len(ops) and ops[k + 1].tag == "replace" and ops[k + 1].src_pos == op.src_pos + 1
                and op.src_pos + 1 < len(a) and op.dest_pos + 1 < len(b)
                and a[op.src_pos] == b[op.dest_pos + 1] and a[op.src_pos + 1] == b[op.dest_pos]):
            out.append("transpose"); skip.add(k + 1)
        else:
            out.append(op.tag)
    return out


# ================================================================================================================
# 2b. name op detection (train gold pairs)
# ================================================================================================================
def detect_name(n1: str, n2: str) -> Tuple[dict, dict]:
    """(signature, vocabulary observations) of the ops turning S1 name n1 into candidate name n2."""
    sig: dict = {}
    voc: Dict[str, list] = collections.defaultdict(list)
    n1, n2 = (n1 or "").strip(), (n2 or "").strip()
    if not n2:
        sig["empty"] = 1
        return sig, voc
    if INDIC_RE.search(n2):
        sig["x"] = 1  # native script: handled by translit.py, never replayed for Latin locales
        return sig, voc
    c = _case_of(n2, n1)
    if c:
        sig["case"] = c
    s = n2
    m = JUNK_RE.match(s)
    if m and not JUNK_RE.match(n1):
        sig["junk"] = m.group(1); voc["junk"].append(m.group(1)); s = s[m.end():]
    m = PHONE_TAIL_RE.search(s)
    if m and not PHONE_TAIL_RE.search(n1):
        sig["phone"] = m.group(1) or ""; s = s[:m.start()].strip()
    m = SEPURL_RE.match(s)
    if m:
        sig["url"] = len(m.group(3)); s = m.group(1).strip()
    m = JSUF_RE.search(s)
    if m and not JSUF_RE.search(n1):
        sig["jsuf"] = m.group(1); voc["jsuf"].append(m.group(1)); s = s[:m.start()].strip()
    m = CONNECTOR_RE.search(s)
    if m and not CONNECTOR_RE.search(n1):
        left, right = s[:m.start()].strip(), s[m.end():].strip()
        f1 = fold(n1)
        sl, sr = fuzz.token_set_ratio(fold(left), f1), fuzz.token_set_ratio(fold(right), f1)
        alias, s, order = (left, right, 0) if sr >= sl else (right, left, 1)
        conn = re.sub(r"\s+", " ", m.group(0).strip())
        sig["dba"] = conn; sig["dbo"] = order
        voc["dba"].append(conn)
        at = alias.split()
        if at:
            pw = re.sub(r"[^\w]+$", "", at[0])
            if pw:
                voc["pseudo"].append(pw[:1].upper() + pw[1:].lower())
            junk = at[0][len(pw):]
            suf = " ".join(at[1:])
            sig["dbs"] = (junk + (" " + suf if suf else "")).strip()
            if suf:
                voc["dbs"].append(suf)
    # whole-name forms -------------------------------------------------------------------------------------------
    f1 = fold(n1)
    t1 = tokenize_name(n1)
    core_cat = "".join(t.key for t in t1 if t.cls == "core")
    all_cat = "".join(t.key for t in t1 if t.cls in ("core", "legal", "stop", "amp") and t.key not in ("&", "+"))
    hon = ""
    parts = s.split()
    if len(parts) == 2 and _tkey(parts[0]) in HON_KEYS and (DOMAIN_RE.match(parts[1]) or HANDLE_RE.match(parts[1])):
        hon, s = parts[0], parts[1]
    stem = None
    m = HANDLE_RE.match(s)
    if m and len(fold(m.group(2))) >= 3:
        sig["hdl"] = m.group(1); stem = m.group(2)
    else:
        m = DOMAIN_RE.match(s)
        if m and " " not in s:
            sig["dom"] = m.group(3).lower(); stem = m.group(2)
            if m.group(1):
                sig["www"] = 1
            voc["tld"].append(m.group(3).lower())
        elif " " not in s and len(s) > 6 and s.lower().endswith("com") and core_cat and \
                fuzz.ratio(_tkey(s[:-3]), core_cat) >= 80:
            sig["dom"] = "com"; sig["nodot"] = 1; stem = s[:-3]
    if stem is not None:
        k = _tkey(stem)
        if hon:
            sig["dhon"] = hon; voc["hon"].append(hon)
        if all_cat != core_cat and fuzz.ratio(k, all_cat) > fuzz.ratio(k, core_cat):
            sig["dlg"] = 1
        return sig, voc
    if re.fullmatch(r"[A-Z]{2,6}", s) and len(t1) >= 2:
        ini = "".join(t.key[:1] for t in t1 if t.cls == "core").upper()
        ini2 = "".join(t.key[:1] for t in t1 if t.cls in ("core", "legal")).upper()
        if s in (ini, ini2):
            sig["acr"] = 1 if s == ini else 2
            return sig, voc
    fs = fold(s)
    if len(s.split()) <= 2 and f1 and fuzz.partial_ratio(fs, f1) < 55 and fuzz.token_set_ratio(fs, f1) < 45:
        sig["rep"] = 1
        w = re.sub(r"[^\w]", "", s.split()[0])
        if w:
            voc["pseudo"].append(w[:1].upper() + w[1:].lower())
        if len(s.split()) == 2:
            sig["rsuf"] = s.split()[1]
        return sig, voc
    # token level ------------------------------------------------------------------------------------------------
    if re.search(r"[A-Za-z]-[A-Za-z]", s) and not re.search(r"[A-Za-z]-[A-Za-z]", n1):
        sig["hyph"] = 1
        s = re.sub(r"(?<=[A-Za-zÀ-ÿ])-(?=[A-Za-zÀ-ÿ])", " ", s)
    if "  " in s and "  " not in n1:
        sig["dsp"] = 1
    if "," in n1 and "," not in s:
        sig["dcom"] = 1
    t2 = tokenize_name(s)
    m = align_tokens(t1, t2)
    inv = {i: j for j, i in m.items()}
    # order
    order = [m[j] for j in sorted(m)]
    if order != sorted(order):
        leg = [j for j in sorted(m) if t1[m[j]].cls == "legal"]
        rest = [m[j] for j in sorted(m) if t1[m[j]].cls != "legal"]
        if leg and leg[0] == min(m) and rest == sorted(rest) and all(j < min(jj for jj in m if t1[m[jj]].cls != "legal")
                                                                    for j in leg if any(t1[m[jj]].cls != "legal" for jj in m)):
            sig["lf"] = 1
        else:
            sig["shuf"] = 1
    # unmatched S1 tokens
    dl = dc = 0
    for i, t in enumerate(t1):
        if i in inv:
            continue
        if t.cls == "legal":
            dl += 1
        elif t.cls == "hon":
            sig["dhon1"] = 1
        elif t.cls == "core":
            dc += 1
    if dl:
        sig["dl"] = dl
    if dc:
        sig["dc"] = dc
    # unmatched candidate tokens
    matched_keys = {t2[j].key for j in m} | {N._deleet(t2[j].key) for j in m}
    adds, al, dup, ld = [], [], 0, 0
    n2t = len(t2)
    for j, t in enumerate(t2):
        if j in m:
            continue
        if t.key in matched_keys or N._deleet(t.key) in matched_keys:
            if t.cls == "legal":
                ld += 1
            else:
                dup += 1
            continue
        if t.cls == "legal":
            canon = N.LEGAL_MAP.get(t.key, t.key)
            if any(N.LEGAL_MAP.get(t2[jj].key, t2[jj].key) == canon for jj in m if t2[jj].cls == "legal"):
                ld += 1
            else:
                al.append(canon); voc["legal_add"].append(canon)
        elif t.cls == "hon":
            sig["hon"] = t.s; voc["hon"].append(t.s)
        elif t.cls in ("core", "stop") and t.key:
            pos = "s" if j == 0 else ("e" if j == n2t - 1 else
                                      ("bl" if all(t2[jj].cls == "legal" for jj in range(j + 1, n2t)) else "m"))
            adds.append([t.key, pos]); voc["add"].append(t.key)
        if _bracket(t):
            sig["lb" if t.cls == "legal" else "cb"] = _bracket(t)
    if adds:
        sig["add"] = adds
    if al:
        sig["al"] = al
    if dup:
        sig["dup"] = dup
    if ld:
        sig["ld"] = ld
    # matched pairs: legal swaps, brackets, leetspeak, typos, accents, amp
    typo, leet, acc, unacc = [], 0, 0, 0
    title_ev = 0
    for j, i in m.items():
        a, b = t1[i], t2[j]
        if _bracket(b) and not _bracket(a):
            sig["lb" if a.cls == "legal" else "cb"] = _bracket(b)
        if b.s != a.s and b.s.lower() == a.s.lower() and b.s == b.s[:1].upper() + b.s[1:].lower():
            title_ev += 1
        if a.cls == "legal" and b.cls == "legal":
            # different legal surface (Inc -> Incorporated, P.C. -> Inc) or dotted form (PLLC -> P.L.L.C.)
            if b.key != a.key or b.s.count(".") != a.s.count("."):
                sig["ls"] = 1
                voc["legal_swap"].append((N.LEGAL_MAP.get(a.key, a.key), b.s))
            if not _ACCENT_RE.search(b.s):
                continue
            # accented legal form (Límited): fall through to the accent counter
        if a.cls == "amp":
            if b.key != a.key:
                sig["amp"] = b.key; voc["amp"].append(b.s)
            continue
        ka, kb = a.key, b.key
        if kb != ka and N._deleet(kb) == ka:
            leet += 1
        elif kb != ka:
            kd = N._deleet(kb)
            ed = _edit_types(ka, kd)
            if ed and len(ed) <= max(3, len(ka) // 2):
                typo.append(len(ed))
                voc["edit"].extend(ed)
                for op in Levenshtein.editops(ka, kd):
                    if op.tag in ("insert", "replace") and op.dest_pos < len(kd):
                        voc["ins"].append(kd[op.dest_pos])
        na = len(_ACCENT_RE.findall(a.s)); nb = len(_ACCENT_RE.findall(b.s))
        if nb > na:
            acc += nb - na
            for ca, cb in zip(a.s.lower(), b.s.lower()):
                if ca != cb and _ACCENT_RE.match(cb) and fold(cb) == ca:
                    voc["accent"].append((ca, cb))
        elif na > nb:
            unacc += na - nb
    if title_ev and "case" not in sig and s == _capwords(s):
        sig["case"] = "T"
    if typo:
        sig["typo"] = typo
    if leet:
        sig["leet"] = leet
    if acc:
        sig["acc"] = acc
    if unacc:
        sig["unacc"] = unacc
    return sig, voc


# ================================================================================================================
# 2c. address segments, op detection
# ================================================================================================================
def split_segs(a: str) -> List[str]:
    return [x.strip() for x in (a or "").split(",") if x.strip()]


def _seg_key(seg: str) -> str:
    s = fold(seg)
    s = N.POBOX_RE.sub(" ", s).replace("#", " ")
    s = N.HOUSE_PREFIX_RE.sub(" ", s)
    s = N.ORDINAL_RE.sub(r"\1", s)
    s = N.LEADZERO_RE.sub(r"\1", s)
    return " ".join(N._canon_addr_tokens(s))


def classify_segments(segs: List[str], is_admin) -> List[str]:
    """street / unit / num / admin / city / loc (other locality) for S1-style segments."""
    cls, street_done = [], False
    for s in segs:
        if is_admin(s):
            c = "admin"
        elif UNIT_RE.match(s):
            c = "unit"
        elif _DIGIT_RE.search(s):
            if _ALPHA2_RE.search(s):
                c = "unit" if street_done else "street"
                street_done = True
            else:
                c = "num"
        else:
            c = "loc"
        cls.append(c)
    locs = [i for i, c in enumerate(cls) if c == "loc"]
    if locs:
        adm = [i for i, c in enumerate(cls) if c == "admin"]
        city = max((i for i in locs if adm and i < adm[0]), default=locs[-1])
        cls[city] = "city"
    return cls


def _state_admin_fn(country: str):
    lk = N.STATE_LOOKUP.get(ckey(country), {})

    def f(s: str) -> bool:
        return bool(INDIC_RE.search(s)) or (bool(lk) and fold(s).strip(" .") in lk)
    return f


def _first_num_tok(toks: List[str]) -> int:
    for i, t in enumerate(toks):
        if _DIGIT_RE.search(t):
            return i
    return -1


def _house_op(seg1: str, seg2: str) -> Tuple[str, str]:
    """(op, prefix template) comparing the house number of the S1 street segment with the candidate's."""
    a, b = seg1.split(), seg2.split()
    ia, ib = _first_num_tok(a), _first_num_tok(b)
    if ia < 0:
        return "", ""
    h1 = a[ia]
    pre = ""
    if ib > 0:
        pre_toks = b[:ib]
        a_pre = {x.lower() for x in a[:ia]}
        extra = [x for x in pre_toks if x.lower() not in a_pre]
        if extra and len(" ".join(extra)) <= 16:
            pre = re.sub(r"\d", "9", " ".join(extra))
    if ib < 0:
        return "d", pre
    h2 = b[ib]
    if h2 == h1:
        if ib + 1 < len(b) and b[ib + 1] == "1/2" and not (ia + 1 < len(a) and a[ia + 1] == "1/2"):
            return "half", pre
        return "", pre
    if h2.startswith("#") and h2.lstrip("#") == h1:
        return "hash" + str(len(h2) - len(h2.lstrip("#"))), pre
    if h2.lstrip("0") == h1.lstrip("0") and h2.startswith("0"):
        return "z%d" % (len(h2) - len(h1)), pre
    if h2 in (h1 + "-", h1 + "."):
        return ("dash" if h2.endswith("-") else "dot"), pre
    if re.fullmatch(re.escape(h1) + r"-\d+", h2):
        return "range", pre
    if re.fullmatch(re.escape(h1) + r"-?[A-Za-z]", h2):
        return "letter", pre
    d1, d2 = re.sub(r"\D", "", h1), re.sub(r"\D", "", h2)
    if d1 and d2 and d1 != d2 and Levenshtein.distance(d1, d2) <= 2:
        return "typo", pre
    return "other", pre


def _street_tok_ops(seg1: str, seg2: str) -> Tuple[int, int, List[Tuple[str, str]], int, int]:
    """(#street-type abbreviations, #typos, abbreviation pairs, ordinal changed, number word) inside a street segment."""
    a = [x.strip(".").lower() for x in fold(seg1).split()]
    b_raw = seg2.split()
    b = [fold(x).strip(".").lower() for x in b_raw]
    ab = ty = ordc = nw = 0
    pairs = []
    used = set()
    for x in a:
        if not x or _DIGIT_RE.search(x):
            if N.ORDINAL_RE.fullmatch(x or ""):
                num = re.match(r"\d+", x).group(0)
                for y in b:
                    if y.startswith(num) and y != x and N.ORDINAL_RE.fullmatch(y):
                        ordc = 1
                if any(y in NUM_WORDS for y in b):
                    nw = 1
            continue
        cx = N.ADDR_CANON.get(x)
        best, bj = 0.0, -1
        for j, y in enumerate(b):
            if j in used or not y or _DIGIT_RE.search(y):
                continue
            if cx is not None and N.ADDR_CANON.get(y) == cx:
                best, bj = 2.0, j
                break
            s = JaroWinkler.normalized_similarity(x, y)
            if s > best:
                best, bj = s, j
        if bj < 0:
            continue
        used.add(bj)
        y = b[bj]
        if best == 2.0:
            if y != x:
                ab += 1
                pairs.append((cx, b_raw[bj]))
        elif 0.8 <= best < 1.0 and y != x:
            ty += 1
    return ab, ty, pairs, ordc, nw


def detect_addr(a1: str, a2: str, country: str) -> Tuple[dict, dict]:
    sig: dict = {}
    voc: Dict[str, list] = collections.defaultdict(list)
    seg2 = split_segs(a2)
    if not seg2:
        sig["E"] = 1
        return sig, voc
    seg1 = split_segs(a1)
    if not seg1:
        return sig, voc
    is_admin = _state_admin_fn(country)
    cls1 = classify_segments(seg1, is_admin)
    lk = N.STATE_LOOKUP.get(ckey(country), {})
    cls2 = []
    for s in seg2:
        f = fold(s).strip(" .")
        if f in N.NULL_TOKENS or s.strip().lower() in ("<null>",):
            cls2.append("null")
        elif POBOX_ANY_RE.search(s) and not POBOX_ANY_RE.search(a1) and len(s.split()) <= 4:
            cls2.append("pobox")
        elif is_admin(s):
            cls2.append("admin")
        else:
            cls2.append("x")
    for s, c in zip(seg2, cls2):
        if c == "null":
            sig["null"] = s; voc["null"].append(s)
        elif c == "pobox":
            t = re.sub(r"\d", "9", s); sig["po"] = t; voc["po"].append(t)
    # admin
    adm1 = [i for i, c in enumerate(cls1) if c == "admin"]
    adm2 = [j for j, c in enumerate(cls2) if c == "admin"]
    match: Dict[int, int] = {}
    if adm1:
        st1 = lk.get(fold(seg1[adm1[0]]).strip(" ."), fold(seg1[adm1[0]]))
        if adm2:
            j = adm2[0]
            match[j] = adm1[0]
            s2 = seg2[j]
            if s2 == seg1[adm1[0]]:
                sig["adm"] = "k"
            else:
                sig["adm"] = "n" if INDIC_RE.search(s2) else ("c" if len(s2.strip()) <= 3 else "f")
                if st1 and not INDIC_RE.search(s2) and lk.get(fold(s2).strip(" .")) not in (None, st1):
                    sig["adm"] = "w"  # a different (wrong) state
        else:
            sig["adm"] = "d"
    elif adm2:
        sig["aadd"] = 1
    keys1 = [_seg_key(s) for s in seg1]
    keys2 = [_seg_key(s) for s in seg2]
    cand = []
    for i, c1 in enumerate(cls1):
        if c1 == "admin" or not keys1[i]:
            continue
        for j, c2 in enumerate(cls2):
            if c2 != "x" or not keys2[j]:
                continue
            sc = fuzz.token_set_ratio(keys1[i], keys2[j])
            if c1 == "street" and _DIGIT_RE.search(seg2[j]):
                n1 = set(re.findall(r"\d+", keys1[i])); n2 = set(re.findall(r"\d+", keys2[j]))
                if n1 & n2:
                    sc += 10
            if sc >= 60:
                cand.append((sc, -abs(i - j), i, j))
    cand.sort(reverse=True)
    used1, used2 = set(match.values()), set(match)
    dup = 0
    for sc, _, i, j in cand:
        if j in used2:
            continue
        if i in used1:
            if sc >= 95 and keys2[j] == keys1[i]:
                dup += 1; used2.add(j)
            continue
        match[j] = i; used1.add(i); used2.add(j)
    if dup:
        sig["dup"] = 1
    order = [match[j] for j in sorted(match)]
    if order != sorted(order):
        sig["ro"] = 1
    # unmatched S1 segments
    unmatched2 = [j for j, c in enumerate(cls2) if c == "x" and j not in match]
    for i, c in enumerate(cls1):
        if i in used1:
            continue
        if c == "street":
            sig["dst"] = 1
        elif c == "unit":
            sig["du"] = 1
        elif c == "city":
            if unmatched2 and not any(_DIGIT_RE.search(seg2[j]) for j in unmatched2[:1]):
                sig["crep"] = 1; unmatched2 = unmatched2[1:]
            else:
                sig["dct"] = 1
        elif c in ("loc", "num"):
            sig["do"] = sig.get("do", 0) + 1
    if unmatched2:
        sig["xtra"] = len(unmatched2)
    # matched segment internals
    cs = cc = ""
    for j, i in match.items():
        c1 = cls1[i]
        if c1 == "admin":
            continue
        s1, s2 = seg1[i], seg2[j]
        c = _case_of(s2, s1)
        if c1 == "street":
            cs = c or cs
            h, pre = _house_op(s1, s2)
            if h:
                sig["h"] = h
            if pre:
                sig["hp"] = pre; voc["hp"].append(pre)
            ab, ty, pairs, ordc, nw = _street_tok_ops(s1, s2)
            if ab:
                sig["ab"] = ab
                voc["abbr"].extend(pairs)
            if ty:
                sig["sty"] = ty
            if ordc:
                sig["ord"] = 1
            if nw:
                sig["nw"] = 1
        elif c1 == "unit":
            cs = cs or c
            if fold(s1) != fold(s2):
                sig["uf"] = 1
        else:
            cc = c or cc
            if c1 == "city":
                f1, f2 = fold(s1), fold(s2)
                if f1 != f2:
                    if f2.startswith(f1) or f1 in f2:
                        suf = s2[len(s1):].strip() if s2.lower().startswith(s1.lower()) else ""
                        if suf:
                            sig["csuf"] = suf; voc["csuf"].append(suf)
                    elif Levenshtein.normalized_similarity(f1, f2) >= 0.7:
                        sig["ctyp"] = 1
    if cs:
        sig["cs"] = cs
    if cc:
        sig["cc"] = cc
    return sig, voc


# ================================================================================================================
# 2d. the noise channel
# ================================================================================================================
_VOC_KEYS = ("junk", "jsuf", "dba", "dbs", "pseudo", "tld", "hon", "legal_add", "add", "amp", "edit", "ins", "null", "po",
             "hp", "csuf")


class NoiseChannel:
    """Empirical S1 -> S2/S3 channel: pools of (name signature, address signature) per source + vocabularies."""

    def __init__(self):
        self.pools: Dict[str, List[list]] = {"2": [], "3": []}
        self.voc: Dict[str, collections.Counter] = {k: collections.Counter() for k in _VOC_KEYS}
        self.legal_swap: Dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
        self.abbr: Dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
        self.accent: Dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
        self.cluster: collections.Counter = collections.Counter()
        self.stats: dict = {}

    # -------------------------------------------------------------------------------------------------------------
    def fit(self, pairs: pd.DataFrame, max_per_source: int = 80000, seed: int = 0,
            gt_sizes: Optional[pd.DataFrame] = None) -> "NoiseChannel":
        """pairs: s1, cand, name1, addr1, name2, addr2, country (one row per gold pair).
        gt_sizes (optional): columns n2, n3 per S1 incl. singletons (0, 0); default = computed from `pairs`."""
        t0 = time.time()
        rng = np.random.default_rng(seed)
        op_count: Dict[str, collections.Counter] = {"name": collections.Counter(), "addr": collections.Counter()}
        n_by_src = collections.Counter()
        for r in pairs.itertuples(index=False):
            src = "3" if str(r.cand).startswith("S3") else "2"
            ns, nv = detect_name(r.name1, r.name2)
            as_, av = detect_addr(r.addr1, r.addr2, r.country)
            n_by_src[src] += 1
            pool = self.pools[src]
            if len(pool) < max_per_source:
                pool.append([ns, as_])
            else:  # reservoir sampling keeps an unbiased subset
                k = rng.integers(0, n_by_src[src])
                if k < max_per_source:
                    pool[k] = [ns, as_]
            for k in ns:
                op_count["name"][k] += 1
            for k in as_:
                op_count["addr"][k] += 1
            for vv in (nv, av):
                for k, items in vv.items():
                    if k == "legal_swap":
                        for canon, surf in items:
                            self.legal_swap[canon][surf] += 1
                    elif k == "abbr":
                        for canon, surf in items:
                            self.abbr[canon][surf] += 1
                    elif k == "accent":
                        for base, accd in items:
                            self.accent[base][accd] += 1
                    elif k in self.voc:
                        self.voc[k].update(items)
        n = max(1, len(pairs))
        self.stats = {"n_pairs": int(len(pairs)), "by_source": dict(n_by_src),
                      "name_op_rate": {k: round(v / n, 5) for k, v in op_count["name"].most_common()},
                      "addr_op_rate": {k: round(v / n, 5) for k, v in op_count["addr"].most_common()}}
        if gt_sizes is None:
            src = pairs["cand"].str[:2]
            g = pd.DataFrame({"s1": pairs["s1"], "n2": (src == "S2").astype(int), "n3": (src == "S3").astype(int)})
            gt_sizes = g.groupby("s1")[["n2", "n3"]].sum()
        self.cluster = collections.Counter(zip(gt_sizes["n2"].astype(int), gt_sizes["n3"].astype(int)))
        log.info("noise channel fit on %d pairs in %.0fs", len(pairs), time.time() - t0)
        return self

    # -------------------------------------------------------------------------------------------------------------
    def sample_sig(self, rng: np.random.Generator, src: str, latin_only: bool = True) -> Tuple[dict, dict]:
        pool = self.pools[src] or self.pools["2"] or self.pools["3"]
        for _ in range(20):
            ns, as_ = pool[int(rng.integers(0, len(pool)))]
            if latin_only and ns.get("x"):
                continue
            return ns, as_
        return {}, {}

    def sample_cluster(self, rng) -> Tuple[int, int]:
        k = _counter_pick(rng, {f"{a},{b}": c for (a, b), c in self.cluster.items()})
        if k is None:
            return 1, 2
        a, b = k.split(",")
        return int(a), int(b)

    # -------------------------------------------------------------------------------------------------------------
    def to_json(self) -> dict:
        return {"pools": self.pools, "voc": {k: _top(v, 5000) for k, v in self.voc.items()},
                "legal_swap": {k: _top(v, 50) for k, v in self.legal_swap.items()},
                "abbr": {k: _top(v, 50) for k, v in self.abbr.items()},
                "accent": {k: _top(v, 20) for k, v in self.accent.items()},
                "cluster": [[a, b, int(c)] for (a, b), c in self.cluster.items()], "stats": self.stats}

    @classmethod
    def from_json(cls, d: dict) -> "NoiseChannel":
        ch = cls()
        ch.pools = d["pools"]
        for k, v in d["voc"].items():
            ch.voc[k] = collections.Counter({a: b for a, b in v})
        for name in ("legal_swap", "abbr", "accent"):
            tgt = getattr(ch, name)
            for k, v in d[name].items():
                tgt[k] = collections.Counter({a: b for a, b in v})
        ch.cluster = collections.Counter({(a, b): c for a, b, c in d["cluster"]})
        ch.stats = d.get("stats", {})
        return ch

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_json(), f, ensure_ascii=False)

    @classmethod
    def load(cls, path: str) -> "NoiseChannel":
        with open(path, encoding="utf-8") as f:
            return cls.from_json(json.load(f))


# ================================================================================================================
# 2e. label-free locale model of the unseen country
# ================================================================================================================
def _street_type_canon(tok: str, nxt: Optional[str]) -> Optional[str]:
    k = fold(tok).strip(".").strip()
    if not k or _DIGIT_RE.search(k):
        return None
    if k == "st":  # street (segment-final) vs saint (followed by a name)
        return "st" if nxt is None else None
    if k == "saint":
        return None
    c = N.ADDR_CANON.get(k)
    return c if c in STREET_TYPES else None


def _mixed_case(s: str) -> bool:
    letters = [c for c in s if c.isalpha()]
    return bool(letters) and not all(c.isupper() for c in letters) and not all(c.islower() for c in letters)


def _is_subseq(a: str, b: str) -> bool:
    it = iter(b)
    return all(ch in it for ch in a)


class Locale:
    """Label-free marginals of one unseen country (from its test S1 and S2/S3 records only)."""

    def __init__(self, country: str = ""):
        self.country = country
        self.d: dict = {}

    # -------------------------------------------------------------------------------------------------------------
    @classmethod
    def fit(cls, s1: pd.DataFrame, cand: pd.DataFrame, country: str, min_count: int = 30) -> "Locale":
        """s1 / cand: raw columns business_name, business_address (one unseen country)."""
        t0 = time.time()
        loc = cls(ckey(country))
        d = loc.d
        n1, n2 = max(1, len(s1)), max(1, len(cand))
        a1 = s1["business_address"].fillna("").tolist()
        a2 = cand["business_address"].fillna("").tolist()
        nm1 = s1["business_name"].fillna("").tolist()
        nm2 = cand["business_name"].fillna("").tolist()
        segs1 = [split_segs(a) for a in a1]
        segs2 = [split_segs(a) for a in a2]
        # ---- S1 admin segments: frequent non-digit values with many distinct partners, mostly last ----------------
        freq, last, surf = collections.Counter(), collections.Counter(), {}
        partners: Dict[str, set] = collections.defaultdict(set)
        for sg in segs1:
            nd = [(fold(x).strip(" ."), x) for x in sg if not _DIGIT_RE.search(x)]
            ks = {k for k, _ in nd if k}
            for k, x in nd:
                if k:
                    freq[k] += 1; surf.setdefault(k, x)
            if sg and not _DIGIT_RE.search(sg[-1]):
                last[fold(sg[-1]).strip(" .")] += 1
            if len(ks) > 1:
                for k in ks:
                    if len(partners[k]) < 64:
                        partners[k] |= ks - {k}
        admin = {k: surf[k] for k, c in freq.items()
                 if c >= max(min_count, 0.002 * n1) and len(partners[k]) >= 5 and last[k] >= 0.4 * c}
        d["admin_values"] = admin
        # ---- city -> admin (S1), admin -> cities --------------------------------------------------------------------
        city_admin, admin_cities = collections.defaultdict(collections.Counter), collections.defaultdict(collections.Counter)
        city_surf = {}
        for sg in segs1:
            ks = [fold(x).strip(" .") for x in sg]
            ai = [i for i, k in enumerate(ks) if k in admin]
            if not ai:
                continue
            ak = ks[ai[0]]
            locs = [i for i, x in enumerate(sg) if not _DIGIT_RE.search(x) and ks[i] not in admin and ks[i]]
            if not locs:
                continue
            ci = max((i for i in locs if i < ai[0]), default=locs[-1])
            city_admin[ks[ci]][ak] += 1
            admin_cities[ak][sg[ci]] += 1
            city_surf.setdefault(ks[ci], sg[ci])
        d["city_admin"] = {c: v.most_common(1)[0][0] for c, v in city_admin.items()}
        d["admin_cities"] = {a: _top(v, 300) for a, v in admin_cities.items()}
        cities = set(d["city_admin"])
        # ---- candidate admin outputs (keep / replace / none), conditioned on the city -------------------------------
        cfreq, cpart, csurf = collections.Counter(), collections.defaultdict(set), collections.defaultdict(collections.Counter)
        for sg in segs2:
            nd = [(fold(x).strip(" ."), x) for x in sg if not _DIGIT_RE.search(x)]
            ks = {k for k, _ in nd if k}
            for k, x in nd:
                if k:
                    cfreq[k] += 1; csurf[k][x] += 1
                    if len(cpart[k]) < 64:
                        cpart[k] |= ks - {k}
        cand_admin = {k for k, c in cfreq.items() if k not in admin and k not in cities
                      and c >= max(min_count, 0.002 * n2) and len(cpart[k]) >= 5
                      and (c / n2) > 5 * (freq.get(k, 0) + 1) / n1}
        out_city = collections.defaultdict(collections.Counter)
        for sg in segs2:
            ks = [fold(x).strip(" .") for x in sg]
            cc = [k for k in ks if k in cities]
            if not cc:
                continue
            city = cc[0]
            outs = [(("k" if k in admin else "r"), x) for k, x in zip(ks, sg) if k in admin or k in cand_admin]
            if not outs:
                out_city[city]["n\x1f"] += 1
            else:
                kind, x = outs[0]
                out_city[city][kind + "\x1f" + x] += 1
        d["admin_out_city"] = {c: _top(v, 20) for c, v in out_city.items()}
        out_admin = collections.defaultdict(collections.Counter)
        for c, v in out_city.items():
            a = d["city_admin"].get(c)
            if a is not None:
                out_admin[a].update(v)
        d["admin_out_admin"] = {a: _top(v, 30) for a, v in out_admin.items()}
        d["cand_admin"] = sorted(cand_admin)
        # ---- street-type surfaces (mixed-case segments only; the case op is replayed separately) --------------------
        st_s, st_1 = collections.defaultdict(collections.Counter), collections.defaultdict(collections.Counter)
        for segs, tgt in ((segs2, st_s), (segs1, st_1)):
            for sg in segs:
                for x in sg:
                    if not _mixed_case(x):
                        continue
                    toks = x.split()
                    for i, t in enumerate(toks):
                        c = _street_type_canon(t, toks[i + 1] if i + 1 < len(toks) else None)
                        if c:
                            tgt[c][t] += 1
        d["street_surf"] = {c: _top(v, 12) for c, v in st_s.items() if sum(v.values()) >= min_count}
        d["street_surf_s1"] = {c: _top(v, 6) for c, v in st_1.items() if sum(v.values()) >= min_count}
        # ---- legal forms, '&' surfaces, TLDs, abbreviations ---------------------------------------------------------
        lg2, lg1c, amp2, amp1 = (collections.defaultdict(collections.Counter), collections.Counter(), collections.Counter(),
                                 collections.Counter())
        tok1, tok2 = collections.Counter(), collections.Counter()
        for name in nm1:
            ks = {_tkey(t.s) for t in tokenize_name(name)}
            tok1.update(ks)
            for t in tokenize_name(name):
                if t.cls == "legal":
                    lg1c[N.LEGAL_MAP.get(t.key, t.key)] += 1
                elif t.cls == "amp":
                    amp1[t.s] += 1
        for name in nm2:
            toks = tokenize_name(name)
            tok2.update({t.key for t in toks})
            mixed = _mixed_case(name)
            for t in toks:
                if t.cls == "legal" and mixed:
                    lg2[N.LEGAL_MAP.get(t.key, t.key)][t.s] += 1
                elif t.cls == "amp" and mixed:
                    amp2[t.s] += 1
        d["legal_surf"] = {c: _top(v, 12) for c, v in lg2.items() if sum(v.values()) >= min_count}
        d["legal_canon_s1"] = _top(lg1c, 30)
        d["amp_surf"] = _top(amp2, 8)
        d["amp_rate_s1"] = sum(amp1.values()) / n1
        d["tld"] = _top(collections.Counter(m.group(3).lower() for x in nm2 for m in [DOMAIN_RE.match(x.split()[-1])]
                                            if x.split() and m), 10)
        # abbreviations mined label-free: a short candidate-only token that is a subsequence of a depleted S1 token
        abbr = {}
        rich = [t for t, c in tok2.items() if c >= min_count and len(t) <= 4 and t.isalpha()
                and (c / n2) > 20 * (tok1.get(t, 0) + 1) / n1]
        full = [t for t, c in tok1.most_common(20000) if c >= min_count and len(t) >= 4 and t.isalpha()
                and (tok2.get(t, 0) / n2) < (c / n1)]
        for a in rich:
            best = max((f for f in full if f[0] == a[0] and len(f) >= len(a) + 2 and _is_subseq(a, f)),
                       key=lambda f: tok1[f], default=None)
            if best is not None:
                p = tok2[a] / (tok2[a] + tok2.get(best, 0) + 1e-9)
                abbr[best] = [a, round(float(p), 4)]
        d["abbr"] = abbr
        # ---- house-number prefix / zero padding / empty / case / accents --------------------------------------------
        def street_seg(sg):
            for x in sg:
                if _DIGIT_RE.search(x) and _ALPHA2_RE.search(x):
                    return x
            return None

        def pre_of(x):
            m = re.match(r"^\s*([^\d,]{1,10}?)\s*(?=\d)", x)
            return m.group(1).strip() if m else ""

        p1 = collections.Counter(pre_of(x).upper() for sg in segs1 for x in [street_seg(sg)] if x)
        p2 = collections.Counter(pre_of(x) for sg in segs2 for x in [street_seg(sg)] if x)
        ns1 = max(1, sum(p1.values())); ns2 = max(1, sum(p2.values()))
        noise_pre = collections.Counter({k: v for k, v in p2.items() if k and len(k) <= 10 and (v / ns2) > 5 * (p1.get(k.upper(), 0) + 1) / ns1})
        d["hprefix"] = {"rate": round(sum(noise_pre.values()) / ns2, 5), "surf": _top(noise_pre, 20)}

        def lead0(sg):
            x = street_seg(sg)
            if not x:
                return None
            m = re.search(r"(?<![\w/])(0+)\d", x)
            return len(m.group(1)) if m else 0

        z1 = [v for v in (lead0(sg) for sg in segs1) if v is not None]
        z2 = [v for v in (lead0(sg) for sg in segs2) if v is not None]
        zr1 = np.mean([v > 0 for v in z1]) if z1 else 0.0
        zr2 = np.mean([v > 0 for v in z2]) if z2 else 0.0
        d["zpad"] = {"rate": round(float(max(0.0, zr2 - zr1)), 5), "width": _top(collections.Counter(v for v in z2 if v > 0), 5)}
        d["empty_rate"] = round(max(0.0, float(np.mean([not x.strip() for x in a2])) - float(np.mean([not x.strip() for x in a1]))), 5)

        def case_share(xs):
            u = l = 0
            for x in xs:
                letters = [c for c in x if c.isalpha()]
                if len(letters) >= 3:
                    u += all(c.isupper() for c in letters); l += all(c.islower() for c in letters)
            return u / max(1, len(xs)), l / max(1, len(xs))

        u1, l1 = case_share(nm1); u2, l2 = case_share(nm2)
        d["name_upper"], d["name_lower"] = round(max(0.0, u2 - u1), 5), round(max(0.0, l2 - l1), 5)
        su1, _ = case_share([street_seg(sg) or "" for sg in segs1]); su2, _ = case_share([street_seg(sg) or "" for sg in segs2])
        d["street_upper"] = round(max(0.0, su2 - su1), 5)
        # accents: noise accents are accented variants of tokens whose S1 form is unaccented; strips are the reverse
        s1_surf_by_fold = collections.defaultdict(collections.Counter)
        for name in nm1:
            for t in name.split():
                s1_surf_by_fold[fold(t)][t.lower()] += 1
        acc_names = 0
        acc_map = collections.defaultdict(collections.Counter)
        strip_hit = strip_tot = 0
        for name in nm2:
            hit = False
            for t in name.split():
                ft = fold(t)
                srf = s1_surf_by_fold.get(ft)
                if not srf:
                    continue
                s1f = srf.most_common(1)[0][0]
                tl = t.lower()
                if _ACCENT_RE.search(tl) and tl != s1f:
                    hit = True
                    for ca, cb in zip(s1f, tl):
                        if ca != cb and _ACCENT_RE.match(cb):
                            acc_map[fold(cb)][cb] += 1
                if _ACCENT_RE.search(s1f):
                    strip_tot += 1
                    strip_hit += int(not _ACCENT_RE.search(tl))
            acc_names += hit
        d["accent"] = {"rate": round(acc_names / n2, 5), "map": {k: _top(v, 8) for k, v in acc_map.items()},
                       "strip": round(strip_hit / max(1, strip_tot), 5)}
        # ---- PO box templates ---------------------------------------------------------------------------------------
        po = collections.Counter(re.sub(r"\d", "9", m.group(0)) for x in a2 for m in [POBOX_ANY_RE.search(x)] if m)
        d["pobox"] = _top(po, 10)
        # ---- enriched (candidate-only) name tokens -> role via address agreement of near-duplicates ----------------
        d.update(_enriched_roles(nm1, a1, nm2, a2, tok1, tok2, n1, n2, min_count))
        d["s1_vocab"] = _top(tok1, 20000)
        d["n_s1"], d["n_cand"] = int(len(s1)), int(len(cand))
        log.info("locale %s fit in %.0fs (%d S1, %d cand)", country, time.time() - t0, len(s1), len(cand))
        return loc

    # -------------------------------------------------------------------------------------------------------------
    def is_admin(self, seg: str) -> bool:
        return fold(seg).strip(" .") in self.d.get("admin_values", {})

    def admin_output(self, rng, city_key: str, admin_key: str) -> Tuple[str, str]:
        """('k', surface) keep, ('r', surface) replace, ('n', '') drop; None when the locale has no model."""
        dist = self.d.get("admin_out_city", {}).get(city_key)
        if not dist or sum(c for _, c in dist) < 5:
            dist = self.d.get("admin_out_admin", {}).get(admin_key)
        if not dist:
            return "", ""
        k = _counter_pick(rng, dist)
        kind, surf = k.split("\x1f")
        return kind, surf

    def to_json(self) -> dict:
        return {"country": self.country, **self.d}

    @classmethod
    def from_json(cls, d: dict) -> "Locale":
        loc = cls(d.get("country", ""))
        loc.d = {k: v for k, v in d.items() if k != "country"}
        return loc

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_json(), f, ensure_ascii=False)

    @classmethod
    def load(cls, path) -> "Locale":
        with open(path, encoding="utf-8") as f:
            return cls.from_json(json.load(f))

    def report(self) -> dict:
        d = self.d
        return {"admin_values": list(d.get("admin_values", {}).values())[:10], "cand_admin": d.get("cand_admin", [])[:12],
                "street_surf": {k: v[:4] for k, v in list(d.get("street_surf", {}).items())[:8]},
                "legal_surf": {k: v[:4] for k, v in list(d.get("legal_surf", {}).items())[:6]},
                "amp_surf": d.get("amp_surf"), "abbr": dict(list(d.get("abbr", {}).items())[:10]),
                "hprefix": d.get("hprefix"), "zpad": d.get("zpad"), "empty_rate": d.get("empty_rate"),
                "name_upper": d.get("name_upper"), "street_upper": d.get("street_upper"),
                "accent": {k: v for k, v in d.get("accent", {}).items() if k != "map"},
                "pos_vocab": d.get("pos_vocab", [])[:15], "neg_vocab": d.get("neg_vocab", [])[:15],
                "pobox": d.get("pobox", [])[:3], "tld": d.get("tld")}


_ROLE_SKIP = {"dba", "formerly", "doing", "business", "aka", "known", "fka", "nee", "com", "www", "as", "t", "a", "k"}


def _enriched_roles(nm1, a1, nm2, a2, tok1, tok2, n1, n2, min_count) -> dict:
    """Candidate-only name tokens and their role. 'pos' means an appended generic word (the near-duplicate keeps the
    S1 address). 'neg' means a distractor insert (the near-duplicate has another address). Label-free."""
    rich = {t: c for t, c in tok2.items() if c >= min_count and len(t) >= 2 and t.isalpha() and t not in _ROLE_SKIP
            and t not in LEGAL_KEYS and t not in HON_KEYS and (c / n2) > 3 * (tok1.get(t, 0) + 1) / n1}
    if not rich:
        return {"pos_vocab": [], "neg_vocab": [], "roles": {}}
    idx: Dict[str, List[int]] = collections.defaultdict(list)
    core1 = []
    for i, name in enumerate(nm1):
        k = " ".join(sorted({t.key for t in tokenize_name(name) if t.cls == "core"}))
        core1.append(k)
        if k:
            idx[k].append(i)
    agree, tot = collections.Counter(), collections.Counter()
    for name, addr in zip(nm2, a2):
        toks = {t.key for t in tokenize_name(name) if t.cls == "core"}
        hit = [t for t in toks if t in rich]
        if len(hit) != 1 or not addr.strip():
            continue
        t = hit[0]
        k = " ".join(sorted(toks - {t}))
        cands = idx.get(k)
        if not cands or len(cands) > 20:
            continue
        ka = _seg_key(addr)
        best = max(fuzz.token_set_ratio(ka, _seg_key(a1[i])) for i in cands[:20])
        tot[t] += 1
        agree[t] += int(best >= 80)
    roles = {t: [int(agree[t]), int(tot[t])] for t in rich if tot[t] >= 10}
    excess = {t: max(0.0, c - n2 * (tok1.get(t, 0) + 1) / n1) for t, c in rich.items()}
    pos = [[t, round(excess[t], 1)] for t, (a, n) in roles.items() if a / n >= 0.5]
    neg = [[t, round(excess[t], 1)] for t, (a, n) in roles.items() if a / n < 0.5]
    pos.sort(key=lambda x: -x[1]); neg.sort(key=lambda x: -x[1])
    return {"pos_vocab": pos, "neg_vocab": neg, "roles": roles}


# ================================================================================================================
# 2f. synthesizer: apply a train signature to an S1 record with the locale's surfaces
# ================================================================================================================
class Synthesizer:
    def __init__(self, channel: NoiseChannel, locale: Optional[Locale] = None, seed: int = 0,
                 locale_rates: bool = True):
        self.ch = channel
        self.loc = locale or Locale("")
        self.rng = np.random.default_rng(seed)
        self.locale_rates = locale_rates
        L = self.loc.d
        self._pseudo = self.ch.voc["pseudo"]
        # mapping of train "added words" to the locale: appended generic words
        self._pos_vocab = L.get("pos_vocab") or []
        self._train_add = self.ch.voc["add"]

    # ------------------------------------------------------------------------------------------------------ names
    def pseudo_word(self) -> str:
        w = _counter_pick(self.rng, self._pseudo)
        return w or "Lyraverasyn"

    def _added_word(self, key: str) -> str:
        """An appended word. Locale vocabulary (label-free positive-role tokens) when available, else the train word."""
        r = self.rng
        if self._pos_vocab and r.random() < 0.7:
            w = _counter_pick(r, self._pos_vocab)
        else:
            w = key if r.random() < 0.5 else (_counter_pick(r, self._train_add) or key)
        return (w or key)[:1].upper() + (w or key)[1:]

    def _legal_surface(self, canon: str, cur: str) -> str:
        L = self.loc.d.get("legal_surf", {})
        if canon in L:
            return _counter_pick(self.rng, L[canon]) or cur
        sw = self.ch.legal_swap.get(canon)
        return (_counter_pick(self.rng, sw) or cur) if sw else cur

    def _legal_new(self) -> str:
        c = _counter_pick(self.rng, self.loc.d.get("legal_canon_s1") or self.ch.voc["legal_add"]) or "ltd"
        return self._legal_surface(c, c.upper())

    def _typo(self, w: str, n: int) -> str:
        r = self.rng
        letters = "abcdefghijklmnopqrstuvwxyz"
        ins = self.ch.voc["ins"]
        for _ in range(n):
            if len(w) < 3:
                break
            op = _counter_pick(r, self.ch.voc["edit"]) or "replace"
            p = int(r.integers(1, len(w)))
            ch = _counter_pick(r, ins) or letters[int(r.integers(0, 26))]
            if not ch.isalpha():
                ch = letters[int(r.integers(0, 26))]
            if w[p - 1].isupper() and (p >= len(w) or w[p].isupper()):
                ch = ch.upper()
            if op == "insert":
                w = w[:p] + ch + w[p:]
            elif op == "delete":
                w = w[:p] + w[p + 1:]
            elif op == "transpose" and p + 1 < len(w):
                w = w[:p] + w[p + 1] + w[p] + w[p + 2:]
            else:
                w = w[:p] + ch + w[p + 1:]
        return w

    def _leet(self, w: str) -> str:
        pos = [i for i, c in enumerate(w) if c.lower() in LEET_FWD]
        if not pos:
            return w
        i = pos[int(self.rng.integers(0, len(pos)))]
        return w[:i] + LEET_FWD[w[i].lower()] + w[i + 1:]

    def _accent(self, w: str, k: int) -> str:
        r = self.rng
        amap = self.loc.d.get("accent", {}).get("map") or {}
        pos = [i for i, c in enumerate(w) if c.lower() in (amap or ACCENT_DEFAULT) or c.lower() in ACCENT_DEFAULT]
        for _ in range(k):
            if not pos:
                break
            i = pos[int(r.integers(0, len(pos)))]
            base = w[i].lower()
            opts = amap.get(base) or [[c, 1] for c in ACCENT_DEFAULT.get(base, "")]
            ch = _counter_pick(r, opts) if opts else None
            if not ch:
                continue
            w = w[:i] + (ch.upper() if w[i].isupper() else ch) + w[i + 1:]
        return w

    def synth_name(self, name: str, sig: dict) -> str:
        r = self.rng
        L = self.loc.d
        if sig.get("empty"):
            return ""
        if sig.get("rep"):
            out = self.pseudo_word() + ((" " + sig["rsuf"]) if sig.get("rsuf") else "")
            return apply_case(out, sig.get("case", ""))
        toks = [t.copy() for t in tokenize_name(name)]
        if not toks:
            return name
        # locale surface resampling (label-free target marginals)
        amp_s = L.get("amp_surf")
        abbr = L.get("abbr", {})
        strip = L.get("accent", {}).get("strip", 0.0) if self.locale_rates else 0.0
        for t in toks:
            if t.cls == "amp" and amp_s and t.key in ("&", "+", "et"):
                t.s = _counter_pick(r, amp_s) or t.s
            ab = abbr.get(t.key)
            if ab and r.random() < ab[1]:
                t.s = ab[0][:1].upper() + ab[0][1:] if t.s[:1].isupper() else ab[0]
            if strip and _ACCENT_RE.search(t.s) and r.random() < strip:
                t.s = "".join(c if not _ACCENT_RE.match(c) else (fold(c).upper() if c.isupper() else fold(c)) for c in t.s)
        # legal ops
        legal = [t for t in toks if t.cls == "legal"]
        if sig.get("dl") and legal:
            k = min(len(legal), sig["dl"])
            drop = set(id(t) for t in legal[-k:])
            toks = [t for t in toks if id(t) not in drop] or toks
            legal = [t for t in toks if t.cls == "legal"]
        for t in legal:
            if sig.get("ls") or (L.get("legal_surf") and t.key in N.LEGAL_MAP and r.random() < 0.5):
                t.s = self._legal_surface(N.LEGAL_MAP.get(t.key, t.key), t.s)
        for c in sig.get("al", []):
            nt = Tok(self._legal_new())
            toks.append(nt)
        for _ in range(sig.get("ld", 0)):
            legal = [t for t in toks if t.cls == "legal"]
            if legal:
                toks.append(legal[-1].copy())
        if sig.get("lf"):
            legal = [t for t in toks if t.cls == "legal"]
            if legal:
                toks = legal + [t for t in toks if t.cls != "legal"]
        if sig.get("lb"):
            for t in toks:
                if t.cls == "legal":
                    t.lead, t.trail = sig["lb"][0], sig["lb"][1] + t.trail.lstrip(")]")
                    break
        # core ops
        core = [t for t in toks if t.cls == "core"]
        dc = min(sig.get("dc", 0), max(0, len(core) - 1))
        if dc:
            drop = set(id(core[i]) for i in r.choice(len(core), dc, replace=False))
            toks = [t for t in toks if id(t) not in drop]
        for _ in range(sig.get("dup", 0)):
            core = [i for i, t in enumerate(toks) if t.cls == "core"]
            if core:
                i = core[int(r.integers(0, len(core)))]
                toks.insert(i + 1, toks[i].copy())
        for key, pos in sig.get("add", []):
            w = Tok(self._added_word(key))
            if pos == "s":
                toks.insert(0, w)
            elif pos == "m" and len(toks) > 1:
                toks.insert(int(r.integers(1, len(toks))), w)
            elif pos == "bl":
                li = [i for i, t in enumerate(toks) if t.cls == "legal"]
                toks.insert(li[0] if li and li[0] > 0 else len(toks), w)
            else:
                toks.append(w)
        if sig.get("hon"):
            toks.insert(0, Tok(sig["hon"]))
        if sig.get("dhon1"):
            toks = [t for t in toks if t.cls != "hon"] or toks
        if sig.get("amp"):
            for t in toks:
                if t.cls == "amp":
                    t.s = sig["amp"] if not amp_s else (_counter_pick(r, amp_s) or sig["amp"])
        if sig.get("shuf") and len(toks) > 1:
            perm = r.permutation(len(toks))
            if (perm == np.arange(len(toks))).all():
                perm = perm[::-1]
            trail_last = toks[-1].trail
            toks = [toks[i] for i in perm]
        if sig.get("cb"):
            core = [t for t in toks if t.cls == "core"]
            if core:
                core[-1].lead, core[-1].trail = sig["cb"][0], sig["cb"][1]
        # char-level ops
        alpha = [t for t in toks if re.search(r"[A-Za-z]{3,}", t.s)]
        for _ in range(sig.get("leet", 0)):
            if alpha:
                t = alpha[int(r.integers(0, len(alpha)))]
                t.s = self._leet(t.s)
        for n_ed in sig.get("typo", []):
            if alpha:
                t = alpha[int(r.integers(0, len(alpha)))]
                t.s = self._typo(t.s, int(n_ed))
        n_acc = sig.get("acc", 0)
        if self.locale_rates and L.get("accent"):  # locale accent-noise rate replaces the train rate
            n_acc = (1 + int(r.random() < 0.2)) if r.random() < L["accent"]["rate"] else 0
        for _ in range(n_acc):
            if alpha:
                t = alpha[int(r.integers(0, len(alpha)))]
                t.s = self._accent(t.s, 1)
        out = render(toks)
        if sig.get("hyph") and len(toks) > 1:
            out = out.replace(" ", "-", 1 if r.random() < 0.5 else -1)
        if sig.get("dsp") and " " in out:
            i = [m.start() for m in re.finditer(" ", out)]
            p = i[int(r.integers(0, len(i)))]
            out = out[:p] + " " + out[p:]
        if sig.get("dcom"):
            out = out.replace(",", "")
        # whole-name forms
        core_keys = [t for t in toks if t.cls == "core"]
        if sig.get("acr"):
            src = core_keys if sig["acr"] == 1 else [t for t in toks if t.cls in ("core", "legal")]
            ini = "".join(fold(t.s)[:1] for t in src).upper()
            if len(ini) >= 2:
                out = ini
        if sig.get("dom") or sig.get("hdl"):
            keep = [t for t in toks if t.cls in ("core", "stop") or (sig.get("dlg") and t.cls == "legal")]
            stem = "".join(re.sub(r"[^\w]", "", t.s) for t in keep).lower() or re.sub(r"\W", "", out).lower()
            if sig.get("hdl"):
                out = sig["hdl"] + stem
            elif sig.get("nodot"):
                out = stem + "com"
            else:
                tld = _counter_pick(r, L.get("tld")) if L.get("tld") else sig["dom"]
                out = ("www." if sig.get("www") else "") + stem + "." + (tld or sig["dom"])
            if sig.get("dhon"):
                out = sig["dhon"] + " " + out
        if sig.get("url"):
            stem = "".join(re.sub(r"[^\w]", "", fold(t.s)) for t in core_keys)[: int(sig["url"])]
            out = out + " | www." + stem + ".com"
        if sig.get("jsuf"):
            out = out + sig["jsuf"]
        if sig.get("dba"):
            alias = self.pseudo_word() + (" " + sig["dbs"] if sig.get("dbs") and not sig["dbs"][:1].isalnum()
                                          else ("" if not sig.get("dbs") else " " + sig["dbs"]))
            alias = alias.replace(" +", "+").strip()
            out = (alias + " " + sig["dba"] + " " + out) if sig.get("dbo", 0) == 0 else (out + " " + sig["dba"] + " " + alias)
        if "phone" in sig:
            out = out + " - " + "".join(str(int(x)) for x in r.integers(0, 10, 10))
        if sig.get("junk"):
            out = sig["junk"] + " " + out
        case = sig.get("case", "")
        if self.locale_rates and ("name_upper" in L):
            u, l = L["name_upper"], L["name_lower"]
            x = r.random()
            case = "U" if x < u else ("L" if x < u + l else ("T" if case == "T" else ""))
        return apply_case(out, case)

    # ------------------------------------------------------------------------------------------------------ address
    def _house(self, tok: str, op: str) -> str:
        r = self.rng
        m = re.match(r"^(\d+)(.*)$", tok)
        if not m:
            return tok
        num, rest = m.group(1), m.group(2)
        if op.startswith("z"):
            k = int(op[1:] or 1) if op[1:].isdigit() else 1
            return "0" * max(1, k) + tok
        if op == "typo" and len(num) >= 2:
            i = int(r.integers(0, len(num)))
            kind = r.random()
            if kind < 0.4:
                num = num[:i] + num[i + 1:]
            elif kind < 0.8:
                num = num[:i] + str(int(r.integers(0, 10))) + num[i + 1:]
            else:
                num = num[:i] + str(int(r.integers(0, 10))) + num[i:]
            return (num or "1") + rest
        if op == "range":
            return tok + "-" + str(int(num) + int(r.integers(1, 9)))
        if op == "letter":
            return tok + ("-" if r.random() < 0.5 else "") + "ABCD"[int(r.integers(0, 4))]
        if op == "dash":
            return tok + "-"
        if op == "dot":
            return tok + "."
        if op.startswith("hash"):
            return "#" * int(op[4:] or 1) + tok
        if op == "half":
            return tok + " 1/2"
        return tok

    def _fill9(self, t: str) -> str:
        return "".join(str(int(self.rng.integers(0, 10))) if c == "9" else c for c in t)

    def synth_addr(self, addr: str, sig: dict) -> str:
        r = self.rng
        L = self.loc.d
        empty = sig.get("E")
        if self.locale_rates and "empty_rate" in L:
            empty = r.random() < L["empty_rate"]
        if empty:
            return ""
        segs = split_segs(addr)
        if not segs:
            return addr
        is_admin = self.loc.is_admin if L.get("admin_values") else (lambda s: False)
        cls = classify_segments(segs, is_admin)
        city_key = next((fold(s).strip(" .") for s, c in zip(segs, cls) if c == "city"), "")
        adm_key = next((fold(s).strip(" .") for s, c in zip(segs, cls) if c == "admin"), "")
        out: List[Tuple[str, str]] = []  # (class, text)
        for s, c in zip(segs, cls):
            if c == "street":
                if sig.get("dst"):
                    continue
                toks = s.split()
                # street-type surfaces from the locale (else the train abbreviation table when the sig says so)
                surf = L.get("street_surf", {})
                for i, t in enumerate(toks):
                    cst = _street_type_canon(t, toks[i + 1] if i + 1 < len(toks) else None)
                    if not cst:
                        continue
                    if cst in surf and self.locale_rates:
                        toks[i] = _counter_pick(r, surf[cst]) or t
                    elif sig.get("ab") and self.ch.abbr.get(cst):
                        toks[i] = _counter_pick(r, self.ch.abbr[cst]) or t
                hi = _first_num_tok(toks)
                if hi >= 0:
                    h = sig.get("h", "")
                    if self.locale_rates and L.get("zpad", {}).get("rate", 0) > 0 and not h:
                        if r.random() < L["zpad"]["rate"]:
                            h = "z" + str(_counter_pick(r, L["zpad"]["width"]) or 1)
                    if h == "d":
                        toks = toks[:hi] + toks[hi + 1:]
                    elif h:
                        toks[hi] = self._house(toks[hi], h)
                    pre = sig.get("hp", "")
                    hp = L.get("hprefix", {})
                    if self.locale_rates and hp.get("surf"):
                        pre = _counter_pick(r, hp["surf"]) if r.random() < hp["rate"] else ""
                    if pre and h != "d":
                        toks.insert(max(0, hi), self._fill9(pre))
                if sig.get("sty"):
                    alpha = [i for i, t in enumerate(toks) if len(t) >= 4 and t.isalpha()]
                    for _ in range(sig["sty"]):
                        if alpha:
                            i = alpha[int(r.integers(0, len(alpha)))]
                            toks[i] = self._typo(toks[i], 1)
                out.append(("street", " ".join(toks)))
            elif c == "unit":
                if not sig.get("du"):
                    out.append(("unit", s))
            elif c == "city":
                if sig.get("dct"):
                    continue
                t = s
                if sig.get("crep"):
                    alt = L.get("admin_cities", {}).get(adm_key or L.get("city_admin", {}).get(city_key, ""))
                    if alt:
                        t = _counter_pick(r, alt) or s
                if sig.get("ctyp"):
                    t = self._typo(t, 1)
                if sig.get("csuf") and self.ch.voc["csuf"] and not L:
                    t = t + " " + sig["csuf"]
                out.append(("city", t))
            elif c == "admin":
                kind, surf = self.loc.admin_output(r, city_key, adm_key) if self.locale_rates else ("", "")
                if not kind:
                    kind = {"d": "n", "k": "k"}.get(sig.get("adm", "k"), "k")
                    surf = s
                if kind == "k":
                    out.append(("admin", surf or s))
                elif kind == "r":
                    out.append(("admin", surf))
            elif c == "num" or c == "loc":
                if sig.get("do") and r.random() < 0.5:
                    continue
                out.append((c, s))
        if not out:
            return ""
        if sig.get("xtra") and L.get("admin_cities"):
            alt = L["admin_cities"].get(adm_key or L.get("city_admin", {}).get(city_key, ""))
            for _ in range(min(2, sig["xtra"])):
                if alt:
                    out.insert(int(r.integers(0, len(out) + 1)), ("loc", _counter_pick(r, alt)))
        if sig.get("dup"):
            i = int(r.integers(0, len(out)))
            out.insert(i, out[i])
        if sig.get("po"):
            tmpl = _counter_pick(r, L["pobox"]) if L.get("pobox") else sig["po"]
            out.insert(min(1, len(out)), ("po", self._fill9(tmpl)))
        if sig.get("null"):
            out.insert(int(r.integers(0, len(out) + 1)), ("null", sig["null"]))
        if sig.get("ro") and len(out) > 1:
            perm = r.permutation(len(out))
            if (perm == np.arange(len(out))).all():
                perm = perm[::-1]
            out = [out[i] for i in perm]
        cs, cc = sig.get("cs", ""), sig.get("cc", "")
        if self.locale_rates and "street_upper" in L:
            cs = "U" if r.random() < L["street_upper"] else ("" if cs == "U" else cs)
            cc = "U" if cs == "U" and r.random() < 0.7 else ("" if cc == "U" else cc)
        res = []
        for c, t in out:
            if c in ("street", "unit"):
                t = apply_case(t, cs)
            elif c in ("city", "loc", "num"):
                t = apply_case(t, cc)
            res.append(t)
        return ", ".join(res)

    # ------------------------------------------------------------------------------------------------------ record
    def synth_record(self, name: str, addr: str, src: str) -> Tuple[str, str, dict, dict]:
        ns, as_ = self.ch.sample_sig(self.rng, src)
        return self.synth_name(name, ns), self.synth_addr(addr, as_), ns, as_


# ================================================================================================================
# 3. distractors (negatives)
# ================================================================================================================
class DistractorModel:
    """How the generator builds UNMATCHED S2/S3 records, learned from train records + gt.

    Kinds: 'ins' = an S1 name with 1-2 inserted tokens; 'rep' = one core token replaced.
    Per kind it stores the insert position and the address relation to the source S1: same address, same street with
    another house number, same locality, or another address. It also stores the df-percentile of the inserted token in
    the S1 vocabulary (transferred to the unseen country's S1 vocabulary) and the rate of such distractors per S1."""

    def __init__(self):
        self.d: dict = {}

    def fit(self, rec: pd.DataFrame, gt: Dict[str, List[str]], max_unmatched: int = 300000, seed: int = 0) -> "DistractorModel":
        t0 = time.time()
        rng = np.random.default_rng(seed)
        matched = {c for v in gt.values() for c in v}
        s1 = rec[rec.src == 1]
        c = rec[(rec.src != 1) & ~rec.entity_id.isin(matched)]
        n_s1, n_unm = len(s1), len(c)
        if len(c) > max_unmatched:
            c = c.iloc[np.sort(rng.choice(len(c), max_unmatched, replace=False))]
        idx = collections.defaultdict(list)
        sub = collections.defaultdict(list)
        s1core = s1["name_core"].tolist()
        for i, k in enumerate(s1core):
            ts = sorted(set(k.split()))
            if not ts:
                continue
            idx[" ".join(ts)].append(i)
            if 2 <= len(ts) <= 6:
                for j in range(len(ts)):
                    sub[" ".join(ts[:j] + ts[j + 1:])].append(i)
        df1 = collections.Counter(t for k in s1core for t in set(k.split()))
        ranks = {t: r for r, (t, _) in enumerate(df1.most_common())}
        nvoc = max(1, len(ranks))
        kinds, pos, rel, pct, hdelta, n_ins = (collections.Counter(), collections.Counter(), collections.Counter(),
                                               [], [], collections.Counter())
        a1 = s1["addr_norm"].tolist(); st1 = s1["addr_street"].tolist(); h1 = s1["addr_house"].tolist()
        lo1 = s1["addr_locality"].tolist()
        for core, an, stn, hn, lon in zip(c["name_core"], c["addr_norm"], c["addr_street"], c["addr_house"], c["addr_locality"]):
            toks = core.split()
            ts = sorted(set(toks))
            if len(ts) < 2:
                continue
            hit, kind, extra = None, None, []
            for j in range(len(ts)):  # one inserted token
                k = " ".join(ts[:j] + ts[j + 1:])
                if k in idx:
                    hit, kind, extra = idx[k], "ins", [ts[j]]
                    break
            if hit is None and len(ts) >= 4:
                for j in range(len(ts)):
                    for jj in range(j + 1, len(ts)):
                        k = " ".join(t for q, t in enumerate(ts) if q not in (j, jj))
                        if k in idx:
                            hit, kind, extra = idx[k], "ins", [ts[j], ts[jj]]
                            break
                    if hit:
                        break
            if hit is None and 2 <= len(ts) <= 6:
                for j in range(len(ts)):
                    k = " ".join(ts[:j] + ts[j + 1:])
                    if k in sub:
                        hit, kind, extra = sub[k], "rep", [ts[j]]
                        break
            if hit is None or len(hit) > 30:
                continue
            kinds[kind] += 1
            n_ins[len(extra)] += 1
            p = toks.index(extra[0]) if extra[0] in toks else 0
            pos["s" if p == 0 else ("e" if p == len(toks) - 1 else "m")] += 1
            for t in extra:
                pct.append(ranks[t] / nvoc if t in ranks else -1.0)
            best = max(hit, key=lambda i: fuzz.token_set_ratio(an, a1[i]))
            if an and fuzz.token_set_ratio(an, a1[best]) >= 90 and hn == h1[best]:
                rel["same"] += 1
            elif stn and st1[best] and fuzz.token_set_ratio(stn, st1[best]) >= 80:
                rel["street"] += 1
                try:
                    hdelta.append(int(re.sub(r"\D", "", hn) or 0) - int(re.sub(r"\D", "", h1[best]) or 0))
                except ValueError:
                    pass
            elif lon and lo1[best] and fuzz.token_set_ratio(lon, lo1[best]) >= 80:
                rel["locality"] += 1
            else:
                rel["other"] += 1
        n_found = sum(kinds.values())
        scale = n_unm / max(1, len(c))
        hd = collections.Counter(int(np.clip(x, -50, 50)) for x in hdelta if x != 0)
        self.d = {"rate_per_s1": round(n_found * scale / max(1, n_s1), 5), "unmatched_per_s1": round(n_unm / max(1, n_s1), 5),
                  "near_dup_share": round(n_found / max(1, len(c)), 5), "kinds": dict(kinds), "pos": dict(pos),
                  "relation": dict(rel), "n_insert": {str(k): v for k, v in n_ins.items()},
                  "pct": [round(float(x), 4) for x in np.quantile([p for p in pct if p >= 0], np.linspace(0, 1, 101))] if any(p >= 0 for p in pct) else [],
                  "unknown_share": round(float(np.mean([p < 0 for p in pct])) if pct else 0.0, 4),
                  "house_delta": _top(hd, 40)}
        log.info("distractor model: %s (%.0fs)", {k: v for k, v in self.d.items() if k not in ("pct", "house_delta")}, time.time() - t0)
        return self

    def to_json(self):
        return self.d

    @classmethod
    def from_json(cls, d):
        m = cls(); m.d = d; return m

    @classmethod
    def default(cls) -> "DistractorModel":
        m = cls()
        m.d = {"rate_per_s1": 0.9, "unmatched_per_s1": 1.2, "kinds": {"ins": 3, "rep": 1}, "pos": {"s": 1, "m": 1, "e": 2},
               "relation": {"same": 1, "street": 2, "locality": 3, "other": 6}, "n_insert": {"1": 9, "2": 1},
               "pct": [], "unknown_share": 0.4, "house_delta": [[1, 1], [-1, 1], [2, 1], [-2, 1]]}
        return m

    # -------------------------------------------------------------------------------------------------------------
    def make(self, syn: Synthesizer, name: str, addr: str, others: Sequence[Tuple[str, str]], vocab: Sequence[str],
             neg_vocab: Sequence[list]) -> Tuple[str, str]:
        """One distractor derived from S1 (name, addr). others = (addr, admin/city key) pool of same-country S1 addresses."""
        r = syn.rng
        d = self.d
        toks = tokenize_name(name)
        kind = _counter_pick(r, d.get("kinds") or {"ins": 1}) or "ins"
        nins = int(_counter_pick(r, d.get("n_insert") or {"1": 1}) or 1)

        def new_word():
            if neg_vocab and r.random() < max(0.3, d.get("unknown_share", 0.4)):
                w = _counter_pick(r, neg_vocab)
            elif d.get("pct") and vocab:
                q = d["pct"][int(r.integers(0, len(d["pct"])))]
                w = vocab[min(len(vocab) - 1, int(q * len(vocab)))]
            else:
                w = vocab[int(r.integers(0, len(vocab)))] if vocab else "Group"
            return (w or "Group")[:1].upper() + (w or "Group")[1:]

        core = [i for i, t in enumerate(toks) if t.cls == "core"]
        if kind == "rep" and len(core) >= 2:
            i = core[int(r.integers(0, len(core)))]
            toks[i] = Tok(new_word())
        else:
            for _ in range(nins):
                p = _counter_pick(r, d.get("pos") or {"e": 1})
                w = Tok(new_word())
                if p == "s":
                    toks.insert(0, w)
                elif p == "m" and len(toks) > 1:
                    toks.insert(int(r.integers(1, len(toks))), w)
                else:
                    li = [i for i, t in enumerate(toks) if t.cls == "legal"]
                    toks.insert(li[0] if li and li[0] > 0 else len(toks), w)
        dname = render(toks)
        rel = _counter_pick(r, d.get("relation") or {"other": 1})
        if rel == "same":
            daddr = addr
        elif rel == "street":
            delta = int(_counter_pick(r, d.get("house_delta") or [[1, 1]]) or 1)
            daddr = re.sub(r"(?<![\w/])(\d+)", lambda m: str(max(1, int(m.group(1)) + delta)), addr, count=1)
        elif others:
            daddr = others[int(r.integers(0, len(others)))][0]
        else:
            daddr = addr
        ns, as_ = syn.ch.sample_sig(r, "2" if r.random() < 0.5 else "3")
        ns = {k: v for k, v in ns.items() if k not in ("rep",)}  # a distractor keeps its (inserted) name
        return syn.synth_name(dname, ns), syn.synth_addr(daddr, as_)


# ================================================================================================================
# 4. pseudo-labels from the pipeline's own test scores
# ================================================================================================================
def select_pseudo_labels(scores: pd.DataFrame, s1_country: Dict[int, str], countries: Iterable[str],
                         p_pos: float = 0.98, margin: float = 0.3, p_neg: float = 0.01, score: str = "p",
                         complete_only: bool = True) -> pd.DataFrame:
    """scores: i1, i2, s1_id, cand_id, p (work_dir/test/scores_gbdt.parquet). s1_country: i1 -> country_norm.

    Positive: p >= p_pos and the pair is this candidate's best S1 (exclusivity) with a margin >= `margin` over the
    runner-up S1, and it is among the S1's confident set. Negative: p <= p_neg. With complete_only, only S1 groups whose
    rows are ALL labelled are kept, so the stage-2 context features see the same group structure as in training.
    Returns the selected rows with a `y` column."""
    cs = {ckey(c) for c in countries}
    df = scores[scores["i1"].map(lambda i: s1_country.get(int(i), "") in cs)].copy()
    if df.empty:
        return df.assign(y=np.int8(0))
    s = df[score].to_numpy(np.float64)
    o = np.lexsort((-s, df["i2"].to_numpy()))
    i2s = df["i2"].to_numpy()[o]
    first = np.r_[True, i2s[1:] != i2s[:-1]]
    rank_c = np.empty(len(df), np.int64)
    grp = np.cumsum(first) - 1
    starts = np.flatnonzero(first)
    rank_c[o] = np.arange(len(df)) - starts[grp]
    second = np.full(len(df), 0.0)
    cnt = np.diff(np.r_[starts, len(df)])
    sec_val = np.where(cnt > 1, s[o][np.minimum(starts + 1, len(df) - 1)], 0.0)
    second[o] = sec_val[grp]
    best_c = rank_c == 0
    pos = (s >= p_pos) & best_c & ((s - second) >= margin)
    neg = s <= p_neg
    df["y"] = np.where(pos, 1, 0).astype(np.int8)
    lab = pos | neg
    if complete_only:
        ok = pd.Series(lab, index=df.index).groupby(df["i1"]).transform("all").to_numpy()
        df = df[ok]
    else:
        df = df[lab]
    return df


# ================================================================================================================
# 5. emission: synthetic split -> records.parquet / gt.json; labels; augmentation
# ================================================================================================================
def _wd(cfg, split, name=None):
    d = os.path.join(cfg["work_dir"], split)
    os.makedirs(d, exist_ok=True)
    return d if name is None else os.path.join(d, name)


def _art(cfg, name):
    os.makedirs(cfg["artifacts_dir"], exist_ok=True)
    return os.path.join(cfg["artifacts_dir"], name)


def read_country_rows(cache_dir: str, split: str, k: int, countries: Iterable[str], max_rows: Optional[int] = None,
                      seed: int = 0) -> pd.DataFrame:
    """Rows of the given countries from the parquet cache, row group by row group (memory-lean)."""
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(os.path.join(cache_dir, f"{split}_s{k}.parquet"))
    rng = np.random.default_rng(seed + k)
    cs = pa.array(sorted({ckey(c) for c in countries}))
    parts = []
    frac = None
    for g in range(pf.num_row_groups):
        t = pf.read_row_group(g)
        t = t.filter(pc.is_in(pc.utf8_lower(pc.utf8_trim_whitespace(t["country"])), value_set=cs))
        if max_rows is not None and t.num_rows:
            if frac is None:
                frac = min(1.0, max_rows / max(1, t.num_rows * pf.num_row_groups))
            keep = np.flatnonzero(rng.random(t.num_rows) < frac)
            t = t.take(pa.array(keep))
        parts.append(t)
    return pa.concat_tables(parts).to_pandas() if parts else pd.DataFrame(columns=["entity_id", "business_name", "business_address", "country"])


def load_gold_pairs(cache_dir: str, n_s1: int, seed: int = 0, countries: Optional[Iterable[str]] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """(pairs with raw text, cluster sizes incl. singletons) for a random sample of train S1 (read-only on the cache)."""
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    rng = np.random.default_rng(seed)
    s1 = pq.read_table(os.path.join(cache_dir, "train_s1.parquet"), columns=["entity_id", "country"])
    if countries is not None:
        cs = pa.array(sorted({ckey(c) for c in countries}))
        s1 = s1.filter(pc.is_in(pc.utf8_lower(s1["country"]), value_set=cs))
    ids = s1["entity_id"].to_numpy(zero_copy_only=False)
    pick = pa.array(ids[rng.choice(len(ids), min(n_s1, len(ids)), replace=False)])
    pairs = pq.read_table(os.path.join(cache_dir, "train_pairs.parquet"))
    pairs = pairs.filter(pc.is_in(pairs["s1"], value_set=pick)).to_pandas()

    def grab(f, idset):
        pf = pq.ParquetFile(os.path.join(cache_dir, f"{f}.parquet"))
        vs = pa.array(sorted(idset))
        out = []
        for g in range(pf.num_row_groups):
            t = pf.read_row_group(g)
            out.append(t.filter(pc.is_in(t["entity_id"], value_set=vs)))
        return pa.concat_tables(out).to_pandas()

    r1 = grab("train_s1", set(pick.to_pylist()))
    cset = set(pairs["cand"])
    rc = pd.concat([grab("train_s2", {c for c in cset if c.startswith("S2")}),
                    grab("train_s3", {c for c in cset if c.startswith("S3")})], ignore_index=True)
    d = pairs.merge(r1.rename(columns={"entity_id": "s1", "business_name": "name1", "business_address": "addr1"}), on="s1")
    d = d.merge(rc.rename(columns={"entity_id": "cand", "business_name": "name2", "business_address": "addr2",
                                   "country": "country2"}), on="cand")
    src = d["cand"].str[:2]
    sizes = pd.DataFrame({"s1": d["s1"], "n2": (src == "S2").astype(int), "n3": (src == "S3").astype(int)})
    sizes = sizes.groupby("s1")[["n2", "n3"]].sum().reindex(r1["entity_id"], fill_value=0)
    return d[["s1", "cand", "name1", "addr1", "name2", "addr2", "country"]], sizes


def s1_neighbours(texts: Sequence[str], seeds: np.ndarray, k: int, dim: int = 64, seed: int = 0) -> np.ndarray:
    """k nearest S1 (char TF-IDF -> SVD, inner product) of each seed row. Returns (len(seeds), k) indices."""
    from .blocking import LSAEncoder, knn

    enc = LSAEncoder(dim=dim, ngram=(2, 4), max_features=200000, seed=seed).fit(list(texts))
    v = enc.transform(list(texts))
    _, I = knn(v, v[seeds], k + 1, "flat")
    return I[:, 1:]


def build_synth_split(cfg: dict, countries: Optional[List[str]] = None, channel: Optional[NoiseChannel] = None) -> dict:
    """Synthetic labelled split for the unseen countries -> work_dir/<synth_split>/{records.parquet, gt.json}."""
    from . import prepare as P
    from . import translit
    from .pipeline import add_key_columns, text_views

    a = acfg(cfg)
    t0 = time.time()
    cache = cfg["cache_dir"]
    countries = countries or (a["countries"] if a["countries"] != "auto" else unseen_from_cache(cache))
    log.info("adapt: unseen countries %s", countries)
    rng = np.random.default_rng(a["seed"])
    if channel is None:
        cp = _art(cfg, "adapt_channel.json")
        if os.path.exists(cp):
            channel = NoiseChannel.load(cp)
        else:
            pairs, sizes = load_gold_pairs(cache, a["channel_s1"], a["seed"])
            channel = NoiseChannel().fit(pairs, gt_sizes=sizes, seed=a["seed"])
            channel.save(cp)
            del pairs
    dm_path = _art(cfg, "adapt_distractors.json")
    if os.path.exists(dm_path):
        with open(dm_path) as f:
            dmod = DistractorModel.from_json(json.load(f))
    elif os.path.exists(os.path.join(cfg["work_dir"], "train", "records.parquet")):
        rec_tr = pd.read_parquet(os.path.join(cfg["work_dir"], "train", "records.parquet"),
                                 columns=["entity_id", "src", "name_core", "addr_norm", "addr_street", "addr_house", "addr_locality"])
        with open(os.path.join(cfg["work_dir"], "train", "gt.json")) as f:
            gt_tr = json.load(f)
        dmod = DistractorModel().fit(rec_tr, gt_tr, seed=a["seed"])
        del rec_tr, gt_tr
        with open(dm_path, "w") as f:
            json.dump(dmod.to_json(), f)
    else:
        dmod = DistractorModel.default()
    tpath = _art(cfg, "translit.json")
    tdict = translit.load(tpath) if os.path.exists(tpath) else {"tokens": {}, "segments": {}, "variants": {}}
    s1_rows, c2_rows, c3_rows, gt = [], [], [], {}
    meta = {"countries": countries, "per_country": {}}
    for country in countries:
        s1 = read_country_rows(cache, "test", 1, [country])
        cand = pd.concat([read_country_rows(cache, "test", k, [country], a["locale_cands"] // 2, a["seed"]) for k in (2, 3)],
                         ignore_index=True)
        lp = _art(cfg, f"adapt_locale_{re.sub(r'[^a-z0-9]+', '_', ckey(country))}.json")
        loc = Locale.fit(s1, cand, country)
        loc.save(lp)
        del cand
        syn = Synthesizer(channel, loc, seed=a["seed"] + zlib.crc32(country.encode()) % 1000)
        # seeds + S1-S1 neighbours (guaranteed non-matches: S1 is deduplicated)
        s1 = s1.reset_index(drop=True)
        nrm = pd.DataFrame([N.normalize_record(n, ad, c) for n, ad, c in zip(s1.business_name, s1.business_address, s1.country)])
        views = text_views(nrm)
        n_seed = min(a["n_seeds"], len(s1))
        seeds = np.sort(rng.choice(len(s1), n_seed, replace=False))
        keep = set(seeds.tolist())
        if a["k_neighbours"] > 0 and len(s1) > a["k_neighbours"] + 1:
            for view in ("name", "both"):
                I = s1_neighbours(views[view], seeds, a["k_neighbours"], seed=a["seed"])
                keep |= set(I.ravel().tolist())
        sel = s1.iloc[sorted(keep)].reset_index(drop=True)
        del nrm, views
        others = list(zip(sel.business_address.tolist(), [""] * len(sel)))
        vocab = [t for t, _ in loc.d.get("s1_vocab", []) if t.isalpha() and len(t) >= 3]
        neg_vocab = loc.d.get("neg_vocab") or []
        n_pos = n_neg = 0
        rate = dmod.d.get("rate_per_s1", 0.9) * a["distractor_scale"]
        for r in sel.itertuples(index=False):
            sid = AUG_PREFIX_SYNTH + r.entity_id
            s1_rows.append({"entity_id": sid, "business_name": r.business_name, "business_address": r.business_address, "country": r.country})
            k2, k3 = channel.sample_cluster(rng)
            ids = []
            for src, kk, rows in (("2", k2, c2_rows), ("3", k3, c3_rows)):
                for _ in range(kk):
                    nm, ad, _, _ = syn.synth_record(r.business_name, r.business_address, src)
                    cid = f"{AUG_PREFIX_SYNTH}S{src}-{len(c2_rows) + len(c3_rows):09d}"
                    rows.append({"entity_id": cid, "business_name": nm, "business_address": ad, "country": r.country})
                    ids.append(cid)
            gt[sid] = ids
            n_pos += len(ids)
            nd = int(rng.poisson(rate))
            for _ in range(nd):
                nm, ad = dmod.make(syn, r.business_name, r.business_address, others, vocab, neg_vocab)
                src = "2" if rng.random() < 0.5 else "3"
                rows = c2_rows if src == "2" else c3_rows
                cid = f"{AUG_PREFIX_SYNTH}S{src}-{len(c2_rows) + len(c3_rows):09d}"
                rows.append({"entity_id": cid, "business_name": nm, "business_address": ad, "country": r.country})
                n_neg += 1
        meta["per_country"][country] = {"s1_total": int(len(s1)), "seeds": int(n_seed), "s1_in_split": int(len(sel)),
                                        "synthetic_positives": n_pos, "distractors": n_neg, "locale": loc.report()}
        log.info("adapt %s: %d S1 (%d seeds), %d synthetic positives, %d distractors", country, len(sel), n_seed, n_pos, n_neg)
    src = {1: pd.DataFrame(s1_rows), 2: pd.DataFrame(c2_rows), 3: pd.DataFrame(c3_rows)}
    for k in (2, 3):
        if src[k].empty:
            src[k] = pd.DataFrame(columns=["entity_id", "business_name", "business_address", "country"])
    rec = P.build_records(src, tdict, cfg.get("n_jobs", 1))
    rec = add_key_columns(rec)
    split = a["synth_split"]
    rec.to_parquet(_wd(cfg, split, "records.parquet"))
    with open(_wd(cfg, split, "gt.json"), "w") as f:
        json.dump(gt, f)
    meta["channel_stats"] = {k: v for k, v in channel.stats.items() if k != "addr_op_rate"}
    meta["distractor_model"] = {k: v for k, v in dmod.d.items() if k not in ("pct", "house_delta")}
    with open(_wd(cfg, split, "adapt_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=1, ensure_ascii=False)
    log.info("adapt: synthetic split %s written (%d records) in %.0fs", split, len(rec), time.time() - t0)
    return meta


def label_split(cfg: dict, split: str) -> None:
    """Add y (from the split's gt.json) and fold (features.fold_of) to work_dir/<split>/feats.parquet -> train schema."""
    from . import features as F
    from .pipeline import label_pairs

    fp = _wd(cfg, split, "feats.parquet")
    df = pd.read_parquet(fp)
    rec = pd.read_parquet(_wd(cfg, split, "records.parquet"), columns=["entity_id"])
    with open(_wd(cfg, split, "gt.json")) as f:
        gt = json.load(f)
    df["y"] = label_pairs(df, rec, gt)
    df["fold"] = F.fold_of(rec["entity_id"].to_numpy(), cfg.get("n_folds", 5))[df.i1.to_numpy()].astype(np.int8)
    df.to_parquet(fp)
    log.info("label %s: %d rows, %d positives", split, len(df), int(df.y.sum()))


def run_synth(cfg: dict) -> dict:
    """build_synth_split + the pipeline's own block and features steps + labels."""
    from . import pipeline as PL

    meta = build_synth_split(cfg)
    split = acfg(cfg)["synth_split"]
    PL.step_block(cfg, split)
    PL.step_features(cfg, split)
    label_split(cfg, split)
    return meta


def pseudo_label_split(cfg: dict) -> dict:
    """Pseudo-labelled rows of the unseen countries -> work_dir/<pseudo_split>/{feats,records}.parquet + gt.json."""
    from . import features as F

    a = acfg(cfg)
    sp = os.path.join(cfg["work_dir"], "test", "scores_gbdt.parquet")
    if not os.path.exists(sp):
        raise FileNotFoundError(f"{sp} missing: run `pipeline predict --split test` first")
    rec = pd.read_parquet(os.path.join(cfg["work_dir"], "test", "records.parquet"))
    countries = a["countries"] if a["countries"] != "auto" else unseen_from_cache(cfg["cache_dir"])
    s1m = (rec["src"] == 1).to_numpy()
    s1_country = dict(zip(rec["idx"].to_numpy()[s1m].tolist(), rec["country_norm"].to_numpy()[s1m].tolist()))
    scores = pd.read_parquet(sp, columns=["i1", "i2", "p"])
    sel = select_pseudo_labels(scores, s1_country, countries, a["pseudo_p_pos"], a["pseudo_margin"], a["pseudo_p_neg"])
    s1_sel = sel["i1"].unique()
    if len(s1_sel) > a["pseudo_max_s1"]:
        s1_sel = np.random.default_rng(a["seed"]).choice(s1_sel, a["pseudo_max_s1"], replace=False)
        sel = sel[sel["i1"].isin(set(s1_sel.tolist()))]
    feats = pd.read_parquet(os.path.join(cfg["work_dir"], "test", "feats.parquet"))
    feats = feats.merge(sel[["i1", "i2", "y"]], on=["i1", "i2"], how="inner")
    used = np.unique(np.r_[feats.i1.to_numpy(), feats.i2.to_numpy()])
    sub = rec.iloc[used].copy()
    remap = pd.Series(np.arange(len(sub)), index=used)
    sub["idx"] = np.arange(len(sub), dtype=np.int64)
    sub["entity_id"] = AUG_PREFIX_PSEUDO + sub["entity_id"].astype(str)
    feats["i1"] = remap.loc[feats.i1.to_numpy()].to_numpy()
    feats["i2"] = remap.loc[feats.i2.to_numpy()].to_numpy()
    feats["s1_id"] = AUG_PREFIX_PSEUDO + feats["s1_id"].astype(str)
    feats["cand_id"] = AUG_PREFIX_PSEUDO + feats["cand_id"].astype(str)
    feats["s1_hash"] = np.fromiter((zlib.crc32(e.encode()) % 1000 for e in feats["s1_id"]), dtype=np.int16, count=len(feats))
    feats["fold"] = F.fold_of(feats["s1_id"].to_numpy(), cfg.get("n_folds", 5)).astype(np.int8)
    split = a["pseudo_split"]
    sub.reset_index(drop=True).to_parquet(_wd(cfg, split, "records.parquet"))
    feats.to_parquet(_wd(cfg, split, "feats.parquet"))
    gt = feats[feats.y == 1].groupby("s1_id")["cand_id"].agg(list).to_dict()
    for s in feats["s1_id"].unique():
        gt.setdefault(s, [])
    with open(_wd(cfg, split, "gt.json"), "w") as f:
        json.dump(gt, f)
    rep = {"pairs": int(len(feats)), "positives": int(feats.y.sum()), "s1": int(feats.s1_id.nunique())}
    log.info("pseudo labels: %s", rep)
    return rep


def _aug_splits(cfg) -> List[str]:
    a = acfg(cfg)
    out = []
    for sp in (a["synth_split"], a["pseudo_split"]):
        if os.path.exists(os.path.join(cfg["work_dir"], sp, "feats.parquet")) and (sp != a["pseudo_split"] or a["pseudo"]):
            out.append(sp)
    return out


def augment_train(cfg: dict, df: pd.DataFrame, rec: pd.DataFrame, gt: Dict[str, List[str]]):
    """In-process hook: append the synthetic (+ pseudo) rows to the train frames (i1 / i2 / idx offset, same columns)."""
    a = acfg(cfg)
    dfs, recs = [df], [rec]
    off = len(rec)
    gt = dict(gt)
    for sp in _aug_splits(cfg):
        f = pd.read_parquet(os.path.join(cfg["work_dir"], sp, "feats.parquet"))
        r = pd.read_parquet(os.path.join(cfg["work_dir"], sp, "records.parquet"))
        f["i1"] += off; f["i2"] += off
        r = r.assign(idx=r["idx"] + off)
        f = f.reindex(columns=df.columns)
        for c in df.columns:
            if c in f and df[c].dtype != f[c].dtype and df[c].dtype.kind in "fiub":
                f[c] = f[c].astype(df[c].dtype) if f[c].notna().all() or df[c].dtype.kind == "f" else f[c]
        dfs.append(f); recs.append(r.reindex(columns=rec.columns))
        if a["tune_on_aug"]:
            with open(os.path.join(cfg["work_dir"], sp, "gt.json")) as fh:
                gt.update(json.load(fh))
        off += len(r)
        log.info("augment_train: +%d rows (%d pos) from %s", len(f), int(f.y.sum()), sp)
    return pd.concat(dfs, ignore_index=True), pd.concat(recs, ignore_index=True), gt


def materialize(cfg: dict) -> str:
    """work_dir/<aug_root>/train/{feats,records}.parquet + gt.json = train + synthetic (+ pseudo), streamed row group by
    row group (never holds the full train feats in memory)."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    a = acfg(cfg)
    root = os.path.join(cfg["work_dir"], a["aug_root"])
    out_dir = os.path.join(root, "train")
    os.makedirs(out_dir, exist_ok=True)
    tr_f = os.path.join(cfg["work_dir"], "train", "feats.parquet")
    schema = pq.read_schema(tr_f)
    rec_tr = pd.read_parquet(os.path.join(cfg["work_dir"], "train", "records.parquet"))
    off = len(rec_tr)
    writer = pq.ParquetWriter(os.path.join(out_dir, "feats.parquet"), schema)
    pf = pq.ParquetFile(tr_f)
    for g in range(pf.num_row_groups):
        writer.write_table(pf.read_row_group(g))
    recs = [rec_tr]
    with open(os.path.join(cfg["work_dir"], "train", "gt.json")) as f:
        gt = json.load(f)
    rep = {"train_rows": int(pf.metadata.num_rows)}
    for sp in _aug_splits(cfg):
        f = pd.read_parquet(os.path.join(cfg["work_dir"], sp, "feats.parquet"))
        r = pd.read_parquet(os.path.join(cfg["work_dir"], sp, "records.parquet"))
        f["i1"] += off; f["i2"] += off
        r = r.assign(idx=r["idx"] + off)
        t = pa.Table.from_pandas(f.reindex(columns=schema.names), preserve_index=False)
        t = t.cast(schema, safe=False)
        writer.write_table(t)
        recs.append(r.reindex(columns=rec_tr.columns))
        if a["tune_on_aug"]:
            with open(os.path.join(cfg["work_dir"], sp, "gt.json")) as fh:
                gt.update(json.load(fh))
        rep[sp] = {"rows": int(len(f)), "pos": int(f.y.sum())}
        off += len(r)
    writer.close()
    pd.concat(recs, ignore_index=True).to_parquet(os.path.join(out_dir, "records.parquet"))
    with open(os.path.join(out_dir, "gt.json"), "w") as f:
        json.dump(gt, f)
    log.info("materialize: %s -> %s", rep, out_dir)
    return root


def recalibrate_real(cfg: dict, aug_root: str) -> dict:
    """Re-fit the isotonic calibrator and the decision layer on REAL train S1 rows only (synthetic rows shift priors)."""
    import pickle

    from sklearn.isotonic import IsotonicRegression

    from . import decision as Dm
    from . import metrics as M

    oof = pd.read_parquet(os.path.join(aug_root, "train", "oof.parquet"))
    real = ~oof["s1_id"].astype(str).str.startswith((AUG_PREFIX_SYNTH, AUG_PREFIX_PSEUDO))
    o = oof[real]
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(o["p_blend"], o["y"])
    cp = _art(cfg, "calibrator.pkl")
    with open(cp, "rb") as f:
        cal = pickle.load(f)
    cal["iso"] = iso
    with open(cp, "wb") as f:
        pickle.dump(cal, f)
    with open(os.path.join(cfg["work_dir"], "train", "gt.json")) as f:
        gt = json.load(f)
    pairs = o[["s1_id", "cand_id"]].rename(columns={"s1_id": "s1", "cand_id": "cand"}).assign(p=iso.predict(o["p_blend"]))
    best = Dm.tune_decision(pairs, gt, list(gt), "p")
    with open(_art(cfg, "decision.json"), "w") as f:
        json.dump(best, f, indent=1)
    rep = M.report(Dm.decide(pairs, best, "p"), gt, list(gt))
    rep["decision"] = best
    log.info("recalibrated on real S1: %s", rep)
    return rep


def train_aug(cfg: dict) -> dict:
    from .models import gbdt

    a = acfg(cfg)
    root = os.path.join(cfg["work_dir"], a["aug_root"])
    res = gbdt.train_stack({**cfg, "work_dir": root})
    if a["recalibrate_real"]:
        res = recalibrate_real(cfg, root)
    return res


# ================================================================================================================
# 6. validation: does the synthetic channel reproduce train marginals? (leave-country-out, e.g. US -> India)
# ================================================================================================================
def pair_stats(rows: pd.DataFrame) -> pd.DataFrame:
    """Per pair statistics on the pipeline's normalised views (name1/addr1/name2/addr2/country)."""
    out = []
    for n1, a1, n2, a2, c in zip(rows.name1, rows.addr1, rows.name2, rows.addr2, rows.country):
        r1 = N.normalize_record(n1, a1, c); r2 = N.normalize_record(n2, a2, c)
        c1 = r1["name_core"] or r1["name_norm"]; c2 = r2["name_core"] or r2["name_norm"]
        ad = detect_addr(a1, a2, c)[0]
        nsig = detect_name(n1, n2)[0]
        out.append({
            "nm_tset": fuzz.token_set_ratio(c1, c2) / 100.0 if c1 and c2 else np.nan,
            "nm_ratio": fuzz.ratio(c1, c2) / 100.0 if c1 and c2 else np.nan,
            "nm_tsort": fuzz.token_sort_ratio(c1, c2) / 100.0 if c1 and c2 else np.nan,
            "nm_core_eq": float(bool(c1) and c1 == c2),
            "nm_legal_eq": float(r1["name_legal"] == r2["name_legal"]),
            "ad_tset": fuzz.token_set_ratio(r1["addr_norm"], r2["addr_norm"]) / 100.0 if r1["addr_norm"] and r2["addr_norm"] else np.nan,
            "ad_house_eq": float(r1["addr_house"] == r2["addr_house"]) if r1["addr_house"] and r2["addr_house"] else np.nan,
            "ad_empty": float(r2["addr_empty"]),
            "ad_reorder": float(bool(ad.get("ro"))),
            "ad_loc_tset": fuzz.token_set_ratio(r1["addr_locality"], r2["addr_locality"]) / 100.0 if r1["addr_locality"] and r2["addr_locality"] else np.nan,
            "nm_upper": float(nsig.get("case") == "U"),
            "nm_dba": float("dba" in nsig), "nm_domain": float("dom" in nsig or "hdl" in nsig),
            "nm_replaced": float("rep" in nsig), "nm_shuffle": float("shuf" in nsig or "lf" in nsig),
            "nm_is_domain": float(r2["name_is_domain"]), "nm_has_dba": float(r2["name_has_dba"]),
        })
    return pd.DataFrame(out)


def ks_stat(a: np.ndarray, b: np.ndarray) -> float:
    a = np.sort(a[~np.isnan(a)]); b = np.sort(b[~np.isnan(b)])
    if not len(a) or not len(b):
        return float("nan")
    grid = np.unique(np.r_[a, b])
    return float(np.max(np.abs(np.searchsorted(a, grid, "right") / len(a) - np.searchsorted(b, grid, "right") / len(b))))


def compare_stats(real: pd.DataFrame, synth: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for c in real.columns:
        x, y = real[c].to_numpy(np.float64), synth[c].to_numpy(np.float64)
        rows.append({"stat": c, "real_mean": np.nanmean(x), "synth_mean": np.nanmean(y), "ks": ks_stat(x, y),
                     "real_p10": np.nanquantile(x, 0.1) if np.isfinite(x).any() else np.nan,
                     "synth_p10": np.nanquantile(y, 0.1) if np.isfinite(y).any() else np.nan})
    return pd.DataFrame(rows).set_index("stat")


def validate_lco(cache_dir: str, out_dir: str, src_country: str = "US", tgt_country: str = "India", n_s1: int = 12000,
                 n_eval: int = 4000, locale_s1: int = 120000, locale_cands: int = 200000, seed: int = 0) -> dict:
    """Leave-country-out: fit the channel on src_country gold pairs and the locale label-free on tgt_country
    records, synthesise tgt_country S2/S3 records for held-out tgt S1 and compare with their REAL gold matches."""
    t0 = time.time()
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(seed)
    pairs_src, sizes_src = load_gold_pairs(cache_dir, n_s1, seed, [src_country])
    pairs_tgt, sizes_tgt = load_gold_pairs(cache_dir, n_s1, seed + 1, [tgt_country])
    log.info("validate: %d %s pairs, %d %s pairs (%.0fs)", len(pairs_src), src_country, len(pairs_tgt), tgt_country, time.time() - t0)
    tgt_ids = pairs_tgt["s1"].unique()
    rng.shuffle(tgt_ids)
    eval_ids = set(tgt_ids[:n_eval]); fit_ids = set(tgt_ids[n_eval:])
    ch_src = NoiseChannel().fit(pairs_src, gt_sizes=sizes_src, seed=seed)
    ch_tgt = NoiseChannel().fit(pairs_tgt[pairs_tgt.s1.isin(fit_ids)], seed=seed)  # in-country upper bound
    # label-free locale of the target country (S1 + unlabelled S2/S3, incl. distractors)
    s1_t = read_country_rows(cache_dir, "train", 1, [tgt_country], locale_s1, seed)
    cand_t = pd.concat([read_country_rows(cache_dir, "train", k, [tgt_country], locale_cands // 2, seed) for k in (2, 3)],
                       ignore_index=True)
    cand_t = cand_t[~cand_t.business_name.str.contains("[\u0900-\u0d7f]", regex=True)]
    loc = Locale.fit(s1_t, cand_t, tgt_country)
    loc.save(os.path.join(out_dir, f"locale_{ckey(tgt_country)}.json"))
    del s1_t, cand_t
    real = pairs_tgt[pairs_tgt.s1.isin(eval_ids)].reset_index(drop=True)
    real_latin = real[~real.name2.str.contains("[\u0900-\u0d7f]", regex=True) &
                      ~real.addr2.str.contains("[\u0900-\u0d7f]", regex=True)].reset_index(drop=True)

    def synth(ch, locale, rates=True):
        syn = Synthesizer(ch, locale, seed=seed, locale_rates=rates)
        rows = []
        for r in real.itertuples(index=False):
            src = "3" if r.cand.startswith("S3") else "2"
            nm, ad, _, _ = syn.synth_record(r.name1, r.addr1, src)
            rows.append({"name1": r.name1, "addr1": r.addr1, "name2": nm, "addr2": ad, "country": r.country})
        return pd.DataFrame(rows)

    variants = {
        f"{src_country}->{tgt_country} channel + {tgt_country} locale": synth(ch_src, loc, True),
        f"{src_country}->{tgt_country} channel, no locale": synth(ch_src, Locale(""), False),
        f"{tgt_country}->{tgt_country} channel (upper bound)": synth(ch_tgt, loc, False),
    }
    src_real = pairs_src.sample(min(len(pairs_src), len(real)), random_state=seed).reset_index(drop=True)
    st_real = pair_stats(real_latin)
    res = {"n_real_pairs": int(len(real)), "n_real_latin": int(len(real_latin)), "variants": {}}
    tables = {}
    for name, sy in variants.items():
        st = pair_stats(sy)
        tab = compare_stats(st_real, st)
        tables[name] = tab
        res["variants"][name] = {"mean_ks": round(float(tab["ks"].mean()), 4), "max_ks": round(float(tab["ks"].max()), 4)}
    tab = compare_stats(st_real, pair_stats(src_real))
    tables[f"real {src_country} pairs (no adaptation)"] = tab
    res["variants"][f"real {src_country} pairs (no adaptation)"] = {"mean_ks": round(float(tab["ks"].mean()), 4),
                                                                   "max_ks": round(float(tab["ks"].max()), 4)}
    for name, tab in tables.items():
        tab.to_csv(os.path.join(out_dir, "lco_" + re.sub(r"[^A-Za-z0-9]+", "_", name) + ".csv"))
    res["channel_src_ops"] = {"name": dict(list(ch_src.stats["name_op_rate"].items())[:20]),
                              "addr": dict(list(ch_src.stats["addr_op_rate"].items())[:20])}
    res["channel_tgt_ops"] = {"name": dict(list(ch_tgt.stats["name_op_rate"].items())[:20]),
                              "addr": dict(list(ch_tgt.stats["addr_op_rate"].items())[:20])}
    res["locale"] = loc.report()
    res["seconds"] = round(time.time() - t0, 1)
    with open(os.path.join(out_dir, "lco_report.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1, ensure_ascii=False, default=float)
    return {"report": res, "tables": tables}


# ================================================================================================================
# CLI
# ================================================================================================================
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["synth", "build", "label", "pseudo", "materialize", "train", "validate", "unseen"])
    ap.add_argument("--config")
    ap.add_argument("--pq", default=None, help="parquet cache dir (validate / unseen)")
    ap.add_argument("--out", default=None, help="output dir (validate)")
    ap.add_argument("--src", default="US")
    ap.add_argument("--tgt", default="India")
    ap.add_argument("--n_s1", type=int, default=12000)
    ap.add_argument("--n_eval", type=int, default=4000)
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    if a.step == "validate":
        r = validate_lco(a.pq, a.out, a.src, a.tgt, a.n_s1, a.n_eval)
        print(json.dumps(r["report"]["variants"], indent=1))
        return
    if a.step == "unseen":
        print(unseen_from_cache(a.pq))
        return
    from .pipeline import load_cfg

    cfg = load_cfg(a.config)
    {"synth": lambda: run_synth(cfg), "build": lambda: build_synth_split(cfg),
     "label": lambda: label_split(cfg, acfg(cfg)["synth_split"]), "pseudo": lambda: pseudo_label_split(cfg),
     "materialize": lambda: materialize(cfg), "train": lambda: train_aug(cfg)}[a.step]()


if __name__ == "__main__":
    main()
