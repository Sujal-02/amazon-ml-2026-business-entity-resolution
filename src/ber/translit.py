"""Learn an Indic-script -> Latin dictionary from the labelled training pairs (no external data).

Facts established on the training data:
  * ~550k matched pairs have a native-script S2/S3 name whose S1 counterpart is Latin; the native name is a word-for-word
    phonetic transliteration (token counts differ in only ~10 pairs), so positional alignment is almost exact.
    A similarity vote (consonant skeleton of the rule romanisation vs S1 token) backs it up for shuffled names.
  * The generator picks Latin spelling variants ~50/50 (laxmi/lakshmi, jay/jai, shree/sree, limited/ltd): those are
    learned as a Latin "variant canon" (applied to BOTH sides) instead of trusting the argmax only.
  * Native address text is mostly the state as ONE comma segment ("महाराष्ट्र", "ఆంధ్రప్రదేశ్"): learned at segment level.
  * Test-only native tokens (~3.6%, mainly category words such as "मोटर्स", "स्टोर्स") are resolved against the closed
    Latin vocabulary of the same split (transliteration mining, xscript.VocabResolver). They matter for precision:
    category words inserted into an existing S1 name are how the generator builds hard-negative distractors.
"""
from __future__ import annotations

import collections
import json
import re
import unicodedata
from typing import Dict, Iterable, List, Tuple

from rapidfuzz.distance import Levenshtein

from . import xscript
from .normalize import INDIC_RE, clean_native, roman_fallback, skeleton

_TOKEN_RE = re.compile("[ऀ-ൿ​‌‍⁠]+|[A-Za-zÀ-ɏ0-9]+")


def _latin_tokens(s: str) -> List[str]:
    s = xscript.fold_latin(unicodedata.normalize("NFKC", s))
    return [t for t in re.split(r"[^a-z0-9]+", s) if t and not t.isdigit()]


def _mixed_tokens(s: str) -> List[Tuple[str, bool]]:
    s = unicodedata.normalize("NFKC", s)
    out = []
    for m in _TOKEN_RE.finditer(s):
        t = m.group(0)
        if INDIC_RE.search(t):
            out.append((clean_native(t), True))
        else:
            out.append((xscript.fold_latin(t), False))
    return out


def _sim(native_roman: str, latin: str) -> float:
    a, b = skeleton(native_roman), skeleton(latin)
    if not a or not b:
        return 0.0
    return Levenshtein.normalized_similarity(a, b)


def collect_token_votes(pairs: Iterable[Tuple[str, str]], min_sim: float = 0.5) -> Dict[str, collections.Counter]:
    """pairs: (latin_text_from_S1, candidate_text_possibly_native)."""
    votes: Dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    for latin, cand in pairs:
        if not INDIC_RE.search(cand):
            continue
        L = _latin_tokens(latin)
        if not L:
            continue
        C = [t for t in _mixed_tokens(cand) if not t[0].isdigit()]
        positional = len(C) == len(L)
        for i, (t, is_n) in enumerate(C):
            if not is_n:
                continue
            if positional:
                votes[t][L[i]] += 2
            r = roman_fallback(t)
            best, bs = None, 0.0
            for l in L:
                s = _sim(r, l)
                if s > bs:
                    best, bs = l, s
            if best is not None and bs >= min_sim:
                votes[t][best] += 1
    return votes


def collect_segment_votes(pairs: Iterable[Tuple[str, str]], min_sim: float = 0.4) -> Dict[str, collections.Counter]:
    """Address: fully-native comma segments of the candidate -> most similar S1 segment."""
    votes: Dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    for latin, cand in pairs:
        if not INDIC_RE.search(cand):
            continue
        lsegs = [xscript.fold_latin(x.strip()) for x in latin.split(",") if x.strip()]
        lsegs = [re.sub(r"[^a-z0-9 ]+", " ", x).strip() for x in lsegs]
        lsegs = [x for x in lsegs if x and not any(c.isdigit() for c in x)]
        if not lsegs:
            continue
        for seg in cand.split(","):
            key = clean_native(seg.strip())
            if not key or not INDIC_RE.search(key) or re.search(r"[A-Za-z0-9]", key):
                continue
            r = "".join(roman_fallback(t) for t in key.split())
            best, bs = None, 0.0
            for l in lsegs:
                s = _sim(r, l.replace(" ", ""))
                if s > bs:
                    best, bs = l, s
            if best is not None and bs >= min_sim:
                votes[key][best] += 1
    return votes


def finalize(votes: Dict[str, collections.Counter], min_count: int = 2, min_share: float = 0.4) -> Dict[str, str]:
    out = {}
    for t, c in votes.items():
        (best, n), tot = c.most_common(1)[0], sum(c.values())
        if n >= min_count and n / tot >= min_share:
            out[t] = best
    return out


def latin_variants(votes: Dict[str, collections.Counter], min_share: float = 0.2, min_count: int = 20) -> Dict[str, str]:
    """Latin spellings that the same native token maps to with substantial shares are variants of one word."""
    parent: Dict[str, str] = {}

    def find(x):
        while parent.get(x, x) != x:
            x = parent[x]
        return x

    weight = collections.Counter()
    for t, c in votes.items():
        tot = sum(c.values())
        if tot < min_count:
            continue
        big = [w for w, n in c.items() if n / tot >= min_share]
        for w in big:
            weight[w] += c[w]
        for w in big[1:]:
            a, b = find(big[0]), find(w)
            if a != b:
                parent[b] = a
    groups = collections.defaultdict(list)
    for w in list(parent) + [p for p in parent.values()]:
        groups[find(w)].append(w)
    out = {}
    for root, members in groups.items():
        members = set(members)
        canon = max(members, key=lambda w: (weight[w], -len(w)))
        for w in members:
            # only true spelling variants (laxmi/lakshmi, jai/jay): guards against alignment accidents (south->private)
            if w != canon and Levenshtein.normalized_similarity(skeleton(w), skeleton(canon)) >= 0.6 \
                    and w not in LEGAL_WORDS and canon not in LEGAL_WORDS and len(w) > 2 and len(canon) > 2:
                out[w] = canon
    return out


LEGAL_WORDS = {"private", "pvt", "limited", "ltd", "llp", "llc", "inc", "corp", "co", "company"}


def learn(s1_name: Dict[str, str], s1_addr: Dict[str, str], cand_name: Dict[str, str], cand_addr: Dict[str, str],
          pairs: List[Tuple[str, str]]) -> dict:
    """Returns {"tokens": {...}, "segments": {...}, "variants": {...}} learned from positive (s1_id, cand_id) pairs."""
    nm = [(s1_name[a], cand_name[b]) for a, b in pairs if b in cand_name and a in s1_name]
    ad = [(s1_addr[a], cand_addr[b]) for a, b in pairs if b in cand_addr and a in s1_addr]
    tv = collect_token_votes(nm)
    av = collect_token_votes(ad)
    sv = collect_segment_votes(ad)
    tokens = finalize(av)
    tokens.update(finalize(tv))
    return {"tokens": tokens, "segments": finalize(sv), "variants": latin_variants(tv)}


def resolve_oov(native_texts: Iterable[str], known: Dict[str, str], latin_vocab: collections.Counter,
                min_vocab_count: int = 3) -> Dict[str, str]:
    """Map native tokens not in `known` to the closed Latin vocabulary of the split (xscript.VocabResolver)."""
    vocab = collections.Counter({w: c for w, c in latin_vocab.items() if c >= min_vocab_count and len(w) > 1})
    res = xscript.VocabResolver(vocab)
    out = {}
    seen = set()
    for s in native_texts:
        for m in re.finditer("[ऀ-ൿ​‌‍⁠]+", s):
            t = clean_native(m.group(0))
            if t in known or t in seen:
                continue
            seen.add(t)
            cand = res.resolve(t, k=1)
            if cand and _confident(t, cand[0]):
                out[t] = cand[0]
            # otherwise keep the rule romanisation: an unknown extra token is SAFER than a wrong common word
            # (e.g. 'एजेंसीज' must not become 'sons' -- that would turn a category-inserted distractor into a match)
    return out


def _confident(native_tok: str, latin: str) -> bool:
    r = roman_fallback(native_tok)
    if not r or min(len(r), len(latin)) / max(len(r), len(latin)) < 0.6:
        return False
    fk = xscript.skeleton(r, latin=False)
    if fk and fk == xscript.skeleton(latin):
        return True
    lev = Levenshtein.normalized_similarity(r, latin)
    ck = xscript.skeleton(r, coarse=True, latin=False)
    if ck and ck == xscript.skeleton(latin, coarse=True) and len(ck) >= 4 and lev >= 0.5:
        return True
    return lev >= 0.75


def latin_vocab(texts: Iterable[str]) -> collections.Counter:
    c = collections.Counter()
    for s in texts:
        if INDIC_RE.search(s):
            continue
        c.update(_latin_tokens(s))
    return c


def save(d: dict, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(d, f, ensure_ascii=False)


def load(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def install(d: dict) -> None:
    """Activate a learned dictionary in the normaliser (tokens + segments share one lookup table)."""
    from . import normalize

    table = dict(d.get("tokens", {}))
    table.update(d.get("segments", {}))
    table.update(d.get("oov", {}))
    normalize.set_translit_dict(table, d.get("variants", {}))


def coverage(texts: Iterable[str], d: Dict[str, str]) -> float:
    hit = tot = 0
    for s in texts:
        for m in re.finditer("[ऀ-ൿ​‌‍⁠]+", s):
            tot += 1
            hit += clean_native(m.group(0)) in d
    return hit / max(1, tot)
