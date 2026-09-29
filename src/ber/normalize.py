"""Text normalisation for business names and addresses (country-agnostic, open-set countries).

Design notes (from EDA of the training data):
* S1 is always Latin script and clean; S2/S3 carry the noise.
* ~9% of S2 / 5% of S3 names are in one of 9 Indic scripts and are phonetic transliterations of the English name
  ("ब्लू लॉजिस्टिक्स प्राइवेट लिमिटेड" == "Blue Logistics Private Limited"). We romanise them with a token dictionary
  LEARNED FROM THE TRAINING PAIRS (see translit.py) and fall back to a rule-based romaniser.
* Name noise: legal-form variants, token shuffles, leetspeak (0->o, 1->l), junk prefixes (***, --, <<, #, @),
  honorifics (M/s, Mr, Shri), phone-number suffixes, "| www.x.com", domain/handle forms, DBA/t/a aliases with
  random fake names, bracketed legal forms, added generic words (Center, Services, Group...).
* Address noise: component reordering, abbreviations (St/Street, R./Rue), state as code/full/native script,
  zero-padded or suffixed house numbers (013623, 905-), ordinal typos (24nd), null/N/A/<NULL>, PO Box / PMB,
  "Door No"/"H.NO"/"#" prefixes.
All maps below are generic linguistic normalisation rules (no external data lookup).
"""
from __future__ import annotations

import re
import unicodedata
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

from . import xscript

# ----------------------------------------------------------------------------------------------------------------
# Script handling (stdlib only: no GPL transliteration libraries)
# ----------------------------------------------------------------------------------------------------------------
INDIC_RE = re.compile("[ऀ-ൿ]")
# Indic run incl. zero-width joiners that sit inside Telugu/Kannada tokens (stripped before lookup)
INDIC_TOKEN_RE = re.compile("[ऀ-ൿ​‌‍⁠]+")
NON_ASCII_RE = re.compile(r"[^\x00-\x7f]")
_ZW_TABLE = dict.fromkeys([0x200B, 0x200C, 0x200D, 0x2060, 0xFEFF], None)

_TRANSLIT_DICT: Dict[str, str] = {}
_LATIN_VARIANTS: Dict[str, str] = {}


def set_translit_dict(d: Dict[str, str], variants: Optional[Dict[str, str]] = None) -> None:
    """Install a learned native-token -> latin dictionary and latin spelling-variant canon (see translit.py)."""
    global _TRANSLIT_DICT, _LATIN_VARIANTS
    _TRANSLIT_DICT = {clean_native(k): v for k, v in d.items()}
    if variants is not None:
        _LATIN_VARIANTS = dict(variants)
    romanize_token.cache_clear()


def clean_native(tok: str) -> str:
    return unicodedata.normalize("NFC", tok).translate(_ZW_TABLE)


_DOUBLE_RE = re.compile(r"([a-z])\1+")


@lru_cache(maxsize=1_000_000)
def roman_fallback(tok: str) -> str:
    """Rule-based romanisation (Inter-Indic offset romaniser, word-final schwa deletion)."""
    r = xscript.romanize(clean_native(tok))
    return re.sub(r"[^a-z0-9]", "", r.lower())


@lru_cache(maxsize=2_000_000)
def romanize_token(tok: str) -> str:
    t = clean_native(tok)
    hit = _TRANSLIT_DICT.get(t)
    if hit is not None:
        return hit
    return roman_fallback(t)


def romanize(s: str) -> str:
    """Replace every run of Indic characters with its romanisation (dictionary first, then rules)."""
    if not INDIC_RE.search(s):
        return s
    return INDIC_TOKEN_RE.sub(lambda m: " " + romanize_token(m.group(0)) + " ", s)


def fold_ascii(s: str) -> str:
    """NFKC -> romanise Indic runs -> fold Latin accents/ligatures (œ->oe, ß->ss) -> drop other non-ASCII."""
    s = unicodedata.normalize("NFKC", s)
    s = romanize(s)
    if NON_ASCII_RE.search(s):
        s = s.translate(xscript._LIG)
        s = unicodedata.normalize("NFKD", s)
        s = "".join(ch for ch in s if not unicodedata.combining(ch))
        s = NON_ASCII_RE.sub(" ", s)
    return s.lower()


# ----------------------------------------------------------------------------------------------------------------
# Names
# ----------------------------------------------------------------------------------------------------------------
# canonical legal forms (value) for surface variants (key), applied after punctuation removal
LEGAL_MAP = {
    "inc": "inc", "incorporated": "inc", "incorporation": "inc",
    "corp": "corp", "corporation": "corp", "corpn": "corp",
    "co": "co", "company": "co", "cos": "co", "compnay": "co",
    "ltd": "ltd", "limited": "ltd", "ltda": "ltd", "limted": "ltd",
    "pvt": "pvt", "private": "pvt", "pvtltd": "pvt ltd", "pte": "pvt",
    "llc": "llc", "llp": "llp", "lp": "lp", "plc": "plc", "pllc": "pllc", "pc": "pc", "pa": "pa", "lllp": "lllp",
    "opc": "opc", "huf": "huf",
    # French / European legal forms (generic, used for the unseen country)
    "sarl": "sarl", "sas": "sas", "sasu": "sasu", "sa": "sa", "sci": "sci", "eurl": "eurl", "snc": "snc",
    "ei": "ei", "eirl": "eirl", "scop": "scop", "scp": "scp", "selarl": "selarl", "sca": "sca", "scs": "scs",
    "gie": "gie", "gmbh": "gmbh", "ag": "ag", "bv": "bv", "nv": "nv", "srl": "srl", "spa": "spa",
}
LEGAL_TOKENS = set(LEGAL_MAP.values())
LEGAL_TOKENS |= {"pvt", "ltd"}
# honorifics / filler prefixes that the generator adds or drops
HONORIFICS = {"the", "ms", "m/s", "mr", "mrs", "shri", "sri", "smt", "messrs", "mssrs", "ets", "etablissements", "les", "le", "la"}
STOP_NAME = {"and", "of", "&", "de", "du", "des", "et", "l", "d"}
# generic business words the noise process appends ("Center", "Services"...). Kept in name_norm, removed in name_key.
GENERIC_WORDS = {"center", "centre", "services", "service", "group", "holdings", "partners", "solutions", "enterprises",
                 "enterprise", "international", "india", "global", "associates", "consultants", "company", "co",
                 "industries", "ventures", "trading", "agency", "france", "usa", "us",
                 "participations", "distribution", "development", "holdings", "group"}

DBA_RE = re.compile(r"\b(?:d\s*/\s*b\s*/\s*a|d\.b\.a\.?|dba|doing business as|t\s*/\s*a|trading as|a\s*/\s*k\s*/\s*a|aka|formerly known as|fka)\b\s*:?", re.I)
PHONE_RE = re.compile(r"(?<![\w/-])\+?\d[\d\s-]{8,}\d(?![\w/])")
URL_RE = re.compile(r"(?:https?://)?(?:www\.)?([a-z0-9][a-z0-9-]*)\.(?:com|net|org|in|co\.in|co|biz|info|fr|us|io|org\.in|net\.in)\b", re.I)
HANDLE_RE = re.compile(r"(?:(?<=\s)|^)[@#]([a-z0-9_]+)", re.I)
LEET_TRANS = str.maketrans({"0": "o", "1": "l", "3": "e", "4": "a", "5": "s", "7": "t", "@": "a", "$": "s"})
MIXED_ALNUM_RE = re.compile(r"\b(?=[a-z]*\d)(?=\d*[a-z])[a-z0-9]{3,}\b")
DOTS_IN_ABBR_RE = re.compile(r"(?<=\b[a-z])\.(?=[a-z]\b)")
NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")
SPACE_RE = re.compile(r"\s+")


def _deleet(tok: str) -> str:
    """Undo leetspeak inside alphabetic tokens (beac0n -> beacon, hea1th -> health). Pure numbers untouched."""
    digits = sum(c.isdigit() for c in tok)
    letters = len(tok) - digits
    if digits and letters >= 2 and digits <= 2:
        return tok.translate(LEET_TRANS)
    return tok


def _join_single_letters(tokens: List[str]) -> List[str]:
    """'l l c' -> 'llc', 'p v t' -> 'pvt' (initialisms split by punctuation)."""
    out, buf = [], []
    for t in tokens:
        if len(t) == 1 and t.isalpha():
            buf.append(t)
            continue
        if buf:
            out.append("".join(buf) if len(buf) > 1 else buf[0])
            buf = []
        out.append(t)
    if buf:
        out.append("".join(buf) if len(buf) > 1 else buf[0])
    return out


# cross-lingual canon for business-name words (generic linguistic rules): French generator/insertion words map onto
# the English words the train-learned tables (inserted-token harm, generic words) already know
NAME_WORD_CANON = {
    "groupe": "group", "groupes": "group", "holding": "holdings", "associes": "associates", "associe": "associates",
    "internationale": "international", "developpement": "development", "fils": "sons", "freres": "brothers",
    "cie": "co", "compagnie": "co", "societe": "co", "entreprise": "enterprise", "entreprises": "enterprises",
    "conseil": "consultants", "conseils": "consultants", "centre": "center",
}


def _tokens(s: str) -> List[str]:
    s = DOTS_IN_ABBR_RE.sub("", s)  # l.l.c -> llc, p.v.t -> pvt
    s = s.replace("&", " and ").replace("+", " and ")
    s = s.replace("'", "")  # orelee's -> orelees, don't split
    toks = [t for t in NON_ALNUM_RE.split(s) if t]
    toks = [_deleet(t) for t in toks]
    if _LATIN_VARIANTS:
        toks = [_LATIN_VARIANTS.get(t, t) for t in toks]
    toks = [NAME_WORD_CANON.get(t, t) for t in toks]
    return _join_single_letters(toks)


def _canon_legal(tokens: List[str]) -> List[str]:
    out = []
    for t in tokens:
        m = LEGAL_MAP.get(t)
        if m is None:
            out.append(t)
        else:
            out.extend(m.split())
    return out


def _dedupe_consecutive(tokens: List[str]) -> List[str]:
    return [t for i, t in enumerate(tokens) if i == 0 or t != tokens[i - 1]]


def norm_name(raw: str) -> Dict[str, object]:
    """Return a dict of name representations used by blocking and features."""
    s = fold_ascii(raw or "")
    had_indic = bool(INDIC_RE.search(raw or ""))
    s = PHONE_RE.sub(" ", s)
    s = s.replace("|", " , ")
    # domains / handles -> keep the stem, flag it
    is_domain = False
    m = URL_RE.search(s)
    if m:
        is_domain = True
        rest = URL_RE.sub(" ", s)
        rest_toks = [t for t in NON_ALNUM_RE.split(rest) if t and t not in HONORIFICS and t not in LEGAL_MAP and t not in STOP_NAME]
        if any(len(t) >= 2 for t in rest_toks):
            s = rest  # "NAME | www.name.com" -> the domain is a suffix, drop it ("Smt palholdings.com" keeps the stem)
        else:
            s = URL_RE.sub(lambda mm: " " + mm.group(1) + " ", s)  # whole name is a domain -> keep its stem
    if HANDLE_RE.search(s):
        is_domain = True
        s = HANDLE_RE.sub(lambda mm: " " + mm.group(1) + " ", s)
    parts = [p for p in DBA_RE.split(s) if p.strip()]
    has_dba = len(parts) > 1
    part_tokens = []
    for p in parts:
        toks = _dedupe_consecutive(_canon_legal(_tokens(p)))
        if toks:
            part_tokens.append(toks)
    if not part_tokens:
        part_tokens = [[]]
    all_toks = [t for pt in part_tokens for t in pt]

    def core_of(toks):
        return [t for t in toks if t not in LEGAL_TOKENS and t not in HONORIFICS and t not in STOP_NAME]

    core = core_of(all_toks)
    legal = sorted({t for t in all_toks if t in LEGAL_TOKENS})
    # the alias after DBA/t-a is usually the true name; keep each part's core for max-over-parts features
    alias_cores = [" ".join(core_of(pt)) for pt in part_tokens]
    key = [t for t in core if t not in GENERIC_WORDS] or core
    return {
        "name_norm": " ".join(all_toks),
        "name_core": " ".join(core),
        "name_key": " ".join(key),                    # core minus generic words (blocking key)
        "name_sorted": " ".join(sorted(set(core))),
        "name_concat": "".join(core),                 # for domain / handle comparisons
        "name_legal": " ".join(legal),
        "name_parts": "|".join(a for a in alias_cores if a),
        "name_is_domain": is_domain,
        "name_has_dba": has_dba,
        "name_had_indic": had_indic,
    }


# ----------------------------------------------------------------------------------------------------------------
# Addresses
# ----------------------------------------------------------------------------------------------------------------
# canonical short forms; both directions collapse to the same token (street/st -> st, saint -> st, road -> rd ...)
ADDR_CANON = {
    # English
    "street": "st", "str": "st", "st": "st", "saint": "st", "road": "rd", "rd": "rd", "avenue": "ave", "av": "ave",
    "ave": "ave", "aven": "ave", "avn": "ave", "drive": "dr", "dr": "dr", "drv": "dr", "lane": "ln", "ln": "ln",
    "court": "ct", "ct": "ct", "crt": "ct", "circle": "cir", "cir": "cir", "circ": "cir", "place": "pl", "pl": "pl",
    "plc": "pl", "boulevard": "blvd", "blvd": "blvd", "boul": "blvd", "bvd": "blvd", "bd": "blvd", "bld": "blvd",
    "highway": "hwy", "hwy": "hwy", "hiway": "hwy", "parkway": "pkwy", "pkwy": "pkwy", "pky": "pkwy",
    "trail": "trl", "trl": "trl", "terrace": "ter", "ter": "ter", "terr": "ter", "square": "sq", "sq": "sq",
    "way": "way", "wy": "way", "expressway": "expy", "expy": "expy", "freeway": "fwy", "fwy": "fwy",
    "point": "pt", "pt": "pt", "crossing": "xing", "xing": "xing", "cove": "cv", "cv": "cv", "hollow": "holw",
    "heights": "hts", "hts": "hts", "junction": "jct", "jct": "jct", "mountain": "mtn", "mtn": "mtn",
    "mount": "mt", "mt": "mt", "fort": "ft", "ft": "ft", "park": "park", "pk": "park", "plaza": "plz", "plz": "plz",
    "run": "run", "row": "row", "walk": "walk", "path": "path", "pass": "pass", "loop": "loop", "alley": "aly",
    "aly": "aly", "creek": "crk", "crk": "crk", "ridge": "rdg", "rdg": "rdg", "valley": "vly", "vly": "vly",
    "view": "vw", "vw": "vw", "village": "vlg", "vlg": "vlg", "estates": "est", "estate": "est", "est": "est",
    "north": "n", "south": "s", "east": "e", "west": "w", "northeast": "ne", "northwest": "nw", "southeast": "se",
    "southwest": "sw", "apartment": "apt", "apartments": "apts", "apt": "apt", "suite": "ste", "ste": "ste",
    "floor": "fl", "fl": "fl", "flr": "fl", "building": "bldg", "bldg": "bldg", "bldng": "bldg", "unit": "unit",
    "room": "rm", "rm": "rm", "department": "dept", "dept": "dept", "township": "twp", "twp": "twp",
    "county": "cnty", "cnty": "cnty", "city": "city", "ity": "city", "cty": "city",
    # India
    "nagar": "nagar", "ngr": "nagar", "colony": "colony", "col": "colony", "sector": "sec", "sec": "sec",
    "opposite": "opp", "opp": "opp", "near": "nr", "nr": "nr", "behind": "bhd", "bhd": "bhd", "marg": "marg",
    "chowk": "chowk", "bazar": "bazaar", "bazaar": "bazaar", "market": "mkt", "mkt": "mkt", "cross": "crs",
    "main": "main", "layout": "layout", "phase": "ph", "ph": "ph", "block": "blk", "blk": "blk", "village": "vill",
    "vill": "vill", "vil": "vill", "post": "po", "po": "po", "district": "dist", "dist": "dist", "distt": "dist",
    "tehsil": "teh", "teh": "teh", "taluk": "tq", "taluka": "tq", "tq": "tq", "industrial": "indl", "indl": "indl",
    "ind": "indl", "area": "area", "society": "soc", "soc": "soc", "complex": "cmplx", "cmplx": "cmplx",
    "ground": "gr", "gr": "gr", "first": "1", "second": "2", "third": "3", "fourth": "4",
    # French
    "rue": "rue", "r": "rue", "allee": "all", "all": "all", "impasse": "imp", "imp": "imp", "chemin": "chem",
    "chem": "chem", "ch": "chem", "route": "rte", "rte": "rte", "quai": "quai", "q": "quai", "cours": "crs",
    "crs": "crs", "faubourg": "fbg", "fbg": "fbg", "fg": "fbg", "residence": "res", "res": "res", "resid": "res",
    "lotissement": "lot", "lot": "lot", "lieu": "ld", "ld": "ld", "hameau": "ham", "ham": "ham", "sente": "sen",
    "passage": "pass", "voie": "voie", "zone": "zone", "za": "za", "zi": "zi", "bis": "bis",
}
ADDR_STOP = {"de", "du", "des", "la", "le", "les", "l", "d", "of", "the", "and", "a", "en", "sur", "et"}
# canonical road-type / street-affix tokens (values of ADDR_CANON for thoroughfares) + directions + French bis/ter
ROAD_TYPES = {"st", "rd", "ave", "dr", "ln", "ct", "cir", "pl", "blvd", "hwy", "pkwy", "trl", "ter", "sq", "way", "expy",
              "fwy", "pt", "xing", "cv", "rue", "all", "imp", "chem", "rte", "quai", "crs", "fbg", "res", "lot", "ld",
              "ham", "sen", "pass", "voie", "zone", "za", "zi", "bis", "b", "t", "n", "s", "e", "w", "ne", "nw", "se", "sw",
              "marg", "path", "walk", "row", "run", "loop", "aly"}
# countries with a built-in state list (captured at import time; learned admin lookups for open-set countries are
# installed later and must not change this set)
BUILTIN_STATE_COUNTRIES = None
NULL_TOKENS = {"null", "none", "nan", "na", "n/a", "<null>", "nil", "unknown", "notavailable", "not available", "-"}
UNIT_WORDS = {"apt", "apts", "ste", "unit", "fl", "rm", "bldg", "flat", "shop", "office", "room", "floor", "no"}
POBOX_RE = re.compile(r"\b(?:p\s*\.?\s*o\s*\.?\s*box|post office box|pmb|p\.m\.b\.?)\s*#?\s*(\d+)", re.I)
HOUSE_PREFIX_RE = re.compile(r"\b(?:door\s*no|d\s*no|h\s*\.?\s*no|house\s*no|hno|plot\s*no|sr\s*\.?\s*no|survey\s*no|s\s*no|kh\s*no|khasra\s*no|gat\s*no|no)\b\s*[.:\-]*\s*", re.I)
ORDINAL_RE = re.compile(r"\b(\d+)(?:st|nd|rd|th)\b")
LEADZERO_RE = re.compile(r"\b0+(\d)")
NUM_RE = re.compile(r"\d+")

US_STATES = {
    "alabama": "al", "alaska": "ak", "arizona": "az", "arkansas": "ar", "california": "ca", "colorado": "co",
    "connecticut": "ct", "delaware": "de", "florida": "fl", "georgia": "ga", "hawaii": "hi", "idaho": "id",
    "illinois": "il", "indiana": "in", "iowa": "ia", "kansas": "ks", "kentucky": "ky", "louisiana": "la",
    "maine": "me", "maryland": "md", "massachusetts": "ma", "michigan": "mi", "minnesota": "mn",
    "mississippi": "ms", "missouri": "mo", "montana": "mt", "nebraska": "ne", "nevada": "nv",
    "new hampshire": "nh", "new jersey": "nj", "new mexico": "nm", "new york": "ny", "north carolina": "nc",
    "north dakota": "nd", "ohio": "oh", "oklahoma": "ok", "oregon": "or", "pennsylvania": "pa",
    "rhode island": "ri", "south carolina": "sc", "south dakota": "sd", "tennessee": "tn", "texas": "tx",
    "utah": "ut", "vermont": "vt", "virginia": "va", "washington": "wa", "west virginia": "wv",
    "wisconsin": "wi", "wyoming": "wy", "district of columbia": "dc", "washington dc": "dc", "puerto rico": "pr",
}
IN_STATES = {
    "andhra pradesh": "ap", "arunachal pradesh": "ar", "assam": "as", "bihar": "br", "chhattisgarh": "cg",
    "chattisgarh": "cg", "goa": "ga", "gujarat": "gj", "haryana": "hr", "himachal pradesh": "hp", "jharkhand": "jh",
    "karnataka": "ka", "kerala": "kl", "keralam": "kl", "madhya pradesh": "mp", "maharashtra": "mh",
    "manipur": "mn", "meghalaya": "ml", "mizoram": "mz", "nagaland": "nl", "odisha": "od", "orissa": "od",
    "punjab": "pb", "rajasthan": "rj", "sikkim": "sk", "tamil nadu": "tn", "tamilnadu": "tn", "telangana": "tg",
    "tripura": "tr", "uttar pradesh": "up", "uttarakhand": "uk", "uttaranchal": "uk", "west bengal": "wb",
    "delhi": "dl", "new delhi": "dl", "nct of delhi": "dl", "jammu and kashmir": "jk", "jammu kashmir": "jk",
    "ladakh": "la", "puducherry": "py", "pondicherry": "py", "chandigarh": "ch", "andaman and nicobar islands": "an",
    "dadra and nagar haveli": "dn", "daman and diu": "dd", "lakshadweep": "ld",
}
IN_CODE_ALIASES = {"ts": "tg", "or": "od", "ct": "cg", "ut": "uk", "orissa": "od"}


def _build_state_lookup():
    """country(lower) -> {surface form -> canonical code}. Unknown countries get an empty map (open set)."""
    us = dict(US_STATES)
    us.update({v: v for v in US_STATES.values()})
    ind = dict(IN_STATES)
    ind.update({v: v for v in IN_STATES.values()})
    ind.update(IN_CODE_ALIASES)
    return {"us": us, "usa": us, "united states": us, "india": ind, "in": ind}


STATE_LOOKUP = _build_state_lookup()


def _canon_addr_tokens(seg: str) -> List[str]:
    seg = DOTS_IN_ABBR_RE.sub("", seg)
    toks = [t for t in re.split(r"[^a-z0-9/\-]+", seg) if t]
    out = []
    for t in toks:
        t = t.strip("-/")
        if not t:
            continue
        if t in NULL_TOKENS:
            continue
        if any(c.isdigit() for c in t):
            out.append(t)
            continue
        c = ADDR_CANON.get(t, t)
        out.append(c)
    return out


def _romanize_addr_segments(raw: str) -> str:
    """Whole-segment lookup first (multi-word state names written as one native run: 'ఆంధ్రప్రదేశ్')."""
    if not INDIC_RE.search(raw):
        return raw
    segs = raw.split(",")
    out = []
    for seg in segs:
        key = clean_native(seg.strip())
        hit = _TRANSLIT_DICT.get(key) if INDIC_RE.search(key) else None
        out.append(" " + hit if hit is not None else seg)
    return ",".join(out)


def _builtin_countries():
    global BUILTIN_STATE_COUNTRIES
    if BUILTIN_STATE_COUNTRIES is None:
        BUILTIN_STATE_COUNTRIES = frozenset(_build_state_lookup())
    return BUILTIN_STATE_COUNTRIES


def norm_addr(raw: str, country: str = "") -> Dict[str, object]:
    s = fold_ascii(_romanize_addr_segments(raw or ""))
    lookup = STATE_LOOKUP.get((country or "").strip().lower(), {})
    pobox = POBOX_RE.findall(s)
    s = POBOX_RE.sub(" ", s)
    s = s.replace("#", " ").replace("<null>", " ")
    s = HOUSE_PREFIX_RE.sub(" ", s)
    s = ORDINAL_RE.sub(r"\1", s)
    s = LEADZERO_RE.sub(r"\1", s)
    if (country or "").strip().lower() not in _builtin_countries():
        s = re.sub(r"\b(\d+)\s+b\b", r"\1 bis", s)
        s = re.sub(r"\b(\d+)\s+t\b", r"\1 ter", s)
    segs = [SPACE_RE.sub(" ", x).strip(" .-") for x in s.split(",")]
    segs = [x for x in segs if x and x not in NULL_TOKENS]
    state = ""
    locality: List[str] = []
    street_tokens: List[str] = []
    unit_tokens: List[str] = []
    all_tokens: List[str] = []
    for seg in segs:
        st = lookup.get(seg)
        if st is None and lookup:
            # "... tamil nadu" / "city tx" at the end of a segment
            for n in (3, 2, 1):
                w = seg.split()
                if len(w) > n and " ".join(w[-n:]) in lookup and n >= 2:
                    st = lookup[" ".join(w[-n:])]
                    seg = " ".join(w[:-n])
                    break
        if st is not None and not state:
            state = st
            continue
        toks = _canon_addr_tokens(seg)
        if not toks:
            continue
        all_tokens.extend(toks)
        has_num = any(any(c.isdigit() for c in t) for t in toks)
        if toks[0] in UNIT_WORDS:
            unit_tokens.extend(toks)
        elif has_num and not street_tokens:
            street_tokens.extend(toks)
        elif has_num:
            unit_tokens.extend(toks)
        else:
            locality.extend(toks)
    nums = []
    for t in street_tokens + unit_tokens:
        nums.extend(NUM_RE.findall(t))
    house = ""
    for t in street_tokens:
        if any(c.isdigit() for c in t):
            house = t
            break
    street_alpha = [t for t in street_tokens if not any(c.isdigit() for c in t) and t not in ADDR_STOP]
    loc = [t for t in locality if t not in ADDR_STOP]
    return {
        "addr_norm": " ".join(all_tokens),
        "addr_street": " ".join(street_alpha),
        "addr_house": house,
        "addr_nums": " ".join(dict.fromkeys(nums)),
        "addr_locality": " ".join(dict.fromkeys(loc)),
        "addr_state": state,
        "addr_unit": " ".join(unit_tokens),
        "addr_pobox": " ".join(pobox),
        "addr_n_segs": len(segs),
        "addr_empty": len(all_tokens) == 0,
    }


def normalize_record(name: str, addr: str, country: str) -> Dict[str, object]:
    d = norm_name(name)
    d.update(norm_addr(addr, country))
    d["country_norm"] = (country or "").strip().lower()
    return d


# ----------------------------------------------------------------------------------------------------------------
# Phonetic / skeleton keys (robust to transliteration vowel loss and Tamil voiced/unvoiced merge)
# ----------------------------------------------------------------------------------------------------------------
_SKEL_MAP = str.maketrans({"w": "v", "q": "k", "c": "k", "g": "k", "z": "j", "d": "t", "b": "p", "f": "p", "x": "k", "y": "i"})
_VOWEL_RE = re.compile(r"(?<!^)[aeiouh]")


@lru_cache(maxsize=2_000_000)
def skeleton(tok: str) -> str:
    """Consonant skeleton: 'silver' -> 'slvr', 'silvr' -> 'slvr', 'limitet' == 'limited'."""
    if not tok:
        return ""
    t = tok.translate(_SKEL_MAP)
    t = _VOWEL_RE.sub("", t)
    return _DOUBLE_RE.sub(r"\1", t)


def skeleton_str(s: str) -> str:
    return " ".join(skeleton(t) for t in s.split())
