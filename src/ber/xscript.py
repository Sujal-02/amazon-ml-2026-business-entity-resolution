"""xscript.py - cross-script name normalisation for the Business Entity Resolution challenge.

Stdlib only (no GPL/AGPL deps). Provides:
  * token_script / is_native           - Unicode-block script detection for the 9 Indic scripts
  * clean_native / fold_latin          - NFC + ZWJ/ZWNJ strip; NFKD accent folding + ligatures (French)
  * canon_latin                        - legal-suffix canonicalisation (pvt->private, ltd->limited ...)
  * romanize                           - Inter-Indic offset romaniser (all 9 scripts -> Devanagari layout -> Latin),
                                         word-final schwa dropped, Malayalam/Tamil/Bengali special letters
  * skeleton(fine|coarse)              - shared phonetic consonant key applied to BOTH sides
  * Lexicon                            - learned native-token -> Latin-token distribution from train GT pairs
                                         (positional alignment; token counts are equal in train pairs)
  * VocabResolver                      - OOV fallback: nearest token of a CLOSED Latin vocabulary (S1 names of the
                                         same split) by skeleton bucket + edit distance + log-frequency
"""
import unicodedata, re, math, collections, json

BASES = {'DEVANAGARI': 0x0900, 'BENGALI': 0x0980, 'GURMUKHI': 0x0A00, 'GUJARATI': 0x0A80, 'ORIYA': 0x0B00,
         'TAMIL': 0x0B80, 'TELUGU': 0x0C00, 'KANNADA': 0x0C80, 'MALAYALAM': 0x0D00}

def _script_cp(cp):
    if 0x0900 <= cp < 0x0D80:
        for k, b in BASES.items():
            if b <= cp < b + 0x80:
                return k
    return None

def is_native(s):
    return any(0x0900 <= ord(ch) < 0x0D80 for ch in s)

def token_script(t):
    c = collections.Counter(_script_cp(ord(ch)) for ch in t)
    c.pop(None, None)
    return c.most_common(1)[0][0] if c else 'LATIN'

_ZW = dict.fromkeys([0x200B, 0x200C, 0x200D, 0x2060, 0xFEFF], None)
_PUNCT = '.,()[]{}"\'-|/<>#*:;!?'

def clean_native(t):
    return unicodedata.normalize('NFC', t).translate(_ZW).strip(_PUNCT)

_LIG = str.maketrans({'œ': 'oe', 'Œ': 'oe', 'æ': 'ae', 'Æ': 'ae', 'ß': 'ss', 'ø': 'o', 'Ø': 'o', 'ł': 'l', 'Ł': 'l',
                      'đ': 'd', 'Đ': 'd', 'ı': 'i', '’': "'", '‘': "'", 'º': 'o', 'ª': 'a'})

def fold_latin(t):
    t = t.translate(_LIG)
    t = unicodedata.normalize('NFKD', t)
    t = ''.join(ch for ch in t if not unicodedata.combining(ch))
    return t.lower().strip(_PUNCT)

LEGAL = {'pvt': 'private', 'pvt.': 'private', 'prv': 'private', 'ltd': 'limited', 'ltd.': 'limited', 'lmtd': 'limited'}

def canon_latin(t):
    t = fold_latin(t)
    return LEGAL.get(t, t)

# ----------------------------------------------------------------------------- romaniser
CONS = {0x15: 'k', 0x16: 'kh', 0x17: 'g', 0x18: 'gh', 0x19: 'n', 0x1A: 'ch', 0x1B: 'chh', 0x1C: 'j', 0x1D: 'jh', 0x1E: 'n',
        0x1F: 't', 0x20: 'th', 0x21: 'd', 0x22: 'dh', 0x23: 'n', 0x24: 't', 0x25: 'th', 0x26: 'd', 0x27: 'dh', 0x28: 'n',
        0x29: 'n', 0x2A: 'p', 0x2B: 'f', 0x2C: 'b', 0x2D: 'bh', 0x2E: 'm', 0x2F: 'y', 0x30: 'r', 0x31: 'r', 0x32: 'l',
        0x33: 'l', 0x34: 'zh', 0x35: 'v', 0x36: 'sh', 0x37: 'sh', 0x38: 's', 0x39: 'h',
        0x58: 'q', 0x59: 'kh', 0x5A: 'g', 0x5B: 'z', 0x5C: 'd', 0x5D: 'rh', 0x5E: 'f', 0x5F: 'y'}
INDEP_V = {0x05: 'a', 0x06: 'aa', 0x07: 'i', 0x08: 'ee', 0x09: 'u', 0x0A: 'oo', 0x0B: 'ri', 0x0D: 'e', 0x0E: 'e',
           0x0F: 'e', 0x10: 'ai', 0x11: 'o', 0x12: 'o', 0x13: 'o', 0x14: 'au', 0x72: 'e', 0x73: 'o'}
MATRA = {0x3E: 'aa', 0x3F: 'i', 0x40: 'ee', 0x41: 'u', 0x42: 'oo', 0x43: 'ri', 0x45: 'e', 0x46: 'e', 0x47: 'e',
         0x48: 'ai', 0x49: 'o', 0x4A: 'o', 0x4B: 'o', 0x4C: 'au', 0x55: '', 0x56: 'ai', 0x57: 'au'}
VIRAMA, NUKTA, ANUSVARA, CANDRA, VISARGA = 0x4D, 0x3C, 0x02, 0x01, 0x03
SPECIAL = {0x0D7A: 'n', 0x0D7B: 'n', 0x0D7C: 'r', 0x0D7D: 'l', 0x0D7E: 'l', 0x0D7F: 'k', 0x0D54: 'm', 0x0D55: 'y',
           0x0D56: 'l', 0x09CE: 't', 0x0A70: 'n', 0x0A71: '', 0x0B71: 'v', 0x0B83: '', 0x09DC: 'r', 0x09DD: 'rh',
           0x09DF: 'y', 0x0B5C: 'd', 0x0B5D: 'rh', 0x0B5F: 'y', 0x0950: 'om'}
# script-specific digraph fixes applied before the offset walk
PRE = [('റ്റ', 'ട്ട'),   # Malayalam റ്റ -> ട്ട  (tt)
       ('ന്റ', 'ന്ട'),   # Malayalam ന്റ -> ന്ട  (nt)
       ('ற்ற', 'ட்ட')]   # Tamil ற்ற -> ட்ட (tt)

def romanize(tok):
    for a, b in PRE:
        tok = tok.replace(a, b)
    out, pend_a = [], False
    cps = [ord(c) for c in tok]
    i = 0
    while i < len(cps):
        cp = cps[i]
        if cp in SPECIAL:
            if cp == 0x0B83 and i + 1 < len(cps) and cps[i + 1] - 0x0B80 == 0x2A:   # Tamil aytham + pa = f
                if pend_a: out.append('a')
                out.append('f'); pend_a = True; i += 2; continue
            if pend_a and SPECIAL[cp]: out.append('a'); pend_a = False
            out.append(SPECIAL[cp]); i += 1; continue
        sc = _script_cp(cp)
        if sc is None:
            if pend_a: out.append('a'); pend_a = False
            ch = chr(cp)
            out.append(fold_latin(ch) if ch.isalnum() else ''); i += 1; continue
        off = cp - BASES[sc]
        if off in CONS:
            if pend_a: out.append('a')
            c = CONS[off]
            if i + 1 < len(cps) and cps[i + 1] - BASES[sc] == NUKTA:
                c = {'k': 'q', 'g': 'g', 'j': 'z', 'd': 'r', 'dh': 'rh', 'f': 'f', 'kh': 'kh'}.get(c, c); i += 1
            out.append(c); pend_a = True
        elif off in MATRA:
            out.append(MATRA[off]); pend_a = False
        elif off == VIRAMA:
            pend_a = False
        elif off in INDEP_V:
            if pend_a: out.append('a'); pend_a = False
            out.append(INDEP_V[off])
        elif off in (ANUSVARA, CANDRA):
            if pend_a: out.append('a'); pend_a = False
            out.append('n')
        elif off == VISARGA:
            if pend_a: out.append('a'); pend_a = False
            out.append('h')
        elif 0x66 <= off <= 0x6F:
            if pend_a: out.append('a'); pend_a = False
            out.append(str(off - 0x66))
        i += 1
    return ''.join(out)          # trailing inherent 'a' is NOT emitted = word-final schwa deletion

# ----------------------------------------------------------------------------- skeleton keys
_LAT = [(r'tion', 'shn'), (r'sion', 'shn'), (r'ture', 'char'), (r'ph', 'f'), (r'ck', 'k'), (r'q', 'k'), (r'x', 'ks'),
        (r'c(?=[eiy])', 's'), (r'ch', '\x01'), (r'c', 'k'), (r'\x01', 'ch'), (r'w', 'v'), (r'z', 'j'), (r'g(?=[eiy])', 'j')]
_COARSE = str.maketrans({'g': 'k', 'd': 't', 'b': 'p', 'j': 'c', 's': 'c', 'f': 'p', 'v': 'p', 'l': 'r', 'n': 'm', 'h': ''})

def skeleton(s, coarse=False, latin=True):
    s = s.lower()
    if latin:
        s = fold_latin(s)
        for a, b in _LAT:
            s = re.sub(a, b, s)
    else:
        s = s.replace('z', 'j')
    s = s.replace('sh', 's').replace('ch', 'c')
    s = re.sub(r'(?<=[bcdfgjklmnprstv])h', '', s)
    s = re.sub(r'[^a-z0-9]', '', s)
    if not s:
        return s
    head = s[0] if s[0] in 'aeiou' else ''
    s = head + re.sub(r'[aeiouy]', '', s)
    if coarse:
        s = re.sub(r'^[aeiou]', '', s).translate(_COARSE)
    return re.sub(r'(.)\1+', r'\1', s)

# ----------------------------------------------------------------------------- learned lexicon
class Lexicon:
    def __init__(self):
        self.tab = collections.defaultdict(collections.Counter)

    def fit(self, pairs):
        """pairs: iterable of (native_or_mixed_name, s1_latin_name) from train ground truth."""
        n_used = n_skip = 0
        for nat, lat in pairs:
            nt, lt = nat.split(), lat.split()
            if len(nt) != len(lt):
                n_skip += 1; continue
            n_used += 1
            for a, b in zip(nt, lt):
                if is_native(a):
                    self.tab[clean_native(a)][canon_latin(b)] += 1
        return n_used, n_skip

    def variants(self, tok):
        return self.tab.get(clean_native(tok))

    def back(self, name, resolver=None):
        """Returns (argmax_string, list_of_variant_sets, n_oov). Variant sets let the matcher accept any
        spelling seen for the native token (shree/sree, jai/jay, laxmi/lakshmi, shiv/shiva)."""
        best, sets, oov = [], [], 0
        for t in name.split():
            if is_native(t):
                c = self.variants(t)
                if c:
                    best.append(c.most_common(1)[0][0]); sets.append(set(c))
                else:
                    oov += 1
                    cand = resolver.resolve(clean_native(t)) if resolver else []
                    if cand:
                        best.append(cand[0]); sets.append(set(cand[:3]))
                    else:
                        r = romanize(clean_native(t)); best.append(r); sets.append({r})
            else:
                w = canon_latin(t); best.append(w); sets.append({w})
        return ' '.join(best), sets, oov

    def save(self, path):
        json.dump({k: dict(v) for k, v in self.tab.items()}, open(path, 'w', encoding='utf-8'), ensure_ascii=False)

    @classmethod
    def load(cls, path):
        lx = cls()
        for k, v in json.load(open(path, encoding='utf-8')).items():
            lx.tab[k] = collections.Counter(v)
        return lx

# ----------------------------------------------------------------------------- OOV fallback
def _lev(a, b):
    if a == b: return 0
    if len(a) < len(b): a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]

class VocabResolver:
    """Map an unseen native token to the most plausible token of a closed Latin vocabulary
    (e.g. all tokens of S1 India names in the same split) - transliteration *mining* instead of generation."""
    def __init__(self, vocab_counts):
        self.cnt = vocab_counts
        self.fine = collections.defaultdict(list); self.coarse = collections.defaultdict(list)
        for w in vocab_counts:
            self.fine[skeleton(w)].append(w); self.coarse[skeleton(w, coarse=True)].append(w)

    def resolve(self, native_tok, k=5):
        r = romanize(native_tok)
        fk, ck = skeleton(r, latin=False), skeleton(r, coarse=True, latin=False)
        cands = set(self.fine.get(fk, [])) | set(self.coarse.get(ck, []))
        scored = []
        for w in cands:
            # weights hand-tuned on a held-out type split (top1 0.835 / top3 0.899); replace by a learned ranker
            s = (2.0 if skeleton(w) == fk else 0.0) - _lev(r, w) / max(len(r), len(w)) * 2.0 + 0.8 * math.log1p(self.cnt[w])
            scored.append((s, w))
        scored.sort(reverse=True)
        return [w for _, w in scored[:k]]
