import os
import sys
import tempfile

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ber import decision as D  # noqa: E402
from ber import io as bio  # noqa: E402
from ber import metrics as M  # noqa: E402
from ber import normalize as N  # noqa: E402


def test_f05_matches_problem_statement_example():
    s = M.f05({"S2-00047", "S2-00193", "S3-00812"}, {"S2-00047", "S3-00812"})
    assert abs(s - 0.714) < 1e-3


def test_singletons():
    assert M.f05(set(), set()) == 1.0
    assert M.f05({"S2-1"}, set()) == 0.0
    assert M.f05(set(), {"S2-1"}) == 0.0


def test_expected_f_prefers_empty_for_weak_single_candidate():
    pairs = pd.DataFrame({"s1": ["a", "b", "b", "c"], "cand": ["x", "y", "z", "w"], "p": [0.3, 0.97, 0.95, 0.7]})
    out = D.select_expected_f(pairs)
    assert "a" not in out  # E[F|empty] = 0.7 > E[F|{x}] = 0.3
    assert set(out["b"]) == {"y", "z"}
    assert out["c"] == ["w"]


def test_expected_f_matches_bruteforce():
    rng = np.random.default_rng(0)
    for _ in range(50):
        p = np.sort(rng.random(5))[::-1]
        best_k, best_f = D._best_prefix(p, 0.25, 0.0, 16)
        # brute force enumeration over outcomes
        import itertools

        def ef(k):
            tot = 0.0
            for ys in itertools.product([0, 1], repeat=len(p)):
                pr = np.prod([pi if y else 1 - pi for pi, y in zip(p, ys)])
                pred = set(range(k)); true = {i for i, y in enumerate(ys) if y}
                tot += pr * M.f05({str(i) for i in pred}, {str(i) for i in true})
            return tot

        bf = [ef(k) for k in range(len(p) + 1)]
        assert abs(max(bf) - best_f) < 1e-9
        assert int(np.argmax(bf)) == best_k


def test_exclusivity():
    pairs = pd.DataFrame({"s1": ["a", "b"], "cand": ["x", "x"], "p": [0.9, 0.8]})
    ex = D.enforce_exclusivity(pairs)
    assert list(ex.s1) == ["a"]


def test_writer_roundtrip():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "m.tsv")
        bio.write_matching(p, ["S1-1", "S1-2", "S1-1"], {"S1-1": ["S2-1", "S2-1", "S3-2", "S1-9"]})
        txt = open(p, encoding="utf-8").read().splitlines()
        assert txt[0] == "source1_entity_id\tmatched_entity_ids"
        assert txt[1] == "S1-1\tS2-1,S3-2"
        assert txt[2] == "S1-2\t"
        assert len(txt) == 3


def test_normalize_examples():
    r = N.norm_name("Ectoorbidrex DBA: Devlin Beacon Emerging Corp")
    assert "devlin beacon emerging" in r["name_parts"].split("|")
    assert N.norm_name("womens hea1th coastal care, inc.")["name_core"] == "womens health coastal care"
    a = N.norm_addr("AZ, 013623 24nd Ln, PHOENIX", "US")
    b = N.norm_addr("13623 24th Lane, Phoenix, AZ", "US")
    assert a["addr_norm"] == b["addr_norm"] and a["addr_state"] == "az"
    fr = N.norm_addr("104 Bis R. Des Prairies, Calais", "France")
    assert fr["addr_house"] == "104" and "rue" in fr["addr_norm"]
    assert N.fold_ascii("CŒUR Frères") == "coeur freres"


def test_pair_features_empty_core_names_do_not_crash():
    from ber import features as F

    a = "Alarca House, Jodhpur, Rajasthan"
    r1 = N.normalize_record("La & Co", a, "India")
    r2 = N.normalize_record("LA & CO.", a, "India")
    cx = {"idf": {}, "dflt": 10.0, "s1df": {}, "x1": set(), "x2": set()}
    f = F.pair_features(r1, r2, cx)
    assert f["nm_core_tset"] == 1.0 and f["ad_tset"] == 1.0


def test_honorific_domain_keeps_stem():
    assert N.norm_name("Smt PALHOLDINGS.COM")["name_core"] == "palholdings"


def test_context_second_best_vectorised():
    from ber import features as F

    df = pd.DataFrame({"i1": [1, 1, 2, 2, 3], "i2": [10, 11, 10, 12, 10], "p": [0.9, 0.2, 0.6, 0.5, 0.1]})
    c = F.context_features(df, "p")
    # candidate 10 is contested by S1 1 (0.9), 2 (0.6), 3 (0.1): margins vs best other S1
    assert np.allclose(c["ctx_margin_other_s1"].to_numpy()[[0, 2, 4]], [0.3, -0.3, -0.8])
