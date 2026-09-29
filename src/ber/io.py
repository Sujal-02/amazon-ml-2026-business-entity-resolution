"""Reading challenge TSVs and writing submission files."""
from __future__ import annotations

import csv
import os
from typing import Dict, Iterable, List

import pandas as pd

SOURCE_COLS = ["entity_id", "business_name", "business_address", "country"]


def read_source(path: str) -> pd.DataFrame:
    """Read one *_sourceK.tsv exactly as-is (all strings, no NA coercion, BOM-safe).

    quoting=QUOTE_NONE matters: names such as `"Joe's" Cafe` must not be parsed as quoted fields.
    """
    df = pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        na_filter=False,
        quoting=csv.QUOTE_NONE,
        encoding="utf-8-sig",
        engine="c",
    )
    df.columns = [c.strip().lower() for c in df.columns]
    for c in SOURCE_COLS:
        if c not in df.columns:
            df[c] = ""
    return df[SOURCE_COLS]


def read_split(data_dir: str, split: str) -> Dict[int, pd.DataFrame]:
    return {k: read_source(os.path.join(data_dir, split, f"{split}_source{k}.tsv")) for k in (1, 2, 3)}


def read_ground_truth(path: str) -> Dict[str, List[str]]:
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, na_filter=False,
                     quoting=csv.QUOTE_NONE, encoding="utf-8-sig")
    df.columns = [c.strip().lower() for c in df.columns]
    return {s1: ([x for x in m.split(",") if x] if m else [])
            for s1, m in zip(df["source1_entity_id"], df["matched_entity_ids"])}


def gt_pairs(gt: Dict[str, List[str]]) -> pd.DataFrame:
    rows = [(s1, m) for s1, ms in gt.items() for m in ms]
    return pd.DataFrame(rows, columns=["s1", "cand"])


def write_id_lists(path: str, s1_ids: Iterable[str], mapping: Dict[str, Iterable[str]], col: str) -> None:
    """Write a matching_results / candidate_pairs TSV that passes utils/validate_submission.py.

    Every S1 id gets exactly one row, lists are de-duplicated preserving order, only S2-/S3- ids kept.
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(f"source1_entity_id\t{col}\n")
        seen_rows = set()
        for s1 in s1_ids:
            if s1 in seen_rows:
                continue
            seen_rows.add(s1)
            ids, seen = [], set()
            for x in mapping.get(s1, ()):
                if x not in seen and (x.startswith("S2-") or x.startswith("S3-")):
                    seen.add(x)
                    ids.append(x)
            f.write(f"{s1}\t{','.join(ids)}\n")


def write_matching(path: str, s1_ids, mapping) -> None:
    write_id_lists(path, s1_ids, mapping, "matched_entity_ids")


def write_candidates(path: str, s1_ids, mapping) -> None:
    write_id_lists(path, s1_ids, mapping, "candidate_entity_ids")
