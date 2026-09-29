"""Build the final submission package exactly as the challenge requires:

Elyptics_submission.zip
├── output/
│   ├── matching_results.tsv
│   └── candidate_pairs.tsv
├── code/
│   └── business_entity_resolution/   (src/, README.md, requirements.txt, configs/, scripts/, tests/)
└── Documentation_template.md          (filled-in methodology)

Runs the official validator (utils/validate_submission.py, --check-ids) first and refuses to package on FAIL.
usage: python scripts/make_submission.py --outputs D:/er_data/work_dev/output/final --team Elyptics \
         --validator D:/er_data/student_resource/utils/validate_submission.py --test-dir D:/er_data/student_resource/dataset/test
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
CODE_ITEMS = ["src", "configs", "scripts", "tests", "README.md", "requirements.txt", "AWS_RUNBOOK.md"]
SKIP_DIRS = {"__pycache__", ".pytest_cache", ".ipynb_checkpoints", "catboost_info"}  # catboost_info: CatBoost's own
# live training-log scratch dir, written wherever training is invoked from (src/, per run_local*.sh) -- not code


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs", required=True, help="dir holding matching_results.tsv + candidate_pairs.tsv")
    ap.add_argument("--team", default="Elyptics")
    ap.add_argument("--validator", required=True)
    ap.add_argument("--test-dir", required=True)
    ap.add_argument("--doc", default=os.path.join(REPO, "Documentation.md"))
    ap.add_argument("--dest", default=os.path.dirname(REPO))
    a = ap.parse_args(argv)
    m = os.path.join(a.outputs, "matching_results.tsv")
    c = os.path.join(a.outputs, "candidate_pairs.tsv")
    for p in (m, c, a.doc):
        if not os.path.exists(p):
            sys.exit(f"missing {p}")
    r = subprocess.run([sys.executable, a.validator, "--matching", m, "--candidate", c, "--test-dir", a.test_dir,
                        "--check-ids"], capture_output=True, text=True)
    print(r.stdout[-4000:], r.stderr[-2000:])
    if r.returncode != 0 or "PASS" not in r.stdout:
        sys.exit("validator did not PASS -> not packaging")
    zpath = os.path.join(a.dest, f"{a.team}_submission.zip")
    with zipfile.ZipFile(zpath, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        z.write(m, "output/matching_results.tsv")
        z.write(c, "output/candidate_pairs.tsv")
        z.write(a.doc, "Documentation_template.md")
        for item in CODE_ITEMS:
            src = os.path.join(REPO, item)
            if os.path.isfile(src):
                z.write(src, f"code/business_entity_resolution/{item}")
                continue
            for root, dirs, files in os.walk(src):
                dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
                for fn in files:
                    if fn.endswith((".pyc", ".pyo")):
                        continue
                    full = os.path.join(root, fn)
                    rel = os.path.relpath(full, REPO).replace(os.sep, "/")
                    z.write(full, f"code/business_entity_resolution/{rel}")
    with zipfile.ZipFile(zpath) as z:
        names = z.namelist()
        bad = z.testzip()
    print(f"wrote {zpath} ({os.path.getsize(zpath) / 1e6:.1f} MB, {len(names)} files, integrity={'OK' if bad is None else bad})")
    for need in ("output/matching_results.tsv", "output/candidate_pairs.tsv", "Documentation_template.md",
                 "code/business_entity_resolution/README.md", "code/business_entity_resolution/requirements.txt",
                 "code/business_entity_resolution/src/ber/pipeline.py"):
        assert need in names, need
    print("structure OK")


if __name__ == "__main__":
    main()
