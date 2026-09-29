# Business Entity Resolution — Amazon ML Challenge 2026 (Team Elyptics)

Link every record of a reference business list (Source 1) to all of its noisy duplicates in two other lists
(Source 2 / Source 3), across **US, India and an unseen country (France)**, at the scale of **~26 million records**,
on a **16 GB CPU laptop** plus free Kaggle GPUs. Metric: macro **F0.5** per Source-1 entity (precision counts twice as much as recall;
an entity with no true match scores 1 only if we predict nothing).

**Final leaderboard score: 0.982** (one day of iteration: 0.977 → 0.979 → 0.980 → 0.981 → 0.982). The top-10 cut-off was ~0.99, so we missed it. This repo is the full, honest record of what we built, what worked, what did not, and why.

| Submission | What changed | Public LB |
|---|---|---|
| A | Blocking v2 + two-stage GBDT stack + expected-F0.5 decision (OOF 0.9899) | **0.977** |
| CE-L6 | + cross-encoder re-ranker (MiniLM-L6) on the uncertain band | 0.979 |
| CE-L12 | bigger re-ranker (MiniLM-L12, wider hard-pair band, 3 epochs) | 0.980 |
| Ensemble | + multilingual re-ranker (mMiniLMv2-L12), weights fitted on held-out folds | 0.981 |
| **Final** | + two re-rankers fine-tuned on label-free **French pseudo-labels**, France re-scored | **0.982** |

---

## 1. The problem, and what the data told us

- Scale: train S1 2.2M / S2 5.0M / S3 5.3M records; test S1 1.73M / S2 4.9M / S3 5.1M. Blocking is mandatory.
- **Every S2/S3 record matches at most one S1** (0 of 7.6M matched ids shared) → the decision layer can enforce exclusivity.
- Matches never cross countries → blocking runs inside each country value. Country is treated as an **open set** (nothing hard-codes US/India).
- 5.6% of S1 have no match ("singletons"); ~3.46 matches per S1; ~26% of S2/S3 are pure distractors.
- The hard negatives are deliberate: an S1 name with an **inserted category word** ("Group", "Motors"...), or the same street with the **house number shifted** by a known set {1,2,3,4,5,7,9,11,13,21}.
- ~9% of S2 names are in **9 Indic scripts** (phonetic transliterations of the Latin S1 name).
- **France is 15% of test S1 (259k) and absent from train** — no labels, dense streets (72% of French S1 have a same-street sibling), 52% of names shared.

## 2. The pipeline (how we moved forward)

```
raw TSV → normalize → block (per country, bidirectional) → ~125 pair features
        → GBDT stack (LGB → LGB+XGB+CatBoost, isotonic) → uncertain-band cross-encoders
        → logistic blend (fit on held-out folds) → exclusivity + expected-F0.5 decision → matching_results.tsv
```

1. **Normalization** (`src/ber/normalize.py`, `prepare.py`, `translit.py`, `xscript.py`): legal forms, DBA aliases, domains/handles, leetspeak, junk prefixes, house-number variants. Indic scripts are romanized with a token dictionary **learned from the training pairs** (1,363 tokens), backed by a stdlib offset romanizer (we replaced a GPL dependency) and confident-only OOV resolution. France: a label-free learned region/département lookup, French word canon (groupe→group, associés→associates...), "13 B" → "13 bis".
2. **Blocking** (`blocking.py`): char-n-gram TF-IDF → SVD (name / address / both) kNN with FAISS in both directions, plus exact composite keys (house+street, name pairs, name×number, skeleton), then a **LightGBM pre-scorer** picks ~50 candidates per S1 under a global budget. Pair recall **0.9727 → 0.9951**, oracle ceiling of a perfect matcher on our candidates **0.9915 → 0.9986**.
3. **Features** (`features.py`, ~125): rapidfuzz batteries, IDF overlaps (per split and country), out-of-fold "inserted-token harm" target encoding, house-number arithmetic, competition features (second-best candidate, per-candidate rivals), cluster-support features.
4. **GBDT stack** (`models/gbdt.py`): S1-level 5-fold OOF, stage-1 LightGBM → context features → stage-2 LGB + XGB + CatBoost (weights 0.3 / 0.4 / 0.3), isotonic calibration. **OOF macro F0.5 0.9820 (v1) → 0.9899 (v2)**, precision 0.9974, recall 0.9757.
5. **Decision layer** (`decision.py`): each candidate goes only to its highest-scoring S1; per S1 we emit the prefix of the ranked list with the highest **exact expected F0.5** (empty set included). Grid-tuned on OOF (temperature 0.7, bias −0.5, missing-mass 0.05); extra −0.3 logit bias for the unseen country.
6. **Label-free prior-shift correction** (`shift.py`): test shards had 2.1× the near-twin density of train (France 1.45 vs 0.69 pairs/S1), so odds are rescaled per (house-shift, extra-token) stratum.
7. **Cross-encoder re-ranking** (`scripts/build_ce_data.py`, `scripts/kaggle/`, `scripts/ce_blend.py`): only the uncertain band (GBDT p ∈ [0.003, 0.997], 4.7M test pairs) is re-read by fine-tuned transformer cross-encoders (Apache-2.0, ≤120M params) on free Kaggle GPUs, and blended with the GBDT logit by a logistic regression fitted on **held-out folds** (held-out gain +0.0007 → +0.0016, which converted to ~+0.002 → +0.005 on the leaderboard because test has **3.6× more uncertain pairs** than train).
8. **France pseudo-labels** (`scripts/final_france_blend.py`): with no French labels, we fine-tuned re-rankers on French pairs where two independent pipelines agreed (both link = positive, both reject = negative) and re-scored France with them (+0.001 LB).

## 3. Findings

**What moved the score (measured):** blocking v2 (+recall), the expected-F0.5 decision with exclusivity, learned transliteration, OOF-safe target encodings, cross-encoder re-ranking (each larger / more diverse model helped: +0.002, +0.001, +0.001), France pseudo-label adaptation (+0.001).

**What we tested and refuted (so you don't have to):**
- Hand-written pruning rules (empty-address rule, common-name penalty): no gain — the calibrated model already carries the signal.
- Cluster-aware decision rule: +0.00015 only.
- Extra prior-shift correction for candidates with no parsed state: changed 0.1–0.2% of links.
- Re-tuning thresholds on the blended probabilities: −0.0001 cross-fitted.
- A third, bigger re-ranker (`bge-reranker-base`, 278M): blend weight 0.04, no gain over the two-model ensemble.
- Global bias sweeps: macro F0.5 is flat within ±0.0002 over a wide range — the loss is **swapped links on look-alike pairs**, not threshold volume.

**Leaderboard forensics.** Probing the leaderboard with the predictions of a second, independent pipeline shared with us (its France rows emptied → 0.840) showed France ≈ 15% of the public score, with that pipeline's US/India ≈ 0.978 and France ≈ 0.937. Those files are not part of this repo. Our volumes (links/S1, empty rate) matched theirs almost exactly, confirming the remaining error is hard, semantic disambiguation rather than calibration.

## 4. Why offline 0.990 became 0.977 → 0.982 online

Submission A's offline number (OOF macro F0.5 **0.9899**) overstated its leaderboard score (0.977) by ~0.013, and the one other pipeline we could compare showed the same pattern (holdout 0.9896 → LB 0.972). Even with the re-rankers, held-out 0.9915 → LB 0.982. Causes we could identify: (1) test is far denser than any sample we trained on (3.6× more uncertain pairs, 2.1× near-twin density), (2) France is unseen and has no labels, (3) our dev sample dropped 92% of unmatched no-state candidates while every test shard carries the whole country pool. **If we started over:** build a density-matched training world first, start the GPU re-ranker on day 1 (it gave five consecutive gains), and pseudo-label the unseen country early.

## 5. Engineering notes (bugs that cost hours)

- France street key: the "first street token" was the road type ("rue"), so only 15% of blocks were usable; skipping road types raised usable blocks to 92%.
- India sharding OOM: shards were balanced on S1 count, but candidate volume per state varies 2× → one shard had 2.7M candidates and died in a pandas consolidation. Fix: balance on candidate count, account for cross-state link duplication, brute-force the last two groups (≈2.6M each), and exclude the empty-state label from bin-packing. Global exclusivity is re-enforced at merge (dropped ~46–51k cross-shard duplicate links).
- Windows: "free physical memory" is misleading (disk cache); watch commit headroom. Kaggle CLI needs relative paths on Windows and datasets mount at nested paths (`glob(..., recursive=True)`); free tier allows 2 GPU sessions at once.
- Heredocs silently corrupted regex backslashes twice — write regex code with a file tool.

## 6. Reproducing

```bash
python -m venv venv && source venv/bin/activate && pip install -r requirements.txt
# put the challenge data under the paths in configs/local_final.yaml (paths are Windows-style and hard-coded; edit them)
bash scripts/run_local2.sh configs/local_final.yaml          # blocking → features → GBDT stack → predict → finalize (validator-gated)
python scripts/build_ce_data.py                               # export the uncertain band for the re-rankers
#   train/score the re-rankers on a GPU with scripts/kaggle/*.py, then:
python scripts/ce_blend.py --config configs/local_final_tuned.yaml --mode eval  --ce-dir <dirs,comma-separated>
python scripts/ce_blend.py --config configs/local_final_tuned.yaml --mode apply --ce-dir <dirs,comma-separated>
python scripts/make_submission.py --outputs <final dir> --team Elyptics --validator <validate_submission.py> --test-dir <test dir>
pytest tests
```

The challenge data, trained models and submissions are **not** in this repo. See `docs/PIPELINE_REFERENCE.md` (module-by-module reference) and `docs/HANDOFF_NOTES.md` (the running engineering log with every measurement).

## 7. Rules followed

Only the provided data (no lookups, geocoding or external datasets); all models Apache-2.0/MIT and ≤ 8B parameters (largest model in the final submission: ~118M); country treated as an open set; every test S1 gets exactly one row; matches ⊂ candidates (official validator: PASS).
