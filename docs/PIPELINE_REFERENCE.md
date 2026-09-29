# Business Entity Resolution — reproducible pipeline

Given the three noisy sources of the ML Challenge 2026, the pipeline links every Source-1 entity to all of its
Source-2 / Source-3 records and writes `output/matching_results.tsv` plus the exact candidate set the matcher
scored, `output/candidate_pairs.tsv`. It uses only the provided data and MIT / Apache-2.0 models of at most 8B
parameters. There is no external lookup: no geocoding, no registries, no external datasets.

```
data ─► prepare ─► block ─► features ─► GBDT stack ─► (bi-encoder re-block) ─► cross-encoder zoo ─► holdout stacker ─► decision ─► TSVs
```

| stage | module | what it does |
|---|---|---|
| prepare | `ber/prepare.py`, `normalize.py`, `translit.py`, `xscript.py` | Normalizes names and addresses: legal forms, DBA aliases, domains and handles, leetspeak, junk prefixes, abbreviations (US / India / French), house numbers, states. Romanizes 9 Indic scripts with a token dictionary **learned from the training pairs**, backed by a stdlib offset romanizer and closed-vocabulary resolution. |
| block | `ber/blocking.py` | Per-country, bidirectional kNN (S1→cand top-30, cand→S1 top-8) over char-TF-IDF→SVD vectors for name, address and both, plus an optional fine-tuned bi-encoder channel and exact composite keys. The union is capped per S1. |
| features | `ber/features.py` | About 110 country-agnostic pair features. Name and address rapidfuzz batteries on several views, IDF overlap with per-split IDF, an out-of-fold "extra-token harm" encoding, numeric and house-number agreement, blocking cosines. |
| GBDT stack | `ber/models/gbdt.py` | Stage-1 LightGBM, then competition and cluster-support features, then stage-2 LightGBM + XGBoost + CatBoost. S1-level 5-fold OOF, isotonic calibration. |
| deep zoo | `ber/models/biencoder.py`, `cross_encoder.py`, `llm_matcher.py` | multilingual-e5 bi-encoder (blocking). XLM-R-large / bge-reranker-v2-m3 / mDeBERTa-v3 / mmBERT cross-encoders with a numeric-feature head. Optional Qwen3-4B / Qwen2.5-7B LoRA on the uncertain band. |
| stacker | `ber/models/stack.py` | LightGBM over GBDT and deep scores, fit on the holdout S1 folds (deep models train on folds 0–2). |
| decision | `ber/decision.py` | Each S2/S3 record goes to at most one S1 (train fact: 0 of 7.6M ids shared). Per S1, the prefix with the highest **expected F0.5** is chosen (exact Poisson-binomial, empty set included). Tuned on OOF macro F0.5. |

## Quick start (AWS, full data)

```bash
# 1. g6e.12xlarge (4x L40S) or p4d.24xlarge, Deep Learning AMI (Ubuntu 22.04)
git clone <this repo> ~/ber && cd ~/ber
bash scripts/aws_setup.sh /path/to/6ab10eb3b23ba_student_resource.zip
source /data/venv/bin/activate
# 2. first submission: CPU pipeline + GBDT stack (~3-4 h)
bash scripts/run_all.sh configs/aws.yaml gbdt
# 3. full model zoo: bi-encoder re-blocking + cross-encoders + holdout stacker (+8-14 GPU-h on 4 GPUs)
bash scripts/run_all.sh configs/aws.yaml deep
```

Outputs land in `/data/work/output/`. Validation reports land in `/data/work/train/*.json`:
`blocking_report.json`, `oof_report_p.json` and `stack_report.json`.

## Step-by-step

```bash
export PYTHONPATH=$PWD/src
python -m ber.pipeline prepare  --config configs/aws.yaml --split train
python -m ber.pipeline block    --config configs/aws.yaml --split train
python -m ber.pipeline features --config configs/aws.yaml --split train
python -m ber.pipeline prepare  --config configs/aws.yaml --split test
python -m ber.pipeline block    --config configs/aws.yaml --split test
python -m ber.pipeline features --config configs/aws.yaml --split test
python -m ber.pipeline train    --config configs/aws.yaml
python -m ber.pipeline predict  --config configs/aws.yaml --split test
# optional deep zoo (each is independent; run one per GPU)
python -m ber.models.biencoder train --config configs/aws.yaml        # then re-run block/features/train/predict
python -m ber.models.cross_encoder train   --config configs/aws.yaml --name xlmr_large
python -m ber.models.cross_encoder predict --config configs/aws.yaml --name xlmr_large --split train
python -m ber.models.cross_encoder predict --config configs/aws.yaml --name xlmr_large --split test
python -m ber.models.llm_matcher  train    --config configs/aws.yaml --name qwen3_4b    # optional
python -m ber.models.stack train   --config configs/aws.yaml
python -m ber.models.stack predict --config configs/aws.yaml --split test
python ../student_resource/utils/validate_submission.py --matching /data/work/output/matching_results.tsv \
       --candidate /data/work/output/candidate_pairs.tsv --test-dir /data/student_resource/dataset/test
```

Local development on a geographic sample: `configs/local_dev.yaml` sets `sample.train: 0.08`, which keeps whole
states so local confuser density stays realistic. Unit tests: `pytest tests`.

## Model zoo (all licenses verified MIT / Apache-2.0, ≤ 8B params)

| id | model | params | license | role |
|---|---|---|---|---|
| lgb / xgb / cat | LightGBM / XGBoost / CatBoost | – | MIT / Apache-2.0 / Apache-2.0 | stage-1 + stage-2 matchers |
| biencoder | intfloat/multilingual-e5-base | 278M | MIT | dense blocking channel |
| xlmr_large | FacebookAI/xlm-roberta-large | 561M | MIT | cross-encoder |
| bge_reranker_m3 | BAAI/bge-reranker-v2-m3 | 568M | Apache-2.0 | cross-encoder |
| mdeberta | microsoft/mdeberta-v3-base | 278M | MIT | cross-encoder |
| mmbert | jhu-clsp/mmBERT-base | 307M | MIT | cross-encoder |
| xlmr_base | FacebookAI/xlm-roberta-base | 278M | MIT | fast cross-encoder |
| qwen3_4b | Qwen/Qwen3-4B-Instruct-2507 | 4.02B | Apache-2.0 | LoRA matcher, uncertain band |
| qwen25_7b | Qwen/Qwen2.5-7B-Instruct | 7.62B | Apache-2.0 | LoRA matcher, uncertain band |

Not used, and why: Qwen3-8B (8.19B params, over the cap), Jellyfish-7B/8B (CC-BY-NC), Llama/Gemma
(non-permissive licenses), Unidecode (GPL, replaced by our own `xscript.py` romanizer), libpostal (its model data
could count as external data).

## Compliance notes
* Country is an open set. Blocking runs per country *value*, no feature encodes the country label, and state
  agreement is NaN when a state cannot be recognized (France).
* IDF tables, TF-IDF/SVD encoders and OOV transliteration resolution are fit on each split's own unlabeled records.
* The transliteration dictionary and "extra-token harm" encodings are learned from train labels only. The
  encodings are out-of-fold during training.
