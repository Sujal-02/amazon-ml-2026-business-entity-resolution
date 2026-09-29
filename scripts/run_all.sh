#!/usr/bin/env bash
# End-to-end reproduction: data -> blocking -> matching -> output/matching_results.tsv + candidate_pairs.tsv
# Usage: bash scripts/run_all.sh [configs/aws.yaml] [phase]
#   phase = gbdt   : CPU pipeline + GBDT stack only (first leaderboard submission, ~3-4 h on g6e.12xlarge)
#   phase = deep   : bi-encoder re-blocking + cross-encoder zoo + holdout stacker (adds ~8-14 GPU-hours on 4 GPUs)
#   phase = all    : both (default)
set -euo pipefail
CFG=${1:-configs/aws.yaml}
PHASE=${2:-all}
cd "$(dirname "$0")/.."
[ -f /data/venv/bin/activate ] && source /data/venv/bin/activate
export PYTHONPATH=$PWD/src
set -o pipefail
PY="python -m"
LOG=${LOG_DIR:-/data/work/logs}; mkdir -p "$LOG"

gbdt_phase () {
  for SPLIT in train test; do
    $PY ber.pipeline prepare  --config "$CFG" --split $SPLIT 2>&1 | tee "$LOG/prepare_$SPLIT.log"
    $PY ber.pipeline block    --config "$CFG" --split $SPLIT 2>&1 | tee "$LOG/block_$SPLIT.log"
    $PY ber.pipeline features --config "$CFG" --split $SPLIT 2>&1 | tee "$LOG/features_$SPLIT.log"
  done
  $PY ber.pipeline train   --config "$CFG"              2>&1 | tee "$LOG/train_gbdt.log"
  $PY ber.pipeline predict --config "$CFG" --split test 2>&1 | tee "$LOG/predict_gbdt.log"
  cp /data/work/output/matching_results.tsv /data/work/output/matching_results.gbdt.tsv
}

deep_phase () {
  # 1) dense bi-encoder -> re-block both splits with the extra dense channel -> refeaturise -> retrain GBDT
  $PY ber.models.biencoder train --config "$CFG" 2>&1 | tee "$LOG/biencoder.log"
  for SPLIT in train test; do
    $PY ber.pipeline block    --config "$CFG" --split $SPLIT 2>&1 | tee "$LOG/block2_$SPLIT.log"
    $PY ber.pipeline features --config "$CFG" --split $SPLIT 2>&1 | tee "$LOG/features2_$SPLIT.log"
  done
  $PY ber.pipeline train   --config "$CFG"              2>&1 | tee "$LOG/train_gbdt2.log"
  $PY ber.pipeline predict --config "$CFG" --split test 2>&1 | tee "$LOG/predict_gbdt2.log"
  # 2) cross-encoder zoo, one model per GPU in parallel
  MODELS=(${CE_MODELS:-xlmr_large bge_reranker_m3 mdeberta mmbert})
  i=0
  for M in "${MODELS[@]}"; do
    ( export CUDA_VISIBLE_DEVICES=$((i % ${N_GPUS:-4}))
      $PY ber.models.cross_encoder train   --config "$CFG" --name "$M"                && \
      $PY ber.models.cross_encoder predict --config "$CFG" --name "$M" --split train  && \
      $PY ber.models.cross_encoder predict --config "$CFG" --name "$M" --split test ) > "$LOG/ce_$M.log" 2>&1 &
    i=$((i + 1))
  done
  wait
  # 3) holdout stacker over GBDT + all cross-encoder scores -> final files
  $PY ber.models.stack train   --config "$CFG"              2>&1 | tee "$LOG/stack_train.log"
  $PY ber.models.stack predict --config "$CFG" --split test 2>&1 | tee "$LOG/stack_predict.log"
}

case $PHASE in
  gbdt) gbdt_phase ;;
  deep) deep_phase ;;
  all)  gbdt_phase; deep_phase ;;
esac
python /data/student_resource/utils/validate_submission.py \
  --matching /data/work/output/matching_results.tsv \
  --candidate /data/work/output/candidate_pairs.tsv \
  --test-dir /data/student_resource/dataset/test
