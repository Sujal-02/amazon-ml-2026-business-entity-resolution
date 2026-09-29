#!/usr/bin/env bash
# Resumable, checkpointed step runner for remote workers (AWS via SSM/tmux, Kaggle).
# usage: bash scripts/worker_run.sh <config> <split> <step> [<step> ...]
#   steps: prepare block features predict  (ber.pipeline steps) | any "module:cmd" e.g. "stack:predict"
# Each finished step writes /data/work/<split>/.done_<step>; reruns skip finished steps (recovery after failures).
# After every step the work dir + logs are synced to S3 (checkpoint), so a fresh box can resume via s3 sync.
set -eo pipefail
CFG=$1; SPLIT=$2; shift 2
S3=${S3_PREFIX:-s3://ber-ml-170848441936-aps1/worker}
WORK=${WORK_DIR:-/data/work}
export PATH=/snap/bin:$PATH
[ -f /data/venv/bin/activate ] && source /data/venv/bin/activate
cd "$(dirname "$0")/.."
export PYTHONPATH=$PWD/src
mkdir -p "$WORK/logs" "$WORK/$SPLIT"
# resume: pull any checkpoint that exists in S3 but not locally
aws s3 sync "$S3/work/$SPLIT" "$WORK/$SPLIT" --only-show-errors || true
aws s3 sync "$S3/artifacts" "$WORK/artifacts" --only-show-errors || true
for STEP in "$@"; do
  MARK="$WORK/$SPLIT/.done_${STEP//:/_}"
  if [ -f "$MARK" ]; then echo "[$(date +%T)] skip $STEP (done)"; continue; fi
  echo "[$(date +%T)] start $STEP" | tee -a "$WORK/logs/progress.log"
  if [[ "$STEP" == *:* ]]; then
    MOD=${STEP%%:*}; CMD=${STEP##*:}
    python -m "ber.models.$MOD" "$CMD" --config "$CFG" --split "$SPLIT" 2>&1 | tee -a "$WORK/logs/${SPLIT}_${MOD}_${CMD}.log"
  else
    python -m ber.pipeline "$STEP" --config "$CFG" --split "$SPLIT" 2>&1 | tee -a "$WORK/logs/${SPLIT}_${STEP}.log"
  fi
  touch "$MARK"
  echo "[$(date +%T)] done $STEP" | tee -a "$WORK/logs/progress.log"
  aws s3 sync "$WORK/$SPLIT" "$S3/work/$SPLIT" --only-show-errors
  aws s3 sync "$WORK/logs" "$S3/logs" --only-show-errors
  [ -d "$WORK/output" ] && aws s3 sync "$WORK/output" "$S3/output" --only-show-errors
done
echo "[$(date +%T)] ALL DONE $*" | tee -a "$WORK/logs/progress.log"
aws s3 sync "$WORK/logs" "$S3/logs" --only-show-errors
