#!/usr/bin/env bash
# Unattended, resumable local pipeline (laptop, CPU) -- risk-first order:
#   train features -> ALL test shards prepare/block/features (never-run code path first) -> tune -> train -> predict -> finalize
# Done-markers in $WORK/.done_<step>; rerun = resume. usage: bash scripts/run_local2.sh [config]
set -eo pipefail
CFG=${1:-configs/local_final.yaml}
PY=${PY:-/d/er_env/Scripts/python.exe}
cd "$(dirname "$0")/.."
ROOT=$PWD
WORK=$($PY -c "import yaml,sys; print(yaml.safe_load(open(sys.argv[1]))['work_dir'])" "$CFG")
LOG=$WORK/logs; mkdir -p "$LOG"
export PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1
run () {  # run <marker> <cmd...>
  local M="$WORK/.done_$1"; shift
  if [ -f "$M" ]; then echo "[$(date +%T)] skip $(basename $M)"; return 0; fi
  echo "[$(date +%T)] START $(basename $M): $*" | tee -a "$LOG/driver.log"
  (cd "$ROOT/src" && "$@") >> "$LOG/$(basename $M).log" 2>&1
  touch "$M"; echo "[$(date +%T)] DONE  $(basename $M)" | tee -a "$LOG/driver.log"
}
SHARDS=("test@france#0of3" "test@france#1of3" "test@france#2of3" "test@us#0of3" "test@us#1of3" "test@us#2of3" "test@india#0of8" "test@india#1of8" "test@india#2of8" "test@india#3of8" "test@india#4of8" "test@india#5of8" "test@india#6of8" "test@india#7of8")
C=../$CFG
run features_train   $PY -m ber.pipeline features --config $C --split train
for S in "${SHARDS[@]}"; do
  T=${S//[@#]/_}
  run prepare_$T   $PY -m ber.pipeline prepare  --config $C --split "$S"
  run block_$T     $PY -m ber.pipeline block    --config $C --split "$S"
  run features_$T  $PY -m ber.pipeline features --config $C --split "$S"
done
# stale OOF scores from an earlier candidate set must not feed the tuner / stage-2 (pairs changed after re-blocking)
if [ ! -f "$WORK/.done_tune_lgb" ] && [ -f "$WORK/train/oof.parquet" ] && [ "$WORK/train/oof.parquet" -ot "$WORK/.done_features_train" ]; then
  mv "$WORK/train/oof.parquet" "$WORK/train/oof_prev.parquet"; echo "archived stale oof.parquet" | tee -a "$LOG/driver.log"
fi
run tune_lgb         $PY -m ber.models.tune --config $C --kind lgb --trials 30 --time-budget-min 60
TUNED=${CFG%.yaml}_tuned.yaml
$PY - "$CFG" "$TUNED" <<'PYEOF'
import sys, os, yaml
c = yaml.safe_load(open(sys.argv[1], encoding="utf-8"))
bp = os.path.join(c["artifacts_dir"], "best_params_lgb.yaml")
if os.path.exists(bp):
    best = yaml.safe_load(open(bp, encoding="utf-8")) or {}
    keep = {k: v for k, v in best.items() if k not in ("num_threads", "n_jobs")}
    c["gbdt_params"]["lgb"].update(keep)
    print("merged tuned lgb params:", keep)
yaml.safe_dump(c, open(sys.argv[2], "w", encoding="utf-8"), sort_keys=False)
PYEOF
C=../$TUNED
run train_stack      $PY -m ber.pipeline train --config $C
for S in "${SHARDS[@]}"; do
  T=${S//[@#]/_}
  run predict_$T   $PY -m ber.pipeline predict  --config $C --split "$S"
done
run finalize         $PY -m ber.pipeline finalize --config $C
# keep a validated safety copy of this submission
SNAP=$WORK/output/submission_A_$(date +%Y%m%d_%H%M)
mkdir -p "$SNAP" && cp "$WORK/output/final/"*.tsv "$SNAP/" && echo "snapshot -> $SNAP" | tee -a "$LOG/driver.log"
echo "[$(date +%T)] PIPELINE COMPLETE" | tee -a "$LOG/driver.log"
