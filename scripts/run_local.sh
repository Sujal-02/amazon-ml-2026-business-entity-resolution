#!/usr/bin/env bash
# Unattended, resumable local pipeline (laptop, CPU). Done-markers in $WORK/.done_<step>; rerun = resume.
# usage: bash scripts/run_local.sh [config]
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
C=../$CFG
run features_train   $PY -m ber.pipeline features --config $C --split train
# stale OOF scores from an earlier candidate set must not feed the tuner / stage-2 (pairs changed after re-blocking)
if [ ! -f "$WORK/.done_tune_lgb" ] && [ -f "$WORK/train/oof.parquet" ] && [ "$WORK/train/oof.parquet" -ot "$WORK/.done_features_train" ]; then
  mv "$WORK/train/oof.parquet" "$WORK/train/oof_prev.parquet"; echo "archived stale oof.parquet" | tee -a "$LOG/driver.log"
fi
run tune_lgb        $PY -m ber.models.tune --config $C --kind lgb --trials 30 --time-budget-min 60
# merge tuned stage-2 LightGBM params (if the tuner produced them) into a derived config
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
for S in "test@france" "test@us#0of3" "test@us#1of3" "test@us#2of3" "test@india#0of4" "test@india#1of4" "test@india#2of4" "test@india#3of4"; do
  T=${S//[@#]/_}
  run prepare_$T   $PY -m ber.pipeline prepare  --config $C --split "$S"
  run block_$T     $PY -m ber.pipeline block    --config $C --split "$S"
  run features_$T  $PY -m ber.pipeline features --config $C --split "$S"
  run predict_$T   $PY -m ber.pipeline predict  --config $C --split "$S"
done
run finalize         $PY -m ber.pipeline finalize --config $C
echo "[$(date +%T)] PIPELINE COMPLETE" | tee -a "$LOG/driver.log"
