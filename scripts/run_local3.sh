#!/usr/bin/env bash
# Unattended, resumable local pipeline -- EARLY-TRAIN order (same done-markers as run_local2.sh, so finished steps are skipped):
#   shards A (France x3, US x3, India #0-#1) -> train -> predict shards A (early LB probes possible with hybrid files)
#   -> shards B (India #2-#3) prepare/block/features -> predict shards B -> finalize -> snapshot
# usage: bash scripts/run_local3.sh [config]
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
SHARDS_A=("test@france#0of3" "test@france#1of3" "test@france#2of3" "test@us#0of3" "test@us#1of3" "test@us#2of3" "test@india#0of4" "test@india#1of4")
SHARDS_B=("test@india#2of4" "test@india#3of4")
C=../$CFG
build () {  # build <shard>
  local T=${1//[@#]/_}
  run prepare_$T   $PY -m ber.pipeline prepare  --config $C --split "$1"
  run block_$T     $PY -m ber.pipeline block    --config $C --split "$1"
  run features_$T  $PY -m ber.pipeline features --config $C --split "$1"
}
run features_train   $PY -m ber.pipeline features --config $C --split train
for S in "${SHARDS_A[@]}"; do build "$S"; done
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
print("exclude_features:", c.get("exclude_features"), "| decision_unseen_bias:", c.get("decision_unseen_bias"))
yaml.safe_dump(c, open(sys.argv[2], "w", encoding="utf-8"), sort_keys=False)
PYEOF
C=../$TUNED
run train_stack      $PY -m ber.pipeline train --config $C
for S in "${SHARDS_A[@]}"; do T=${S//[@#]/_}; run predict_$T $PY -m ber.pipeline predict --config $C --split "$S"; done
echo "[$(date +%T)] EARLY PREDICTIONS READY (France, US, India #0-#1)" | tee -a "$LOG/driver.log"
C=../$CFG
for S in "${SHARDS_B[@]}"; do build "$S"; done
C=../$TUNED
for S in "${SHARDS_B[@]}"; do T=${S//[@#]/_}; run predict_$T $PY -m ber.pipeline predict --config $C --split "$S"; done
run finalize         $PY -m ber.pipeline finalize --config $C
SNAP=$WORK/output/submission_A_$(date +%Y%m%d_%H%M)
mkdir -p "$SNAP" && cp "$WORK/output/final/"*.tsv "$SNAP/" && echo "snapshot -> $SNAP" | tee -a "$LOG/driver.log"
# unattended packaging: fill Documentation.md placeholders from the reports, then the validator-gated zip
(cd "$ROOT" && $PY scripts/fill_doc.py --config "$TUNED") >> "$LOG/package.log" 2>&1 \
  || echo "[$(date +%T)] fill_doc failed (see package.log)" | tee -a "$LOG/driver.log"
if (cd "$ROOT" && $PY scripts/make_submission.py --outputs "$WORK/output/final" --team Elyptics \
      --validator D:/er_data/student_resource/utils/validate_submission.py --test-dir D:/er_data/student_resource/dataset/test) >> "$LOG/package.log" 2>&1; then
  echo "[$(date +%T)] ZIP BUILT: $(dirname "$ROOT")/Elyptics_submission.zip" | tee -a "$LOG/driver.log"
else
  echo "[$(date +%T)] make_submission FAILED (see package.log)" | tee -a "$LOG/driver.log"
fi
echo "[$(date +%T)] PIPELINE COMPLETE" | tee -a "$LOG/driver.log"
