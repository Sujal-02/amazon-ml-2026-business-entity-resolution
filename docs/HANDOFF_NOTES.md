# HANDOFF: state of the Elyptics ER pipeline (written Sat 26 Sep 2026 ~16:00 IST)

The **final deadline is Sun 27 Sep 23:59 IST** (leaderboard closes; max 5 uploads/day), and this is the **last submission**. Team **Elyptics** (Sujal
Kaware, Harshit Verma, Shivansh Choudhary, Mridul Mittal).

## 0. USER RUNBOOK: no LLM needed (Sun 27 Sep)
Everything below runs by itself. Run the commands in Git Bash.
1. **Progress:** `tail -5 D:/er_data/work_dev/logs/driver.log`. `driver.pid` holds the driver PID.
   - About 03:40: after the India #1 features, the watcher `D:/er_data/work_dev/logs/ctl/switch_to_run3.sh` logs "switched to run_local3.sh".
   - About 05:40: train done. The OOF report is `D:/er_data/work_dev/train/oof_report_p.json`, key `macro_f05`.
   - About 06:40: "EARLY PREDICTIONS READY" means France, US and India #0–1 are predicted.
   - About 11:30: finalize runs, then "ZIP BUILT", then "PIPELINE COMPLETE".
     - Upload file: `D:/er_data/work_dev/output/final/matching_results.tsv`. The validator result is in `logs/.done_finalize.log`.
     - Zip: `C:/Users/rakes/OneDrive/Desktop/ML/ML Challenge/Elyptics_submission.zip`.
2. **If the driver died** (no new log lines, and `ps -p $(cat D:/er_data/work_dev/logs/driver.pid)` fails):
   read the traceback at the end of the newest `D:/er_data/work_dev/logs/.done_*.log`, then relaunch. It resumes and skips finished steps:
   `cd "C:/Users/rakes/OneDrive/Desktop/ML/ML Challenge/business_entity_resolution" && nohup bash scripts/run_local2.sh configs/local_final.yaml >> D:/er_data/work_dev/logs/driver_stdout.log 2>&1 & echo $! > D:/er_data/work_dev/logs/driver.pid`
   (Use `run_local2.sh`, not `run_local3.sh` — the early-train reorder was abandoned, see §3g.)
3. **Optional early leaderboard probes** (after "EARLY PREDICTIONS READY"). Measurements only; never the final file.
   - P1, our France + v7 US/India: `/d/er_env/Scripts/python.exe scripts/build_probe.py --out D:/er_data/work_dev/output/probes/P1.tsv --france ours --us ref --india ref`
     Then F_France(ours) ≈ 0.937 + (LB(P1) − 0.972) / 0.1498.
   - P2, our US + v7 France/India: `... --out .../P2.tsv --france ref --us ours --india ref`. Then F_US(ours) − F_US(v7) ≈ (LB(P2) − 0.972) / 0.383.
4. **France bias** (only if P1 shows our France is below 0.937). France volume per bias, compared with v7:
   `/d/er_env/Scripts/python.exe scripts/redecide.py --config configs/local_final_tuned.yaml --splits "test@france*" --sweep -0.7 -0.3 0 0.3 --reference D:/er_data/ref/other_v7_0972_matching_results.tsv`
   To apply a bias B: `--apply B`. Then re-run finalize and the zip:
   `cd src && /d/er_env/Scripts/python.exe -m ber.pipeline finalize --config ../configs/local_final_tuned.yaml && cd .. && /d/er_env/Scripts/python.exe scripts/make_submission.py --outputs D:/er_data/work_dev/output/final --team Elyptics --validator D:/er_data/student_resource/utils/validate_submission.py --test-dir D:/er_data/student_resource/dataset/test`
5. **Fallback:** the first finalized output is also copied to `D:/er_data/work_dev/output/submission_A_<timestamp>/`.
   Documentation placeholders are filled automatically (`Documentation.template.md` is the unfilled backup). Add the LB score later with `scripts/fill_doc.py --lb <score>`.

## 1. What is running unattended (survives Claude session limits)
- Driver: `bash scripts/run_local2.sh configs/local_final.yaml` (nohup, bash PID in `D:/er_data/work_dev/logs/driver.pid`).
  - Progress: `D:/er_data/work_dev/logs/driver.log` (START/DONE per step). Step logs:
    `D:/er_data/work_dev/logs/.done_<step>.log`. Done-markers: `D:/er_data/work_dev/.done_<step>`.
  - Order: features_train ✅ → for 10 test shards [france#0-2of3, us#0-2of3, india#0-3of4]: prepare → block →
    features → tune_lgb (Optuna 60 min) → train_stack (LGB/XGB/CAT stage 2 + OOF blend weights + isotonic + decision
    grid) → predict ×10 → finalize → snapshot `output/submission_A_<ts>`.
  - ETA: test shards ~22:40 Sat, tune ~23:40, train ~02:50 Sun, **submission A ~03:50 Sun** in
    `D:/er_data/work_dev/output/final/{matching_results,candidate_pairs}.tsv` (validator run by finalize, see
    `.done_finalize.log`).
- **Resume after any crash:** re-run the same command. It skips finished steps. If a python step crashed,
  read its `.done_<step>.log` traceback, fix, and relaunch:
  `cd "C:/Users/rakes/OneDrive/Desktop/ML/ML Challenge/business_entity_resolution" && nohup bash scripts/run_local2.sh configs/local_final.yaml >> D:/er_data/work_dev/logs/driver_stdout.log 2>&1 &`
  then write the new bash PID into `driver.pid`. **Never run two memory-heavy jobs at once** (16 GB laptop; OOM twice).

## 1b. v2 final OOF (Sun 27 Sep 15:39, after the India fixes): **macro F0.5 = 0.9899** (India 0.9893, US 0.9904),
precision 0.9974 / recall 0.9757. Decision grid: temperature 0.7, bias **−0.5** (OOF picked a conservative bias on its
own), missing_mass 0.05. Up from v1's 0.9820. Same caveat as always: dev-sample OOF, no France, likely optimistic
vs the real leaderboard (see §3e/3f/3g — the other pipeline's holdout was 0.9896 but LB came in at 0.951-0.972).
Predict started 15:39 (France #0 first); 14 shards total.

## 2. Key results so far
- v1 (old blocking): OOF macro F0.5 **0.9820** (US 0.9836, India 0.9801), micro precision 0.997, recall 0.953. Loss
  by type: partial recall 0.0113, missed-all 0.0035, extra FP 0.0022, singleton FP 0.0010.
- Blocking v2 (LightGBM pre-scorer + budget + name_pair/name_x_num keys + IVF): pair recall **0.9951** (was 0.9727),
  entity recall 0.9835, 50 candidates per S1.
- v2 OOF will be in `D:/er_data/work_dev/train/oof_report_p.json` after train_stack.
- Other team (LB 0.951 → 0.963 with a hand-set near-twin penalty; leader 0.99): their file is at
  `D:/er_data/ref/other_v6_matching_results.tsv` (empty 5.7–6.3%, 3.27–3.40 links per S1, the same in every
  country). Their "test has 4–7× more near-twins" claim is partly an artefact of their distractor-subsampled
  training set. Our whole-state dev sample already has 0.86 (US) / 0.50 (India) near-twin pairs per S1.

## 3. Built-in safety and corrections (already in code)
- `ber/shift.py` + `gbdt.predict_stack`: a label-free per-stratum prior correction (house shift d, extra-token
  flag) using `artifacts/strata_train.json`. λ is about 1 unless the test shard is enriched. It logs `shift <split>:`
  lines and writes `work/<split>/shift_lambda.json`.
  - Main output = corrected. `matching_results_nocorr.tsv` = uncorrected (merge it via `finalize.variant: nocorr`).
- Unseen-country guard: `decision_unseen_bias: -0.3` (France; was −0.7, see 3d). Calibrate it with `scripts/loco.py` (US→India).
- Global exclusivity across state sub-shards in `pipeline.step_finalize`.

## 3b. Updates Sat 26 Sep evening (FRANCE-FOCUS: intel says the leaderboard is weighted towards France)
- Tuning is skipped (`.done_tune_lgb` touched) and CatBoost uses 800 iterations. Revised ETA for submission A is about 06:00 Sun.
- **France admin lookup** (`prepare.learn_admin_lookup`, used in `pipeline.step_prepare` for countries without a
  state list). It learns 3 regions and 4 départements label-free (Gironde→NA, Loire-Atlantique→PdL,
  Nord/Pas-de-Calais→HdF). These are stripped from the address and compared as `addr_state`, exactly like US/India
  states. Before the fix, strong French pairs had address IDF-Jaccard 0.926 and LSA cosine 0.878 (train positives 1.0),
  so true French matches were under-scored.
- France ×3 shards are being RE-RUN with this fix (markers and `artifacts/lsa_test_france_*.pkl` deleted
  automatically when US#1 features finished). The driver order is now France×3 (redo) → US#2 → India×4 → train →
  predict → finalize.
- The France #0 test near-twin density is 1.383 per S1 vs 0.691 in train (pooled). The enrichment sits in in-set shifts
  (≈1.8×) and "+extra token" strata (≈2.1×); out-of-set shifts are not enriched, so λ is about 0.43–0.53 there. This is
  handled automatically by the shift correction at predict.
- The first clue-hunt swarm failed entirely (session limit). A second compact swarm (Sonnet) was launched at 19:25: decision rule,
  empty-address/singletons, France residual gaps. Findings are persisted to `C:/Users/rakes/AppData/Local/Temp/claude/clues2/<key>/findings.md`;
  journal: `.../subagents/workflows/wf_ae4aeebe-d5b/journal.jsonl`.
- French name canon added (`normalize.NAME_WORD_CANON`: groupe→group, holding→holdings, associés→associates, fils→sons, cie→co, ...).


## 3c. Leaderboard feedback from the other team (Sat 20:10)
- Their 0.963 file: `D:/er_data/ref/other_v6_matching_results.tsv`. Their **0.972** file: `D:/er_data/ref/other_v7_0972_matching_results.tsv`.
  Going from 0.963 to 0.972 meant MORE links (France 3.29→3.35 per S1, India 3.27→3.37) and fewer empties (≈5.6%), i.e. less conservative.
  Their hand penalty over-pruned.
- Their handover (`D:/er_data/ref/HANDOVER_density_fix.md`) confirms the root cause was a random S1 subsample that lost sibling
  density. French S1: 71.6% have a same-street sibling with an in-set house shift (259k S1 on 38.8k streets).
- **TODO after submission A: label-free volume calibration.** Choose France's `decision_unseen_bias` so France's predicted
  links/S1 and empty rate match the ratio our model reaches on US/India (target ≈3.3–3.4 links/S1, ≈5.6–6% empty),
  and compare our per-country volume against their 0.972 file.

## 3d. Swarm-2 fixes applied Sat ~20:30 (journal `.../subagents/workflows/wf_ae4aeebe-d5b/journal.jsonl`)
- **France street-key defect:** `street_first` was the road type ("rue", "avenue"), so only 15.1% of French blocks were usable
  (US 90.1%). `pipeline.add_key_columns` now skips `normalize.ROAD_TYPES` for open-set countries, raising usable blocks to 91.9%.
  US/India keys are unchanged, so train is unaffected and no retrain is needed.
- **French bis/ter:** in `normalize.norm_addr` (open-set countries only), "13 B rue X" becomes "13 bis rue x" and "5 T" becomes "5 ter".
- `decision_unseen_bias` is now **−0.3** (swarm-measured best; −0.7 cost about 0.0004). The `_tuned` config inherits it at merge.
- France #2 picks both fixes up directly. A watcher (`scratchpad/relaunch_fr.sh`) stops the driver after France #2 features,
  clears the France #0/#1 markers, relaunches, and writes the new PID to `driver.pid`. Order: France #0, #1 redo → US #2 → India ×4 → train …
  This adds about 1 h, so the new ETA for submission A is about 07:00 Sun.
  **Done 21:44:** all 3 France shards were re-processed with both fixes; keys verified (e.g. `20|prmntr` for rue Parmentier).
  The new driver PID is 20573 and US #2 started at 21:44.
- Refuted by the swarm (do NOT apply): an empty-address rule and a common-name penalty. Cluster-aware decision is worth only about +0.00015 (optional).

## 3e. Leaderboard probes from the other pipeline (Sat 22:10) and the gap hunt
- Files in `D:/er_data/ref/`: `probeA_v6_usin092_fr095_0973.tsv` (LB **0.973**) and `probeB_v6_france_empty_0840.tsv` (LB **0.840**),
  alongside v6 (0.963) and v7 (0.972).
- Arithmetic (w = France share of S1 = 0.1498): probe B gives US/India F0.5 ≈ **0.978** for its base, although their holdout said 0.9896.
  France F0.5 is ≈ 0.877 if the base is v6, or 0.937 if the base is v7. Feasibility forces w into [0.13, 0.17], so France is about 15% of the public LB, not the majority.
  The US/India holdout-to-LB gap costs as much as France does.
- Suspected shared cause: train/test density. Our geo sample keeps only 8% of the unmatched no-state distractors, but every test state-shard carries its
  country's whole no-state pool (US #2: 9.5 candidates per S1 vs about 4.6 in train). So the kNN ranks (`blk_rank_lsa_*`) and frequency and rank features shift.
- New hook: config `exclude_features: [...]` (exact names or `prefix*`) in `configs/local_final.yaml` drops columns from both stages
  (`gbdt.config_excluded`). Predict follows automatically. It must be set **before train_stack starts (~04:30 Sun)**; the `_tuned` config is regenerated from local_final.
- Gap-hunt workflow `wf_fbe670e4-41a` (5 investigators + verifiers + synthesis; journal under `.../subagents/workflows/wf_fbe670e4-41a/`;
  agent scratch `C:/Users/rakes/AppData/Local/Temp/claude/clues3/`). Heavy analysis goes through `bash D:/er_data/locks/run_heavy.sh <cmd>` (lock + RAM guard).
- Plan for Sunday's 5 uploads: A, then "A with France emptied" (or a hybrid with their US/India) to isolate France, then France-bias and threshold variants via `scripts/redecide.py`.

## 3g. Sun 01:40: India OOM crash, fixed (driver PID 26185, back on run_local2.sh)
- **Crash:** `test@india#0of4` block hit `MemoryError` (4.08 GiB alloc) in `blocking.merge_channels`. Root cause: `prepare.state_groups`
  bin-packed India's 4 state-shards by **S1 count only**, but candidate volume per state is very uneven (one group had 22.8 cand/S1
  vs Maharashtra's 10.4), so one shard got 2.73M candidates — well past what fits in 16 GB alongside pandas' consolidation overhead.
- **Fix** (`src/ber/prepare.py`, `state_groups`/`state_shard`, contained to those two functions, only caller): bin-pack by
  **candidate count** instead (`weight_labels` param), and India is now split **8-way** (`test@india#0of8`..`#7of8`, was `#Kof4`) —
  more groups didn't help beyond 8 (Maharashtra's own group floors at 1.83M candidates: its own volume + the 22.9% of India
  candidates with no parseable state, which duplicate into every group regardless of group count). Verified end-to-end against
  real data: all 8 groups now sit at 1.35M–1.84M candidates (in the range that already worked for US shards), S1 coverage sums
  correctly (809,985 of 809,986 — 1 pre-existing orphan record with no parseable state gets an empty prediction, negligible).
  Caught and fixed a second bug along the way: the empty-state label `""` was briefly being treated as an assignable "state"
  with the *entire* no-state candidate pool as its weight, which would have isolated it into its own degenerate shard — excluded
  explicitly now. All 13 unit tests pass. `state_links.json` (cross-state affinity) is keyed by `country|state`, so it's
  unaffected by the group-count change.
- **Abandoned** the `run_local3.sh` early-train reorder and the LB-probe workflow: per user request (token/time conservation,
  switched to Sonnet 5), we're back on the simpler, proven `run_local2.sh` order (shards → train → predict → finalize).
  Reordering doesn't save wall-clock anyway (jobs are strictly sequential on this machine), it only would have enabled earlier
  LB probing, which isn't the current priority. Ignore any stale `test@india#Kof4` references in `run_local3.sh`.
- Driver relaunched fresh: PID **26185**, resumed correctly at `india#0of8 prepare` (skipped all done France/US/train_features steps).
  New ETA: India ×8 shards done ~05:30, train ~08:00, predict+finalize+zip ~10:30 Sun.

## 3f. Sun 01:00: salvaged gap-hunt results, and the driver reorder
- The first gap-hunt run died at the session limit. Its partial outputs are in `C:/Users/rakes/AppData/Local/Temp/claude/clues3/`:
  - `probes/p2_pairwise.json`: **probe B = v7 with France emptied**, and **probe A = v7 minus 34.6k low-confidence US/India links**.
    So the other pipeline's v7 scores US/India ≈ 0.978 and France ≈ 0.937.
  - `shift/density_all.json`: near-twin candidates per S1 are 0.43 in train vs 0.89–1.01 in US test. The empty-address candidate share is 3.3% vs 6.4–7.0%.
    Candidates per S1: train 4.6, US test 7.2–7.9, France 5.9, India #0 **14**.
  - Adversarial AUC between train and test "obvious matches" is 0.95. The top shifted features are `nm_idf_min_s1` and `ad_state_eq` (the NaN comes from train-only state dropout),
    then `nm_c_samename_n`, `ad_extra_*` and `nm_idf_shared`. Ranks are not shifted.
  - `shift_check`: France λ is about 0.45 in the in-set strata. The bias grid on train is flat from −0.5 to +0.25.
- The continuation workflow is `wf_b88e0687-a44` (4 investigators + verifiers + synthesis). Each agent checkpoints to `clues3/<key>/PROGRESS.jsonl`.
- **Driver reorder:** a watcher (`scratchpad/switch_to_run3.sh`) stops `run_local2` when `.done_features_test_india_1of4` appears (about 03:40)
  and starts **`scripts/run_local3.sh`**. The new order is train → predict France/US/India #0–1 → India #2–3 build → predict → finalize.
  Early predictions are expected around 06:30 and submission A around 11:30. `exclude_features` in `local_final.yaml` must be decided before the switch.

## 4. NEXT STEPS (in order)
1. When submission A exists:
   - Run `python scripts/qa_submission.py --matching D:/er_data/work_dev/output/final/matching_results.tsv --candidates D:/er_data/work_dev/output/final/candidate_pairs.tsv --test-dir D:/er_data/student_resource/dataset/test --scores-glob "D:/er_data/work_dev/test@*/scores_gbdt.parquet"`.
   - Compare with the other team's file (links/S1, empty rate per country, agreement rate).
2. Read the v2 OOF report (`oof_report_p.json`, per-model `oof_report_p2_*.json`) and the shift logs (`grep "shift test"` in the `.done_predict_*.log` files).
3. `python scripts/loco.py --config configs/local_final_tuned.yaml --source us --target india` (≈40–60 min). Set
   `decision_unseen_bias` to the best bias, then re-run predict for the France shards only (delete
   `.done_predict_test_france_*`, `.done_finalize`) and relaunch the driver. It only predicts and finalizes.
4. Clue-hunt swarm results: workflow journal
   `C:/Users/rakes/.claude/projects/C--Users-rakes-OneDrive-Desktop-ML-ML-Challenge/9ff8afd4-e41e-425b-9dc5-7da6338db416/subagents/workflows/wf_3cd9a9d0-ae1/journal.jsonl`
   (one `{"type":"result"}` line per finished agent). Tasks: shift, generator, france, oof_errors,
   empty_singletons, decision + verifiers + synthesis. If the run died, re-run only the missing tasks (script in
   `.../workflows/scripts/er-clue-hunt-wf_3cd9a9d0-ae1.js`). Agent scratch: `C:/Users/rakes/AppData/Local/Temp/claude/clues/`.
5. Apply only verified improvements. Predict-level changes are cheap (~1 h). Feature changes need re-features for train
   and 10 shards (~5 h), retrain (~3 h) and predict (~1 h), so start them by Sun ~08:00 at the latest.
6. New helper scripts (tested Sat 21:00):
   - `scripts/redecide.py --config configs/local_final_tuned.yaml --splits "test@france*" --sweep -0.7 -0.5 -0.3 0 0.3 --reference D:/er_data/ref/other_v7_0972_matching_results.tsv`
     prints France volume per bias. `--apply B` rewrites the shard outputs; then run `cd src && python -m ber.pipeline finalize --config ../configs/local_final_tuned.yaml`.
   - `scripts/compare_ref.py --config ... --matching D:/er_data/work_dev/output/final/matching_results.tsv --reference D:/er_data/ref/other_v7_0972_matching_results.tsv --test-dir D:/er_data/student_resource/dataset/test`
     reports per-country links, empty rate, overlap and our p on disagreements.
   - `scripts/fill_doc.py --config configs/local_final_tuned.yaml --lb <score>` fills the Documentation.md placeholders.
     The oracle ceiling of blocking v2 is **0.9986** (v1 was 0.9915).
   - Team v7 (0.972) volume: France 3.35 links/S1 with 5.56% empty, India 3.37 / 5.66%, US 3.40 / 5.55%. That is almost exactly the train prior (3.46 / 5.58%).
7. Final: `python scripts/make_submission.py --outputs D:/er_data/work_dev/output/final --team Elyptics --validator D:/er_data/student_resource/utils/validate_submission.py --test-dir D:/er_data/student_resource/dataset/test`
   → `Elyptics_submission.zip` (validator-gated). Fill the `{{…}}` placeholders in `Documentation.md` from the reports first.

## 5. Environment
- venv `D:/er_env/Scripts/python.exe`; code `src/ber`; configs `configs/local_final.yaml` (+ `_tuned` after tuning).
- Data cache `D:/er_data/pq`; raw data `D:/er_data/student_resource`; work dir `D:/er_data/work_dev`.
- AWS: dropped (free plan; the worker instance is stopped). Kaggle CLI is authenticated but unused (GPU models skipped).
