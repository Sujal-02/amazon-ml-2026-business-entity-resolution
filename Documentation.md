# ML Challenge 2026: Business Entity Resolution Solution

**Team Name:** Elyptics
**Team Members:** Sujal Kaware, Harshit Verma, Shivansh Choudhary, Mridul Mittal
**Submission Date:** 27 Sep 2026
**Submitted configuration:** `gbdt` (either `gbdt`: GBDT stack only, or `deep`: GBDT stack plus the optional deep zoo and holdout stacker)

> `{{NAME}}` fields are filled in from the run reports listed in §5.1 after the final run. Figures marked
> **dev snapshot** were measured on the 8% geographic development sample of train (204,331 S1, 944,326 S2/S3
> records). They show where the pipeline stood while it was being developed. They are not the final scores.

---

## 1. Executive Summary

For each Source-1 (S1) entity, the pipeline finds all of its Source-2/3 records in four stages:
1. **Normalisation.** Rule-based name and address normalisation. This includes a transliteration dictionary for 9 Indic scripts that is learned from the training pairs.
2. **Blocking.** Per-country, bidirectional, multi-view kNN over character TF-IDF→SVD vectors, plus exact composite-key joins.
3. **Matching.** A two-stage GBDT stack: LightGBM, then LightGBM, XGBoost and CatBoost with competition and cluster-support features. The output is isotonic-calibrated.
4. **Decision.** A decision layer that uses two structural facts of the data. First, every S2/S3 record belongs to at most one S1, so each candidate is given only to its best S1. Second, the metric is per-entity F0.5 and a singleton scores only when its list is empty. So, for each S1, the layer emits the prefix of the probability-sorted candidate list (possibly empty) that has the highest exact expected F0.5.

Optional transformer members plug in through a holdout stacker: a bi-encoder retrieval channel, cross-encoders and a LoRA-tuned LLM. They are implemented but **not part of the submitted run**, which is the CPU-only GBDT stack (`gbdt` configuration), because no GPU was available. Validation OOF macro F0.5 is **0.9899**, at a blocking pair recall of **0.9951**.

---

## 2. Methodology

### 2.1 Problem Analysis

EDA of the training data and the unlabelled test data found the following.

| # | Observation | Evidence | Design consequence |
|---|---|---|---|
| 1 | Scale | train S1 2.2M / S2 5.0M / S3 5.3M; test S1 1.73M / S2 4.9M / S3 5.1M | Blocking is mandatory. Every stage is chunked and memory-lean so it runs on a 16 GB CPU machine (§2.2). |
| 2 | Match density | 7.64M matched pairs (US 4.58M, India 3.06M), about 3.5 per S1. Dev: mean 3.46, max 11. | Per-S1 lists are short. The subset selection (§4.4) caps the list length at 16. |
| 3 | Country always agrees | 100% of matched pairs have the same `country` value | Blocking runs inside each country value. The label is never used as a feature (open set). |
| 4 | **Exclusivity** | Each S2/S3 id matches **at most one** S1 (0 of 7.6M matched ids are shared) | The decision layer gives each candidate only to its best S1. The reverse kNN direction (candidate→S1) is a natural channel. The model gets competition features. |
| 5 | **Singletons** | 5.6% of S1 have no match. They score 1 only for an empty prediction. | "Emit nothing" is an explicit option in the expected-F0.5 decision. |
| 6 | **Distractors** | 26% of S2/S3 records match no S1. Many are an existing S1 name with an inserted **category word** ("Motors", "Stores", "Group", ...). | Insertion features, S1 document frequency of the inserted token, and an out-of-fold "extra-token harm" target encoding (§4.1). In the dev harm table, the most harmful inserted tokens are `group`, `holdings`, `public`, `hotel`, `steel`, `overseas`, `stores`, `motors` (harm logit ≈ 4.9–6.4). Alias and typo tokens (`fka`, `formerly`, `nee`, `lnc`) score below zero, i.e. benign. |
| 7 | Shared names | 30% of S1 share their exact name with another S1 (chains, generic names) | The address decides these. Same-name counts, within-candidate score margins and exclusivity handle chains. |
| 8 | **Indic scripts** | About 9% of S2 and 5% of S3 names (dev sample: 10.9% / 6.1%) use one of 9 scripts. Devanagari dominates, followed by Telugu, Kannada, Tamil, Bengali, Gujarati, Malayalam, Oriya and Gurmukhi. The names are word-for-word phonetic transliterations of the Latin S1 name. About 550k train pairs align. About 3.6% of the native tokens in test are unseen in train, mostly category words. | A dictionary learned from train pairs, a stdlib offset romaniser as fallback, and closed-vocabulary resolution of unseen (OOV) tokens (§2.3). Native state names are resolved at segment level. |
| 9 | Name replaced | Some S2/S3 records have a random pseudo-word as the name ("Lyraverasyn"). The address is exact. | Address-only kNN view, house+street and house+locality keys, cluster-support features. |
| 10 | Empty address | 3.3% of S2 and S3 records in the dev sample | A name-only kNN view and name keys. `ad_empty_c` gates the address features (NaN instead of 0). |
| 11 | Unseen country | France is 15% of test S1 and does not occur in train. S1 carries the region ("Nouvelle-Aquitaine"). S2/S3 often lack it or give other admin tokens. French abbreviations ("R." for Rue, "Bd") and legal forms (SARL, SAS, EURL, EI) appear as prefix or suffix, with accents. | Generic French street-type and legal-form rules and accent folding. For a country without a built-in state table, an admin lookup (regions plus départements, e.g. Gironde→Nouvelle-Aquitaine) is **learned label-free** from the S1 address segments. The admin tokens are then stripped and compared as `addr_state`, the same way US/India states are. For open-set countries, "13 B"/"5 T" become "13 bis"/"5 ter", and the street key skips the road type ("rue", "avenue"), so exact-key blocks stay specific. French name words are mapped to their train-side equivalents (groupe→group, associés→associates). `ad_state_eq` is NaN when no state is recognised, and train-time state dropout teaches the model that NaN. Source-specific admin tokens are removed label-free. IDF is computed per split and per country. |
| 12 | Ordering / IDs | Rows are not shuffled (they cluster geographically). ID numbers carry no signal: corr(S1 id, matched id) = 1.4e-4. | No row-position or ID feature. Dev sampling keeps whole states, not row ranges. |

**Name noise** seen in S2/S3 (S1 is clean, Latin and deduplicated):
* legal-form variants and bracketing ("[Inc.]", "Pvt"/"Private", "Incorporated");
* token shuffles ("Corp Devlin Beacon Emerging");
* leetspeak ("Beac0n", "hea1th");
* OCR-like typos ("Cioasirl");
* junk prefixes ("***", "--", "#");
* honorifics ("M/s", "Shri", "The");
* phone suffixes;
* domains and handles ("capitalexpresszeo.com", "@lornedevries");
* DBA / "doing business as" aliases with a random fake first name ("Ectoorbidrex DBA: Devlin Beacon Emerging Corp");
* dropped or added generic words ("Center", "Services").

**Address noise:**
* component reordering;
* street-type abbreviations;
* the state as a code, as a full name or in native script;
* zero-padded or suffixed house numbers ("013623", "905-");
* ordinal typos ("24nd");
* off-by-one house numbers (975 / 976);
* `null` / `N/A` tokens;
* PO Box / PMB;
* "Door No" / "H.NO" / "#" prefixes;
* garbage appended to locality names ("Clayton ITY");
* missing components.

### 2.2 Solution Strategy

**Approach Type:** Hybrid. The stages are:
1. multi-channel blocking;
2. a two-stage GBDT pair classifier with contextual features;
3. optionally, a transformer zoo combined through holdout stacking;
4. a constrained decision layer that optimises the metric directly.

**Core Innovation:**
1. **Decision layer that uses exclusivity and optimises expected F0.5 exactly** (§4.4). A candidate goes only to the S1 that scores it highest. Then, for each S1, the layer picks the prefix of the probability-sorted list that maximises the exact expected F0.5. TP and FN counts are computed as Poisson-binomial distributions, and blocking losses can be modelled as a Poisson term. The empty set is always one of the options. The decision's temperature, bias and mode are grid-searched on OOF macro F0.5.
2. **Features aimed at the distractors.** The generator builds hard negatives by inserting category words into an S1 name. Near-duplicate "superset" pairs get dedicated features: position of the inserted token, its S1 document frequency, a generic-word flag, and an out-of-fold harm encoding that learns *which* inserted tokens turn a record into a different business (§4.1).
3. **Transliteration learned from labels.** The token dictionary and the Latin spelling-variant canon are learned from the train pairs by positional alignment. At inference, native tokens unseen in train are resolved against the closed Latin vocabulary of the same split. This is transliteration mining rather than generation, and it is conservative: a token is resolved only when the match is confident (§2.3).
4. **Bidirectional blocking that follows exclusivity.** Every candidate always keeps its top-3 reverse neighbours (candidate→S1), so the true owner of a record survives the per-S1 cap (§3).
5. **Open-set country handling.** Everything runs per country *value*. Statistics are recomputed on each split without labels. Unknown states are NaN, and train applies state dropout. Admin tokens that appear only in one source are dropped without labels, so the French region vs département mismatch does not look like disagreement. At inference, two label-free guards protect an unseen country:
   * **prior-shift correction** (`ber/shift.py`): per near-twin stratum (house-number shift × extra-token flag), the negative density of the test shard is compared with train, and the odds are rescaled;
   * **unseen-country logit bias** (`decision_unseen_bias`): extra caution because thresholds tuned on US/India cannot be validated on France.

**Pipeline:**

```
raw TSV ─► prepare ───────────► block ─────────────────► features ─► GBDT stack ─────────► decision ─► TSVs
           normalise names/     per-country kNN          126 pair     stage-1 LGB             exclusivity +
           addresses, Indic     (name/addr/both,         features     stage-2 LGB+XGB+CAT     expected-F0.5
           romanisation         fwd+rev) + 5 exact                    isotonic calibration    prefix selection
                                keys, per-S1 cap
                                   ▲ (optional) dense bi-encoder channel
optional deep zoo: cross-encoders, LoRA LLM (uncertain band) ─► holdout stacker ─► decision ─► TSVs
```

**Validation protocol.**
* **Folds.** A single S1-level fold assignment, `crc32(s1_entity_id) % 5` (`features.fold_of`), is used everywhere: the OOF target encoding, GBDT CV, deep-model training folds and the stacker. A candidate always sits in the fold of its S1, so no S1 appears on both sides of a split.
* **Metric.** OOF macro F0.5 is computed over **all** S1 of the validation set with the challenge formula (`metrics.py`). This includes singletons and S1 that have no candidate.
* **Development sample.** Development runs use a geographic sample: whole states are kept (`prepare.geo_sample`, `sample.train: 0.08` gave 8 states). This keeps the local density of confusable records realistic. Kept S2/S3 records are the true matches of kept S1s, plus unmatched records from the kept states. Unmatched records with no parseable state are kept at random at the sampling rate.

**Compute profile.** The whole GBDT path is CPU-only and was developed on an 8-core / 16 GB laptop. The memory-lean parts are:
* chunked normalisation in 50k-record worker tasks;
* LSA fit on a 1.5M-text sample, SVD on a 600k-row subsample, transform in 200k chunks;
* FAISS HNSW for large per-country indexes;
* merging channels through int64 pair keys (`i1·N+i2`, `np.unique` + `searchsorted`) instead of pivots;
* feature workers that receive only the records their 50k pairs touch;
* float32/float16 storage;
* negative downsampling for GBDT training, plus an optional hashed S1 fraction (`gbdt_train_s1_frac`).

The deep zoo needs a GPU box (`AWS_RUNBOOK.md`: g6e.12xlarge) and is **optional**.

### 2.3 Record normalisation and transliteration (prepare stage)

`normalize.normalize_record(name, address, country)` returns the name and address views below. Blocking and features only ever compare these views.

| Field group | Views |
|---|---|
| Name | `name_norm` (all tokens), `name_core` (minus legal forms, honorifics, stop words), `name_key` (core minus generic business words), `name_sorted`, `name_concat` (for domains and handles), `name_legal` (canonical legal forms), `name_parts` (cores of each DBA/aka part), flags `name_is_domain`, `name_has_dba`, `name_had_indic` |
| Address | `addr_norm`, `addr_street` (alphabetic street tokens), `addr_house`, `addr_nums`, `addr_locality`, `addr_state`, `addr_unit`, `addr_pobox`, `addr_n_segs`, `addr_empty` |
| Keys (`pipeline.add_key_columns`) | `key_house_street`, `key_house_loc`, `key_name`, `key_concat`, `key_skel` (§3.2) |

**Names.** The steps, in order:
1. NFKC normalisation.
2. Romanisation of Indic runs.
3. Latin folding: ligatures (œ→oe, ß→ss) and accents.
4. Phone numbers are removed.
5. A URL is reduced to its stem if it is the whole name, and dropped if it is a suffix. `@handle` / `#handle` become the stem.
6. The name is split into parts on DBA / t/a / aka / fka markers.
7. Tokenisation: dotted initialisms (`l.l.c`), `&`/`+`→`and`, leetspeak is reversed inside alphabetic tokens (≤ 2 digits), the learned Latin variant canon is applied, and single letters are joined (`p v t`→`pvt`).
8. Legal forms are made canonical: US, Indian, and French/European (`sarl`, `sas`, `eurl`, `gmbh`, ...).
9. Consecutive duplicate tokens are removed.

**Addresses.** The steps, in order:
1. Native-script segments are looked up whole (a state written as one run).
2. PO Box / PMB is extracted.
3. House-number prefixes (Door No, H.NO, Plot No, #), ordinal suffixes, leading zeros and null tokens are stripped.
4. The state is detected per country value, including a trailing multi-word state inside a segment. Only US and Indian state names and codes are tabulated. Any other country value gets an empty table, so the state is simply unknown (open set).
5. Each segment is classified as street, unit or locality. Street, unit and direction abbreviations are made canonical for English, Indian and French patterns (`street/st`, `nagar/ngr`, `rue/r`, `boulevard/bd`, ...).

**Phonetic skeleton.** `skeleton(token)` is a consonant skeleton that tolerates vowel loss in transliteration and the Tamil voiced/unvoiced merge ("limitet" = "limited"). It is used for keys and features.

**Transliteration** (`translit.py`, `xscript.py`, all stdlib):
* **Token dictionary.** Learned from positive train pairs where the candidate is in native script and the S1 name is Latin. Each native token gets votes from two sources:
  * positional alignment when the token counts are equal (weight 2);
  * the best Latin token by skeleton-Levenshtein similarity (≥ 0.5) of the rule romanisation (weight 1).

  A mapping is kept when it has ≥ 2 votes and ≥ 40% share. Address segments that are fully native are aligned the same way at segment level.
* **Latin variant canon.** When one native token maps to several Latin spellings with ≥ 20% share each (laxmi/lakshmi, jai/jay, shree/sree), those spellings are merged into one canonical form. The canon is applied to both sides. Legal words and pairs with skeleton similarity below 0.6 are excluded.
* **Fallback romaniser.** Handles all 9 scripts through their shared ISCII-style layout: code point minus the script's Unicode block base, then a Devanagari table. It deletes the word-final schwa and has fixes for special letters in Malayalam, Tamil and Bengali.
* **OOV resolution** (`prepare.extend_translit_oov`). Runs on each split, label-free. Native tokens not in the dictionary are mapped to the closed Latin vocabulary of the same split (tokens seen ≥ 3 times). Candidates come from fine and coarse skeleton buckets. They are scored by 2·[fine skeleton equal] − 2·normalised Levenshtein + 0.8·log(1+freq). A mapping is accepted only when it is confident. Otherwise the rule romanisation is kept, because an unknown extra token is safer for precision than a wrong common word. For example, `एजेंसीज` ("agencies") must not become `sons`, which would turn a category-inserted distractor into a match.
* **Dev snapshot.** The learned artefact holds 1,363 token entries, 16 address-segment entries and 4 variant canons. 137 OOV tokens were resolved on the dev split.

---

## 3. Candidate Generation (Blocking)

`pipeline.step_block` → `blocking.generate_candidates` → `blocking.merge_channels`. All channels run **separately for each country value**, over every value present in the split.

### 3.1 Channels

The three text views are:
* **name:** `name_core`, or `name_norm` when the core is empty;
* **addr:** house + street + locality;
* **both:** name + " | " + addr.

| Channel | Representation | Search | Default k (fwd S1→cand / rev cand→S1) | What it recovers |
|---|---|---|---|---|
| `lsa_name` | char_wb 2–4-gram TF-IDF (sublinear tf, min_df 3, ≤ 300k features in dev, 400k in aws) → randomized TruncatedSVD 128-d → L2 | FAISS inner product | 30 / 8 | name noise; records with an empty address |
| `lsa_addr` | same, on the address view | same | 30 / 8 | names replaced by a pseudo-word; renamed or aliased names |
| `lsa_both` | same, on the combined view | same | 30 / 8 | general case |
| `dense` (**optional**) | fine-tuned `multilingual-e5-base` mean-pooled 768-d, float16 (§4.3) | same | 40 / 10 (`k_fwd_dense`, `k_rev_dense`) | semantic and transliteration residue |
| 5 exact keys | §3.2 | hash join inside the country | all pairs of a block | exact evidence that survives heavy text noise |

* **Encoders.** The TF-IDF/SVD encoders are fit, without labels, on the split's own S1 + candidate texts (transductive). The vectorizer is fit on a 1.5M-text sample and the SVD on 600k rows.
* **Index choice.** Flat inner-product search is used when a GPU is visible or the index has ≤ 200k vectors. Otherwise HNSW is used (M = 32, efConstruction 80, efSearch = max(128, 2k)).
* **Directions.** Both directions are searched. The reverse direction reflects exclusivity: a candidate's true S1 should be among its nearest S1s. It also supplies the reverse-rank features used by the matcher.

### 3.2 Blocking keys

| Key | Definition | Target noise |
|---|---|---|
| `house_street` | alphanumeric house number + skeleton(first street token) | the name is replaced or aliased; the address is reordered |
| `house_loc` | house number + skeleton(first locality token) | the street is abbreviated or misspelled |
| `name` | sorted set of `name_key` tokens (core minus generic words) | token shuffles; generic words inserted or dropped |
| `concat` | concatenated core, if ≥ 6 characters | domains and handles (`capitalexpresszeo.com`, `@lornedevries`) |
| `skel` | sorted consonant skeletons of the `key_name` tokens | vowel loss in transliteration; typos |

Keys are joined inside the country value, and empty keys are skipped. A block is dropped when more than `max_block` = 50 S1s share the key, or more than 400 candidates do, because such blocks carry no information and explode the pair count.

### 3.3 Merge and cap

1. All channel/direction hits are merged into unique (S1, candidate) pairs.
2. Each channel/direction gets its own columns: `blk_sim_*` (best cosine), `blk_rank_*` (best rank) and `blk_key_*` (key hit flags), plus `blk_n_channels`.
3. The pre-score is the maximum kNN cosine over channels plus 0.05 per key hit.
4. Each S1 keeps its top `max_cand_per_s1` = 50 pairs by pre-score. A pair is **always kept** when the S1 is among the candidate's top-3 reverse neighbours on any channel (`always_keep_rev_rank`).
5. Every vector channel's cosine (`cos_lsa_*`, `cos_dense`) is then recomputed for **every** surviving pair. This way no feature depends on which channel happened to retrieve the pair.

The surviving set is exactly what the matcher scores, and it is written as `candidate_pairs.tsv`.

### 3.4 How true matches are protected

* **Complementary views.** Name-only, address-only and combined views each cover a failure mode of the others: a replaced name, an empty address, heavy noise on both. Exact keys cover cases that survive noise badly in n-gram space: domains and skeleton-equal transliterations.
* **Two directions.** A record matched to a dense chain or a generic name may not be in the S1's top-30. Its own top-8 reverse neighbours still contain its owner.
* **Exemption from the cap.** The top-3 reverse neighbours of every candidate survive the per-S1 cap, even when the S1 already has 50 candidates.
* **Normalisation before retrieval.** Romanisation, legal-form canon, de-leeting and house-number canon run before retrieval, so the kNN space compares like with like. In the dev snapshot, Indic-script candidates are *under*-represented among blocking misses: 0.35% of misses vs 8.65% of retrieved pairs.
* **Measured recall.** `blocking_report.json` reports on train:
  * `pair_recall`: share of true pairs retrieved;
  * `entity_full_recall`: share of S1 with every match retrieved;
  * `avg_candidates`.

  The runbook target is pair recall ≥ 0.995. Configuration levers: `k_fwd`, `k_rev`, per-channel `k_fwd_<chan>` / `k_rev_<chan>`, `max_cand_per_s1`, `always_keep_rev_rank`, the dense channel.
* **Losses priced in the decision.** The decision layer can price in the remaining losses through `missing_mass` (§4.4).

### 3.5 Candidate pairs generated

| Quantity | Test (submitted) | Train (validation) | Dev snapshot (8% geo sample, LSA + keys, no dense channel) |
|---|---|---|---|
| candidate pairs | 89,696,272 | 10,216,550 | 13,995,124 (32,396,725 unique pairs before the cap) |
| candidates per S1 | 51.8 | 50.0 | 68.5 |
| reduction ratio vs within-country Cartesian product | 99.9987% (of 6724.57B pairs) | – | 99.9856% (of 96.96B pairs) |
| pair recall | – | 0.9951 | 0.9727 (S2 0.9801, S3 0.9657) |
| entity full recall | – | 0.9835 | 0.9150 |
| ceiling: macro F0.5 of a perfect matcher on the candidate set | – | {{ORACLE_F05_CEILING}} | 0.9915 (India 0.9885, US 0.9941) |

In the dev snapshot, the key channels contributed 0.9M (`house_street`) to 2.75M (`skel`) pairs each. Every S1 with a true match hits the cap of 50. The mean of 68.5 is above the cap because of the reverse-rank exemption. The ceiling row is the macro F0.5 of predicting exactly the true matches that are present in the candidate set.

---

## 4. Matching Model

### 4.1 Features

All features are country-agnostic, and none encodes the country label. IDF tables, S1 token document frequencies and source-specific token sets are computed **per split and per country value** from that split's own unlabelled records. Missing evidence is NaN, never 0.

| Group | # (stage 1) | Features |
|---|---|---|
| Blocking | 21 | `blk_sim_*` / `blk_rank_*` for each LSA view and direction (12), `blk_key_*` (5), `blk_n_channels`, `cos_lsa_{name,addr,both}` |
| Name: string similarity | 23 | rapidfuzz battery (ratio, token-set, token-sort, partial, Jaro-Winkler, Levenshtein) on the core, norm and key views; exact core and sorted equality; concatenated-name ratio and partial (domains and handles, also against the full normalised S1 name); common-prefix share; skeleton token-set and ratio; Monge-Elkan; soft-TFIDF in both directions (Jaro-Winkler ≥ 0.85) |
| Name: IDF / insertion | 19 | IDF-weighted Jaccard and coverage per side; unmatched ("extra") IDF mass per side and max extra-token IDF; shared IDF mass and each side's IDF mass (a rare shared name is strong evidence, a generic one is weak); min S1 token IDF; token counts; `nm_is_superset` (candidate = S1 + inserted tokens); relative position of the first inserted token; max S1 doc-frequency of inserted tokens (typos ≈ 0, category words high); inserted-generic-word flag |
| Name: structure | 9 | best token-set and ratio over DBA parts; DBA, domain and Indic flags; legal-form equality, Jaccard and conflict; length ratio |
| Address | 32 | rapidfuzz battery on `addr_norm`; IDF Jaccard, coverage and extra mass; token-set with source-specific admin tokens removed (`_x`); number-set Jaccard, first-number equality, Levenshtein and any-match (tolerates 975/976, 11/6 vs 1/6); house-number equality and Levenshtein; missing-house flag; street token-set and ratio; locality token-set, partial, best-token Jaro-Winkler, and token-set with admin tokens removed; `ad_state_eq` (NaN if a state is unknown); PO-box, unit and unit-similarity flags; segment count; empty-address flag |
| Cross | 1 | share of candidate name tokens that appear in the S1 address |
| OOF extra-token harm | 3 | max and sum of harm over inserted tokens, count of unknown inserted tokens (definition below) |
| Frequency / competition | 18 | same-name counts: S1 name among S1s, candidate name among S1s, candidate name among candidates (chains). For 5 key similarities, the rank and gap within the candidate's S1 competitors and the rank within the S1's candidates (score-free exclusivity signal). |
| **Stage-1 total** | **126** | (dev `feats.parquet`: 134 columns, of which 8 are ids and bookkeeping) |
| Stage-2 context | +11 | on the stage-1 OOF score p1: rank within the S1 and within the candidate, gap to the best, **margin over the best other S1 competing for the same candidate**, margin to the next candidate, candidates per S1, S1s per candidate, strong-candidate count, score sum, mutual-best flag |
| Stage-2 cluster support | +3 | token-set similarity (address and name) of each candidate to the S1's *other* confident candidates (top 3 with p1 ≥ 0.5), and the number of such anchors. S2/S3 duplicates of one entity corroborate each other, e.g. a pseudo-word-named record whose address equals the address of the confident members. |
| Stage-2 total | **141** | 126 + 11 + 3 + p1 |

**Extra-token harm.** Near-duplicate pairs are those where the candidate's core tokens are a superset of the S1 core with 1–2 extra tokens. For each inserted token, the harm is logit(base rate) − logit(p̂). Here p̂ is the positive rate of pairs containing the token, smoothed towards the base rate (prior weight 20). The harm is only defined when the token occurs ≥ 30 times. It is estimated **out of fold** on the shared S1 folds. The test table is the mean of the fold tables, restricted to tokens known in a majority of folds (dev: 855 tokens).

**Train-only augmentation.** `ad_state_eq` is set to NaN for 25% of S1 (by hash, `state_dropout`). This teaches the model that "state unknown" also happens on full addresses, which is the regime of the unseen country.

### 4.2 Model types and training

| Stage | Learner | Inputs | Output |
|---|---|---|---|
| 1 | LightGBM (binary) | 126 features | p1 (OOF on train) |
| 2 | LightGBM + XGBoost (hist) + CatBoost | 141 features | `p2_lgb`, `p2_xgb`, `p2_cat` (OOF) |
| Blend | equal-weight mean of the stage-2 logits | – | `p_blend` |
| Calibration | isotonic regression fit on the OOF `p_blend` | – | calibrated `p` (the decision needs probabilities) |
| (**optional**) holdout stacker | LightGBM (31 leaves, lr 0.03) + isotonic | GBDT logits, deep scores, context features | `p_final` (§4.3) |

**Out-of-fold fitting** (`models/gbdt.py`). For each of the 5 S1 folds:
* The model trains on the other folds, after negative downsampling: all positives, the top-15 pairs by pre-score per S1 (hard negatives), and 30% of the remaining negatives.
* Early stopping (100 rounds, logloss) uses a hashed subset of the training-fold S1s: buckets 8–9 of `crc32 % 10`. Because those buckets overlap the fold hash, this is 25% of the training S1s for folds 0–2 and 12.5% for folds 3–4. It never overlaps the held-out fold.
* The model predicts the held-out fold.

OOF predictions therefore cover every train pair. A separate full-data model, with 10% of S1 held out for early stopping, is saved for test inference. On test, the stage-2 context and support features are computed from the full stage-1 model's scores.

**Hyperparameters.**
* LightGBM: 127 leaves, lr 0.05, min_data_in_leaf 100, feature and bagging fraction 0.8, L2 1.
* XGBoost: depth 8, eta 0.05.
* CatBoost: depth 8, lr 0.08.
* `configs/aws.yaml` raises LightGBM to 255 leaves and min_data_in_leaf 200, XGBoost to depth 9, and allows up to 4000 rounds.
* At full scale each fit uses a hashed 40% of S1 (`gbdt_train_s1_frac`). The OOF still covers all rows.
* XGBoost and CatBoost switch to the GPU when one is visible.

### 4.3 Optional deep zoo (GPU; skipped in the CPU-only configuration)

Deep models train only on the S1 folds `deep_train_folds` = {0, 1, 2}. Their scores on those folds are inflated, so GBDT feature selection excludes every `cos_dense*`, `blk_*_dense*`, `ce_*` and `llm_*` column. Deep scores reach the decision **only** through the holdout stacker, which is fit on folds {3, 4}.

* **Bi-encoder** (`models/biencoder.py`, dense blocking channel).
  * Base: `intfloat/multilingual-e5-base`.
  * Text: "query: name legal | address", built from the normalised, romanised views.
  * Loss: symmetric InfoNCE (τ = 0.05). Each anchor S1 is paired with one random true match per epoch, and one hard negative is drawn from its top-5 non-matching blocking candidates. S1 is deduplicated, so in-batch S1s are true negatives.
  * Training: AdamW 3e-5, 2 epochs, max_len 64, batch 384, warmup 5%, AMP.
  * After training, `block` and `features` are re-run with the extra channel.
* **Cross-encoders** (`models/cross_encoder.py`).
  * Architecture: a transformer [CLS] vector concatenated with an MLP over 29 standardised numeric pair features plus their missing flags, then a 2-layer head.
  * Text for each side: raw name (with the romanised core in brackets for Indic names) `[ADDR]` raw address `[NORM]` normalised address.
  * Training pairs: all positives, plus negatives with p1 ≥ 0.001 sampled 4:1, capped at 4M pairs.
  * Optimisation: AdamW with 20× the learning rate on the head, 6% linear warmup, EMA 0.999 of the weights, random A/B swap. Inference averages the logits of both A/B orders.
  * Scoring band: pairs with `p_blend` ≥ 0.002. Pairs outside the band count as confident negatives (logit −12).
  * Bases: XLM-R-large, bge-reranker-v2-m3, mDeBERTa-v3-base (fp32) and mmBERT-base run by default. XLM-R-base is configured but not run by default.
* **LLM matcher** (`models/llm_matcher.py`, optional).
  * Model: Qwen3-4B-Instruct-2507 or Qwen2.5-7B-Instruct with a sequence-classification head (1 logit).
  * LoRA r = 16, α = 32, dropout 0.05 on all attention and MLP projections; bf16; gradient checkpointing.
  * Trained and applied only on the uncertain band, 0.02 ≤ `p_blend` ≤ 0.98. Outside the band the stacker uses the GBDT logit.
* **Holdout stacker** (`models/stack.py`).
  * Fit on folds 3–4 with an inner 2-way S1 split, then isotonic calibration of the inner OOF.
  * Inputs:
    * logits of `p_blend` and each `p2_*`;
    * the `ce_*` / `llm_*` logits, their mean, and a combined logit;
    * dense cosines;
    * 10 strong raw features;
    * candidate-side competition on `p_blend`, computed over **all** S1 before the holdout filter (the same regime as test);
    * S1-side competition on the combined score.
  * During tuning, exclusivity also sees competitors from folds 0–2, scored by the calibrated GBDT (`oth_fold_best_p`).
  * The report includes `gbdt_only_same_holdout`, the reference score on the same S1s.
  * At test time, it averages the two inner models' calibrated outputs and fails loudly if a deep score file is missing.

### 4.4 Decision: exclusivity + expected-F0.5 subset selection

`decision.decide` applies the following steps to the calibrated pair probabilities.

1. **Exclusivity.** Each candidate is kept only for its highest-scoring S1, optionally subject to a margin (`enforce_exclusivity`). This enforces the fact that each S2/S3 record matches at most one S1.
2. **Optional recalibration.** Temperature and bias on the logit: p′ = σ(logit(p)/T + b).
3. **Expected-F0.5 prefix selection** (`select_expected_f`, numba-jitted). For each S1:
   * Candidates with p ≥ 1e-4 are sorted in descending order.
   * For each k = 0 … min(n, 16), TP ~ PoissonBinomial(p₁…p_k) and FN ~ PoissonBinomial(p_{k+1}…p_n). The FN term can be convolved with Poisson(`missing_mass`), truncated at 3, for matches lost in blocking.
   * The expected score is

     E[F0.5 | k] = Σ_a Σ_b P(TP=a)·P(FN=b) · 1.25·a / (k + 0.25·(a+b)).

     For k = 0 the score is P(no true match) = Π(1−p_i)·e^(−missing_mass), which is the singleton case.
   * The k with the highest expected F0.5 is emitted. Under independence, the F-optimal set is a prefix of the probability-sorted list, so this selection is exact. `tests/test_core.py` checks it against brute-force enumeration over outcomes.
4. **Tuning** (`tune_decision`). The config is grid-searched on OOF macro F0.5 over all validation S1:
   * exclusivity on or off;
   * mode `threshold` ∈ {0.3 … 0.9};
   * mode `expected_f` × T ∈ {0.7, 1.0, 1.4} × b ∈ {−1, −0.5, 0, 0.5} × `missing_mass` ∈ {0, 0.05}.

   That is 62 configurations. The winner is stored in `decision.json` (or in `stacker.pkl`) and reused unchanged on test.

Because of the precision weighting (β² = 0.25), the rule emits an uncertain extra candidate only when its probability is high relative to what is already emitted. An empty list wins for an S1 whose best candidate is weak. For example, one candidate at p = 0.3 gives E[F|∅] = 0.7 > E[F|{x}] = 0.3.

---

## 5. Results & Error Analysis

### 5.1 Scores

| Metric | Value | Source (after `train` / `stack train`) |
|---|---|---|
| **F0.5 macro, OOF, GBDT stack (all validation S1)** | **0.9899** | `work/train/oof_report_p.json` → `macro_f05` |
| F0.5 on singletons / non-singletons | 0.9859 / 0.9901 | same → `f05_singletons`, `f05_nonsingletons` |
| micro precision / micro recall | 0.9974 / 0.9757 | same → `micro_precision`, `micro_recall` |
| F0.5 by country (US / India) | 0.9904 / 0.9893 | same → `by_country` |
| single stage-2 models, default decision (LGB / XGB / CAT) | 0.9897 / 0.9897 / 0.9895 | `oof_report_p2_{lgb,xgb,cat}.json` |
| selected decision config | `{"exclusive": true, "mode": "expected_f", "temperature": 0.7, "bias": -0.5, "missing_mass": 0.05, "unseen_bias": -0.3}` | `artifacts/decision.json` |
| (optional) stacker, holdout folds 3–4 | not used (GBDT-only submission) (GBDT only on the same S1s: n/a) | `work/train/stack_report.json` |
| blocking pair recall / entity full recall | 0.9951 / 0.9835 | `work/train/blocking_report.json` |
| public leaderboard | 0.982 | portal |
| validation set | 204,331 S1 (whole-state geographic sample of train) | config `sample.train` |

**Blocking ceiling (dev snapshot).** A perfect matcher on the current dev candidate set would score 0.9915 macro F0.5. 0.18% of non-singleton S1 have none of their matches in the candidate set. With the LSA and key configuration, blocking recall, not matching, sets the upper bound. That is why the dense channel and the k / cap levers of §3.4 matter.

**Profile of blocking misses (dev snapshot).** The table compares 19,291 true pairs missed by blocking with 687,186 retrieved.

| Property of the (S1, true candidate) pair | share among misses | share among retrieved |
|---|---|---|
| candidate address empty | 26.2% | 3.7% |
| name-core token Jaccard < 0.5 **and** address token Jaccard < 0.5 | 40.6% | 1.5% |
| no shared name-core token and address Jaccard < 0.5 (pseudo-word name plus a noisy address) | 11.1% | 0.7% |
| address token Jaccard < 0.5 | 73.2% | 15.6% |
| candidate is a domain or handle | 7.7% | 6.2% |
| both house numbers present and different | 23.6% | 19.7% |
| candidate name in an Indic script | 0.35% | 8.65% |
| candidate name has a DBA alias | 0.03% | 1.58% |

Misses concentrate where *both* fields are degraded, or where the only field present (the name) is noisy. Recall is also lower for S3 (0.966) than for S2 (0.980). Transliteration and DBA handling are not recall bottlenecks.

### 5.2 Expected false positives (wrong merges)

| Type | Mechanism | Guard in the pipeline | Residual risk |
|---|---|---|---|
| Category-word distractor | An unmatched S2/S3 record = S1 name + an inserted category word, often at a similar address | superset, insertion-position and S1-document-frequency features; OOF extra-token harm; generic-word flag | inserted tokens unseen in train, in particular French category words and native-script category words left as rule romanisations. `nm_extra_unknown` marks them, but they have no learned harm. |
| Chain / generic name, wrong branch | 30% of S1 share a name; an address that is partial, empty or off by one points to a sibling | exclusivity; `ctx_margin_other_s1`; same-name counts; address competition ranks | the true branch is outside the candidate set, so exclusivity cannot redirect the record |
| Co-located businesses | different businesses at the same house+street (malls, office blocks) with overlapping name tokens | name IDF mass (rare vs generic shared tokens); legal-form conflict | short, generic names at the same address |
| Singleton with a near-namesake | any emitted id scores 0 for a singleton | the expected-F0.5 empty option; calibrated probabilities | calibration shift on the unseen country |
| Over-extended lists | a low-probability tail appended to a confident list | prefix selection with β² = 0.25; temperature/bias tuned on OOF | correlated errors break the independence assumption, e.g. several distractors built from the same S1 |

### 5.3 Expected false negatives (missed matches)

| Type | Mechanism | Guard | Residual risk |
|---|---|---|---|
| Lost in blocking | both name and address heavily degraded; empty address plus a noisy name; per-S1 cap in dense areas (dev profile, §5.1) | multi-view kNN in both directions; keys; exemption from the cap by reverse rank; optional dense channel | the dev ceiling is 0.9915; this is the dominant FN source |
| Pseudo-word name | no name evidence, so only the address and corroboration from other records support the match | address kNN view and keys; cluster-support features | isolated records, where no other confident member confirms the address |
| Deliberate precision trade | an uncertain extra true match is dropped because β = 0.5 rewards precision | by design; `missing_mass` and bias are tuned | small recall loss on large clusters |
| Unseen country (France) | region vs département admin tokens, French abbreviations, legal-form prefixes, unknown state | French canon rules; accent folding; learned admin lookup (regions/départements); bis/ter and street-head keys; state NaN + dropout; label-free source-specific admin token removal; per-split IDF; prior-shift correction; unseen-country bias. France is sharded by learned region, and global exclusivity is enforced across shards. | probabilities for France cannot be validated offline; under-confidence would show up as FNs. The bias was checked label-free: France's links per S1 and empty rate were compared with the train prior (3.46 links / 5.6% empty) and with US/India. |
| Indic OOV tokens | native tokens unseen in train that the resolver rejects as unconfident | fallback romaniser + skeleton features | a small precision-first loss on names with several OOV tokens |

### 5.4 Caveats of the offline estimate
* The training data has no France, so the OOF report cannot measure it. The public leaderboard is the only check on the unseen-country regime.
* The transliteration dictionary is learned on all train pairs. The OOF estimate is therefore mildly optimistic for Indic records: about 3.6% of test native tokens are unseen in train and depend on OOV mining.
* The isotonic calibrator and the 62-point decision grid are fit on the same OOF predictions. The grid is small, so the resulting optimism is minor.
* Dev numbers come from 8 whole states. The full-train numbers ({{…}} above) are the reference.

---

## 6. Conclusion

The solution treats ER here as three problems: recall-oriented blocking, calibrated pair scoring, and a per-entity decision under a precision-weighted metric.
* **Blocking** is multi-view and bidirectional, with exact keys, and runs per country.
* **Scoring** is a two-stage GBDT stack. Its features target how the generator builds noise and distractors: category-word insertion, name replacement, transliteration.
* **The decision** optimises exact expected F0.5 under an exclusivity constraint. That removes threshold guesswork and credits singletons correctly.

Lessons:
1. Structural facts of the data (exclusivity, 100% country agreement, word-for-word transliteration) are worth more than extra model capacity.
2. The recall ceiling from blocking, not the classifier, is the binding constraint near the top of the leaderboard.
3. For precision, an unknown token must stay unknown: forced OOV guesses create false merges.

---

## Appendix

### A. Code Artefacts

```
business_entity_resolution/
├── README.md                 reproduction guide (stages, quick start, model zoo)
├── AWS_RUNBOOK.md            EC2 GPU runbook for the full-scale / deep run
├── Documentation.md          this document
├── requirements.txt          pinned dependencies
├── configs/
│   ├── local_dev.yaml        8% geographic train sample, CPU laptop, GBDT path
│   ├── aws.yaml              full data, 4-GPU box, all stages and model zoo settings
│   └── smoke.yaml            1% train+test samples, every code path with tiny random models (minutes)
├── scripts/
│   ├── run_all.sh            end-to-end driver: phases gbdt | deep | all, then validate_submission.py
│   ├── aws_setup.sh          one-time EC2 setup (instance store, venv, deps, pytest)
│   └── package_code.ps1      zips the code folder without caches (LF line endings for .sh)
├── src/ber/
│   ├── pipeline.py           CLI prepare | block | features | train | predict | all; text views + key columns; recall report
│   ├── io.py                 exact TSV reader (QUOTE_NONE, BOM-safe) and validator-safe writers
│   ├── normalize.py          name / address normalisation, state parsing, phonetic skeleton
│   ├── translit.py           transliteration dictionary + variant canon learning, OOV resolution
│   ├── xscript.py            stdlib Indic romaniser, Latin folding, skeleton keys, VocabResolver
│   ├── prepare.py            source loading + parquet cache, geo sampling, parallel normalisation
│   ├── blocking.py           LSA encoder, FAISS kNN, key joins, channel merge + per-S1 cap
│   ├── features.py           pair features, IDF, frequency/competition, OOF harm, context/support, folds
│   ├── metrics.py            F0.5, macro F0.5, diagnostic report, candidate recall
│   ├── decision.py           exclusivity, expected-F0.5 prefix selection, decision tuning
│   └── models/
│       ├── gbdt.py           two-stage GBDT stack, isotonic calibration, predict → TSVs
│       ├── biencoder.py      (optional) e5 bi-encoder training + encode() for the dense channel
│       ├── cross_encoder.py  (optional) cross-encoder zoo train / predict
│       ├── llm_matcher.py    (optional) LoRA LLM matcher on the uncertain band
│       └── stack.py          (optional) holdout stacker train / predict → TSVs
└── tests/test_core.py        10 unit tests: metric, singletons, expected-F vs brute force, exclusivity, writer, normalisation, context features
```

**Entry points.** Run from the repository root with `PYTHONPATH=src`, or from `src/` on Windows.

```bash
# GBDT path (CPU): produces output/matching_results.tsv + output/candidate_pairs.tsv
for S in train test; do
  python -m ber.pipeline prepare  --config configs/aws.yaml --split $S
  python -m ber.pipeline block    --config configs/aws.yaml --split $S
  python -m ber.pipeline features --config configs/aws.yaml --split $S
done
python -m ber.pipeline train   --config configs/aws.yaml
python -m ber.pipeline predict --config configs/aws.yaml --split test
# or: bash scripts/run_all.sh configs/aws.yaml gbdt

# optional deep zoo (GPU), then the holdout stacker overwrites the two TSVs
python -m ber.models.biencoder train --config configs/aws.yaml        # then re-run block/features/train/predict
python -m ber.models.cross_encoder train   --config configs/aws.yaml --name xlmr_large
python -m ber.models.cross_encoder predict --config configs/aws.yaml --name xlmr_large --split train   # holdout folds
python -m ber.models.cross_encoder predict --config configs/aws.yaml --name xlmr_large --split test
python -m ber.models.llm_matcher  train|predict --config configs/aws.yaml --name qwen3_4b [--split ...]
python -m ber.models.stack train   --config configs/aws.yaml
python -m ber.models.stack predict --config configs/aws.yaml --split test
# or: bash scripts/run_all.sh configs/aws.yaml deep

python student_resource/utils/validate_submission.py --matching output/matching_results.tsv \
       --candidate output/candidate_pairs.tsv --test-dir student_resource/dataset/test
```

**Work files** are written to `work_dir/<split>/`:
* data: `records.parquet` (normalised records), `cands.parquet` (blocking output), `feats.parquet` (the scored pair set, which equals `candidate_pairs.tsv`);
* scores: `oof.parquet`, `scores_gbdt.parquet`, `score_ce_*.parquet`, `score_llm_*.parquet`;
* reports: `blocking_report.json`, `oof_report_*.json`, `stack_report.json`.

**Artifacts** are written to `artifacts_dir`: `translit.json`, `extra_harm.json`, `model_stage1.pkl`, `model_stage2_{lgb,xgb,cat}.pkl`, `calibrator.pkl`, `decision.json`, `biencoder/`, `ce_<name>/`, `llm_<name>/`, `stacker.pkl`.

**Output guarantees** (`io.write_id_lists`):
* exactly one row for every test S1, including S1 with no candidate;
* ids de-duplicated in order, and only `S2-` / `S3-` ids;
* exact header names, UTF-8, LF line endings;
* matches are a subset of candidates by construction, because both files come from the same scored pair frame;
* `run_all.sh` ends with the official validator.

### B. Model zoo

Pretrained weights are used **only** by the optional deep zoo. Everything else is trained from scratch on the challenge data: the GBDT models, the isotonic calibrators, the TF-IDF/SVD projections, the transliteration dictionary and the harm table.

| id | Model | Params | License | Role | Status |
|---|---|---|---|---|---|
| lgb / xgb / cat | LightGBM 4.7 / XGBoost 3.4 / CatBoost 1.2 | trained from scratch | MIT / Apache-2.0 / Apache-2.0 | stage-1 and stage-2 matchers, stacker | **core** |
| lsa | TF-IDF + TruncatedSVD (scikit-learn) | fitted projections: 3 views × ≤ 400k × 128 ≈ 154M coefficients, no pretraining | BSD-3 (library) | blocking vectors | **core** |
| biencoder | intfloat/multilingual-e5-base | 278M | MIT | dense blocking channel | optional (deep phase) |
| xlmr_large | FacebookAI/xlm-roberta-large | 561M | MIT | cross-encoder | optional (deep default) |
| bge_reranker_m3 | BAAI/bge-reranker-v2-m3 | 568M | Apache-2.0 | cross-encoder | optional (deep default) |
| mdeberta | microsoft/mdeberta-v3-base | 278M | MIT | cross-encoder | optional (deep default) |
| mmbert | jhu-clsp/mmBERT-base | 307M | MIT | cross-encoder | optional (deep default) |
| xlmr_base | FacebookAI/xlm-roberta-base | 278M | MIT | fast cross-encoder | optional (configured, not in the default list) |
| qwen3_4b | Qwen/Qwen3-4B-Instruct-2507 | 4.02B | Apache-2.0 | LoRA matcher, uncertain band | optional |
| qwen25_7b | Qwen/Qwen2.5-7B-Instruct | 7.6B | Apache-2.0 | LoRA matcher, uncertain band | optional |

Alternatives supported by the code but not configured:
* `Alibaba-NLP/gte-multilingual-reranker-base` (Apache-2.0, 306M), as a cross-encoder base;
* `intfloat/multilingual-e5-small` (MIT, 118M), `BAAI/bge-m3` (MIT, 568M) and `sentence-transformers/LaBSE` (Apache-2.0), as bi-encoder bases.

`hf-internal-testing/tiny-random-XLMRobertaModel` is used only by `configs/smoke.yaml` to exercise code paths and is never part of a submission.

**Parameter budget.** The 8B cap is applied conservatively to the *whole* ensemble: the sum of all pretrained parameters.

| Configuration | Pretrained parameters | Within 8B |
|---|---|---|
| `gbdt` (CPU) | 0 | yes |
| `deep` default: bi-encoder + 4 cross-encoders | 278 + 561 + 568 + 278 + 307 = 1.99B | yes |
| `deep` default + qwen3_4b | 6.01B | yes |
| `deep` default + xlmr_base + qwen3_4b | 6.29B | yes |
| `deep` default + qwen25_7b | 9.6B | **no**: not allowed |
| bi-encoder + qwen25_7b only | 7.9B | yes |

**Excluded, and why:**
* Qwen3-8B: 8.19B parameters, over the cap.
* Jellyfish-7B/8B: CC-BY-NC.
* Llama / Gemma families: licenses that are neither MIT nor Apache.
* Unidecode: GPL; replaced by the stdlib romaniser in `xscript.py`.
* libpostal: its model data could count as external data.

**Libraries** (`requirements.txt`), all permissive open source:
* MIT: lightgbm, rapidfuzz, faiss, PyYAML, pytest;
* Apache-2.0: xgboost, catboost, pyarrow, transformers, accelerate, peft, sentencepiece;
* BSD: numpy, pandas, scipy, scikit-learn, numba, torch, protobuf;
* MPL-2.0 / MIT: tqdm.

### C. Compliance notes
* **No external data lookup.** There is no geocoding, registry, API, gazetteer or external dataset. The normalisation maps in `normalize.py` are hand-written general linguistic rules embedded in the code: legal-form variants, street and unit abbreviations for English, Indian and French patterns, and US and Indian state names and codes. The transliteration dictionary is learned from the provided train pairs only. The fallback romaniser is a Unicode code-point table.
* **Label usage.** Train labels are used for:
  * the GBDT and deep models;
  * the transliteration dictionary and variant canon;
  * the extra-token harm table, which is out of fold on train.

  The test split contributes only *label-free, transductive* statistics computed on its own records:
  * TF-IDF/SVD encoders;
  * IDF and S1 document-frequency tables;
  * source-specific admin-token sets;
  * closed-vocabulary OOV resolution.
* **Country as an open set.** Blocking, IDF and competition features are computed per country *value* for whatever values occur. No feature encodes the label, and nothing filters or one-hot-encodes countries. A country value without a state table (France, or any new value) gets NaN state agreement, which train-time state dropout makes a regime the model has seen. Every test S1 is written, whatever its country.
* **Leakage controls.**
  * The S1-level fold hash is shared by all OOF steps.
  * Candidates stay in their S1's fold.
  * Deep scores are used only by a stacker fit on folds the deep models never saw.
  * No ID or row-order feature is used; IDs are uncorrelated with matches.
* **Model licenses and size.** See Appendix B. Every pretrained model is MIT or Apache-2.0 and ≤ 8B parameters. The submitted ensemble stays within 8B pretrained parameters in total.
* **Submission format.** One row per S1 and no duplicates. Only S2/S3 ids that exist in the test split appear, and every match also appears in `candidate_pairs.tsv`. The official validator passes: PASS.

### D. Additional results (dev snapshot, 8% geographic sample, laptop: 8 cores, 16 GB, 6 workers)

| Stage | Volume | Wall-clock |
|---|---|---|
| prepare | 1,148,657 records (S1 204,331 / S2 460,615 / S3 483,711; 8 states) | 961 s |
| block | 3 LSA views: encoding 1,586 s. Per-country kNN in both directions (India 95,478 × 435,899; US 108,853 × 508,427), 5 key joins, merge of 32.4M pairs into 14.0M | 3,257 s |
| features | 13,995,124 pairs × 134 columns (126 model features) | 1,439 s |
| train / predict | – | 1.8 h / 6 min (last shard) |

Learned artefacts on dev:
* transliteration: 1,363 tokens, 16 segments, 4 variant canons, and 137 OOV tokens resolved on the split;
* extra-token harm table: 855 tokens;
* source-specific locality tokens found without labels: 91 for US and 30 for India, candidate side only.


## Final submission (27 Sep 2026): LB 0.982
GBDT stack (OOF 0.9899, LB 0.977) + cross-encoder re-ranking of the uncertain band (GBDT p in [0.003, 0.997], ~5% of test pairs):
fine-tuned `cross-encoder/ms-marco-MiniLM-L-12-v2` and `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` (Apache-2.0, ~33M/118M params)
on train hard pairs (folds 0-2), blended with the GBDT logit by a logistic regression fitted on held-out folds 3-4
(held-out macro F0.5 0.99010 -> 0.99152). For France (unseen country), two extra cross-encoders were fine-tuned on
label-free French pseudo-labels and averaged into the multilingual score. Scripts: scripts/build_ce_data.py, the kernel
script (Kaggle GPU), scripts/ce_blend.py. LB progression: 0.977 (GBDT) -> 0.979 (+CE L6) -> 0.980 (L12) -> 0.981
(L12+multilingual) -> 0.982 (+France-adapted CE).
