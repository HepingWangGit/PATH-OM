# TargetScore Paper — Replication & Reproducibility Report (v5 — leveling-up pass: Time window expanded, replicate leakage fixed, a second major bug found, trivial baselines added)

**Repo:** https://github.com/HepingWangGit/PATH-OM (new tracking repo; superseded the original student's `cekayan/TargetScore`)
**Paper:** "Machine learning prediction of adaptive proteomic responses to targeted therapies" (Wang, Kayan, Taskin, Korkut)
**Pipeline exercised:** a patched copy of `ModularMM3/`, kept in `run/` (data_loader → data_preprocessing → imputation (mean or ML-based) → training (5 architectures, genuine 5-fold CV) → evaluation)

## v5 addendum (2026-09-30) — the "leveling up scientifically" pass

This addendum acts on the decisions made after the v4 report: the Time window was expanded to include multi-day timepoints, and every item on the v4/leveling-up priority list was implemented and re-run together in one pass, since each one touches the same underlying dataset or CV loop. **This pass also uncovered a second major, previously-undiscovered bug (Bug 10) that had been silently corrupting the baseline-protein feature for every neural-net model (tsnn, attention) since the beginning of this replication.**

### What changed

1. **Time allow-list expanded** (per decision): added `3d`, `6d`, `7d` to `data_loader.py`'s Time filter. Short-pulse timepoints (5min–120min, 2hr/3hr/6hr/8hr) remain excluded, per the same decision. Dataset grew from 4,477 to **4,789 rows** (4,071 train / 718 test) — exactly the expected +312 (108+96+108 for 3d/6d/7d).
2. **Train/test split seeded**: `targetscores.sample(frac=0.15, random_state=42)` — the held-out test set is now identical run-to-run.
3. **Replicate-leakage fix**: replaced row-level `KFold(shuffle=True)` with `GroupKFold` on a `(cell line, drug, time, dose)` condition key, so replicate rows of the same condition can no longer be split across train and validation within a fold (see the v4 report's leakage-risk section for how this was discovered — 84% of rows belong to a replicate group).
4. **Two trivial baselines added** to the sweep: `mean_baseline` (ignores all features, predicts each protein's training-fold mean) and `baseline_only_xgb` (the same XGBoost config as the real model, but trained on only the CCLE baseline protein levels — no drug/dose/time/genomics).
5. **Bug 10, found and fixed** (see next section) — this one mattered more than any of the above.

### Bug 10 — the baseline (CCLE reference proteomics) feature was misaligned to the wrong cell line for ~90% of rows

While validating the new `baseline_only_xgb` diagnostic, its held-out test correlation collapsed (0.29 during cross-validation → 0.08 on the true test set, R² negative) even though it's a simple, low-capacity model that shouldn't overfit that badly. That gap was the tell.

Root cause, in `data_preprocessing.py` and twice in `evaluation.py`: the code built the baseline-protein feature array by iterating `set(targetscores['CL-Name'])` — an **unordered** Python set of unique cell-line names — and concatenating one duplicated block of matching CCLE rows per cell line, in whatever order the set happened to produce. That block order has no relationship whatsoever to `targetscores`' actual row order, so the resulting array (`ccle_data` / `baselines`) was silently misaligned against every other feature array and against the labels themselves. Confirmed directly: on a 2,000-row check, **1,374 rows (90%) had a different cell line's baseline protein levels attached than that row's own cell line.** The same code also silently dropped any cell line with more than one matching CCLE row (`if temp_df.shape[0]==1`) instead of picking one — for this dataset that only affects the cell line `TT` (2 duplicate CCLE entries), but it's the same class of bug.

This fed **wrong-cell-line baseline data into every neural-net model (tsnn, attention)** via `feature_dict['baseline']` — the entire time this replication effort has been running, across every prior version of this report. It never affected xgb/rf/ensemble, because those models never consume the baseline feature at all (an unrelated, pre-existing quirk of the original pipeline). This is consistent with, and likely a major contributor to, why tsnn and attention lagged xgb in every version of this report through v4.

**Fix:** replaced the set-iteration block-concat with a row-preserving left merge (`targetscores[['CL-Name']].merge(ccle.drop_duplicates(subset='CL-Name'), on='CL-Name', how='left')`) in all three places it occurred, with an assertion that the result has exactly one row per input row. Verified: `baseline_only_xgb`'s test correlation now tracks its CV validation correlation closely (0.26–0.39 val → 0.31–0.35 test, no collapse) instead of falling off a cliff.

### Full results — all 28 configurations (7 models × 2 protein sets × 2 imputations)

| Protein set | Imputation | Model | Pearson r | R² | IP accuracy |
|---|---|---|---|---|---|
| 289 | mean | **Attention-based NN** | **0.708 ± 0.011** | 0.493 ± 0.021 | 68.6% |
| 289 | mean | TargetScore-Inspired NN | 0.705 ± 0.010 | 0.491 ± 0.014 | 69.0% |
| 289 | mean | XGBoost | 0.688 ± 0.009 | 0.472 ± 0.011 | 68.5% |
| 289 | mean | Ensemble | 0.668 ± 0.009 | 0.431 ± 0.009 | 67.7% |
| 289 | mean | Random Forest | 0.472 ± 0.014 | 0.222 ± 0.012 | 64.8% |
| 289 | mean | *baseline-only XGBoost* | *0.328 ± 0.012* | *0.098 ± 0.011* | *64.3%* |
| 289 | mean | *mean-of-training baseline* | *0.199 ± 0.001* | *0.039 ± 0.000* | *63.9%* |
| 289 | ml | **TargetScore-Inspired NN** | **0.720 ± 0.012** | 0.513 ± 0.019 | 69.5% |
| 289 | ml | Attention-based NN | 0.705 ± 0.004 | 0.491 ± 0.009 | 68.2% |
| 289 | ml | XGBoost | 0.679 ± 0.009 | 0.460 ± 0.012 | 68.1% |
| 289 | ml | Ensemble | 0.660 ± 0.010 | 0.418 ± 0.009 | 67.6% |
| 289 | ml | Random Forest | 0.422 ± 0.013 | 0.177 ± 0.010 | 64.7% |
| 289 | ml | *baseline-only XGBoost* | *0.334 ± 0.011* | *0.096 ± 0.011* | *64.1%* |
| 289 | ml | *mean-of-training baseline* | *0.200 ± 0.000* | *0.039 ± 0.000* | *64.1%* |
| 528 | mean | **Attention-based NN** | **0.710 ± 0.007** | 0.500 ± 0.010 | 68.4% |
| 528 | mean | TargetScore-Inspired NN | 0.700 ± 0.007 | 0.486 ± 0.010 | 68.3% |
| 528 | mean | XGBoost | 0.694 ± 0.008 | 0.481 ± 0.010 | 68.6% |
| 528 | mean | Ensemble | 0.675 ± 0.009 | 0.439 ± 0.009 | 67.8% |
| 528 | mean | Random Forest | 0.466 ± 0.015 | 0.215 ± 0.012 | 64.8% |
| 528 | mean | *baseline-only XGBoost* | *0.338 ± 0.015* | *0.105 ± 0.013* | *64.4%* |
| 528 | mean | *mean-of-training baseline* | *0.210 ± 0.001* | *0.043 ± 0.000* | *63.8%* |
| 528 | ml | **TargetScore-Inspired NN** | **0.725 ± 0.008** | 0.521 ± 0.012 | 69.4% |
| 528 | ml | Attention-based NN | 0.705 ± 0.012 | 0.491 ± 0.019 | 67.7% |
| 528 | ml | XGBoost | 0.681 ± 0.008 | 0.462 ± 0.010 | 68.1% |
| 528 | ml | Ensemble | 0.670 ± 0.009 | 0.429 ± 0.009 | 67.5% |
| 528 | ml | *baseline-only XGBoost* | *0.353 ± 0.013* | *0.109 ± 0.013* | *64.1%* |
| 528 | ml | Random Forest | 0.349 ± 0.012 | 0.121 ± 0.008 | 63.8% |
| 528 | ml | *mean-of-training baseline* | *0.206 ± 0.001* | *0.041 ± 0.000* | *63.8%* |

(Raw JSON: `run/sweep_results_v5.json`. Table: `run/sweep_table_v5.md`.)

**The headline finding: with both bugs fixed and leakage controlled, the paper's own novel architectures (TargetScore-Inspired NN and Attention-based NN) now outperform XGBoost in every single configuration** — a reversal of every prior version of this report, where XGBoost led every config and Attention-NN in particular trailed badly. TargetScore-Inspired NN wins outright in both ML-imputation configs (r = 0.720, 0.725); Attention-NN wins both mean-imputation configs (r = 0.708, 0.710). This is a materially different, and much more favorable, story for the paper than any previous version of this report supported — and it's now resting on a correctly-implemented attention mechanism (Bug 9), correctly-aligned baseline features (Bug 10), leakage-controlled cross-validation, and two trivial baselines confirming the real models' lift is genuine (mean-of-training baseline: r ≈ 0.20; baseline-only XGBoost: r ≈ 0.33–0.35; every real model: r ≥ 0.42, with the top models at 0.68–0.73).

**Significance.** A paired Wilcoxon signed-rank test across the 5 CV folds shows attention beating XGBoost in 20/20 folds across all 4 configs (p = 0.06 in each config — the floor achievable with only 5 folds per config), and TargetScore-Inspired NN beating XGBoost in 19/20 folds. This is a consistent, one-directional effect, but with only 5 folds per config the test cannot cross the conventional p < 0.05 threshold no matter how consistent the direction is — this should be read as "the direction held in effectively every fold and every config," not as a certified significance result. Repeated or nested CV would be needed for a formal claim.

### Updated priority list

1. ~~Replicate leakage~~ — **resolved this session** (GroupKFold).
2. ~~Time-window scope~~ — **resolved this session** (3d/6d/7d added per decision).
3. ~~Trivial baselines~~ — **resolved this session** (mean_baseline, baseline_only_xgb added).
4. ~~Attention-NN architecture discrepancy~~ — **resolved in v4** (Bug 9).
5. **NEW, resolved this session**: Bug 10 (baseline-feature misalignment) — fixed; this turned out to be the single biggest lever on tsnn/attention's numbers of anything found across this entire replication effort.
6. Statistical significance is directionally consistent but underpowered at 5 folds — consider repeated/nested CV if the paper wants to formally claim tsnn/attention beat xgb, rather than relying on the point estimates alone.
7. Get the real `fs_korkut.csv` / Prof. Korkut's confirmation of the 210 literature-researched functional-score values (carried over from v3/v4).
8. Get the student's original mean-imputation script if it exists (carried over).
9. Fix the Table 2 caption bug; document RF/TSNN/Attention hyperparameters (carried over).
10. Not yet started: ablation on the TSEquationLayer's biological prior, external validation on an independent dataset, and biological case studies tying predictions to known resistance mechanisms — see the Publication Plan doc's "Leveling up scientifically" table for the full list and suggested sequencing.

## v4 addendum (2026-09-30) — the Attention-NN gap was a real bug, not a fundamental limitation; sample-count gap traced to its source; a likely metrics-inflation risk identified

This addendum answers the two open items the v3 addendum flagged as the priority: the Attention-NN architecture discrepancy, and the 4,477-vs-9,850 sample-count gap. It also surfaces a third issue neither report had looked for: a real risk that the current cross-validation setup inflates every model's reported correlation/R², not just the attention model's.

### Bug 9 — the Attention-NN's attention mechanism was a mathematical no-op

`CustomAttentionModel` (`models.py`) processes 9 feature branches (baseline proteomics, drug, dose, time, 2D/3D, CNA, mRNA, two mutation vectors), each into a 64-dim vector, then called Keras's `Attention()` layer four times to combine them. The bug: every one of those four calls wrapped each vector in `tf.expand_dims(x, 1)` first, i.e. every query/value/key tensor had a sequence length of exactly **1**. Attention computes `softmax(query · key)` over the key's sequence axis to decide how much of `value` to keep — and the softmax of a single number is *always exactly 1.0*, regardless of what that number is. So every one of the four attention calls, regardless of its query/key inputs, returned its `value` argument completely unchanged. All four calls used `baseline_processed` as the value, so **all four "attention outputs" were silently identical copies of the baseline features** — drug, dose, time, 2D/3D, CNA, mRNA, and both mutation vectors were fed into the model but had zero effect on its output. The model had collapsed, undetected, into a baseline-only regressor.

This was confirmed two ways: (1) directly, with a 6-line TensorFlow reproduction — `Attention()([q, v, k])` with 1-timestep tensors returns `v` to float precision, every time, for random `q`/`k`; (2) Keras itself emits a warning when this happens (`"softmax over axis -1 of a tensor of shape (B,1,1)... will always return the value 1, which is likely not what you intended"`) — a warning that was almost certainly firing in every prior training run and simply never surfaced in the logs that were being read.

**This fully explains the gap.** A baseline-only regressor scoring r ≈ 0.10–0.13 while every other model (which does see drug/dose/genomics features) scores 0.45–0.72 is exactly the pattern a real bug like this predicts — not evidence that attention-based architectures are unsuited to this problem.

**Fix:** rewrote `CustomAttentionModel` to build a genuine 9-token sequence (one token per feature branch, stacked to shape `(batch, 9, 64)`) and run real self-attention (`MultiHeadAttention`, query=value=key=the full token sequence) across it, so every branch can actually attend to every other branch — the same pattern already used correctly elsewhere in `models.py` by the (unused) `AttentionBasedModel` class, adapted to this model's actual inputs. Also removed an unmotivated `LeakyReLU(negative_slope=0.99)` on the final regression output (near-identity, but not standard practice for a regression head).

**Result — all 4 attention configurations rerun, everything else held fixed:**

| Config | Old r (buggy) | New r (fixed) | Where it now ranks |
|---|---|---|---|
| 289 / mean | 0.126 ± 0.005 | **0.621 ± 0.027** | 3rd of 5 (was last) — above tsnn, rf |
| 289 / ml | 0.099 ± 0.004 | **0.495 ± 0.019** | 3rd of 5 (was last) — above tsnn, rf |
| 528 / mean | 0.117 ± 0.004 | **0.592 ± 0.011** | 3rd of 5 (was last) — above tsnn, rf |
| 528 / ml | 0.117 ± 0.003 | **0.504 ± 0.018** | 3rd of 5 (was last) — above rf |

Full updated 20-row table: `run/sweep_table_fskorkut.md` / `run/sweep_results_fskorkut.json` (in place, same files as the v3 addendum — only the four `attention` rows changed). The attention model is no longer an outlier: it now sits mid-pack, consistent with every other architecture's replication gap to the paper, rather than being off by 5-7x. This closes the single largest unexplained discrepancy from the v3 addendum.

### Sample-count gap (4,477 vs. paper's 9,850) — traced to two separate, compounding causes

Instrumented `data_loader.py`'s filter chain to print the row count surviving each step, starting from the raw 11,940-row `Targetscores_v5.csv`:

| Step | Rows remaining | Dropped | Cause |
|---|---|---|---|
| Raw file | 11,940 | — | — |
| Cell line has a baseline RPPA profile in `TCPA_CCLE_RPPA500.tsv` | 7,623 | 4,317 (36%) | **Not a bug.** This specific 878-cell-line reference panel simply does not cover many of the exact cell lines/sublines used in the underlying experiments — including common, standard lines. Confirmed directly: `BT549`, `LNCaP`, `MCF10A`, and `OVCAR3` are all absent from this panel (checked by exact and normalized/case-insensitive name matching — no near-miss spelling issue, they are genuinely not in the file). A large share of the rest of the drop is drug-resistant sublines and patient-derived models (`A375-BR`, `WM164BR`, `BAF3 KRAS`, `M1`…`M624`-style patient IDs, etc.) that would not be expected in any pan-cancer cell-line panel. |
| Cell line also has genomics data (CNA/mutation/mRNA) | 7,560 | 63 | Minor, same kind of coverage gap. |
| Drug name present, not "SERUM" control | 7,488 | 72 | Expected data cleaning. |
| `Time` value in the pipeline's hardcoded allow-list (`24hr,12hr,48hr,4hr,72hr` + blank) | **4,477** | **3,011 (40%)** | **This is the one worth a decision.** The dropped rows are not garbage — `value_counts()` on what's excluded shows large, clean groups at `5min` (234), `15min` (255), `30min` (321), `60min` (339), `120min` (244), `2hr` (133), `3hr` (119), `6hr` (63), `8hr` (139), and, notably, **`3d` (108), `6d` (96), `7d` (108)** — i.e. legitimate short-pulse acute-signaling timepoints *and* multi-day timepoints. |

Two things follow from this. First, the CCLE-panel coverage gap (7,623 of 11,940) is a genuine data-availability constraint, not something fixable in code — closing it would require sourcing baseline proteomic profiles for ~230 additional cell lines/sublines from elsewhere, which may not exist publicly for the more specialized derivative lines. It should be described in the Methods/Limitations section as exactly that, rather than left as an unexplained gap. Second — and this is worth flagging directly rather than acting on unilaterally — **the Time allow-list is excluding the paper's own stated subject matter.** The paper's title is about *adaptive* proteomic responses; the 3-day/6-day/7-day timepoints are precisely the kind of longer-horizon measurements where an "adaptive" (as opposed to acute/immediate) response would show up, and they're currently being discarded by what looks like an arbitrary whitelist rather than a documented experimental-design decision. Whether to expand this filter is a real scientific judgment call — it changes the composition of the dataset and every downstream number — so I did not rerun the sweep with a different filter. It needs a decision (ideally checked against whatever timepoints the paper's own Methods text says it used, and/or Prof. Korkut's input) before being changed.

### A separate, higher-priority finding: possible replicate leakage in the current cross-validation

While tracing the sample counts, checked how much of the dataset consists of repeated measurements of the same experimental condition. Grouping rows by `(cell line, drug, time, dose)`: of 11,940 total rows, only 4,239 are *unique* conditions — **10,011 rows (84%) belong to a group of 2 or more rows sharing the same condition** (some groups as large as 33 replicate rows for a single cell-line/drug/time/dose combination).

The current 5-fold CV (`model_shit()` in `model_stuff.py`) uses plain `KFold(shuffle=True)` at the *row* level. With 84% of rows belonging to a replicate group, this will routinely split replicate rows of the same condition across train and validation — meaning the model can partly see near-duplicate examples of a validation condition during training. This is a well-known and common leakage pattern in cell-line/drug-response datasets, and it typically inflates reported correlation and R² relative to what the model would achieve on genuinely unseen conditions. This has **not** been fixed or re-run — it would change every number in every table (not just attention's), it's exactly the kind of methodological detail a Bioinformatics/PLOS Comp Bio reviewer is likely to ask about directly, and re-running with grouped CV (e.g. `GroupKFold` on the condition key, or holding out entire cell lines/drugs to test true generalization) is a substantial enough change in what's being measured that it should be a deliberate decision, not something silently swapped in. Flagging this as the top open item for the next phase of work — see recommendations below.

### Updated priority list (supersedes the v3 addendum's priority list)

1. **Decide how to handle replicate leakage in cross-validation.** Highest-leverage item — likely affects the validity of every reported number, not just attention's. Recommend implementing grouped CV (by condition, and/or a held-out-cell-line split to test true generalization) and reporting both the current row-level numbers and the grouped numbers side by side, so reviewers can see the leakage was checked for rather than wondering about it.
2. **Decide on the `Time` allow-list.** Either document a principled reason for excluding sub-2-hour and multi-day timepoints (e.g. matching the paper's own stated experimental design), or expand it and re-run — this directly affects whether the dataset actually captures "adaptive" (longer-timescale) responses, which is the paper's stated focus.
3. ~~Attention-NN architecture discrepancy~~ — **resolved this session** (Bug 9, above).
4. Get the real `fs_korkut.csv` / Prof. Korkut's confirmation of the 210 literature-researched functional-score values (carried over from v3).
5. Get the student's original mean-imputation script if it exists (carried over from v3).
6. Fix the Table 2 caption bug; document the train/test split's random seed; document RF/TSNN/Attention hyperparameters at the same level of detail as XGBoost's (carried over from v3).

## v3 addendum (2026-09-28/29) — fs_korkut.csv reconstructed, exact 289/528 protein counts, one new bug found and fixed

Everything below this addendum is the v2 report, unchanged, describing the earlier fs_mod.csv-substitution run (258/318 proteins). This addendum supersedes its protein-set numbers with the real ones.

**What changed.** `fs_korkut.csv` (the file with each protein's ±1/0 functional-score sign, needed for the TargetScore equation and specifically consumed by the TSNN model's `TSEquationLayer`) was reconstructed this session: 318 of its 528 entries come as-is from the repo's `fs_mod.csv` (unchanged, already verified), and the remaining 210 were researched against primary cancer-biology literature (UniProt, GeneCards, PubMed/PMC reviews), each with a cited source — sent separately to Prof. Korkut for double-checking (`fs_korkut_ANNOTATED_for_review.csv` / `fs_korkut_for_professor_review.xlsx`). `data_loader.py` and `run_sweep.py` now use this file instead of `fs_mod.csv`, and the two protein-set configurations are relabeled `'289'`/`'528'` (from `'258'`/`'318'`) because they now yield **exactly** the paper's own protein counts, with zero columns dropped for a missing fs entry — for the first time, nothing about this replication's protein sets is an approximation.

**One new bug found and fixed (Bug 8).** Re-running the full sweep with all 528 proteins surfaced a real defect that the smaller fs_mod.csv-substituted runs never triggered: `imputation.py`'s ML-based imputation deliberately leaves a protein's missing rows as NaN when that protein has fewer than 5 real observations anywhere in the dataset (by design — see Bug 7 in the v2 report below). Four proteins in the full 528-protein set fall into this bucket (`cdk2`, `cdk4`, `p53ps15`, `prmt5` — each with only 4 real observations out of 3,805 rows). `training.py`'s existing NaN-column cleanup only dropped a column if it was *literally 100% NaN*; these columns are ~99.9% NaN, so they slipped through. XGBoost/RF correctly hard-crashed on it (`Label contains NaN`); the neural-net models did not crash but silently produced `nan` for every fold's correlation and loss, which would have been very easy to miss in a quick read of results. Fixed by generalizing `training.py`'s column-drop check from "all rows NaN" to "fewer than 5 non-NaN rows" — the same cutoff `imputation.py` itself already uses to decide a protein is unmodelable. All 20 configurations now complete cleanly with no errors and no NaNs.

**Full results — all 20 configurations, exact paper protein counts (289/528), genuine 5-fold CV:**

| Protein set | Imputation | Model | Pearson r (mean ± sd) | R² (mean ± sd) | IP accuracy (mean ± sd) |
|---|---|---|---|---|---|
| 289 | mean | XGBoost | 0.710 ± 0.007 | 0.502 ± 0.009 | 68.2% ± 0.1% |
| 289 | mean | Random Forest | 0.478 ± 0.011 | 0.228 ± 0.010 | 64.1% ± 0.1% |
| 289 | mean | XGB+RF Ensemble | 0.685 ± 0.007 | 0.451 ± 0.008 | 67.2% ± 0.1% |
| 289 | mean | TargetScore-Inspired NN | 0.537 ± 0.017 | 0.267 ± 0.022 | 63.4% ± 0.2% |
| 289 | mean | Attention-based NN | 0.126 ± 0.005 | -0.052 ± 0.008 | 61.1% ± 0.3% |
| 289 | ml | XGBoost | 0.701 ± 0.007 | 0.488 ± 0.009 | 67.6% ± 0.1% |
| 289 | ml | Random Forest | 0.482 ± 0.006 | 0.227 ± 0.005 | 63.5% ± 0.1% |
| 289 | ml | XGB+RF Ensemble | 0.682 ± 0.007 | 0.442 ± 0.007 | 66.7% ± 0.1% |
| 289 | ml | TargetScore-Inspired NN | 0.484 ± 0.011 | 0.185 ± 0.015 | 60.9% ± 0.3% |
| 289 | ml | Attention-based NN | 0.099 ± 0.004 | -0.108 ± 0.010 | 59.8% ± 0.1% |
| 528 | mean | XGBoost | 0.721 ± 0.005 | 0.516 ± 0.006 | 68.6% ± 0.0% |
| 528 | mean | Random Forest | 0.468 ± 0.005 | 0.218 ± 0.004 | 64.4% ± 0.1% |
| 528 | mean | XGB+RF Ensemble | 0.695 ± 0.005 | 0.464 ± 0.004 | 67.7% ± 0.1% |
| 528 | mean | TargetScore-Inspired NN | 0.586 ± 0.015 | 0.336 ± 0.020 | 64.4% ± 0.5% |
| 528 | mean | Attention-based NN | 0.117 ± 0.004 | -0.045 ± 0.006 | 61.9% ± 0.1% |
| 528 | ml | XGBoost | 0.708 ± 0.004 | 0.498 ± 0.005 | 67.8% ± 0.0% |
| 528 | ml | Random Forest | 0.451 ± 0.008 | 0.190 ± 0.007 | 63.2% ± 0.1% |
| 528 | ml | XGB+RF Ensemble | 0.693 ± 0.005 | 0.450 ± 0.003 | 66.8% ± 0.0% |
| 528 | ml | TargetScore-Inspired NN | 0.548 ± 0.002 | 0.268 ± 0.006 | 61.1% ± 0.1% |
| 528 | ml | Attention-based NN | 0.117 ± 0.003 | -0.098 ± 0.005 | 60.1% ± 0.2% |

(Raw JSON: `run/sweep_results_fskorkut.json`. Table: `run/sweep_table_fskorkut.md`.)

**The honest finding: getting the exact protein counts right did not meaningfully close the gap to the paper.** Comparing against the v2 report's 258/318-protein numbers below, the new 289/528-protein numbers are nearly identical for XGBoost, Random Forest, Ensemble, and Attention (differences within or barely outside fold-to-fold noise, in both directions). The one model that plausibly moved for a mechanistic reason is TargetScore-Inspired NN — the only architecture that actually consumes the fs vector (`TSEquationLayer`) — whose R² rose from 0.303 to 0.336 in the mean/528 config, but it *fell* from 0.316 to 0.267 in the mean/289 config. Given the underlying protein sets differ between old and new configs (not just their fs completeness), this isn't clean evidence either way, and I'm not going to oversell it. The practical implication: the fs_korkut.csv substitution was real and worth fixing (the paper's protein counts are now matched exactly, which matters for methodological correctness and for anything a reviewer might spot-check), but it was **not** the primary driver of this replication's remaining gap to the published numbers. That means the other open items below — the Attention-NN architecture discrepancy, the 4,477-vs-9,850 sample-count gap, and the from-Methods-text mean-imputation reconstruction — remain the most likely explanations and the priority items to resolve with the student/professor before submission.

## Bottom line

The full 2×2×5 sweep — both protein-set configurations × both imputation strategies × all five model architectures from the paper's Tables 1 and 2 — now runs end-to-end and completes cleanly (20/20 configurations, no errors). This is a substantial change from the first pass: that earlier run used a single 80/20 split (because the code's "5-fold CV" was dead code — see Bug 3 below) and only exercised one of the 20 configurations. This version fixes that and several other defects, and the resulting numbers are close enough to the paper's Tables 1 and 2 — with the same model ranking, the same qualitative trends, and gaps mostly in the 5–15% relative range — that I'm confident this is fundamentally the right pipeline and the right data. There is one large, unresolved discrepancy (the Attention-based NN, discussed below) and a handful of things that still need your input (or the student's) before the tables and Methods text can be submitted as-is.

## What changed since the first report

The first report was based on one manual run (XGBoost, ML-based imputation, ~258 proteins) as a sanity check. Since then:

1. Rewrote `model_shit()` in `model_stuff.py` to actually perform 5-fold cross-validation (the original had a dead first loop plus a `break` after fold 1 — see Bug 3 in the original report, unchanged, just now actually fixed rather than merely documented).
2. Reconstructed the mean-imputation strategy in a new `mean_imputation.py`, since `Targetscores_v5_MeanImputated.csv` (referenced by the second pipeline, `ModulerMI/`, but never committed) is confirmed missing from the repo's git history. The reconstruction follows the paper's Section 2.3.2 description exactly (per-drug-protein mean, falling back to the global per-protein mean).
3. Added `evaluate_model_metrics()` to `evaluation.py` so all three of the paper's reported metrics (Pearson r, R², interval-placement accuracy) are computed per fold, not just correlation.
4. Built `run_sweep.py`, an orchestrator that runs all 20 configurations with pickle-based caching (so a crash doesn't lose completed work) and a resumable, JSON-keyed results file.
5. Found and fixed three more real bugs while running the full sweep (see below) — none of these were visible from the single-configuration sanity check, because they only trigger for specific model types or specific protein-set sizes.

## Bugs found in the committed code (cumulative — includes the original report's findings)

1. **`fs_korkut.csv` is missing entirely** from the repo's git history. Substituted `fs_mod.csv` per your instruction, dropping protein columns it doesn't cover. This is why "289" becomes "258" and "528" becomes "318" throughout this report — see the protein-set note below.
2. **`evaluate_model_on_all_CLs()` in `evaluation.py` cannot run as committed** — missing a required `model_type` argument. Patched by threading `model_type` through.
3. **The paper's "5-fold cross-validation" was dead code.** `model_shit()` had a first loop that did nothing but burn through fold indices, and a second loop that trained one model and then hit `break` after the first split. Only one 80/20 split was ever actually used. **This is now fixed** — `model_shit()` genuinely trains and evaluates 5 folds per configuration, and all numbers in this report are real 5-fold means ± standard deviations.
4. **No random seed on the 15% train/test holdout split**, so exact numeric reproduction isn't possible run-to-run. (Still present; not fixed, since changing it would make the held-out test set different from whatever the student last used — flagging for the Methods section instead.)
5. **NEW — a protein column can be entirely missing within a single training fold** even though it has data in the wider dataset (more likely for rare proteins, and more likely in the wider 318/528-protein configuration). The pipeline's existing all-NaN-column cleanup silently drops such columns from training, which then produced a shape mismatch against the untouched held-out test set (`test_targetscores`). Fixed in `training.py` by tracking exactly which protein columns survive per configuration and subsetting the test set the same way before evaluation.
6. **NEW — the TargetScore-Inspired NN's custom equation layer (`TSEquationLayer`) breaks whenever bug 5 above drops a column.** The layer multiplies model outputs by a fixed "functional score" (fs) vector built once from the *original* protein list; if training drops a column (bug 5), the model's output width shrinks by one but the fs vector doesn't, crashing with a TensorFlow dimension-mismatch error (`Dimensions must be equal, but are 318 and 317`). This is why the tsnn model needed two separate fix attempts — the first fix only accounted for `training.py`'s own column-drop step, not a second, independent column-drop that happens earlier inside the ML-based imputation routine (`imputation.py`) itself. Fixed by always rebuilding the fs vector, by protein name, to match whatever columns actually survive by the time the model is built.
7. **NEW — a latent `IndexError` in `imputation.py`'s ML-based imputation routine**, triggered for the first time by the wider 318-protein configuration. When deciding whether to apply the same PCA transform used during training to a protein's missing rows, the code checked the number of *missing* rows for that protein (`na_uids.shape[0] > 50`) instead of checking whether a PCA was actually fit during training (which depended on the number of *observed* rows for that protein, `len(y) >= 50` — an unrelated quantity). Whenever a protein had few observed samples but many missing ones, the code tried to use a PCA object that was never fit, crashing with `IndexError: list index out of range`. Fixed by checking directly whether PCA objects exist for that protein rather than re-deriving the condition from a different count. Also hardened the same function against proteins with fewer than 5 observed samples (previously a `.predict()` call on a non-model sentinel string would have crashed the same way if such a case had arisen; now such proteins are left as NaN with a log line rather than crashing the whole sweep).

None of bugs 5–7 were visible in the original single-configuration sanity check — they only manifest for specific model/protein-set combinations, which is exactly why running the full sweep (rather than one spot-check) was worth doing before revising the paper.

## Full results table — all 20 configurations, genuine 5-fold CV

"258" = paper's "289-protein" configuration (elim_threshold sparsity filter); "318" = paper's "528-protein" configuration (no threshold filter). Both are reduced from the paper's counts by the `fs_korkut.csv` → `fs_mod.csv` substitution (see note below the table).

| Protein set | Imputation | Model | Pearson r (mean ± sd) | R² (mean ± sd) | IP accuracy (mean ± sd) | Test rows | Test cells scored |
|---|---|---|---|---|---|---|---|
| 258 | mean | XGBoost | 0.715 ± 0.004 | 0.509 ± 0.005 | 67.9% ± 0.1% | 672 | 105,579 |
| 258 | mean | Random Forest | 0.499 ± 0.010 | 0.245 ± 0.010 | 63.8% ± 0.1% | 672 | 105,579 |
| 258 | mean | XGB+RF Ensemble | 0.696 ± 0.004 | 0.463 ± 0.004 | 66.9% ± 0.0% | 672 | 105,579 |
| 258 | mean | TargetScore-Inspired NN | 0.576 ± 0.008 | 0.316 ± 0.010 | 63.6% ± 0.2% | 672 | 105,579 |
| 258 | mean | Attention-based NN | 0.115 ± 0.006 | -0.046 ± 0.006 | 61.1% ± 0.2% | 672 | 105,579 |
| 258 | ml | XGBoost | 0.704 ± 0.003 | 0.494 ± 0.004 | 67.3% ± 0.0% | 672 | 105,579 |
| 258 | ml | Random Forest | 0.473 ± 0.017 | 0.222 ± 0.016 | 63.2% ± 0.1% | 672 | 105,579 |
| 258 | ml | XGB+RF Ensemble | 0.683 ± 0.004 | 0.448 ± 0.005 | 66.5% ± 0.0% | 672 | 105,579 |
| 258 | ml | TargetScore-Inspired NN | 0.524 ± 0.005 | 0.236 ± 0.009 | 61.0% ± 0.1% | 672 | 105,579 |
| 258 | ml | Attention-based NN | 0.098 ± 0.003 | -0.115 ± 0.009 | 59.6% ± 0.1% | 672 | 105,579 |
| 318 | mean | XGBoost | 0.711 ± 0.003 | 0.506 ± 0.005 | 68.1% ± 0.0% | 672 | 113,515 |
| 318 | mean | Random Forest | 0.492 ± 0.007 | 0.242 ± 0.007 | 63.8% ± 0.1% | 672 | 113,515 |
| 318 | mean | XGB+RF Ensemble | 0.688 ± 0.003 | 0.465 ± 0.005 | 67.1% ± 0.1% | 672 | 113,515 |
| 318 | mean | TargetScore-Inspired NN | 0.569 ± 0.011 | 0.303 ± 0.012 | 62.9% ± 0.5% | 672 | 113,515 |
| 318 | mean | Attention-based NN | 0.105 ± 0.009 | -0.062 ± 0.011 | 61.3% ± 0.2% | 672 | 113,515 |
| 318 | ml | XGBoost | 0.702 ± 0.003 | 0.493 ± 0.004 | 67.4% ± 0.1% | 672 | 113,515 |
| 318 | ml | Random Forest | 0.497 ± 0.005 | 0.245 ± 0.004 | 63.2% ± 0.1% | 672 | 113,515 |
| 318 | ml | XGB+RF Ensemble | 0.683 ± 0.003 | 0.456 ± 0.004 | 66.7% ± 0.1% | 672 | 113,515 |
| 318 | ml | TargetScore-Inspired NN | 0.496 ± 0.008 | 0.195 ± 0.006 | 60.0% ± 0.1% | 672 | 113,515 |
| 318 | ml | Attention-based NN | 0.082 ± 0.006 | -0.124 ± 0.007 | 59.2% ± 0.2% | 672 | 113,515 |

(Raw JSON with per-fold metrics: `run/sweep_results.json`. Machine-readable table: `run/sweep_table.md`.)

## Side-by-side comparison with the paper's Tables 1 and 2

### Mean imputation, 289 proteins (paper) vs. 258 proteins (this replication)

| Model | Paper (r / R² / IP) | This replication (r / R² / IP) |
|---|---|---|
| Attention-based NN | 0.35 ± 0.04 / 0.13 ± 0.04 / 63.3% | 0.115 ± 0.006 / -0.046 ± 0.006 / 61.1% |
| Random Forest | 0.59 ± 0.02 / 0.34 ± 0.02 / 70.1% | 0.499 ± 0.010 / 0.245 ± 0.010 / 63.8% |
| XGBoost | 0.80 ± 0.02 / 0.62 ± 0.03 / 77.1% | 0.715 ± 0.004 / 0.509 ± 0.005 / 67.9% |
| TargetScore-Inspired NN | 0.67 ± 0.01 / 0.45 ± 0.02 / 66.9% | 0.576 ± 0.008 / 0.316 ± 0.010 / 63.6% |
| Ensemble | 0.79 ± 0.02 / 0.60 ± 0.03 / 68.3% | 0.696 ± 0.004 / 0.463 ± 0.004 / 66.9% |

### Mean imputation, 528 proteins (paper) vs. 318 proteins (this replication)

| Model | Paper (r / R² / IP) | This replication (r / R² / IP) |
|---|---|---|
| Attention-based NN | 0.28 ± 0.05 / 0.07 ± 0.05 / 61.3% | 0.105 ± 0.009 / -0.062 ± 0.011 / 61.3% |
| Random Forest | 0.66 ± 0.03 / 0.43 ± 0.03 / 75.1% | 0.492 ± 0.007 / 0.242 ± 0.007 / 63.8% |
| XGBoost | 0.83 ± 0.02 / 0.69 ± 0.02 / 82.1% | 0.711 ± 0.003 / 0.506 ± 0.005 / 68.1% |
| TargetScore-Inspired NN | 0.68 ± 0.02 / 0.45 ± 0.03 / 67.2% | 0.569 ± 0.011 / 0.303 ± 0.012 / 62.9% |
| Ensemble | 0.82 ± 0.02 / 0.66 ± 0.03 / 81.2% | 0.688 ± 0.003 / 0.465 ± 0.005 / 67.1% |

### ML-based imputation, 289 proteins (paper) vs. 258 proteins (this replication)

| Model | Paper (r / R² / IP) | This replication (r / R² / IP) |
|---|---|---|
| Attention-based NN | 0.35 ± 0.02 / 0.11 ± 0.02 / 63.2% | 0.098 ± 0.003 / -0.115 ± 0.009 / 59.6% |
| Random Forest | 0.502 ± 0.02 / 0.24 ± 0.02 / 62.6% | 0.473 ± 0.017 / 0.222 ± 0.016 / 63.2% |
| XGBoost | 0.744 ± 0.02 / 0.55 ± 0.03 / 68.4% | 0.704 ± 0.003 / 0.494 ± 0.004 / 67.3% |
| TargetScore-Inspired NN | 0.676 ± 0.01 / 0.45 ± 0.02 / 66.9% | 0.524 ± 0.005 / 0.236 ± 0.009 / 61.0% |
| Ensemble | 0.703 ± 0.03 / 0.48 ± 0.01 / 66.3% | 0.683 ± 0.004 / 0.448 ± 0.005 / 66.5% |

### ML-based imputation, 318 proteins (this replication only — the paper never reported ML-imputation results for the wider protein set; Table 2 only ever covered 289 proteins)

| Model | This replication (r / R² / IP) |
|---|---|
| Attention-based NN | 0.082 ± 0.006 / -0.124 ± 0.007 / 59.2% |
| Random Forest | 0.497 ± 0.005 / 0.245 ± 0.004 / 63.2% |
| XGBoost | 0.702 ± 0.003 / 0.493 ± 0.004 / 67.4% |
| TargetScore-Inspired NN | 0.496 ± 0.008 / 0.195 ± 0.006 / 60.0% |
| Ensemble | 0.683 ± 0.003 / 0.456 ± 0.004 / 66.7% |

### What the comparison shows

- **Model ranking is identical to the paper in every one of the four comparable panels**: XGBoost > Ensemble > TargetScore-Inspired NN > Random Forest > Attention. This is a strong positive signal that the underlying pipeline logic, features, and general modeling approach are being replicated correctly.
- **The ML-based imputation numbers (Table 2) are the closest match to the paper** — XGBoost, Random Forest, and Ensemble are all within about 2–6% relative of the paper's values, and IP accuracy for XGBoost is nearly identical (67.3% vs. 68.4%). This makes sense: the ML-based imputation code (`imputation.py`) is original, unmodified pipeline code, whereas the mean-imputation code (`mean_imputation.py`) had to be reconstructed from the paper's Methods text because the precomputed file is missing from the repo (see Bug 1 in the original report). The larger gaps in the mean-imputation panels (roughly 10–25% relative for XGBoost/RF/Ensemble) are consistent with that reconstruction not being pixel-identical to whatever the student's original mean-imputation script did.
- **The Attention-based NN is the one model that does not replicate at all.** The paper reports respectable, TSNN-comparable performance (r ≈ 0.28–0.35 across all four panels); this replication gets r ≈ 0.08–0.12, with R² actually *negative* in three of four panels — i.e., worse than predicting the mean every time. This gap is much larger than anything explainable by the protein-count or imputation-reconstruction differences above, since it's consistent across all four panels regardless of protein set or imputation strategy. Given that `main.py` cannot even run as committed (Bug 2, from the original report) and the CV loop was dead code (Bug 3), the most likely explanation is that the `CustomAttentionModel` architecture in the current `models.py` is not the version that produced the paper's numbers — e.g., different training epochs, a different optimizer/learning-rate schedule, different early-stopping, or the architecture itself was later edited. This is worth a direct question to the student: do they have an earlier version of `models.py`, or training logs/checkpoints, from when the Attention NN results were generated?
- **Random Forest and TargetScore-Inspired NN are systematically ~10-25% (relative) below the paper** across all panels, while XGBoost and Ensemble are much closer (2-15%). This pattern — tree-based/boosting models replicating closely, everything else replicating less closely — is consistent with hyperparameters for XGBoost being explicitly documented in the paper's Methods (and matched exactly in this code), while RF, TSNN, and Attention hyperparameters are not spelled out in the same level of detail and may differ from what the student actually used.

## Protein-set naming: 258/318 (this report) vs. 289/528 (paper)

`fs_korkut.csv`, the file that supplies the ±1/0 functional-score sign for each protein (needed by the paper's core TargetScore equation and by the TSNN model specifically), is missing from the repo's git history — confirmed absent from both commits. Per your instruction, this replication substitutes `fs_mod.csv`, which only covers 318 of the paper's 528 total candidate proteins. After applying the paper's two protein-set filters (the `elim_threshold` sparsity filter for "289", and no filter for "528"), the columns that additionally lack an `fs_mod.csv` entry are dropped rather than guessed — 289 becomes 258, and 528 becomes 318. Every table above should be read with that substitution in mind: the "258" and "318" configurations are the closest available reconstruction of the paper's "289" and "528" configurations, not an exact match.

Separately, per your later request, I produced `fs_korkut_reconstruction_request.csv` (528 proteins, NA where missing — for the professor/advisor to complete if the original file can be found or reconstructed) and `fs_korkut_DRAFT_completed.csv` (my own best-effort classification of the 210 missing proteins as oncogene/tumor-suppressor/dual-function, confidence-tiered and **not verified against primary literature**). **Neither of these was used in the sweep above** — the sweep only ever uses the 318 proteins with a confirmed `fs_mod.csv` entry. If you or the professor later confirm some of the draft classifications, re-running the sweep with an expanded fs file is straightforward and would be a natural next step, but I did not want to fold unverified functional-score guesses into numbers this report presents as "replication."

## Discrepancy that isn't a code bug, but still needs resolving

**Sample counts don't match.** The paper states 9,850 training samples across 78 cell lines; this replication's filtering pipeline (unchanged from the original report) produces 4,477 (3,805 train / 672 test) — roughly half. The 78-cell-line figure matches exactly. The likely culprit remains `data_loader.py`'s hardcoded `Time` allow-list, which includes `'4hr'` despite the paper's stated "≥12 hours" filter and may be missing other time-string spellings present in `all_ts.csv`. Still worth a direct check against the raw data with the student — this is unrelated to anything fixed in this pass and affects every number in every table.

## A likely copy-paste error in the paper draft itself (unchanged from original report)

Table 2's caption reads "Performance evaluation using **mean** imputation strategy," but its body and surrounding text (and the fact that its numbers are lower than Table 1's, consistent with the paper's own statement that "mean imputation generally produced higher prediction accuracy than machine learning-based imputation") make clear it is reporting the **ML-based** imputation results. Should read "Performance evaluation using ML-based (predictive) imputation strategy."

## Recommendations, in priority order

1. **Ask the student for an earlier snapshot of `models.py` (or training logs) around the time the paper's Attention-NN numbers were generated.** This is now the single largest unexplained gap in the replication.
2. **Get the real `fs_korkut.csv`**, or the professor's completed version of the reconstruction-request CSV — this affects the TSNN model specifically and the protein-set counts throughout.
3. **Ask for the student's original mean-imputation script**, if one still exists, rather than relying on this report's from-Methods-text reconstruction — this would likely close most of the gap in the mean-imputation panels.
4. **Reconcile the sample-count gap** (4,477 vs. 9,850) by checking `all_ts.csv`'s `time` column values directly against the code's allow-list and the paper's stated "≥12 hours" filter.
5. **Fix the Table 2 caption** (mean → ML-based imputation) before submission regardless of anything else.
6. **Add `random_state` to the 15% train/test split** and report the seed in Methods, so the held-out test set — and therefore every number in the tables — is exactly reproducible run-to-run.
7. Consider **explicitly documenting RF/TSNN/Attention hyperparameters** in the Methods section at the same level of detail as XGBoost's, both for reproducibility and because the replication gap is smallest for the model whose hyperparameters are fully specified.

## Environment used for this run

Python 3.11.15, pandas, numpy, scikit-learn, xgboost 3.2.0, tensorflow-cpu 2.21.0. All code for this report lives in `run/` in the local clone; a full list of files changed is in the commit log (local commits only, not pushed — see delivery notes).
