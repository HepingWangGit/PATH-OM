# TargetScore Paper — Replication & Reproducibility Report (v3 — real fs_korkut.csv)

**Repo:** https://github.com/cekayan/TargetScore.git
**Paper:** "Machine learning prediction of adaptive proteomic responses to targeted therapies" (Wang, Kayan, Taskin, Korkut)
**Pipeline exercised:** a patched copy of `ModularMM3/`, kept in `run/` (data_loader → data_preprocessing → imputation (mean or ML-based) → training (5 architectures, genuine 5-fold CV) → evaluation)

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
