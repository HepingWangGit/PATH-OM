# TargetScore Paper — Replication & Reproducibility Report

**Repo:** https://github.com/cekayan/TargetScore.git
**Paper:** "Machine learning prediction of adaptive proteomic responses to targeted therapies" (Wang, Kayan, Taskin, Korkut)
**Pipeline exercised:** `ModularMM3/` (data_loader → data_preprocessing → imputation (ML-based) → training (XGBoost, `model_type='c-ml'`) → evaluation)

## Bottom line

I was able to get the pipeline running end-to-end and produced a real number to compare against the paper, but **it does not reproduce the paper's reported numbers exactly**, for reasons that are fixable but need decisions from you/the student before the paper's Methods and Tables can be trusted as-is. This is not a "the code is broken, start over" situation — it's a "the code as pushed to GitHub has several gaps between what it does and what the paper says it does" situation, which is exactly the kind of thing worth catching before submission/revision.

## What I ran

Using your instruction to substitute `fs_mod.csv` for the missing `fs_korkut.csv`, and after fixing one blocking bug (below), I ran the full pipeline once for the **XGBoost regressor, ML-based imputation** configuration — the same configuration reported in the paper's Table 2.

**Result:** overall Pearson correlation on the held-out test set = **0.693** (672 test samples, 63 of 78 cell lines represented in a scoreable way).

**Paper's reported value for this exact configuration (Table 2, XGBoost Regressor, 289 proteins, ML-based imputation):** 0.744 ± 0.02.

These are in the same ballpark (both "moderate-good, XGBoost best-in-class" results) but not an exact match — about a 7% relative gap, in the direction you'd expect given the issues below (fewer proteins, no true cross-validation averaging, roughly half the sample count).

## Bugs found in the committed code

These aren't just my patches being sloppy — they're real defects in what's on GitHub right now:

1. **`fs_korkut.csv` is missing entirely.** Confirmed absent from both commits in the repo's git history (`aa7ab73 Initial commit` and `a260eb0 try`). The repo has `fs.csv` and `fs_mod.csv` instead, and neither is a clean substitute — `fs.csv` covers 189/528 proteins, `fs_mod.csv` covers 318/528 (by column-name overlap with `Targetscores_v5.csv`). Per your instruction I used `fs_mod.csv` and dropped the ~31 protein columns it doesn't cover (258 of 289 remained) rather than guessing functional-score signs for them. **You should ask the student directly whether they still have the original `fs_korkut.csv` on their own machine** — this single file is the biggest source of uncertainty in any replication attempt.

2. **`evaluate_model_on_all_CLs()` in `evaluation.py` cannot run as committed.** It calls `evaluate_model(trained_model, data_dict, test_targetscores, non_NA_mask, print_check=True, pca_list=pca_list)`, but `evaluate_model()`'s signature requires a `model_type` argument with no default. This throws `TypeError: evaluate_model() missing 1 required positional argument: 'model_type'` immediately — meaning **`main.py`, as committed, cannot complete a run.** I patched this locally by threading `model_type` through. This strongly suggests the numbers in the paper were produced by a different (earlier, or since-modified) version of this function than what's currently on GitHub.

3. **The "5-fold cross-validation" described in the paper's Table 1 and Table 2 captions does not match what the code does.** In `model_stuff.py`'s `model_shit()`, there are two `for ... in kf.split(...)` loops back to back: the first loop only increments a `fold` counter and does nothing else (dead code — you can see this in the log output, where the real training loop starts at "Fold 6" instead of "Fold 1", because the dead loop already ran fold 1–5); the second loop trains a model, evaluates it, and then hits `break` after the very first split. **The net effect is that only one 80/20 train/validation split is ever used — not an aggregated 5-fold result — despite the paper stating "Values represent the mean ± standard deviation from 5-fold cross-validation."** The same `break` pattern (explicitly commented `#ONLY FOR CODE DEBUG PURPOSES`) appears in `imputation.py`'s per-protein XGBoost training loop too.

4. **No random seed on the train/test split.** `data_loader.py`'s `targetscores.sample(frac=0.15)` (the 15% holdout described in the paper) has no `random_state`. Every run produces a different held-out test set, so exact numeric reproduction isn't possible unless the student fixed a seed elsewhere that isn't in this repo.

## Discrepancy that isn't a code bug, but needs resolving

**Sample counts don't match.** The paper states the final filtered dataset is 9,850 samples across 78 cell lines. Running the code's filtering steps in order:

| Step | Rows remaining |
|---|---|
| `Targetscores_v5.csv` raw | 11,940 |
| After CL-Name ∈ CCLE | 7,623 (80 cell lines) |
| After CL-Name ∈ genomics_data | 7,560 (**78 cell lines — matches paper**) |
| After Drug-Name filters | 7,488 |
| After the `Time` allow-list filter | **4,477** |

The 78-cell-line figure matches the paper exactly, and the natural protein-column count from the `elim_threshold` filter is exactly **289** (before the `fs_mod.csv` substitution trims it further) — both good signs that this is fundamentally the right code and data. But the final sample count (4,477, split ~3,805 train / 672 test) is roughly **half** the paper's claimed 9,850. The likely culprit is `data_loader.py`'s hardcoded time-string allow-list (`['24hr', '24hrs', '12hr', ..., '#', np.nan, '4hr']`) — note it explicitly includes `'4hr'`, which contradicts the paper's stated "≥12 hours" filter, and it may simply be missing other time-string spellings that exist in `all_ts.csv`. This is worth a direct check against the raw `all_ts.csv` `time` column values with the student.

## A likely copy-paste error in the paper draft itself

Table 1's caption reads "Performance evaluation using mean imputation strategy" and Table 2's caption reads the *same thing* ("Table 2: **Performance evaluation using mean imputation strategy.**"), but Table 2's body text and surrounding paragraph describe it as the **ML-based imputation** results for the 289-protein set (0.744 correlation, matching what I replicated). Worth fixing before submission regardless of anything else.

## What I have NOT yet verified

The paper's Table 1/2 report 5 models (Attention NN, Random Forest, XGBoost, TargetScore-Inspired NN, Ensemble) × 2 protein sets (289, 528) × 2 imputation strategies (mean, ML-based) — 20 configurations in total (some cells only reported for 289). I've only run one of these (XGBoost, ML-based imputation, ~258-289 proteins) as a sanity check, since each full run takes roughly 8–10 minutes of compute here. I have not touched the "mean imputation" pathway (there's a `mean-filled-resps/` folder of per-cell-line CSVs in the repo that looks like it feeds that strategy, but no script in `ModularMM3/`/`ModulerMI/` visibly consumes it — worth asking the student which script does), the 528-protein configuration, or the other four model architectures (Random Forest, Ensemble, Attention NN, TargetScore-Inspired NN).

## Recommendations, in priority order

1. **Get the real `fs_korkut.csv` from the student**, or confirm it's genuinely lost — this affects every number in both tables.
2. **Ask the student for the exact script/notebook version that produced the numbers in the paper's Tables 1 and 2.** Given that `main.py` as committed cannot even complete a run (bug #2 above), the paper's numbers were produced by something other than the current GitHub state. That earlier version is what actually needs replicating.
3. **Fix the 5-fold CV bug** (or rewrite the Methods text to accurately describe what the code does — a single held-out split — if that's what actually generated the numbers). Reviewers who read "5-fold cross-validation, mean ± std" and then see the code will flag this immediately if it's not fixed.
4. **Add `random_state` to the 15% train/test split** so results are exactly reproducible run-to-run, and report the seed in the Methods section.
5. **Reconcile the sample-count gap** (4,477 vs. 9,850) — check the `Time` column values in `all_ts.csv` directly against the paper's stated "≥12 hours" filter and the code's actual allow-list.
6. **Fix the Table 2 caption** (mean vs. ML-based imputation).
7. Once 1–4 are resolved, I'm glad to re-run the full sweep (all 5 models × both protein sets × both imputation strategies) to regenerate Tables 1 and 2 from scratch with a documented, versioned, seeded pipeline — that's the version of "replication" that will hold up under peer review.

## Environment used for this run

Python 3.11.15, pandas, numpy, scikit-learn, xgboost 3.2.0, tensorflow-cpu 2.21.0 (matches `requirements.txt`'s inferred dependency list, which itself was auto-generated and should be pinned to exact versions used for the paper before submission).
