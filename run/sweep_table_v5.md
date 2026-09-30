# TargetScore sweep results — v5 (expanded Time window, GroupKFold, trivial baselines, Bug 10 fix)

Dataset: 4,789 rows (4,071 train / 718 test) — up from 4,477 after adding 3d/6d/7d timepoints to the Time allow-list. CV: GroupKFold (5 folds) on (cell line, drug, time, dose) condition, replacing row-level KFold, to eliminate replicate leakage. Train/test split now seeded (random_state=42).

| Protein set | Imputation | Model | Pearson r | R² | IP accuracy |
|---|---|---|---|---|---|
| 289 | mean | **attention** | **0.708 ± 0.011** | 0.493 ± 0.021 | 68.6% |
| 289 | mean | tsnn | 0.705 ± 0.010 | 0.491 ± 0.014 | 69.0% |
| 289 | mean | xgb | 0.688 ± 0.009 | 0.472 ± 0.011 | 68.5% |
| 289 | mean | ensemble | 0.668 ± 0.009 | 0.431 ± 0.009 | 67.7% |
| 289 | mean | rf | 0.472 ± 0.014 | 0.222 ± 0.012 | 64.8% |
| 289 | mean | baseline_only_xgb | 0.328 ± 0.012 | 0.098 ± 0.011 | 64.3% |
| 289 | mean | mean_baseline | 0.199 ± 0.001 | 0.039 ± 0.000 | 63.9% |
| 289 | ml | **tsnn** | **0.720 ± 0.012** | 0.513 ± 0.019 | 69.5% |
| 289 | ml | attention | 0.705 ± 0.004 | 0.491 ± 0.009 | 68.2% |
| 289 | ml | xgb | 0.679 ± 0.009 | 0.460 ± 0.012 | 68.1% |
| 289 | ml | ensemble | 0.660 ± 0.010 | 0.418 ± 0.009 | 67.6% |
| 289 | ml | rf | 0.422 ± 0.013 | 0.177 ± 0.010 | 64.7% |
| 289 | ml | baseline_only_xgb | 0.334 ± 0.011 | 0.096 ± 0.011 | 64.1% |
| 289 | ml | mean_baseline | 0.200 ± 0.000 | 0.039 ± 0.000 | 64.1% |
| 528 | mean | **attention** | **0.710 ± 0.007** | 0.500 ± 0.010 | 68.4% |
| 528 | mean | tsnn | 0.700 ± 0.007 | 0.486 ± 0.010 | 68.3% |
| 528 | mean | xgb | 0.694 ± 0.008 | 0.481 ± 0.010 | 68.6% |
| 528 | mean | ensemble | 0.675 ± 0.009 | 0.439 ± 0.009 | 67.8% |
| 528 | mean | rf | 0.466 ± 0.015 | 0.215 ± 0.012 | 64.8% |
| 528 | mean | baseline_only_xgb | 0.338 ± 0.015 | 0.105 ± 0.013 | 64.4% |
| 528 | mean | mean_baseline | 0.210 ± 0.001 | 0.043 ± 0.000 | 63.8% |
| 528 | ml | **tsnn** | **0.725 ± 0.008** | 0.521 ± 0.012 | 69.4% |
| 528 | ml | attention | 0.705 ± 0.012 | 0.491 ± 0.019 | 67.7% |
| 528 | ml | xgb | 0.681 ± 0.008 | 0.462 ± 0.010 | 68.1% |
| 528 | ml | ensemble | 0.670 ± 0.009 | 0.429 ± 0.009 | 67.5% |
| 528 | ml | baseline_only_xgb | 0.353 ± 0.013 | 0.109 ± 0.013 | 64.1% |
| 528 | ml | rf | 0.349 ± 0.012 | 0.121 ± 0.008 | 63.8% |
| 528 | ml | mean_baseline | 0.206 ± 0.001 | 0.041 ± 0.000 | 63.8% |

Bolded rows are the best model in each of the 4 protein-set/imputation configs — **tsnn or attention wins all four**, reversing the earlier finding (pre-v5, xgb led every config). Both trivial baselines (mean_baseline: ignores all features; baseline_only_xgb: sees only CCLE baseline protein levels) score well below every real model in every config, confirming genuine predictive signal beyond a naive floor.

## Significance (Wilcoxon signed-rank, paired across the 5 CV folds)

| Comparison | Direction consistent across configs? | Typical p (5 folds/config) |
|---|---|---|
| tsnn vs. xgb | Yes, tsnn higher in 4/4 configs (19/20 folds) | 0.06–0.44 |
| attention vs. xgb | Yes, attention higher in 4/4 configs (20/20 folds) | 0.06 in all 4 configs |
| tsnn vs. attention | Mixed — each wins 2/4 configs | 0.06–0.81 |
| tsnn / attention vs. ensemble | Yes, higher in 4/4 configs (20/20 folds) | 0.06 in all 4 configs |

Caveat: with only 5 folds per config, the Wilcoxon test's minimum achievable two-sided p-value is 0.0625 (when all 5 folds agree in direction) — it cannot cross the conventional 0.05 threshold no matter how consistent the effect. Read this table as "the direction is consistent 19-20 times out of 20 across every config and fold," not as a formal significance claim. A stronger claim would need more folds (repeated or nested CV).
