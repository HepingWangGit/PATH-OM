# TargetScore sweep results — fs_korkut.csv, post-Bug-9-fix (2026-09-30)

| Protein set | Imputation | Model | Pearson r | R² | IP accuracy |
|---|---|---|---|---|---|
| 289 | mean | xgb | 0.710 ± 0.007 | 0.502 ± 0.009 | 68.2% ± 0.1% |
| 289 | mean | ensemble | 0.685 ± 0.007 | 0.451 ± 0.008 | 67.2% ± 0.1% |
| 289 | mean | **attention** | **0.621 ± 0.027** | **0.371 ± 0.035** | **65.1% ± 0.6%** |
| 289 | mean | tsnn | 0.537 ± 0.017 | 0.267 ± 0.022 | 63.4% ± 0.2% |
| 289 | mean | rf | 0.478 ± 0.011 | 0.228 ± 0.010 | 64.1% ± 0.1% |
| 289 | ml | xgb | 0.701 ± 0.007 | 0.488 ± 0.009 | 67.6% ± 0.1% |
| 289 | ml | ensemble | 0.682 ± 0.007 | 0.442 ± 0.007 | 66.7% ± 0.1% |
| 289 | ml | **attention** | **0.495 ± 0.019** | **0.198 ± 0.031** | **60.8% ± 0.5%** |
| 289 | ml | tsnn | 0.484 ± 0.011 | 0.185 ± 0.015 | 60.9% ± 0.3% |
| 289 | ml | rf | 0.482 ± 0.006 | 0.227 ± 0.005 | 63.5% ± 0.1% |
| 528 | mean | xgb | 0.721 ± 0.005 | 0.516 ± 0.006 | 68.6% ± 0.0% |
| 528 | mean | ensemble | 0.695 ± 0.005 | 0.464 ± 0.004 | 67.7% ± 0.1% |
| 528 | mean | **attention** | **0.592 ± 0.011** | **0.327 ± 0.013** | **64.2% ± 0.4%** |
| 528 | mean | tsnn | 0.586 ± 0.015 | 0.336 ± 0.020 | 64.4% ± 0.5% |
| 528 | mean | rf | 0.468 ± 0.005 | 0.218 ± 0.004 | 64.4% ± 0.1% |
| 528 | ml | xgb | 0.708 ± 0.004 | 0.498 ± 0.005 | 67.8% ± 0.0% |
| 528 | ml | ensemble | 0.693 ± 0.005 | 0.450 ± 0.003 | 66.8% ± 0.0% |
| 528 | ml | **attention** | **0.504 ± 0.018** | **0.200 ± 0.024** | **61.0% ± 0.3%** |
| 528 | ml | tsnn | 0.548 ± 0.002 | 0.268 ± 0.006 | 61.1% ± 0.1% |
| 528 | ml | rf | 0.451 ± 0.008 | 0.190 ± 0.007 | 63.2% ± 0.1% |

Bolded rows are the attention model, rerun after the Bug 9 fix (see report addendum). Before the fix, attention scored r = 0.10–0.13 across all four configs — the worst model by a wide margin. After the fix it is mid-pack, beating rf in 3/4 configs and tsnn in 2/4 configs — no longer an outlier, consistent with a real architectural bug rather than a fundamental limitation of attention for this task.
