import numpy as np
import pandas as pd

# ADDED: the paper's "Mean Imputation" strategy (Section 2.3.2) was implemented in the
# original codebase by reading a precomputed 'Targetscores_v5_MeanImputated.csv' file
# (referenced in ModulerMI/data_loader.py), which is not present in the GitHub repo and
# was never committed to it (confirmed via full git history search). This reconstructs
# the described method directly from the paper's text:
#   "For each drug-protein pair, missing TargetScores were replaced with the empirical
#    mean calculated from all available non-missing responses under the corresponding
#    experimental condition. When no observations were available for a given
#    drug-protein pair, missing values were instead imputed using the global mean
#    TargetScore for that protein across all perturbation conditions."


def mean_impute_targetscores(targetscores):
    """targetscores: DataFrame with the 8 metadata columns (incl. 'Drug-Name') followed
    by protein columns, some containing NaN. Returns a new DataFrame, same shape, with
    NaNs filled per the two-tier strategy above. Also returns non_NA_mask (True where
    the ORIGINAL value was present, i.e. not imputed) over the protein columns only,
    matching the shape/semantics of what train_predictive_models() returns."""

    ts = targetscores.copy()
    protein_cols = ts.columns[8:]

    non_NA_mask = ts[protein_cols].notna().to_numpy()

    for col in protein_cols:
        drug_group_mean = ts.groupby('Drug-Name')[col].transform('mean')
        ts[col] = ts[col].fillna(drug_group_mean)

        # Fallback: drug-protein pairs with zero observed values still leave NaNs after
        # the groupby-mean fill (transform('mean') on an all-NaN group is NaN). Fill
        # those with the global per-protein mean across all conditions.
        if ts[col].isna().any():
            global_mean = ts[col].mean()
            ts[col] = ts[col].fillna(global_mean)

    return ts, non_NA_mask
