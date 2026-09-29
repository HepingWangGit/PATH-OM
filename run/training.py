import numpy as np
import pandas as pd
import random
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA
from utils import multivariate_r2
from model_stuff import model_shit

def training_machine(targetscores, dict_list, features, model_type, model_name):

    test_targetscores = dict_list['test_targetscores']
    ccle_data = features['ccle']

    genomics_data, drug2target, dose_dict, dim_dict, time_dict, stimuli_dict = dict_list['dicts']

    count = 0
    remcol = []
    for column in targetscores.columns:
        n_valid = targetscores[column].notna().sum()
        if targetscores[column].isna().sum() > 0:
            # UPDATED 2026-09-28 (Bug 8, found running the 528-protein config with the new
            # fs_korkut.csv, which for the first time includes very sparse proteins that
            # the old fs_mod.csv substitution had simply dropped): the original check only
            # caught a column that is LITERALLY all-NaN. But imputation.py's own ML-based
            # imputation deliberately leaves a protein's missing rows as NaN, un-predicted,
            # whenever that protein has fewer than 5 real observations in the whole dataset
            # (its own "#samples < 5" cutoff -- see imputation.py). Such a column is not
            # ALL NaN (the handful of real rows survive), but it is still >99% NaN and
            # unfit to train a label on: XGBoost/RF hard-crash on it ("Label contains NaN"),
            # while the nn models silently produce NaN correlations/losses for the whole
            # fold without ever raising -- which is how this surfaced (528|ml|tsnn logging
            # "Train Corr = nan" every fold instead of erroring like 528|ml|xgb did).
            # Generalized to the same standard imputation.py itself already uses: fewer
            # than 5 real (non-NaN) values anywhere in the column means this protein can't
            # be reliably modeled from this data, all-NaN or not.
            if n_valid < 5:
                count += 1
                remcol.append(column)

    targetscores = targetscores.drop(columns=remcol)
    # PATCHED: a protein column can end up entirely NaN within just the training split
    # (e.g. a rare protein with zero observations in this particular fold of rows) even
    # though it has data in the wider dataset. Dropping it here changes the model's
    # output width vs. the untouched data_dict['test_targetscores'], which previously
    # caused a shape mismatch at evaluation time ("size of axis is N but size of
    # corresponding boolean axis is N+1"). used_protein_columns records exactly which
    # protein columns survive, so the caller can subset the held-out test set the same
    # way before evaluating.
    used_protein_columns = list(targetscores.columns[8:])

    # PATCHED: dict_list['fs_list'] (built once in load_data() from the ORIGINAL,
    # pre-imputation targetscores) must be re-aligned to whatever protein columns actually
    # survive by the time we get here -- columns can already have been dropped upstream by
    # the ml-imputation step's own all-NaN-column cleanup (imputation.py), on top of
    # whatever this function's own remcol drop removes above. Either way, `targetscores`
    # at this point (post-remcol-drop) is the authoritative source of truth for which
    # columns survived and in what order, so we always rebuild fs_list by NAME lookup
    # against the original data_dict['targetscores'] columns (which is a strict superset
    # and was never touched by imputation), rather than assuming counts only ever change
    # inside this function. Only the tsnn path consumes fs_list (via CustomTSModel's
    # TSEquationLayer), which is why xgb/rf/ensemble/attention never surfaced this.
    col_to_fs = dict(zip(list(dict_list['targetscores'].columns[8:]), dict_list['fs_list']))
    fs_list_used = [col_to_fs[c] for c in used_protein_columns]
    dict_list = dict(dict_list)  # shallow copy; don't mutate the shared/cached data_dict
    dict_list['fs_list'] = fs_list_used

    baselines = ccle_data.iloc[:,1:].to_numpy().astype(np.float32)  # PATCHED: was dtype=object (from
    # the empty pd.DataFrame(columns=...) used to build ccle_data), which crashes Keras
    # with "Invalid dtype: object" for the nn model types (tsnn, attention).
    labels = targetscores.iloc[:,8:].to_numpy()

    drug_vecs = np.array([drug2target[drug] for drug in targetscores['Drug-Name']])

    stimuli_vecs = np.array([stimuli_dict[sti] for sti in targetscores['Stimuli']])
    stimuli_vecs = np.reshape(stimuli_vecs, (stimuli_vecs.shape[0],1))
    stimuli_vecs = stimuli_vecs.astype(np.float32)

    time_vecs = np.array([time_dict[time] for time in targetscores['Time']])
    time_vecs = np.reshape(time_vecs, (time_vecs.shape[0],1))
    time_vecs = time_vecs.astype(np.float32)

    dose_vecs = np.array([dose_dict[dose] for dose in targetscores['Dose']])
    dose_vecs = np.reshape(dose_vecs, (dose_vecs.shape[0],1))
    dose_vecs = dose_vecs.astype(np.float32)

    dim_vecs = np.array([dim_dict[dim] for dim in targetscores['2D-3D']])
    dim_vecs = np.reshape(dim_vecs, (dim_vecs.shape[0],1))
    dim_vecs = dim_vecs.astype(np.float32)

    vectors_cna = np.array([genomics_data[key]['CNA'] for key in targetscores['CL-Name']])
    vectors_mexp = np.array([genomics_data[key]['mRNA'] for key in targetscores['CL-Name']])

    vectors_mut = [[item for item in genomics_data[key]['Mutation'].to_numpy()] for key in targetscores['CL-Name']]

    mut_vec_1 = np.array(vectors_mut)[:,:,0]
    mut_vec_2 = np.array(vectors_mut)[:,:,1]

    feature_list = [drug_vecs, time_vecs, dose_vecs, dim_vecs, vectors_cna, vectors_mexp, mut_vec_1, mut_vec_2]
    feature_name_list = ['drug', 'time', 'dose', 'dim', 'cna', 'mrna', 'hotspot', 'mut_type']

    feature_dict = {}
    for item1, item2 in zip(feature_name_list, feature_list):
        feature_dict[item1] = item2

    if model_type == 'nn':
        feature_dict['baseline'] = baselines

    # PATCHED: model_shit() now returns a list of per-fold results (one dict per real
    # CV fold) instead of a single (model, pcas) pair, since the original 5-fold CV was
    # broken (see model_stuff.py). Callers that want the old single-model behavior
    # should use fold_results[-1]['model'], fold_results[-1]['pcas'].
    fold_results = model_shit(data=dict_list, model_type=model_type, model_name=model_name, features=feature_dict, labels=labels)

    return fold_results, used_protein_columns