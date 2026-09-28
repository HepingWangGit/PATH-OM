"""
Best-effort RECONSTRUCTION of Figures 10-14 ("Feature importance difference for
the PCA features of <modality>" / raw features): no code for these plots exists
anywhere in the repo (confirmed by grep across all .py files), so this is not a
byte-for-byte reproduction like Figures 15/16/18 -- it is a defensible
re-implementation using the standard technique the paper's own text describes
("training importances and their differences reveal whether a feature is
overfitting"): per-feature permutation importance (R^2 drop when that column is
shuffled), computed separately on a training sample and on the true held-out test
set, using the SAME real trained XGBoost model (258-protein/mean config) as
Figures 15/18.

IMPORTANT CAVEAT, and itself a genuine finding: the paper's Figure 14 ("baseline
RPPA landscape modality") cannot be reproduced from this model at all, because
baseline RPPA features are only fed into the 'nn' models (Attention/TSNN) in the
current architecture -- model_stuff.py's feature_dict only adds 'baseline' when
model_type == 'nn' (see training.py line ~90). The tree-based XGBoost/RF/Ensemble
models never see baseline RPPA as an input feature. Reproducing Figure 14 exactly
would require permutation importance on a trained NN model instead (not done here
due to the much higher compute cost of TensorFlow permutation importance -- each
repeat needs a forward pass through Keras, not a cheap XGBoost predict()).

Figures reproduced here: 10 (CNA, 10 PCA cols), 11 (Mutation, 10+10 PCA cols),
12 (mRNA, 716 raw cols), 13 (Drug Targets, 196 raw cols).
"""
import sys, os, pickle, time
sys.path.insert(0, os.path.dirname(__file__))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from utils import multivariate_r2


def build_matrix(targetscores, data_dict, pcas):
    """Builds the same concatenated c-ml feature matrix training.py/evaluation.py
    build, returning (X, y, block_slices) where block_slices maps a modality name
    to its column range within X."""
    genomics_data, drug2target, dose_dict, dim_dict, time_dict, stimuli_dict = data_dict['dicts']

    y = targetscores.iloc[:, 8:].to_numpy()
    drug_vecs = np.array([drug2target[d] for d in targetscores['Drug-Name']])
    time_vecs = np.reshape(np.array([time_dict[t] for t in targetscores['Time']]), (-1, 1)).astype(np.float32)
    dose_vecs = np.reshape(np.array([dose_dict[d] for d in targetscores['Dose']]), (-1, 1)).astype(np.float32)
    dim_vecs = np.reshape(np.array([dim_dict[d] for d in targetscores['2D-3D']]), (-1, 1)).astype(np.float32)
    vectors_cna = np.array([genomics_data[k]['CNA'] for k in targetscores['CL-Name']])
    vectors_mexp = np.array([genomics_data[k]['mRNA'] for k in targetscores['CL-Name']])
    vectors_mut = [[i for i in genomics_data[k]['Mutation'].to_numpy()] for k in targetscores['CL-Name']]
    mut_vec_1 = np.array(vectors_mut)[:, :, 0]
    mut_vec_2 = np.array(vectors_mut)[:, :, 1]

    cna_pca = pcas[0].transform(vectors_cna)
    mut1_pca = pcas[1].transform(mut_vec_1)
    mut2_pca = pcas[2].transform(mut_vec_2)

    blocks = [('drug_target', drug_vecs), ('time', time_vecs), ('dose', dose_vecs),
              ('dim', dim_vecs), ('cna_pca', cna_pca), ('mrna', vectors_mexp),
              ('mut_pca_1', mut1_pca), ('mut_pca_2', mut2_pca)]
    X = np.concatenate([b[1] for b in blocks], axis=1)

    slices, pos = {}, 0
    for name, arr in blocks:
        w = arr.shape[1]
        slices[name] = slice(pos, pos + w)
        pos += w
    return X, y, slices


def permutation_importance_r2(model, X, y, col_slice, n_repeats=3, rng=None, max_cols=None):
    """Per-column permutation importance (R^2 drop), matching the paper's
    'training importance vs test importance difference' framing. y may contain
    NaN (real, non-imputed TargetScores do) -- masked out exactly as
    evaluate_model_metrics() does before scoring."""
    rng = rng or np.random.default_rng(0)
    non_na = ~np.isnan(y)

    def score(Xin):
        pred = np.asarray(model.predict(Xin))[non_na].flatten()
        true = y[non_na].flatten()
        return multivariate_r2(y_pred=pred, y_true=true)

    baseline_r2 = score(X)
    cols = range(col_slice.start, col_slice.stop)
    if max_cols is not None:
        cols = list(cols)[:max_cols]
    importances = []
    for c in cols:
        drops = []
        for _ in range(n_repeats):
            Xp = X.copy()
            Xp[:, c] = rng.permutation(Xp[:, c])
            drops.append(baseline_r2 - score(Xp))
        importances.append(np.mean(drops))
    return np.array(importances)


def plot_pair(train_imp, test_imp, feature_names, title, out_path):
    diff = train_imp - test_imp
    fig, axes = plt.subplots(2, 1, figsize=(max(6, 0.35 * len(feature_names)), 6), sharex=True)
    axes[0].bar(feature_names, diff, color='blue', label='Difference')
    axes[0].set_ylabel('Training Imp - Test Imp')
    axes[0].axhline(0, color='black', linewidth=0.5)
    axes[0].legend()
    axes[1].bar(feature_names, train_imp, color='indianred', label='Train Feature Importance')
    axes[1].set_ylabel('Training Imp')
    axes[1].set_xlabel('Feature')
    axes[1].legend()
    plt.setp(axes[1].get_xticklabels(), rotation=90, fontsize=6)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved {out_path}")


def main():
    with open('cache/fig_model_bundle.pkl', 'rb') as f:
        bundle = pickle.load(f)
    with open('cache/data_dict_258.pkl', 'rb') as f:
        data_dict = pickle.load(f)
    with open('cache/imputed_258_mean.pkl', 'rb') as f:
        targetscores_imp, non_NA_mask = pickle.load(f)

    model, pcas = bundle['model'], bundle['pcas']
    used_cols = bundle['used_protein_columns']

    meta_cols = list(targetscores_imp.columns[:8])
    train_ts = targetscores_imp[meta_cols + used_cols].sample(n=400, random_state=0)  # subsample for speed
    test_ts = bundle['test_aligned']

    X_train, y_train, slices_train = build_matrix(train_ts, data_dict, pcas)
    X_test, y_test, slices_test = build_matrix(test_ts, data_dict, pcas)

    jobs = [
        ('cna_pca', 'CNA', [f'cna_pca_{i}' for i in range(10)], 'Figure_10_reproduced.png'),
        ('mut_pca_1', 'Mutation', [f'mut_pca1_{i}' for i in range(10)], None),  # combined into Fig 11 below
        ('drug_target', 'Drug Targets', None, 'Figure_13_reproduced.png'),
        ('mrna', 'mRNA', None, 'Figure_12_reproduced.png'),
    ]

    t0 = time.time()
    # --- Figure 10: CNA ---
    tr = permutation_importance_r2(model, X_train, y_train, slices_train['cna_pca'])
    te = permutation_importance_r2(model, X_test, y_test, slices_test['cna_pca'])
    plot_pair(tr, te, [f'cna_pca_{i}' for i in range(10)], 'Difference of Feature Imp. for CNA (Reproduced)', 'Figure_10_reproduced.png')
    print(f"  CNA done at {time.time()-t0:.0f}s")

    # --- Figure 11: Mutation (both PCA blocks concatenated) ---
    tr1 = permutation_importance_r2(model, X_train, y_train, slices_train['mut_pca_1'])
    te1 = permutation_importance_r2(model, X_test, y_test, slices_test['mut_pca_1'])
    tr2 = permutation_importance_r2(model, X_train, y_train, slices_train['mut_pca_2'])
    te2 = permutation_importance_r2(model, X_test, y_test, slices_test['mut_pca_2'])
    names = [f'mut1_pca_{i}' for i in range(10)] + [f'mut2_pca_{i}' for i in range(10)]
    plot_pair(np.concatenate([tr1, tr2]), np.concatenate([te1, te2]), names,
              'Difference of Feature Imp. for Mutation (Reproduced)', 'Figure_11_reproduced.png')
    print(f"  Mutation done at {time.time()-t0:.0f}s")

    # --- Figure 12: mRNA (full 716 cols) ---
    tr = permutation_importance_r2(model, X_train, y_train, slices_train['mrna'])
    te = permutation_importance_r2(model, X_test, y_test, slices_test['mrna'])
    plot_pair(tr, te, [f'mrna_{i}' for i in range(tr.shape[0])],
              'Difference of Feature Imp. for mRNA (Reproduced, all 716 genes)', 'Figure_12_reproduced.png')
    print(f"  mRNA (full) done at {time.time()-t0:.0f}s")

    # --- Figure 13: Drug Targets (196 cols) ---
    tr = permutation_importance_r2(model, X_train, y_train, slices_train['drug_target'])
    te = permutation_importance_r2(model, X_test, y_test, slices_test['drug_target'])
    plot_pair(tr, te, [f'drugtarget_{i}' for i in range(196)],
              'Difference of Feature Imp. for Drug Targets (Reproduced)', 'Figure_13_reproduced.png')
    print(f"  Drug Targets done at {time.time()-t0:.0f}s")

    print("\nFigure 14 (baseline RPPA) intentionally skipped -- see module docstring: "
          "baseline RPPA is not an input feature for the c-ml (XGBoost) model at all.")


if __name__ == '__main__':
    main()
