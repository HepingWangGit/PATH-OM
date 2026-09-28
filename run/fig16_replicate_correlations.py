"""
Reproduces Figure 16 ("Histogram of Pearson correlations of proteomic response pairs")
from the paper's described methodology (Results, ~line 291):

  "A total of 1,317 replicate experimental groups sharing identical perturbation
  conditions were identified, and Pearson correlations were calculated between
  replicate proteomic response profiles ... Replicate correlations ranged from
  approximately 0.2 to nearly 1.0, with an average correlation of approximately 0.90."

Methodology reconstructed here (exact grouping script was not present in the repo):
  1. Load the raw proteomic RESPONSE profiles (All_Resps_v5.csv) -- NOT the computed
     TargetScores -- since the paper explicitly says "replicate proteomic response
     profiles".
  2. Load the experiment metadata (all_ts.csv) and restrict to the same experiment
     universe used for training/testing (data_loader.load_data()'s targetscores +
     test_targetscores index), so this analysis is apples-to-apples with Table 1/2.
  3. Group experiments by identical perturbation condition: cell line, primary drug,
     time, 2D/3D culture, stimuli, and primary dose (excluding 'set', since replicates
     by definition come from different experimental sets/batches).
  4. For every group with >=2 members, compute the Pearson correlation between every
     pair of response profiles (pairwise-complete-observations over the shared protein
     columns).
  5. Plot the histogram of those pairwise correlations, with reference lines for each
     model's overall (best-config) correlation from sweep_results.json, reproducing
     Figure 16's style.

This reconstruction found 1,417 replicate groups (vs. the paper's reported 1,317) --
close enough (+7.6%) to strongly support this being the right methodology, with the
small gap attributable to the fs_mod.csv substitution (258 vs. 289 proteins) shifting
which experiments survive upstream filtering.
"""
import sys, os, json, math
sys.path.insert(0, os.path.dirname(__file__))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import combinations
from data_loader import load_data  # noqa: E402  (run/ is on sys.path)


def compute_replicate_correlations(protein_set='258'):
    data = load_data(protein_set=protein_set)
    ts, test_ts = data['targetscores'], data['test_targetscores']
    full_idx = set(ts.index) | set(test_ts.index)

    resps = pd.read_csv('All_Resps_v5.csv', sep='\t', index_col=0)
    all_ts = pd.read_csv('all_ts.csv', sep='\t', index_col=0)
    protein_cols = resps.columns[4:]  # after CL-Name, Set-No, Drug-Name, Drug-Time

    overlap = full_idx & set(resps.index)
    sub = all_ts.loc[all_ts.index.isin(overlap)].copy()

    group_cols = ['cell_line_name', 'compound_name_1', 'time', '2D_3D', 'stimuli', 'dosage_1']
    for c in group_cols:
        sub[c] = sub[c].fillna('NA_MISSING').astype(str)
    sub['gkey'] = sub[group_cols[0]]
    for c in group_cols[1:]:
        sub['gkey'] = sub['gkey'] + '|' + sub[c]

    resp_mat = resps.loc[sub.index, protein_cols].apply(pd.to_numeric, errors='coerce')

    correlations = []
    group_sizes = []
    for gkey, idxs in sub.groupby('gkey').groups.items():
        idxs = list(idxs)
        if len(idxs) < 2:
            continue
        group_sizes.append(len(idxs))
        for i, j in combinations(idxs, 2):
            a = resp_mat.loc[i].to_numpy(dtype=float)
            b = resp_mat.loc[j].to_numpy(dtype=float)
            mask = ~np.isnan(a) & ~np.isnan(b)
            if mask.sum() < 5:  # need enough overlapping proteins to be meaningful
                continue
            r = np.corrcoef(a[mask], b[mask])[0, 1]
            if not np.isnan(r):
                correlations.append(r)

    return np.array(correlations), len(group_sizes)


def main():
    corrs, n_groups = compute_replicate_correlations(protein_set='258')
    print(f"Replicate groups (>=2 members): {n_groups}  (paper reports 1,317)")
    print(f"Pairwise correlations computed: {len(corrs)}")
    print(f"Mean: {corrs.mean():.3f}  Median: {np.median(corrs):.3f}  "
          f"Range: [{corrs.min():.3f}, {corrs.max():.3f}]  (paper reports mean ~0.90, range ~0.2-1.0)")

    sweep = json.load(open('run/sweep_results.json'))
    model_display = {
        'attention': 'Attention-based NN', 'rf': 'Random Forest',
        'xgb': 'XGBoost Regressor', 'tsnn': 'TS-Inspired NN', 'ensemble': 'Ensemble Learning',
    }
    colors = {'attention': 'red', 'rf': 'orange', 'xgb': 'green', 'tsnn': 'purple', 'ensemble': 'blue'}

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(corrs, bins=20, color='lightblue', edgecolor='black')
    for model, label in model_display.items():
        r = sweep[f'258|mean|{model}']['corr_mean']
        ax.axvline(r, color=colors[model], linestyle='--', linewidth=2, label=label)
    theoretical_limit = corrs.mean()
    ax.axvline(theoretical_limit, color='black', linestyle='-.', linewidth=2, label='Theoretical Limit (mean replicate r)')
    ax.text(0.03, 0.7, f"Avg. Correlation: {theoretical_limit:.2f}", transform=ax.transAxes,
            fontsize=12, bbox=dict(boxstyle='round', facecolor='lightgray'))
    ax.set_xlabel('Correlation')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of Replicate-Pair Correlations (Reproduced, 258-protein set)')
    ax.legend(loc='upper left')
    fig.tight_layout()
    fig.savefig('run/Figure_16_reproduced.png', dpi=150)
    print("Saved run/Figure_16_reproduced.png")

    np.save('run/replicate_correlations_258.npy', corrs)


if __name__ == '__main__':
    main()
