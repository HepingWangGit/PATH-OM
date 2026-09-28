"""
Reproduces Figure 18 ("Histogram of absolute Pearson correlation and number of
samples for cell lines present in the test set") using the ORIGINAL repo's
evaluation.evaluate_model_on_all_CLs() (only patched, as documented in the earlier
replication report, to pass model_type since the committed call site omits the
required argument), run on the real held-out predictions from the 258-protein /
mean-imputation XGBoost model (train_for_figures.py).
"""
import sys, os, pickle
sys.path.insert(0, os.path.dirname(__file__))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from evaluation import evaluate_model_on_all_CLs


def main():
    with open('cache/fig_model_bundle.pkl', 'rb') as f:
        bundle = pickle.load(f)
    with open('cache/data_dict_258.pkl', 'rb') as f:
        data_dict = pickle.load(f)

    # evaluate_model_on_all_CLs reads data_dict['test_targetscores'] internally --
    # swap in the column-aligned test set (same alignment run_sweep.py/fig15 use) so
    # its width matches what the model was actually trained on.
    data_dict = dict(data_dict)
    data_dict['test_targetscores'] = bundle['test_aligned']

    sample_nums, corrs, overall_corr, cl_names = evaluate_model_on_all_CLs(
        bundle['model'], data_dict, bundle['non_NA_mask'], pca_list=bundle['pcas'], model_type='c-ml')

    order = np.argsort(cl_names)
    cl_names = [cl_names[i] for i in order]
    corrs = [corrs[i] for i in order]
    sample_nums = [sample_nums[i] for i in order]

    colors = ['red' if c < 0 else 'blue' for c in corrs]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
    ax1.bar(cl_names, corrs, color=colors)
    ax1.axhline(overall_corr, color='black', linewidth=1.5, label=f'Overall R: {overall_corr:.3f}')
    ax1.set_ylabel('Pearson Correlation')
    ax1.legend(loc='upper right')

    ax2.bar(cl_names, sample_nums, color='blue')
    ax2.set_ylabel('#Samples')
    ax2.set_xlabel('Cell Line Name')
    plt.setp(ax2.get_xticklabels(), rotation=90, fontsize=7)

    fig.suptitle('Per-Cell-Line Correlation and Sample Count (Reproduced, XGBoost 258/mean)')
    fig.tight_layout()
    fig.savefig('Figure_18_reproduced.png', dpi=150)
    print(f"Saved Figure_18_reproduced.png; overall R={overall_corr:.3f}, "
          f"{len(cl_names)} cell lines, corr range [{min(corrs):.3f}, {max(corrs):.3f}]")


if __name__ == '__main__':
    main()
