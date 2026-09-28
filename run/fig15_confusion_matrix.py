"""
Reproduces Figure 15 ("Confusion matrix for interval placement with 5 intervals")
using REAL held-out predictions from a model trained by the replicated pipeline
(258-protein / mean-imputation XGBoost config -- see train_for_figures.py), fed
through the ORIGINAL repo's confusion-matrix plotting code
(ModulerMI/plotting2.py:classify_predicted_true), adapted only to save to a file
instead of plt.show() and to accept pre-computed predicted/true arrays.

The 5 interval-placement bins [-inf,-1,-0.25,0.25,1,inf] and their labels match
utils.classification_metric(), which is what the paper's reported IP-Accuracy
numbers (Table 1/2, and this figure) are computed from.
"""
import sys, os, pickle
sys.path.insert(0, os.path.dirname(__file__))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix


def build_predictions(bundle, data_dict):
    """Re-implements the feature-construction + predict() steps of
    evaluation.evaluate_model_metrics(), returning the raw (pred, true) arrays
    instead of aggregated metrics, so we can feed them straight into the
    original confusion-matrix plotting code."""
    model, pcas = bundle['model'], bundle['pcas']
    test_targetscores = bundle['test_aligned']

    genomics_data, drug2target, dose_dict, dim_dict, time_dict, stimuli_dict = data_dict['dicts']
    ccle = data_dict['ccle']

    test_ccle_data = pd.DataFrame(columns=ccle.columns)
    order_test_ts = test_targetscores['CL-Name'].value_counts().to_dict()
    for cl_name in set(test_targetscores['CL-Name']):
        if cl_name in set(ccle['CL-Name']):
            num = order_test_ts[cl_name]
            temp_df = ccle.loc[ccle['CL-Name'] == cl_name, :]
            if temp_df.shape[0] == 1:
                dup = pd.concat([temp_df] * num, axis=0)
                test_ccle_data = pd.concat([test_ccle_data, dup], ignore_index=True)

    test_labels = test_targetscores.iloc[:, 8:].to_numpy()

    test_drug_vecs = np.array([drug2target[d] for d in test_targetscores['Drug-Name']])
    test_time_vecs = np.reshape(np.array([time_dict[t] for t in test_targetscores['Time']]), (-1, 1)).astype(np.float32)
    test_dose_vecs = np.reshape(np.array([dose_dict[d] for d in test_targetscores['Dose']]), (-1, 1)).astype(np.float32)
    test_dim_vecs = np.reshape(np.array([dim_dict[d] for d in test_targetscores['2D-3D']]), (-1, 1)).astype(np.float32)
    test_vectors_cna = np.array([genomics_data[k]['CNA'] for k in test_targetscores['CL-Name']])
    test_vectors_mexp = np.array([genomics_data[k]['mRNA'] for k in test_targetscores['CL-Name']])
    test_vectors_mut = [[i for i in genomics_data[k]['Mutation'].to_numpy()] for k in test_targetscores['CL-Name']]
    test_mut_vec_1 = np.array(test_vectors_mut)[:, :, 0]
    test_mut_vec_2 = np.array(test_vectors_mut)[:, :, 1]

    test_data_list = [test_drug_vecs, test_time_vecs, test_dose_vecs, test_dim_vecs,
                       pcas[0].transform(test_vectors_cna), test_vectors_mexp,
                       pcas[1].transform(test_mut_vec_1), pcas[2].transform(test_mut_vec_2)]
    test_data_list = np.concatenate(test_data_list, axis=1)  # model_type == 'c-ml'

    pred = model.predict(test_data_list)
    non_na = ~np.isnan(test_labels)
    return np.asarray(pred)[non_na].flatten(), test_labels[non_na].flatten()


def classify_predicted_true(predicted, true, bins, labels, plot_title, out_path):
    """Verbatim adaptation of ModulerMI/plotting2.py:classify_predicted_true(),
    changed only to savefig() instead of plt.show()."""
    predicted, true = np.array(predicted), np.array(true)
    predicted_classes = np.digitize(predicted, bins) - 1
    true_classes = np.digitize(true, bins) - 1
    num_bins = len(bins) - 1
    predicted_classes[predicted_classes == num_bins] = num_bins - 1
    true_classes[true_classes == num_bins] = num_bins - 1

    accuracy = accuracy_score(true_classes, predicted_classes)
    cm = confusion_matrix(true_classes, predicted_classes)
    row_acc = np.diag(cm) / np.sum(cm, axis=1)
    col_acc = np.diag(cm) / np.sum(cm, axis=0)
    cm_ext = np.zeros((cm.shape[0] + 1, cm.shape[1] + 1), dtype=float)
    cm_ext[:-1, :-1] = cm
    cm_ext[:-1, -1] = row_acc * 100
    cm_ext[-1, :-1] = col_acc * 100
    cm_ext[-1, -1] = accuracy * 100

    mask = np.zeros_like(cm_ext, dtype=bool)
    mask[-1, :-1] = True
    mask[:-1, -1] = True
    mask[-1, -1] = False

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_ext, annot=True, fmt='.1f', cmap='Reds', mask=~mask, cbar=False,
                xticklabels=labels + ['Row Acc (%)'], yticklabels=labels + ['Col Acc (%)'])
    sns.heatmap(cm_ext, annot=True, fmt='.1f', cmap='Blues', mask=mask, cbar=False,
                xticklabels=labels + ['Row Acc (%)'], yticklabels=labels + ['Col Acc (%)'])
    plt.xlabel('Predicted Class')
    plt.ylabel('True Class')
    plt.title(f"{plot_title}\nOverall Accuracy: {accuracy * 100:.2f}%")
    plt.xticks(rotation=0)
    plt.yticks(rotation=0)
    plt.gca().xaxis.tick_top()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Saved {out_path}, overall accuracy={accuracy*100:.2f}%")
    return accuracy


def main():
    with open('cache/fig_model_bundle.pkl', 'rb') as f:
        bundle = pickle.load(f)
    with open('cache/data_dict_258.pkl', 'rb') as f:
        data_dict = pickle.load(f)

    pred, true = build_predictions(bundle, data_dict)
    bins = [-np.inf, -1, -0.25, 0.25, 1, np.inf]
    labels = ['Very High\nNegative', 'Negative', 'Neutral', 'Positive', 'Very High\nPositive']
    classify_predicted_true(pred, true, bins, labels,
                             "Confusion Matrix with Individual Accuracies (Reproduced, XGBoost 258/mean)",
                             "Figure_15_reproduced.png")


if __name__ == '__main__':
    main()
