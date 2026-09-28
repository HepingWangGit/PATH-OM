"""
Trains one real XGBoost model on the 258-protein / mean-imputation config (the same
config used throughout Tables 1-2 and the reproducibility section) and caches the
fold-5 fitted model + PCA list + aligned held-out test set, so Figures 15 and 18 can
be built from genuine predictions rather than re-described from the paper's numbers.

Reuses the exact same cached data_dict/features/imputed-targetscores and
training_machine()/evaluate_model_metrics() pipeline as run_sweep.py, so results are
consistent with the sweep_results.json numbers already in Table 1.
"""
import sys, os, pickle, time
sys.path.insert(0, os.path.dirname(__file__))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from training import training_machine
from evaluation import evaluate_model_metrics

CACHE_DIR = 'cache'


def load_cache(name):
    with open(os.path.join(CACHE_DIR, name), 'rb') as f:
        return pickle.load(f)


def main():
    t0 = time.time()
    data_dict = load_cache('data_dict_258.pkl')
    features = load_cache('features_258.pkl')
    targetscores_imp, non_NA_mask = load_cache('imputed_258_mean.pkl')

    print("Training XGBoost on 258-protein / mean-imputation config ...", flush=True)
    fold_results, used_protein_columns = training_machine(
        targetscores_imp, data_dict, features, model_type='c-ml', model_name='xgb')
    print(f"Trained {len(fold_results)} folds in {time.time()-t0:.1f}s", flush=True)

    fr = fold_results[-1]  # one representative fold's fitted model + pcas
    model, pcas = fr['model'], fr['pcas']

    meta_cols = list(data_dict['test_targetscores'].columns[:8])
    test_aligned = data_dict['test_targetscores'][meta_cols + used_protein_columns]

    m = evaluate_model_metrics(model, data_dict, test_aligned, pca_list=pcas, model_type='c-ml')
    print("Sanity check against sweep_results.json (258|mean|xgb):", m)

    with open('cache/fig_model_bundle.pkl', 'wb') as f:
        pickle.dump({
            'model': model, 'pcas': pcas, 'used_protein_columns': used_protein_columns,
            'test_aligned': test_aligned, 'non_NA_mask': non_NA_mask,
        }, f)
    print("Saved cache/fig_model_bundle.pkl")


if __name__ == '__main__':
    main()
