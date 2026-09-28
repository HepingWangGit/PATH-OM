import matplotlib
matplotlib.use('Agg')
import numpy as np
import time, json, os, pickle, sys, traceback

# UPDATED 2026-09-28: fresh cache dir + results path for the fs_korkut.csv-based rerun,
# so this never silently reuses pickles built from the old fs_mod.csv substitution
# (protein_set labels also changed from '258'/'318' to '289'/'528' to match, since the
# reconstructed fs_korkut.csv now yields the paper's exact protein counts).
CACHE_DIR = '/home/claude/TargetScore/run/cache_fskorkut'
os.makedirs(CACHE_DIR, exist_ok=True)
RESULTS_PATH = '/home/claude/TargetScore/run/sweep_results_fskorkut.json'

def cpath(name):
    return os.path.join(CACHE_DIR, name)

def save_cache(name, obj):
    with open(cpath(name), 'wb') as f:
        pickle.dump(obj, f)

def load_cache(name):
    with open(cpath(name), 'rb') as f:
        return pickle.load(f)

def have_cache(name):
    return os.path.exists(cpath(name))

def load_results():
    if os.path.exists(RESULTS_PATH):
        with open(RESULTS_PATH) as f:
            return json.load(f)
    return {}

def save_results(results):
    with open(RESULTS_PATH, 'w') as f:
        json.dump(results, f, indent=2)


from data_loader import load_data
from data_preprocessing import data_processing
from imputation import train_predictive_models
from mean_imputation import mean_impute_targetscores
from training import training_machine
from evaluation import evaluate_model_metrics

PROTEIN_SETS = ['289', '528']
IMPUTATIONS = ['mean', 'ml']
MODELS = [('c-ml', 'xgb'), ('c-ml', 'rf'), ('c-ml', 'ensemble'), ('nn', 'tsnn'), ('nn', 'attention')]

t_start = time.time()
results = load_results()

def log(msg):
    print(f"[{time.time()-t_start:8.1f}s] {msg}", flush=True)

for protein_set in PROTEIN_SETS:
    log(f"=== protein_set={protein_set} : load_data ===")
    dd_cache = f'data_dict_{protein_set}.pkl'
    if have_cache(dd_cache):
        data_dict = load_cache(dd_cache)
        log("(loaded data_dict from cache)")
    else:
        data_dict = load_data(protein_set=protein_set)
        save_cache(dd_cache, data_dict)
    n_proteins = data_dict['targetscores'].shape[1] - 8
    log(f"protein_set={protein_set}: {n_proteins} protein columns, "
        f"{data_dict['targetscores'].shape[0]} train rows, {data_dict['test_targetscores'].shape[0]} test rows")

    feat_cache = f'features_{protein_set}.pkl'
    if have_cache(feat_cache):
        features = load_cache(feat_cache)
        log("(loaded features from cache)")
    else:
        features = data_processing(data_dict)
        save_cache(feat_cache, features)

    for imputation in IMPUTATIONS:
        log(f"--- protein_set={protein_set} imputation={imputation}: building imputed targetscores ---")
        imp_cache = f'imputed_{protein_set}_{imputation}.pkl'
        if have_cache(imp_cache):
            targetscores_imp, non_NA_mask = load_cache(imp_cache)
            log("(loaded imputed targetscores from cache)")
        else:
            if imputation == 'mean':
                targetscores_imp, non_NA_mask = mean_impute_targetscores(features['targetscores'])
            else:
                targetscores_imp, non_NA_mask = train_predictive_models(features=features, dict_list=data_dict['dicts'])
            save_cache(imp_cache, (targetscores_imp, non_NA_mask))
        log(f"imputed targetscores shape: {targetscores_imp.shape}")

        for model_type, model_name in MODELS:
            key = f"{protein_set}|{imputation}|{model_name}"
            if key in results:
                log(f"SKIP (already have result): {key}")
                continue

            log(f">>> RUN {key} (model_type={model_type})")
            t0 = time.time()
            try:
                fold_results, used_protein_columns = training_machine(targetscores_imp, data_dict, features, model_type=model_type, model_name=model_name)

                # Subset the held-out test set to the same protein columns the model was
                # actually trained/evaluated on (see training.py's used_protein_columns note).
                meta_cols = list(data_dict['test_targetscores'].columns[:8])
                test_targetscores_aligned = data_dict['test_targetscores'][meta_cols + used_protein_columns]
                if len(used_protein_columns) < n_proteins:
                    log(f"    NOTE: {n_proteins - len(used_protein_columns)} protein column(s) dropped from training "
                        f"(all-NaN in this training split); test set aligned to {len(used_protein_columns)} columns for evaluation")

                fold_metrics = []
                for fr in fold_results:
                    m = evaluate_model_metrics(fr['model'], data_dict, test_targetscores_aligned,
                                                pca_list=fr['pcas'], model_type=model_type)
                    m['fold'] = fr['fold']
                    m['train_corr'] = fr['train_corr']
                    m['val_corr'] = fr['val_corr']
                    fold_metrics.append(m)
                    log(f"    fold {fr['fold']}: test corr={m['corr']:.3f} r2={m['r2']:.3f} ipacc={m['ipacc']:.3f}")

                corrs = [m['corr'] for m in fold_metrics]
                r2s = [m['r2'] for m in fold_metrics]
                ipaccs = [m['ipacc'] for m in fold_metrics]

                results[key] = {
                    'protein_set': protein_set,
                    'n_proteins': int(n_proteins),
                    'imputation': imputation,
                    'model_name': model_name,
                    'model_type': model_type,
                    'corr_mean': float(np.mean(corrs)), 'corr_std': float(np.std(corrs)),
                    'r2_mean': float(np.mean(r2s)), 'r2_std': float(np.std(r2s)),
                    'ipacc_mean': float(np.mean(ipaccs)), 'ipacc_std': float(np.std(ipaccs)),
                    'n_test': fold_metrics[0]['n'],
                    'n_test_scored': fold_metrics[0]['n_scored'],
                    'fold_metrics': fold_metrics,
                    'elapsed_sec': time.time() - t0,
                }
                save_results(results)
                log(f"<<< DONE {key} in {time.time()-t0:.1f}s : corr={results[key]['corr_mean']:.3f}+/-{results[key]['corr_std']:.3f} "
                    f"r2={results[key]['r2_mean']:.3f}+/-{results[key]['r2_std']:.3f} ipacc={results[key]['ipacc_mean']:.3f}+/-{results[key]['ipacc_std']:.3f}")
            except Exception as e:
                log(f"!!! ERROR on {key}: {e}")
                traceback.print_exc()
                results[key] = {'error': str(e)}
                save_results(results)

log("=== SWEEP COMPLETE ===")
