import matplotlib
matplotlib.use('Agg')
import numpy as np
import time, json, os, pickle, sys, traceback

CACHE_DIR = '/home/claude/TargetScore/run/cache_fskorkut'
RESULTS_PATH = '/home/claude/TargetScore/run/sweep_results_fskorkut.json'

def cpath(name):
    return os.path.join(CACHE_DIR, name)

def load_cache(name):
    with open(cpath(name), 'rb') as f:
        return pickle.load(f)

def load_results():
    with open(RESULTS_PATH) as f:
        return json.load(f)

def save_results(results):
    with open(RESULTS_PATH, 'w') as f:
        json.dump(results, f, indent=2)


from training import training_machine
from evaluation import evaluate_model_metrics

PROTEIN_SETS = ['289', '528']
IMPUTATIONS = ['mean', 'ml']

t_start = time.time()
results = load_results()

def log(msg):
    print(f"[{time.time()-t_start:8.1f}s] {msg}", flush=True)

for protein_set in PROTEIN_SETS:
    data_dict = load_cache(f'data_dict_{protein_set}.pkl')
    features = load_cache(f'features_{protein_set}.pkl')
    n_proteins = data_dict['targetscores'].shape[1] - 8

    for imputation in IMPUTATIONS:
        targetscores_imp, non_NA_mask = load_cache(f'imputed_{protein_set}_{imputation}.pkl')

        model_type, model_name = 'nn', 'attention'
        key = f"{protein_set}|{imputation}|{model_name}"
        old = results.get(key, {})
        old_corr = old.get('corr_mean')

        log(f">>> RERUN {key} (Bug 9 fix) -- old corr_mean={old_corr}")
        t0 = time.time()
        try:
            fold_results, used_protein_columns = training_machine(targetscores_imp, data_dict, features, model_type=model_type, model_name=model_name)

            meta_cols = list(data_dict['test_targetscores'].columns[:8])
            test_targetscores_aligned = data_dict['test_targetscores'][meta_cols + used_protein_columns]

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
                'note': 'rerun 2026-09-30 after Bug 9 fix (CustomAttentionModel self-attention rewrite)',
            }
            save_results(results)
            log(f"<<< DONE {key} in {time.time()-t0:.1f}s : corr={results[key]['corr_mean']:.3f}+/-{results[key]['corr_std']:.3f} "
                f"(was {old_corr}) r2={results[key]['r2_mean']:.3f}+/-{results[key]['r2_std']:.3f} ipacc={results[key]['ipacc_mean']:.3f}+/-{results[key]['ipacc_std']:.3f}")
        except Exception as e:
            log(f"!!! ERROR on {key}: {e}")
            traceback.print_exc()

log("=== ATTENTION RERUN COMPLETE ===")
