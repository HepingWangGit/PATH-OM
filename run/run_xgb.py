import matplotlib
matplotlib.use('Agg')
import numpy as np
import time, json, traceback, pickle, os

t0 = time.time()
CACHE_DIR = '/home/claude/TargetScore/run/cache'
os.makedirs(CACHE_DIR, exist_ok=True)

def cache_path(name):
    return os.path.join(CACHE_DIR, name)

def save_cache(name, obj):
    with open(cache_path(name), 'wb') as f:
        pickle.dump(obj, f)

def load_cache(name):
    with open(cache_path(name), 'rb') as f:
        return pickle.load(f)

def have_cache(name):
    return os.path.exists(cache_path(name))

try:
    from data_loader import load_data
    from data_preprocessing import data_processing
    from imputation import train_predictive_models
    from training import training_machine
    from evaluation import evaluate_model_on_all_CLs

    print("=== load_data ===", flush=True)
    if have_cache('data_dict.pkl'):
        data_dict = load_cache('data_dict.pkl')
        print("(loaded from cache)", flush=True)
    else:
        data_dict = load_data()
        save_cache('data_dict.pkl', data_dict)
    print("elapsed:", time.time()-t0, flush=True)

    print("=== data_processing ===", flush=True)
    if have_cache('features.pkl'):
        features = load_cache('features.pkl')
        print("(loaded from cache)", flush=True)
    else:
        features = data_processing(data_dict)
        save_cache('features.pkl', features)
    print("elapsed:", time.time()-t0, flush=True)

    print("=== train_predictive_models (ML-based imputation, XGBoost per protein) ===", flush=True)
    if have_cache('imputed.pkl'):
        targetscores, non_NA_mask = load_cache('imputed.pkl')
        print("(loaded from cache)", flush=True)
    else:
        targetscores, non_NA_mask = train_predictive_models(features=features, dict_list=data_dict['dicts'])
        save_cache('imputed.pkl', (targetscores, non_NA_mask))
    print("elapsed:", time.time()-t0, flush=True)

    print("=== training_machine (model_type=c-ml, model_name=xgb) ===", flush=True)
    if have_cache('trained.pkl'):
        trained_model, pcas = load_cache('trained.pkl')
        print("(loaded from cache)", flush=True)
    else:
        trained_model, pcas = training_machine(targetscores, data_dict, features, model_type='c-ml', model_name='xgb')
        save_cache('trained.pkl', (trained_model, pcas))
    print("elapsed:", time.time()-t0, flush=True)

    print("=== evaluate_model_on_all_CLs ===", flush=True)
    sample_nums, cl_results, actual_corr, keys = evaluate_model_on_all_CLs(trained_model, data_dict, non_NA_mask, pca_list=pcas, model_type='c-ml')
    print("elapsed:", time.time()-t0, flush=True)

    result = {
        "overall_correlation": float(actual_corr),
        "n_cell_lines_in_test": len(keys),
        "total_test_samples": int(sum(sample_nums)),
        "per_cl_corr_mean": float(np.nanmean(cl_results)),
        "per_cl_corr_median": float(np.nanmedian(cl_results)),
    }
    print("=== RESULT ===")
    print(json.dumps(result, indent=2))
    with open('/home/claude/TargetScore/run/result_xgb.json', 'w') as f:
        json.dump(result, f, indent=2)

except Exception as e:
    print("ERROR:", e)
    traceback.print_exc()
