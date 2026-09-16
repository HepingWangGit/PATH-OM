import matplotlib
matplotlib.use('Agg')
import time, sys
sys.path.insert(0, '/home/claude/TargetScore/run')

from data_loader import load_data
from data_preprocessing import data_processing
from mean_imputation import mean_impute_targetscores
from training import training_machine
from evaluation import evaluate_model_metrics

t0 = time.time()
print("load_data(258)...", flush=True)
data_dict = load_data(protein_set='258')
print("elapsed", time.time()-t0, "n_proteins", data_dict['targetscores'].shape[1]-8, flush=True)

print("data_processing...", flush=True)
features = data_processing(data_dict)
print("elapsed", time.time()-t0, flush=True)

print("mean_impute_targetscores...", flush=True)
targetscores_imp, non_NA_mask = mean_impute_targetscores(features['targetscores'])
print("remaining NaNs after mean-impute:", targetscores_imp.iloc[:,8:].isna().sum().sum(), flush=True)
print("elapsed", time.time()-t0, flush=True)

print("training_machine (c-ml, xgb, real 5-fold)...", flush=True)
fold_results = training_machine(targetscores_imp, data_dict, features, model_type='c-ml', model_name='xgb')
print("num folds returned:", len(fold_results), flush=True)
print("elapsed", time.time()-t0, flush=True)

print("evaluate_model_metrics per fold...", flush=True)
for fr in fold_results:
    m = evaluate_model_metrics(fr['model'], data_dict, data_dict['test_targetscores'], pca_list=fr['pcas'], model_type='c-ml')
    print(fr['fold'], m, flush=True)
print("elapsed", time.time()-t0, flush=True)
print("SMOKE TEST OK", flush=True)
