from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.model_selection import KFold, GroupKFold
from sklearn.decomposition import PCA
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from tensorflow.keras.optimizers import Adam
from models import CustomTSModel, CustomAttentionModel, WeightedAverageEnsemble


class MeanBaselineRegressor:
    # UPDATED 2026-09-30: trivial baseline -- ignores all features, predicts each
    # protein's training-fold mean for every row. Added so the real models' reported
    # r/R2 can be read against a naive floor rather than assumed to reflect skill.
    def fit(self, X, y):
        self.means_ = np.nanmean(y, axis=0)
        return self

    def predict(self, X):
        n = X.shape[0] if hasattr(X, 'shape') else len(X)
        return np.tile(self.means_, (n, 1))


# UPDATED 2026-09-30 (replicate-leakage fix): 84% of rows in this dataset share a
# (cell line, drug, time, dose) "condition" with at least one other row (some groups
# have 30+ replicates). The original KFold(shuffle=True) split at the ROW level, so
# replicate rows of the same condition routinely ended up in both train and
# validation within a fold -- letting the model partly see near-duplicates of a
# validation example during training, which inflates reported correlation/R2 relative
# to genuinely unseen conditions. Fixed by switching to GroupKFold on a `groups` array
# (the same condition key), so every row from a given condition falls entirely in one
# fold. `groups=None` falls back to the old row-level KFold for any caller that
# doesn't have a group key yet, but training.py now always supplies one.
#
# PATCHED (see TargetScore_replication_report.md): the original model_shit() had two
# bugs that together meant genuine 5-fold CV never happened:
#   1. A dead first "for train_idx, val_idx in kf.split(X_train): fold += 1" loop that
#      did nothing but burn through 5 fold indices (explaining the "Fold 6" seen in logs
#      for what should have been fold 1 of the real loop).
#   2. A `break` at the end of the first (and, because of bug 1, sixth-numbered)
#      iteration of the real loop, so only one 80/20 split was ever used.
# This version removes both, actually iterates all 5 folds, and returns one
# (model, pcas) pair per fold instead of a single model, so the caller can evaluate
# each fold's model on the true held-out test set and report a genuine mean +/- std,
# matching what the paper's Table 1/2 captions claim ("5-fold cross-validation").
def model_shit(data, model_type, model_name, features, labels, groups=None):

    y_train = labels

    if groups is not None:
        kf = GroupKFold(n_splits=5, shuffle=True, random_state=42)
        split_iter = kf.split(y_train, groups=groups)
    else:
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        split_iter = kf.split(y_train)
    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(split_iter, start=1):

        train_features = []
        val_features = []
        pcas = []

        for key, value in features.items():
            if key in ['cna', 'hotspot', 'mut_type']:
                pca = PCA(n_components=10)
                train_feature = pca.fit_transform(value[train_idx])
                val_feature = pca.transform(value[val_idx])
                pcas.append(pca)
            else:
                train_feature = value[train_idx]
                val_feature = value[val_idx]

            if key != 'baseline':
                train_features.append(train_feature)
                val_features.append(val_feature)

        if model_type == 'c-ml':
            train_features = np.concatenate(train_features, axis=1)
            val_features = np.concatenate(val_features, axis=1)

        if model_type == 'nn':
            train_features.append(features['baseline'][train_idx])
            val_features.append(features['baseline'][val_idx])

        X_tr, X_val, y_tr, y_val = train_features, val_features, y_train[train_idx], y_train[val_idx]

        # UPDATED 2026-09-30: baseline_only_xgb ignores everything built above and
        # trains the same XGBoost config used for 'xgb' on ONLY the baseline (CCLE
        # protein-level) features -- a second trivial-ish baseline showing how much
        # of xgb's real performance comes from baseline protein levels alone vs. the
        # drug/dose/time/genomics features the full model also sees.
        if model_name == 'baseline_only_xgb':
            X_tr = features['baseline'][train_idx]
            X_val = features['baseline'][val_idx]

        if model_name == 'mean_baseline':
            model_cv = MeanBaselineRegressor()
        elif model_name == 'xgb' or model_name == 'baseline_only_xgb':
            model_cv = xgb.XGBRegressor(
                objective='reg:squarederror',
                max_depth=3,
                learning_rate=0.3,
                n_estimators=30,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42
            )
        elif model_name == 'rf':
            model_cv = RandomForestRegressor(n_estimators=30, criterion='squared_error', max_depth=3, random_state=42)
        elif model_name == 'ensemble':
            model_xgb = xgb.XGBRegressor(
                objective='reg:squarederror',
                max_depth=3,
                learning_rate=0.3,
                n_estimators=30,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42
            )
            model_rf = RandomForestRegressor(n_estimators=30, criterion='squared_error', max_depth=3, random_state=42)
            models = [('xgb', model_xgb), ('rf', model_rf)]
            model_cv = WeightedAverageEnsemble(models=models, avg_check=False)
        elif model_name == 'tsnn':
            model_cv = CustomTSModel(nprots=y_train.shape[1], num_categories=7, embedding_dim=10, fs_list=data['fs_list'], bionet=data['bionetwork'])
        elif model_name == 'attention':
            model_cv = CustomAttentionModel(nprots=y_train.shape[1], num_categories=7, embedding_dim=10)

        if model_type == 'c-ml':
            model_cv.fit(X_tr, y_tr)
        elif model_type == 'nn':
            model_cv.compile(optimizer=Adam(learning_rate=1e-3), loss='mse')
            model_cv.fit(X_tr, y_tr,
                epochs=100,
                batch_size=32,
                shuffle=True,
                validation_data=(X_val, y_val),
                verbose=0)

        y_pred_tr = model_cv.predict(X_tr)
        y_pred_val = model_cv.predict(X_val)

        train_corr = np.corrcoef(y_pred_tr.flatten(), y_tr.flatten())[0, 1]
        val_corr = np.corrcoef(y_pred_val.flatten(), y_val.flatten())[0, 1]

        print(f"Fold {fold}: Train Corr = {train_corr:.3f}, Val Corr = {val_corr:.3f}", flush=True)

        fold_results.append({
            'fold': fold,
            'model': model_cv,
            'pcas': pcas,
            'train_corr': train_corr,
            'val_corr': val_corr,
        })

    return fold_results
