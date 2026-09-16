from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from tensorflow.keras.optimizers import Adam
from models import CustomTSModel, CustomAttentionModel, WeightedAverageEnsemble

def model_shit(data, model_type, model_name, features, labels):

    X_train, y_train = features, labels
    
    # --- 5-Fold Cross-Validation on Training Data ---
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_train_corr = []
    cv_val_corr = []
    fold = 0
    for train_idx, val_idx in kf.split(X_train):
        fold += 1
        train_features = []
    val_features = []
    pcas = []

    for train_idx, val_idx in kf.split(labels):
        fold += 1
        
        for key,value in features.items():
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
        
        # Instantiate an XGBoost regressor (you can adjust hyperparameters as needed)
        if model_name == 'xgb':
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
            model_cv = RandomForestRegressor(n_estimators=30, criterion='squared_error', max_depth=3)
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
            model_rf = RandomForestRegressor(n_estimators=30, criterion='squared_error', max_depth=3)
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
            history = model_cv.fit(X_tr, y_tr,
                epochs=100,
                batch_size=32,
                shuffle=True,
                validation_data=(X_val, y_val),
                verbose=0)
        
        y_pred_tr = model_cv.predict(X_tr)
        y_pred_val = model_cv.predict(X_val)
        
        train_corr = np.corrcoef(y_pred_tr.flatten(), y_tr.flatten())[0, 1]
        val_corr = np.corrcoef(y_pred_val.flatten(), y_val.flatten())[0, 1]
        cv_train_corr.append(train_corr)
        cv_val_corr.append(val_corr)
        
        print(f"Fold {fold}: Train Corr = {train_corr:.2f}, Val Corr = {val_corr:.2f}")
        break
    
    return model_cv, pcas