import numpy as np
import pandas as pd
import random
import xgboost as xgb
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score

def train_predictive_models(features, dict_list):

    drug_vecs    = features['drug']
    time_vecs    = features['time']
    dose_vecs    = features['dose']
    dim_vecs     = features['dim']
    vectors_cna  = features['cna']
    vectors_mexp = features['mrna']
    mut_vec_1    = features['hotspot']
    mut_vec_2    = features['mut_type']
    ccle_data    = features['ccle']
    targetscores = features['targetscores']

    baselines = ccle_data.iloc[:,1:].to_numpy()
    labels = targetscores.iloc[:,8:]

    feature_list = [drug_vecs, time_vecs, dose_vecs, dim_vecs, vectors_cna, vectors_mexp, mut_vec_1, mut_vec_2]
    feature_name_list = ['drug', 'time', 'dose', 'dim', 'cna', 'mrna', 'hotspot', 'mut_type']

    models = {}
    pcas_all = {}

    MSE_train = []
    MSE_test = []

    R_train = []
    R_test = []

    R2_train = []
    R2_test = []

    prob_columns = []
    num_samples = []

    y_tests = []
    y_pred_tests = []

    count = 0

    for i in range(labels.shape[1]):

        #print(len(num_samples), len(R_train))
        
        pca_dim = 10
        random_integer = random.randint(0, 9999)
        kf = KFold(n_splits=5, shuffle=True, random_state=random_integer)
        fold = 0

        non_na_uids = labels[labels.iloc[:,i].notna()].index
        non_na_idx = np.array(labels.index.isin(non_na_uids))

        baselines = ccle_data.iloc[non_na_idx, 1:].to_numpy()    
        y = labels.iloc[non_na_idx, i].to_numpy()
        count = count + len(y)
        print("Protein",i,'#Samples:',len(y))

        if len(y) < 5:
                print('HEYYYYYYYYYYYYY')
                prob_columns.append(targetscores.columns[8+i])
                MSE_train.append(np.nan)
                MSE_test.append(np.nan)
                R_train.append(np.nan)
                R_test.append(np.nan)
                R2_train.append(np.nan)
                R2_test.append(np.nan)
                models[i] = '#samples < 5'
                pcas_all[i] = '#samples < 5'
                continue

        for train_index, test_index in kf.split(baselines):
            pcas = []
            fold += 1
            print("Fold", fold)

            train_features = [baselines[train_index]]
            test_features = [baselines[test_index]]

            y_train, y_test = y[train_index], y[test_index]

            for feature, name in zip(feature_list, feature_name_list):

                if (name in ['cna', 'hotspot', 'mut_type']):

                    if len(y) >= 50:
                        pca = PCA(n_components=pca_dim)
                        train_feature = pca.fit_transform(feature[train_index])
                        test_feature = pca.transform(feature[test_index])
                        pcas.append(pca)
                    else:
                        train_feature = feature[train_index]
                        test_feature = feature[test_index]
                else:
                    train_feature = feature[train_index]
                    test_feature = feature[test_index]
                
                train_features.append(train_feature)
                test_features.append(test_feature)
            
            model = xgb.XGBRegressor(
            objective='reg:squarederror',
            max_depth=3,
            learning_rate=0.3,
            n_estimators=30,
            subsample=0.8,
            colsample_bytree=0.8)

            train_features = np.concatenate(train_features, axis=1)
            test_features = np.concatenate(test_features, axis=1)

            model.fit(train_features, y_train)

            y_pred_train = model.predict(train_features)
            y_pred_test = model.predict(test_features)

            print('R Train:', np.corrcoef(y_pred_train, y_train)[0, 1])
            print('R Test:', np.corrcoef(y_pred_test, y_test)[0, 1])
            print('R2 Train:',r2_score(y_pred=y_pred_train, y_true=y_train))
            print('R2 Test:', r2_score(y_pred=y_pred_test, y_true=y_test))
            print('\n')

            R_train.append(np.corrcoef(y_pred_train, y_train)[0, 1])
            R_test.append(np.corrcoef(y_pred_test, y_test)[0, 1])
            R2_train.append(r2_score(y_pred=y_pred_train, y_true=y_train))
            R2_test.append(r2_score(y_pred=y_pred_test, y_true=y_test))

            num_samples.append(y.shape[0])

            y_pred_tests.append(y_pred_test)
            y_tests.append(y_test)
            
            break #ONYL FOR CODE DEBUG PURPOSES

        models[i] = model
        pcas_all[i] = pcas

    labels_t = targetscores.iloc[:,8:].to_numpy()
    non_NA_mask = (np.isnan(labels_t) == 0)

    genomics_data, drug2target, dose_dict, dim_dict, time_dict, stimuli_dict = dict_list

    labels = targetscores.iloc[:,8:]

    for i in range(labels.shape[1]):
        print("Protein",i)

        non_na_uids = labels[labels.iloc[:,i].notna()].index
        na_uids = labels[labels.iloc[:,i].isna()].index
        non_na_idx = np.array(labels.index.isin(non_na_uids))
        na_idx = np.array(labels.index.isin(na_uids))

        if (non_na_uids.shape[0]) != 0:
        
            baseline_test = ccle_data.iloc[na_idx, 1:].to_numpy()

            drug_test = np.array([drug2target[drug] for drug in targetscores.loc[na_uids,'Drug-Name']])

            stimuli_test = np.array([stimuli_dict[sti] for sti in targetscores.loc[na_uids, 'Stimuli']])
            stimuli_test = np.reshape(stimuli_test, (stimuli_test.shape[0],1))

            time_test = np.array([time_dict[time] for time in targetscores.loc[na_uids,'Time']])
            time_test = np.reshape(time_test, (time_test.shape[0],1))

            dose_test = np.array([dose_dict[dose] for dose in targetscores.loc[na_uids,'Dose']])
            dose_test = np.reshape(dose_test, (dose_test.shape[0],1))

            dim_test = np.array([dim_dict[dim] for dim in targetscores.loc[na_uids,'2D-3D']])
            dim_test = np.reshape(dim_test, (dim_test.shape[0],1))
            dim_test = dim_test.astype(np.float32)

            cna_test = np.array([genomics_data[cl]['CNA'] for cl in targetscores.loc[na_uids,'CL-Name']])
            mrna_test = np.array([genomics_data[cl]['mRNA'] for cl in targetscores.loc[na_uids,'CL-Name']])

            mut_test = [[item for item in genomics_data[cl]['Mutation'].to_numpy()] for cl in targetscores.loc[na_uids,'CL-Name']]

            mut_1_test = np.array(mut_test)[:,:,0]
            mut_2_test = np.array(mut_test)[:,:,1]

            # PATCHED: the original condition `na_uids.shape[0] > 50` checked the number of
            # MISSING samples (the rows we're about to predict), but whether PCA objects
            # actually exist in pcas_all[i] was decided during training by `len(y) >= 50`
            # (the number of NON-missing/training samples for this protein) -- a different,
            # uncorrelated quantity. Whenever a protein had few observed samples but many
            # missing ones (common for the wider 318-protein set), this mismatch made the
            # code try pcas_all[i][0] on an empty list, crashing with
            # "IndexError: list index out of range". Fixed by checking whether PCA was
            # actually fit for this protein (pcas_all[i] is a non-empty list of 3 fitted
            # PCA objects) rather than re-deriving the condition from a different count.
            fitted_pcas = pcas_all.get(i)
            if isinstance(fitted_pcas, list) and len(fitted_pcas) == 3:
                pca_cna = fitted_pcas[0].transform(cna_test)
                pca_mut1 = fitted_pcas[1].transform(mut_1_test)
                pca_mut2 = fitted_pcas[2].transform(mut_2_test)
            else:
                pca_cna = cna_test
                pca_mut1 = mut_1_test
                pca_mut2 = mut_2_test

            x_for_pred = np.concatenate([baseline_test, drug_test, time_test, dim_test, mrna_test, pca_cna, pca_mut1, pca_mut2, dose_test], axis=1)

            # PATCHED: models[i] is the sentinel string '#samples < 5' when this protein had
            # fewer than 5 observed training samples (see the `continue` branch above), which
            # has no .predict(). Rather than crash the whole sweep on such a protein, leave
            # its missing values as NaN (they'll be dropped later by the existing
            # all-NaN-column `remcol` cleanup, or survive as legitimately-unimputable cells
            # if only some rows are NaN) and note it.
            if isinstance(models[i], str):
                print(f"    [imputation] skipping protein {i} ({targetscores.columns[i+8]!r}): "
                      f"{models[i]}, leaving {na_uids.shape[0]} missing values as NaN")
            else:
                y_pred = models[i].predict(x_for_pred)
                targetscores.loc[na_uids, targetscores.columns[i+8]] = y_pred
        else:
            print(i)
    
    print(targetscores.iloc[:,8:].isna().sum().sum())
    print("Train Avg. R:",np.nanmean(np.abs(R_train)))
    print("Test Avg. R:",np.nanmean(np.abs(R_test)))
    print("Train Avg. R2:",np.nanmean(R2_train))
    print("Test Avg. R2:",np.nanmean(R2_test))
    
    count = 0
    remcol = []
    for column in targetscores.columns:
        if targetscores[column].isna().sum() > 0:
            if targetscores[column].isna().sum() == targetscores.shape[0]:
                count += 1
                remcol.append(column)

    targetscores = targetscores.drop(columns=remcol)
    
    return(targetscores, non_NA_mask)