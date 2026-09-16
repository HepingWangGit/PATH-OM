import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils import multivariate_r2, classification_metric

def evaluate_model(trained_model, data_dict, test_targetscores, non_NA_mask, pca_list, model_type, print_check=False):

    genomics_data, drug2target, dose_dict, dim_dict, time_dict, stimuli_dict = data_dict['dicts']

    #test_targetscores = data_dict['test_targetscores']
    ccle = data_dict['ccle']

    test_ccle_data = pd.DataFrame(columns = ccle.columns)

    order_test_ts = test_targetscores['CL-Name'].value_counts().to_dict()

    count = 0

    for cl_name in set(test_targetscores['CL-Name']):
        
        if cl_name in set(ccle['CL-Name']):

            num = order_test_ts[cl_name]
            temp_df = ccle.loc[ccle['CL-Name']==cl_name,:]  

            if temp_df.shape[0] == 1:
                duplicated_rows = pd.concat([temp_df] * (num), axis=0)
                test_ccle_data = pd.concat([test_ccle_data, duplicated_rows], ignore_index=True)

                count += num

    test_baselines = test_ccle_data.iloc[:,1:].to_numpy().astype(np.float32)
    test_labels = test_targetscores.iloc[:,8:].to_numpy()

    test_drug_vecs = np.array([drug2target[drug] for drug in test_targetscores['Drug-Name']])

    test_stimuli_vecs = np.array([stimuli_dict[sti] for sti in test_targetscores['Stimuli']])
    test_stimuli_vecs = np.reshape(test_stimuli_vecs, (test_stimuli_vecs.shape[0],1))
    test_stimuli_vecs = test_stimuli_vecs.astype(np.float32)

    test_time_vecs = np.array([time_dict[time] for time in test_targetscores['Time']])
    test_time_vecs = np.reshape(test_time_vecs, (test_time_vecs.shape[0],1))
    test_time_vecs = test_time_vecs.astype(np.float32)

    test_dose_vecs = np.array([dose_dict[dose] for dose in test_targetscores['Dose']])
    test_dose_vecs = np.reshape(test_dose_vecs, (test_dose_vecs.shape[0],1))
    test_dose_vecs = test_dose_vecs.astype(np.float32)

    test_dim_vecs = np.array([dim_dict[dim] for dim in test_targetscores['2D-3D']])
    test_dim_vecs = np.reshape(test_dim_vecs, (test_dim_vecs.shape[0],1))
    test_dim_vecs = test_dim_vecs.astype(np.float32)

    test_vectors_cna = np.array([genomics_data[key]['CNA'] for key in test_targetscores['CL-Name']])
    test_vectors_mexp = np.array([genomics_data[key]['mRNA'] for key in test_targetscores['CL-Name']])

    test_vectors_mut = [[item for item in genomics_data[key]['Mutation'].to_numpy()] for key in test_targetscores['CL-Name']]

    test_mut_vec_1 = np.array(test_vectors_mut)[:,:,0]
    test_mut_vec_2 = np.array(test_vectors_mut)[:,:,1]

    test_data_list = [test_drug_vecs,
                      test_time_vecs,
                      test_dose_vecs,
                      test_dim_vecs,
                      pca_list[0].transform(test_vectors_cna),
                      test_vectors_mexp,
                      pca_list[1].transform(test_mut_vec_1),
                      pca_list[2].transform(test_mut_vec_2)]
    
    if model_type == 'nn':
        test_data_list.append(test_baselines)

    if model_type == 'c-ml':
        test_data_list = np.concatenate(test_data_list, axis=1)


    test_data_pred = trained_model.predict(test_data_list)

    non_NA_mask_test = (np.isnan(test_labels) == 0)

    if print_check == True:
        print("\nTest Data Results")
        print("Correlation:",np.corrcoef(test_data_pred[non_NA_mask_test].flatten(), test_labels[non_NA_mask_test].flatten())[0, 1])
        print("R² Score:",multivariate_r2(y_pred=test_data_pred[non_NA_mask_test], y_true=test_labels[non_NA_mask_test]))
        print("Accuracy:",classification_metric(predicted=test_data_pred[non_NA_mask_test],
                                    true=test_labels[non_NA_mask_test],
                                    bins=[-np.inf, -1, -0.25,  0.25, 1, np.inf]))

        print('\n')
        print("#Organic Data Points (Train):", np.sum(non_NA_mask))
        print("#Organic Data Points (Test):", np.sum(non_NA_mask_test))
    return(test_targetscores.shape[0], np.corrcoef(test_data_pred[non_NA_mask_test].flatten(), test_labels[non_NA_mask_test].flatten())[0, 1])

def evaluate_model_metrics(trained_model, data_dict, test_targetscores, pca_list, model_type):
    # ADDED: same feature construction as evaluate_model(), but returns Correlation,
    # R^2, and interval-placement accuracy together (evaluate_model() only ever
    # returned correlation; the other two metrics were computed but only printed when
    # print_check=True). Needed to reproduce Table 1/2's three reported metrics per
    # CV fold.
    genomics_data, drug2target, dose_dict, dim_dict, time_dict, stimuli_dict = data_dict['dicts']
    ccle = data_dict['ccle']

    test_ccle_data = pd.DataFrame(columns=ccle.columns)
    order_test_ts = test_targetscores['CL-Name'].value_counts().to_dict()

    for cl_name in set(test_targetscores['CL-Name']):
        if cl_name in set(ccle['CL-Name']):
            num = order_test_ts[cl_name]
            temp_df = ccle.loc[ccle['CL-Name'] == cl_name, :]
            if temp_df.shape[0] == 1:
                duplicated_rows = pd.concat([temp_df] * num, axis=0)
                test_ccle_data = pd.concat([test_ccle_data, duplicated_rows], ignore_index=True)

    test_baselines = test_ccle_data.iloc[:, 1:].to_numpy().astype(np.float32)
    test_labels = test_targetscores.iloc[:, 8:].to_numpy()

    test_drug_vecs = np.array([drug2target[drug] for drug in test_targetscores['Drug-Name']])

    test_time_vecs = np.array([time_dict[time] for time in test_targetscores['Time']])
    test_time_vecs = np.reshape(test_time_vecs, (test_time_vecs.shape[0], 1)).astype(np.float32)

    test_dose_vecs = np.array([dose_dict[dose] for dose in test_targetscores['Dose']])
    test_dose_vecs = np.reshape(test_dose_vecs, (test_dose_vecs.shape[0], 1)).astype(np.float32)

    test_dim_vecs = np.array([dim_dict[dim] for dim in test_targetscores['2D-3D']])
    test_dim_vecs = np.reshape(test_dim_vecs, (test_dim_vecs.shape[0], 1)).astype(np.float32)

    test_vectors_cna = np.array([genomics_data[key]['CNA'] for key in test_targetscores['CL-Name']])
    test_vectors_mexp = np.array([genomics_data[key]['mRNA'] for key in test_targetscores['CL-Name']])

    test_vectors_mut = [[item for item in genomics_data[key]['Mutation'].to_numpy()] for key in test_targetscores['CL-Name']]
    test_mut_vec_1 = np.array(test_vectors_mut)[:, :, 0]
    test_mut_vec_2 = np.array(test_vectors_mut)[:, :, 1]

    test_data_list = [test_drug_vecs,
                       test_time_vecs,
                       test_dose_vecs,
                       test_dim_vecs,
                       pca_list[0].transform(test_vectors_cna),
                       test_vectors_mexp,
                       pca_list[1].transform(test_mut_vec_1),
                       pca_list[2].transform(test_mut_vec_2)]

    if model_type == 'nn':
        test_data_list.append(test_baselines)
    if model_type == 'c-ml':
        test_data_list = np.concatenate(test_data_list, axis=1)

    test_data_pred = trained_model.predict(test_data_list)
    non_NA_mask_test = (np.isnan(test_labels) == 0)

    pred = np.asarray(test_data_pred)[non_NA_mask_test].flatten()
    true = test_labels[non_NA_mask_test].flatten()

    corr = np.corrcoef(pred, true)[0, 1]
    r2 = multivariate_r2(y_pred=pred, y_true=true)
    ipacc = classification_metric(predicted=pred, true=true, bins=[-np.inf, -1, -0.25, 0.25, 1, np.inf])

    return {'n': int(test_targetscores.shape[0]), 'n_scored': int(non_NA_mask_test.sum()),
            'corr': float(corr), 'r2': float(r2), 'ipacc': float(ipacc)}


def evaluate_model_on_all_CLs(trained_model, data_dict, non_NA_mask, pca_list=None, model_type='c-ml'):
    # PATCHED: original call below omitted the required 'model_type' argument to
    # evaluate_model(), which raises TypeError as committed (missing 1 required
    # positional argument: 'model_type'). Added a model_type parameter here (default
    # 'c-ml', matching this run's XGBoost model_type) and threading it through.

    test_targetscores = data_dict['test_targetscores']

    tot_samples, actual_corr = evaluate_model(trained_model, data_dict, test_targetscores, non_NA_mask, pca_list=pca_list, model_type=model_type, print_check=True)

    cl_results = {}
    sample_nums = []

    for cl_name in list(set(data_dict['test_targetscores']['CL-Name'])):
        test_targetscores2 = test_targetscores[test_targetscores['CL-Name']==cl_name]

        num_samples, corr_cl = evaluate_model(trained_model, data_dict, test_targetscores2, non_NA_mask, pca_list=pca_list, model_type=model_type)
        cl_results[cl_name] = corr_cl
        sample_nums.append(num_samples)

    # Create a list of colors: red for negative values, blue for non-negative values
    return(sample_nums, list(cl_results.values()), actual_corr, list(cl_results.keys()))
