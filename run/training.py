import numpy as np
import pandas as pd
import random
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA
from utils import multivariate_r2
from model_stuff import model_shit

def training_machine(targetscores, dict_list, features, model_type, model_name):

    test_targetscores = dict_list['test_targetscores']
    ccle_data = features['ccle']

    genomics_data, drug2target, dose_dict, dim_dict, time_dict, stimuli_dict = dict_list['dicts']

    count = 0
    remcol = []
    for column in targetscores.columns:
        if targetscores[column].isna().sum() > 0:
            if targetscores[column].isna().sum() == targetscores.shape[0]:
                count += 1
                remcol.append(column)

    targetscores = targetscores.drop(columns=remcol)

    count = 0
    remcol = []
    for column in test_targetscores.columns:
        if test_targetscores[column].isna().sum() > 0:
            if test_targetscores[column].isna().sum() == test_targetscores.shape[0]:
                count += 1
                remcol.append(column)

    test_targetscores = test_targetscores.drop(columns=remcol)

    baselines = ccle_data.iloc[:,1:].to_numpy()
    labels = targetscores.iloc[:,8:].to_numpy()

    drug_vecs = np.array([drug2target[drug] for drug in targetscores['Drug-Name']])

    stimuli_vecs = np.array([stimuli_dict[sti] for sti in targetscores['Stimuli']])
    stimuli_vecs = np.reshape(stimuli_vecs, (stimuli_vecs.shape[0],1))
    stimuli_vecs = stimuli_vecs.astype(np.float32)

    time_vecs = np.array([time_dict[time] for time in targetscores['Time']])
    time_vecs = np.reshape(time_vecs, (time_vecs.shape[0],1))
    time_vecs = time_vecs.astype(np.float32)

    dose_vecs = np.array([dose_dict[dose] for dose in targetscores['Dose']])
    dose_vecs = np.reshape(dose_vecs, (dose_vecs.shape[0],1))
    dose_vecs = dose_vecs.astype(np.float32)

    dim_vecs = np.array([dim_dict[dim] for dim in targetscores['2D-3D']])
    dim_vecs = np.reshape(dim_vecs, (dim_vecs.shape[0],1))
    dim_vecs = dim_vecs.astype(np.float32)

    vectors_cna = np.array([genomics_data[key]['CNA'] for key in targetscores['CL-Name']])
    vectors_mexp = np.array([genomics_data[key]['mRNA'] for key in targetscores['CL-Name']])

    vectors_mut = [[item for item in genomics_data[key]['Mutation'].to_numpy()] for key in targetscores['CL-Name']]

    mut_vec_1 = np.array(vectors_mut)[:,:,0]
    mut_vec_2 = np.array(vectors_mut)[:,:,1]

    feature_list = [drug_vecs, time_vecs, dose_vecs, dim_vecs, vectors_cna, vectors_mexp, mut_vec_1, mut_vec_2]
    feature_name_list = ['drug', 'time', 'dose', 'dim', 'cna', 'mrna', 'hotspot', 'mut_type']

    feature_dict = {}
    for item1, item2 in zip(feature_name_list, feature_list):
        feature_dict[item1] = item2

    if model_type == 'nn':
        feature_dict['baseline'] = baselines

    trained_model, pcas = model_shit(data=dict_list, model_type=model_type, model_name=model_name, features=feature_dict, labels=labels)

    return trained_model, pcas