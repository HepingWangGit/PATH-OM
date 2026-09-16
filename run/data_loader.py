import pickle
import pandas as pd
import numpy as np
from utils import count_consecutive_elements

def load_data():
        
    all_ts = pd.read_csv('all_ts.csv', sep='\t', index_col=0)
    all_ts = all_ts.fillna('#')

    resps = pd.read_csv('All_Resps_v5.csv', sep='\t', index_col=0)
    resps = resps.loc[:, '14-3-3-beta':]

    targetscores = pd.read_csv('Targetscores_v5.csv', sep='\t', low_memory=False, index_col=0)

    ccle = pd.read_csv('TCPA_CCLE_RPPA500.tsv', sep='\t', low_memory=False)
    cl_names = [item.split('_')[0] for item in ccle['sample_id']]
    ccle['sample_id'] = cl_names
    ccle.columns = ['CL-Name'] + ccle.columns[1:].to_list()

    with open('genomics_data.pkl', 'rb') as file:
        genomics_data = pickle.load(file)

    weights_dict = {}

    for key2 in genomics_data['BT474'].keys():
        weights_dict[key2] = []
        for key in genomics_data.keys():
                weights_dict[key2].append(genomics_data[key][key2])

    weights_dict['Mutation-1'] = []
    weights_dict['Mutation-2'] = []

    for item in weights_dict['Mutation']:
        item2 = item.to_numpy()
        mut_list1 = []
        mut_list2 = []

        for elmnt in item2:
            mut_list1.append(elmnt[0])
            mut_list2.append(elmnt[1])
        weights_dict['Mutation-1'].append(np.array(mut_list1))
        weights_dict['Mutation-2'].append(np.array(mut_list1))

    with open('exp_data_v1.pkl', 'rb') as file:
        loaded_dict_list = pickle.load(file)

    drug2target, dose_dict, dim_dict, time_dict = loaded_dict_list

    set_info = []
    cl_info = []
    drug_info = []
    time_info = []
    dim_info = []
    stimuli_info = []
    drug2_info = []
    dose_info = []

    targetscores = targetscores[targetscores.columns[(targetscores.isna().all()==0)].to_list()]
    elim_threshold = ((targetscores.isna()==0).sum().mean()/2) #+ 2000
    targetscores = targetscores[targetscores.columns[(targetscores.isna()==0).sum()>elim_threshold].to_list()]

    # PATCHED: original 'fs_korkut.csv' is missing from the GitHub repo (confirmed absent
    # from both commits in git history). Substituting 'fs_mod.csv' per user instruction.
    # fs_mod.csv only covers a subset of protein columns (~318/528 in a header sample),
    # so targetscore columns without a matching fs entry are dropped here rather than
    # guessed, to avoid silently fabricating functional-score signs.
    fs_file = pd.read_csv('fs_mod.csv', sep=',', index_col=0)
    fs_file.index = [str(i).lower() for i in fs_file.index]
    fs_file = fs_file[~fs_file.index.duplicated(keep='first')]  # PATCHED: 'x53bp1' is duplicated in fs_mod.csv
    available_cols = [col for col in targetscores.columns if col.lower() in fs_file.index]
    dropped_cols = [col for col in targetscores.columns if col.lower() not in fs_file.index]
    print(f"[fs_mod.csv patch] keeping {len(available_cols)} / {len(targetscores.columns)} protein columns; "
          f"dropping {len(dropped_cols)} columns with no fs_mod.csv entry")
    targetscores = targetscores[available_cols]
    fs_list = [int(fs_file.loc[col.lower(), 'fs']) for col in targetscores.columns]

    for uid in targetscores.index:
        set_info.append(all_ts.loc[uid, 'set'])
        cl_info.append(all_ts.loc[uid, 'cell_line_name'])
        drug_info.append(all_ts.loc[uid, 'compound_name_1'])
        time_info.append(all_ts.loc[uid, 'time'])
        dim_info.append(all_ts.loc[uid, '2D_3D'])
        stimuli_info.append(all_ts.loc[uid, 'stimuli'])
        drug2_info.append(all_ts.loc[uid, 'compound_name_2'])
        dose_info.append(all_ts.loc[uid, 'dosage_1'])

    targetscores.insert(value=set_info, column='SetNo', loc=0)
    targetscores.insert(value=cl_info, column='CL-Name', loc=1)
    targetscores.insert(value=drug_info, column='Drug-Name', loc=2)
    targetscores.insert(value=time_info, column='Time', loc=3)
    targetscores.insert(value=dim_info, column='2D-3D', loc=4)
    targetscores.insert(value=stimuli_info, column='Stimuli', loc=5)
    targetscores.insert(value=drug2_info, column='Drug-2', loc=6)
    targetscores.insert(value=dose_info, column='Dose', loc=7)

    targetscores = targetscores.loc[(targetscores['CL-Name'].isin(ccle['CL-Name'])),:]
    targetscores = targetscores.loc[targetscores['CL-Name'].isin(genomics_data.keys()),:]

    targetscores = targetscores.loc[targetscores['Drug-Name'].isna() == 0]
    targetscores = targetscores.loc[targetscores['Drug-Name']!='SERUM', :]

    targetscores = targetscores.loc[targetscores['Time'].isin(['24hr', '24hrs', '12hr', '12h', '12hrs', '48hrs', '48hr', '48h', '#', np.nan, '72hr', '4hr']), :]

    time_dict[targetscores.loc['set105_481', 'Time']] = 0
    dose_dict[targetscores.loc['set105_481', 'Time']] = 0
    dim_dict[targetscores.loc['set105_481', 'Time']] = 0

    test_targetscores = targetscores.sample(frac=0.15)
    targetscores = targetscores.drop(test_targetscores.index)

    bionetwork = pd.read_csv('./WK_bionetwork.csv', sep=',')
    bionetwork.index = bionetwork['Unnamed: 0']

    stimuli_types = {
        '10% serum': 1, '2% serum': 1, '2% serum (IFC) and 2% matrigel': 1, '2% serum (IFC) and 2% matrigel + EGF': 1,
        'FBS': 1, 'SERUM': 1, '2%MG 2%serum': 1,
        'EGF': 2, 'FGF': 2, 'FGF1': 2, 'HGF': 2, 'HGF1': 2, 'IGF1': 2, 'NRG': 2, 'NRG1': 2,
        'INSULIN': 3, 'Insulin': 3,
        'PBS': 4,
        'WIT EGF': 5, 'WIT-YZ': 5,
        'attached': 6, 'suspension': 6,
        'Foxo DBM 4HT ': 7, "No IL3": 7, 'Foxo 3xA EtOH': 7, 'with IL3': 7, 'Foxo 3xA 4HT ': 7, 'Foxo DBM EtOH ': 7, '4days+1d induction 4HT+2d drug treatment': 7,
        np.nan: 0, '#': 0,
        }
    stimuli_dict = {}

    for key in stimuli_types.keys():
        stimuli_dict[key] = stimuli_types[key]

    data = {
        'all_ts': all_ts,
        'bionetwork': bionetwork,
        'targetscores': targetscores,
        'resps': resps,
        'ccle': ccle,
        'dicts': [genomics_data, drug2target, dose_dict, dim_dict, time_dict, stimuli_dict],
        'fs_list': fs_list,
        'test_targetscores': test_targetscores
    }
    
    return data

