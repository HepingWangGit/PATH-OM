import numpy as np
import pandas as pd

def data_processing(data):

    genomics_data, drug2target, dose_dict, dim_dict, time_dict, stimuli_dict = data['dicts']

    targetscores = data['targetscores']
    ccle = data['ccle']
    
    for item in set(targetscores['Drug-Name']):
        if (item in drug2target.keys()) == 0:
            print(item)

    ccle_data = pd.DataFrame(columns = ccle.columns)

    order_ts = targetscores['CL-Name'].value_counts().to_dict()

    count = 0

    for cl_name in set(targetscores['CL-Name']):
        
        if cl_name in set(ccle['CL-Name']):

            num = order_ts[cl_name]
            temp_df = ccle.loc[ccle['CL-Name']==cl_name,:]  

            if temp_df.shape[0] == 1:
                duplicated_rows = pd.concat([temp_df] * (num), axis=0)
                ccle_data = pd.concat([ccle_data, duplicated_rows], ignore_index=True)

                count += num

    drug_vecs = np.array([drug2target[drug] for drug in targetscores['Drug-Name']])

    stimuli_vecs = np.array([stimuli_dict[sti] for sti in targetscores['Stimuli']])
    stimuli_vecs = np.reshape(stimuli_vecs, (stimuli_vecs.shape[0],1))

    time_vecs = np.array([time_dict[time] for time in targetscores['Time']])
    time_vecs = np.reshape(time_vecs, (time_vecs.shape[0],1))

    dose_vecs = np.array([dose_dict[dose] for dose in targetscores['Dose']])
    dose_vecs = np.reshape(dose_vecs, (dose_vecs.shape[0],1))

    dim_vecs = np.array([dim_dict[dim] for dim in targetscores['2D-3D']])
    dim_vecs = np.reshape(dim_vecs, (dim_vecs.shape[0],1))
    dim_vecs = dim_vecs.astype(np.float32)

    vectors_cna = np.array([genomics_data[key]['CNA'] for key in targetscores['CL-Name']])
    vectors_mexp = np.array([genomics_data[key]['mRNA'] for key in targetscores['CL-Name']])

    vectors_mut = [[item for item in genomics_data[key]['Mutation'].to_numpy()] for key in targetscores['CL-Name']]

    mut_vec_1 = np.array(vectors_mut)[:,:,0]
    mut_vec_2 = np.array(vectors_mut)[:,:,1]

    cols = []
    for item in ccle.columns[1:]:
        cols.append(item.lower())

    ccle_data.columns = ['CL-Name'] + cols

    features = {
        'ccle': ccle_data,
        'targetscores': targetscores,
        'hotspot': mut_vec_1,
        'mut_type': mut_vec_2,
        'mrna': vectors_mexp,
        'cna': vectors_cna,
        'drug': drug_vecs,
        'dim': dim_vecs,
        'time': time_vecs,
        'dose': dose_vecs
    }

    return features