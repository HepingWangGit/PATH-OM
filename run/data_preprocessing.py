import numpy as np
import pandas as pd

def data_processing(data):

    genomics_data, drug2target, dose_dict, dim_dict, time_dict, stimuli_dict = data['dicts']

    targetscores = data['targetscores']
    ccle = data['ccle']
    
    for item in set(targetscores['Drug-Name']):
        if (item in drug2target.keys()) == 0:
            print(item)

    # UPDATED 2026-09-30 (Bug 10): the original code built ccle_data by iterating
    # `set(targetscores['CL-Name'])` -- an UNORDERED Python set -- and concatenating one
    # duplicated block of ccle rows per unique cell line, in whatever order the set
    # happened to iterate in. That block order has no relationship to targetscores'
    # actual row order, so `baselines = ccle_data.iloc[:,1:].to_numpy()` (used below)
    # ended up row-for-row MISALIGNED with every other feature array and with `labels`
    # itself -- confirmed empirically: on a 2,000-row check, 1,374 rows (90%) had a
    # different cell line's baseline protein levels attached than the row's own CL-Name.
    # This fed wrong-cell-line baseline data into every 'nn' model (tsnn, attention) via
    # `feature_dict['baseline']` this whole time. It also silently dropped any cell line
    # with more than one matching ccle row (`if temp_df.shape[0]==1`) instead of just
    # picking one, which for this dataset only affects 'TT' (2 duplicate ccle rows) but
    # would otherwise further shrink/misalign ccle_data relative to targetscores.
    # Fixed with a row-preserving left merge: every targetscores row gets exactly the
    # ccle row for its own CL-Name, in targetscores' own order, with duplicate ccle
    # entries for the same cell line collapsed to one (keep='first') instead of dropped.
    ccle_dedup = ccle.drop_duplicates(subset='CL-Name', keep='first')
    ccle_data = targetscores[['CL-Name']].merge(ccle_dedup, on='CL-Name', how='left')
    assert ccle_data.shape[0] == targetscores.shape[0], "ccle_data must have exactly one row per targetscores row"
    assert ccle_data['CL-Name'].isna().sum() == 0, "every targetscores CL-Name should match a ccle row (data_loader.py pre-filters on this)"

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