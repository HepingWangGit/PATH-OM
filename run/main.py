from data_loader import load_data
from data_preprocessing import data_processing
from imputation import train_predictive_models
from training import training_machine
from evaluation import evaluate_model, evaluate_model_on_all_CLs
import matplotlib.pyplot as plt
import numpy as np

def main():

    data_dict = load_data()
    features = data_processing(data_dict)

    targetscores, non_NA_mask = train_predictive_models(features=features, dict_list=data_dict['dicts'])
    
    trained_model, pcas = training_machine(targetscores, data_dict, features, model_type='nn',  model_name='tsnn')

    sample_nums, cl_results, actual_corr, keys = evaluate_model_on_all_CLs(trained_model, data_dict, non_NA_mask, pca_list=pcas)

    return(sample_nums, cl_results, actual_corr, keys)
    
    #evaluate_model(trained_model, data_dict, non_NA_mask)

if __name__ == '__main__':

    sample_nums, cl_results, actual_corr, keys = main()

    colors = ['red' if value < 0 else 'blue' for value in cl_results]

    fig, ax = plt.subplots(ncols=1, nrows=2, sharex=True)

    ax[0].bar(x=keys, height=np.abs(cl_results), color=colors)
    ax[0].axhline(y=actual_corr, color='black', label=f'Overall R: {np.round(actual_corr, 2)}') #
    ax[0].set_ylabel('Pearson Correlation')
    ax[0].legend()

    ax[1].bar(x=keys, height=sample_nums, color='blue')
    ax[1].set_xlabel('Cell Line Name')
    ax[1].set_ylabel('#Samples')
    ax[1].tick_params(axis='x', rotation=90)

    plt.show()