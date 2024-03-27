
import torch
import pandas as pd
import numpy as np
from pathlib import Path


from helper_tools import prepare_sub_dfs

"""
Read and Copy
-------------------------------------------------------------------------------------------------------------------------------------------
"""
result_directory = Path('./Experiment_Results')
experiment_name = 'POLY_RL_ML'
experiment_dir = result_directory / experiment_name

result_name = 'results_complete.csv'
#result_name = 'results.csv'
#result_name = 'results_part2.csv'

result_file_path = experiment_dir.joinpath(result_name)

results_df = pd.read_csv(result_file_path)

ana_df = results_df.copy()


"""
Split
-------------------------------------------------------------------------------------------------------------------------------------------

split_dict = {}

filter_cols = {
    'Sampler': 'ULASampler',
    'Likelihood': 'Recovery'
}

comparison_columns = ['Epsilon']

for col, val in filter_cols.items():
    filter_mask = ana_df[col] == val
    ana_df = ana_df.loc[filter_mask]

unique_combos = ana_df[comparison_columns].drop_duplicates()
#print(unique_combos)

for index, row_value in unique_combos.iterrows():
    
    entry_name = ", ".join([f"{col}: {val}" for col, val in zip(comparison_columns, row_value)])
    
    matching_mask = (ana_df[comparison_columns] == row_value).all(axis=1)
    split_dict[entry_name] = ana_df.loc[matching_mask].copy()

"""  



"""
Overview
-------------------------------------------------------------------------------------------------------------------------------------------
"""
def average_iter_time(df: pd.DataFrame, return_df: pd.DataFrame, comparison_columns):

    unique_settings = df[comparison_columns].drop_duplicates()

    return_df['Iter. Time mean'] = None
    return_df['Iter. Time std'] = None

    for index, settings in enumerate(unique_settings.itertuples(index = False, name= None)):

        matching_mask = (df[comparison_columns] == settings).all(axis=1)
        setting_slice = df.loc[matching_mask]

        setting_slice_diffs = setting_slice.groupby('training_run_id')['Iteration Timestamp'].diff()
        
        run_means = setting_slice_diffs.mean()
        run_stds = setting_slice_diffs.std()
        
        return_df.loc[index, 'Iter. Time mean'] = run_means
        return_df.loc[index, 'Iter. Time std'] = run_stds

    
    return return_df



def average_estimate_error(df: pd.DataFrame, return_df: pd.DataFrame, comparison_columns: list[str], metric_col: str):

    unique_settings = df[comparison_columns].drop_duplicates()
    
    return_df[f'{metric_col} mean'] = None
    return_df[f'{metric_col} std'] = None
    
    for index, settings in enumerate(unique_settings.itertuples(index = False, name= None)):

        matching_mask = (df[comparison_columns] == settings).all(axis=1)
        setting_slice = df.loc[matching_mask]

        setting_groups = setting_slice.groupby('training_run_id')
        setting_errors = setting_groups[metric_col].tail(1)
        
        return_df.loc[index, f'{metric_col} mean'] = setting_errors.mean()
        return_df.loc[index, f'{metric_col} std'] = setting_errors.std()
        
    
    return return_df



comparison_columns = ['Sampler', 'Epsilon', 'Likelihood', 'Perturbation Variance', 'Burnin Offset']

return_df = ana_df[comparison_columns].drop_duplicates().reset_index(drop=True)
return_df = average_iter_time(ana_df, return_df, comparison_columns)
overview_df = average_estimate_error(ana_df, return_df, comparison_columns, 'W L2-Error')
print(overview_df)

overview_out_path = experiment_dir.joinpath('final_overview.csv')
overview_df.to_csv(overview_out_path, index = False)

#print(overview_df[overview_df['Perturbation Variance'] == 0.1])


"""
Overview Plots
-------------------------------------------------------------------------------------------------------------------------------------------

def overview_plot(overview_df):
    from plotting_components import PlotMatrix, AggregateLinePlot
    import matplotlib.pyplot as plt

    #overview_df = overview_df.loc['Perturbation Variance' == 1]
    mask = lambda col, val: overview_df[col] == val

    plot_slice_dict = {
        f'{likelihood}, {sampler}': overview_df.loc[mask('Sampler', sampler) & mask('Likelihood', likelihood)]
        for sampler in ['ULASampler', 'MALASampler', 'HMCSampler']
        for likelihood in ['Marginal', 'Recovery']
    }

    for label, slice in plot_slice_dict.items():
        plt.plot(slice['Perturbation Variance'], slice['W L2-Error mean'], label = label)
        #plt.plot(slice['Burnin Offset'], slice['W L2-Error mean'], label = label)

    plt.show()


overview_plot(overview_df)
"""