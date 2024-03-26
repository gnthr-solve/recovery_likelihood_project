
import torch
import pandas as pd

from pathlib import Path


from helper_tools import prepare_sub_dfs

# check computation backend to use
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("-device:", device)

### Set Paths ###-------------------------------------------------------
result_directory = Path('./Experiment_Results')
experiment_name = 'POLY_RL_ML'
experiment_dir = result_directory / experiment_name

result_name = 'results.csv'

result_file_path = experiment_dir.joinpath(result_name)

results_df = pd.read_csv(result_file_path)

ana_df = results_df.copy()

"""
Overview
-------------------------------------------------------------------------------------------------------------------------------------------
"""
interest_columns = ['Sampler', 'Epsilon', 'Likelihood', 'Burnin Offset', 'Perturbation Variance']
combinations = ana_df[interest_columns].drop_duplicates()
print(combinations)

"""
Split
-------------------------------------------------------------------------------------------------------------------------------------------
"""  
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



#print(len(results_df))
#results_df.to_csv(result_file_path, index = False)