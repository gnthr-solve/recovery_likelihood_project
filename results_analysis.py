
import torch
import pandas as pd
import numpy as np
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





"""
Average Iteration time
-------------------------------------------------------------------------------------------------------------------------------------------
"""
def average_iter_time(df):

    average_times_dict = {}
    comparison_columns = ['Sampler', 'Epsilon', 'Likelihood', 'Burnin Offset', 'Perturbation Variance']
    unique_combinations = df[comparison_columns].drop_duplicates()

    for _, setting in unique_combinations.iterrows():

        entry_name = ", ".join([f"{col}: {val}" for col, val in zip(comparison_columns, setting)])

        matching_mask = (df[comparison_columns] == setting).all(axis=1)
        df_slice = df.loc[matching_mask]


        



"""
avg_times_list = []
for training_run_id in df_slice['training_run_id'].unique():

    id_mask = df['training_run_id'] == training_run_id
    run_df = df[id_mask]

    run_timestamps = pd.to_datetime(run_df.loc['Iteration Timestamp'])

    time_deltas = run_timestamps.diff()
    avg_times_list.append(time_deltas.mean())

average_times_dict[entry_name] = (np.mean(avg_times_list))
        
"""