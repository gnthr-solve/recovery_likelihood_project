
import torch
import pandas as pd

from hydra import initialize, compose
from hydra.utils import instantiate
from pathlib import Path
from experiment import Experiment
from training_observer import TimingObserver, ParameterObserver, LikelihoodObserver
from result_manager import ResultManager
from metrics import FrobeniusError, LpError

from test_distributions import UnivPolynomial

"""
File for disconnected short operations that don't warrant their own module.
"""


"""
Present results overview
-------------------------------------------------------------------------------------------------------------------------------------------
"""
result_directory = Path('./Experiment_Results')
experiment_name = 'COS_RL_ML'
experiment_dir = result_directory / experiment_name

result_name = 'results_RL_MALA.csv'
result_file_path = experiment_dir.joinpath(result_name)

ana_df = pd.read_csv(result_file_path)

comparison_columns = ['Sampler', 'Epsilon', 'Likelihood', 'Perturbation Variance', 'Burnin Offset', 'Model Batch Size']
unique_settings = ana_df[comparison_columns].drop_duplicates()
print(unique_settings)

for index, setting in enumerate(unique_settings.itertuples(index = False, name= None)):

    matching_mask = (ana_df[comparison_columns] == setting).all(axis=1)
    setting_slice = ana_df.loc[matching_mask]
    print(setting)
    print(len(setting_slice))

    
"""
Concatenate all result dfs
-------------------------------------------------------------------------------------------------------------------------------------------

result_directory = Path('./Experiment_Results')
experiment_name = 'COS_RL_ML'
experiment_dir = result_directory / experiment_name

result_names = [
    'results_ML_ULA.csv', 
    'results_ML_MALA.csv', 
    'results_ML_HMC.csv',
    'results_RL_ULA.csv', 
    'results_RL_MALA.csv', 
    'results_RL_HMC.csv',
]

result_df_list = []
for result_name in result_names:

    result_file_path = experiment_dir.joinpath(result_name)

    result_df_list.append(pd.read_csv(result_file_path))


complete_results_df = pd.concat(result_df_list, axis=0)
print(len(complete_results_df))
#interest_columns = ['Sampler', 'Epsilon', 'Likelihood', 'Burnin Offset', 'Perturbation Variance']
#combinations = complete_results_df[interest_columns].drop_duplicates()
#print(combinations)

result_out_path = experiment_dir.joinpath('results_complete.csv')
complete_results_df.to_csv(result_out_path, index = False)
"""


"""
Drop failed experiments
-------------------------------------------------------------------------------------------------------------------------------------------

result_directory = Path('./Experiment_Results')
experiment_name = 'COS_RL_ML'
experiment_dir = result_directory / experiment_name

result_name = 'results_ML_ULA.csv'
result_file_path = experiment_dir.joinpath(result_name)

results_df = pd.read_csv(result_file_path)

print(len(results_df))
#results_df.drop(results_df[results_df['Likelihood'] == 'Recovery'].index, inplace=True)
#results_df['Perturbation Variance'] = results_df['Perturbation Variance'].round(4)
#results_df['Epsilon'] = results_df['Epsilon'].round(4)
id_list = ['28032024_1933_run_1', '28032024_1926_run_1', '28032024_1925_run_1', '28032024_1922_run_1']

results_df.drop(results_df[results_df['training_run_id'].isin(id_list)].index, inplace=True)

print(len(results_df))
results_df.to_csv(result_file_path, index = False)
"""



