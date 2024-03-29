
import torch
import pandas as pd

from hydra import initialize, compose
from hydra.utils import instantiate
from pathlib import Path
from experiment import Experiment
from training_observer import TimingObserver, ParameterObserver, LikelihoodObserver
from result_manager import ResultManager
from metrics import apply_param_metric_to_df, FrobeniusError, LpError

from test_distributions import UnivPolynomial



"""
Present results overview
-------------------------------------------------------------------------------------------------------------------------------------------
"""
result_directory = Path('./Experiment_Results')
experiment_name = 'COS_RL_ML'
experiment_dir = result_directory / experiment_name

result_name = 'results_RL_HMC.csv'
result_file_path = experiment_dir.joinpath(result_name)

ana_df = pd.read_csv(result_file_path)

comparison_columns = ['Sampler', 'Epsilon', 'Likelihood', 'Perturbation Variance', 'Burnin Offset']
unique_settings = ana_df[comparison_columns].drop_duplicates()
print(unique_settings)



"""
Concatenate all result dfs
-------------------------------------------------------------------------------------------------------------------------------------------

result_directory = Path('./Experiment_Results')
experiment_name = 'POLY_RL_ML'
experiment_dir = result_directory / experiment_name

result_names = ['results.csv', 'results_part2.csv', 'results_part3.csv', 'results_part4.csv']

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
experiment_name = 'POLY_RL_ML'
experiment_dir = result_directory / experiment_name

result_name = 'results_part2.csv'
result_file_path = experiment_dir.joinpath(result_name)

results_df = pd.read_csv(result_file_path)

#results_df.drop(results_df[results_df['Likelihood'] == 'Recovery'].index, inplace=True)
#results_df['Perturbation Variance'] = results_df['Perturbation Variance'].round(4)
#results_df['Epsilon'] = results_df['Epsilon'].round(4)

drop_dict ={
    'Sampler': 'MALASampler',
    'Perturbation Variance': 0.5
}
#joint_drop_mask = results_df['Sampler'] == 'MALASampler' & results_df['Perturbation Variance'] == 0.5

#interest_columns = ['Sampler', 'Epsilon', 'Likelihood', 'Burnin Offset', 'Perturbation Variance']
#combinations = results_df[interest_columns].drop_duplicates()
#print(combinations)

results_df.to_csv(result_file_path, index = False)
"""



