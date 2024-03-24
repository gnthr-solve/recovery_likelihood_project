
import torch
import pandas as pd

from hydra import initialize, compose
from hydra.utils import instantiate
from pathlib import Path
from experiment import Experiment
from training_observer import TimingObserver, ParameterObserver, LikelihoodObserver
from result_manager import ResultManager
from metrics import FrobeniusError, LpError, SimplexLpError, ParameterAssessor

from test_distributions import UnivPolynomial


# check computation backend to use
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("-device:", device)

### Set Paths ###-------------------------------------------------------
result_directory = Path('./Experiment_Results')
experiment_name = 'MVG_RL_ML'
experiment_dir = result_directory / experiment_name

result_name = 'results.csv'

result_file_path = experiment_dir.joinpath(result_name)

results_df = pd.read_csv(result_file_path)



#print(len(results_df))
results_df.to_csv(result_file_path, index = False)