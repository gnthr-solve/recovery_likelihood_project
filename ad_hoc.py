
import torch

from hydra import initialize, compose
from hydra.utils import instantiate
from pathlib import Path
from experiment import Experiment
from training_observer import TimingObserver, ParameterObserver, LikelihoodObserver
from result_manager import ResultManager
from metrics import apply_param_metric_to_df, FrobeniusError, LpError

# check computation backend to use
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("-device:", device)

### Set Paths ###
result_directory = Path('./Experiment_Results')
experiment_name = 'MVG_RL_ML'
experiment_dir = result_directory / experiment_name

config_name = 'recovery_config.yaml'

initialize(config_path= str(experiment_dir), version_base = None)
cfg = compose(config_name = config_name)


### Experiment Setup ###
hyper_parameters = instantiate(cfg.HyperParameters)
model_parameters = instantiate(cfg.ModelParameters)
sampling_parameters = instantiate(cfg.SamplingParameters)

print(hyper_parameters['scheduler_class'])