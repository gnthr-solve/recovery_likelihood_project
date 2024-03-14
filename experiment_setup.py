
import torch

from abc import ABC, abstractmethod
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
from omegaconf import DictConfig, OmegaConf

from experiment_params import ModelParameters, SamplingParameters, HyperParameters


result_directory = Path('./Experiment_Results')
"""
Experiment Meta
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
"""
experiment_name = 'MVG_RL_ML'
# Create the experiment directory (safe creation with mkdir)
experiment_dir = result_directory / experiment_name
experiment_dir.mkdir(parents=True, exist_ok=True)  # Create parent directories if needed

config_name = 'recovery_config.yaml'
config_path = experiment_dir.joinpath(config_name)
print(config_path)

"""
Tensor Parameters - Change tensors here
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
"""
### Model ###
target_params = {
    'mu': torch.tensor([3, 3], dtype = torch.float32),
    'Sigma': 2* torch.diag(torch.ones(size = (2,), dtype=torch.float32)),
}

start_params = {
    'mu_0': torch.tensor([2, 2], dtype = torch.float32),
    'Sigma_0': torch.tensor(
        [[2, 0],
         [0, 1],],
        dtype=torch.float32,
        )
}

perturbation_var = torch.tensor(1, dtype = torch.float32)


### Sampler ###
epsilon = torch.tensor(1e-1, dtype = torch.float32)
M = torch.eye(n = 2)

"""
Parameters config
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Remark: 
The model_batch_size parameter specifies how many samples should be generated 
to estimate the expectation for the model component ( E[grad_theta f] ) of the likelihood gradient.
While the number of samples for the data component is fixed with batch size we can adjust the precision of the model component, 
by sampling a larger batch.
ATTENTION: 
For recovery likelihood the number of MC chains is fixed to the batch_size. 
Thus the model_batch_size MUST be an integer multiple of the batch_size to avoid an error. 
"""
experiment_likelihood_class = 'RecoveryLikelihood'

hyper_params = HyperParameters(
    batch_size = 200,
    epochs = 10,
    burnin_offset = 50,
    model_batch_size = 200,
    likelihood_class = experiment_likelihood_class,
    optimizer_class = 'Adam',
    optimizer_params = {
        'lr': 1e-1,
    },
    scheduler_class = 'ExponentialLR',
    scheduler_params = {
        'gamma': 0.5
    },
)

sampling_params = SamplingParameters(
    sampler_class = 'MALASampler',
    epsilon = epsilon,
    L = 3,
    M = M,
)

model_params = ModelParameters(
    model_class = 'MultivariateGaussianModel',
    target_params = target_params,
    start_params = start_params,
    requires_adapter = (experiment_likelihood_class == 'RecoveryLikelihood'),
    perturbation_var = perturbation_var,
)

paramsets = [
    hyper_params, 
    model_params, 
    sampling_params, 
]

paramsets_config = [params.as_config() for params in paramsets]

params_config = OmegaConf.merge(*paramsets_config)
OmegaConf.save(config = params_config, f = config_path)


"""
Large Tensors
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
"""
### Dataset ###
from torch.distributions import MultivariateNormal
data_mv_normal = MultivariateNormal(
    loc = target_params['mu'], 
    covariance_matrix = target_params['Sigma'],
)

dataset = data_mv_normal.sample(sample_shape = (10000,))

dataset_name = 'dataset.pt'
dataset_path = experiment_dir.joinpath(dataset_name)
torch.save(obj = dataset, f = dataset_path)


### Sampler Start Batch. IMPORTANT: For recovery likelihood batch_size and sampler_start_batch.shape[0] must coincide ###
sampler_start_batch = torch.zeros(size = (200,2))

start_batch_name = 'start_batch.pt'
start_batch_path = experiment_dir.joinpath(start_batch_name)
torch.save(obj = sampler_start_batch, f = start_batch_path)