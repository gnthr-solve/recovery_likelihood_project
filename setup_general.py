
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
experiment_name = 'COS_RL_ML'
# Create the experiment directory (safe creation with mkdir)
experiment_dir = result_directory / experiment_name
experiment_dir.mkdir(parents=True, exist_ok=True)  # Create parent directories if needed

config_name = 'recovery_config.yaml'
config_path = experiment_dir.joinpath(config_name)
print(config_path)


"""
Experiment General
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
"""

dim = 1

model_class = 'UnivModeratedCosine'
experiment_likelihood_class = 'RecoveryLikelihood'
batch_size = 200
model_batch_size = batch_size * 50
learning_rate = 1e-3 * batch_size
print('De-facto lr: ', learning_rate/batch_size)


"""
Tensor Parameters - Change tensors here
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
"""
### Model ###
target_params = {
    'W': torch.tensor(1, dtype = torch.float32),
    'mu': torch.tensor(2, dtype = torch.float32),
}

start_params = {
    'W': torch.tensor(-0.5, dtype = torch.float32),
    'mu': torch.tensor(-2, dtype = torch.float32),
}

perturbation_var = torch.tensor(1, dtype = torch.float32)


### Sampler ###
epsilon = torch.tensor(1e-1, dtype = torch.float32)
M = torch.eye(n = dim)

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


hyper_params = HyperParameters(
    batch_size = batch_size,
    epochs = 10,
    burnin_offset = 100,
    model_batch_size = model_batch_size,
    likelihood_class = experiment_likelihood_class,
    optimizer_class = 'Adam',
    optimizer_params = {
        'lr': learning_rate,
    },
    #scheduler_class = 'ExponentialLR',
    scheduler_params = {
        'gamma': 0.9
    },
)

sampling_params = SamplingParameters(
    sampler_class = 'MALASampler',
    epsilon = epsilon,
    L = 3,
    M = M,
)

model_params = ModelParameters(
    model_class = model_class,
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
from dataset_gen import cos_dataset

dataset = torch.tensor(cos_dataset, dtype = torch.float32)
dataset_name = 'dataset.pt'
dataset_path = experiment_dir.joinpath(dataset_name)
#torch.save(obj = dataset, f = dataset_path)


### Sampler Start Batch. IMPORTANT: For recovery likelihood batch_size and sampler_start_batch.shape[0] must coincide ###
from torch.distributions import MultivariateNormal

mv_normal = MultivariateNormal(loc = torch.zeros(dim), covariance_matrix = 2*torch.eye(dim))
sampler_start_batch = mv_normal.sample(sample_shape = (batch_size,))
start_batch_name = 'normal_start_batch.pt'

#sampler_start_batch = torch.zeros(size = (batch_size, dim))
#start_batch_name = 'zeros_start_batch.pt'

start_batch_path = experiment_dir.joinpath(start_batch_name)
torch.save(obj = sampler_start_batch, f = start_batch_path)