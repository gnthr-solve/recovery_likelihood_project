import torch
import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from hydra import initialize, compose
from hydra.utils import instantiate
from pathlib import Path

from test_models import UnivPolynomial
from experiment_params import SamplingParameters
from experiment import ExperimentBuilder

# check computation backend to use
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("-device:", device)

### Set Paths ###
result_directory = Path('./Experiment_Results')
experiment_name = 'POLY_RL_ML'
experiment_dir = result_directory / experiment_name

config_name = 'marginal_config.yaml'
start_batch_name = 'start_batch.pt'

print(experiment_dir)
### Load from directory ###
start_batch = torch.load(experiment_dir.joinpath(start_batch_name))

initialize(config_path= str(experiment_dir), version_base = None)
cfg = compose(config_name = config_name)


"""
Setup
-------------------------------------------------------------------------------------------------------------------------------------------
"""  
hyper_parameters = instantiate(cfg.HyperParameters)

### Sampler ###
epsilon = torch.tensor(1e-1, dtype = torch.float32)
M = 1*torch.eye(n = 1)
sampler_class = 'MALASampler'
sampling_parameters = SamplingParameters(
    sampler_class = sampler_class,
    epsilon = epsilon,
    L = 5,
    M = M,
)

### Model ###
W = torch.tensor([-1.2, -0.7, 2, 1], dtype = torch.float32)
model = UnivPolynomial(W)

builder = ExperimentBuilder()
sampler = builder.setup_sampler(model, start_batch, sampling_parameters)
likelihood = builder.setup_likelihood(model, sampler, hyper_parameters)


batch_size = 1e+4
burnin_offset = 10

"""
Create pdf to plot
-------------------------------------------------------------------------------------------------------------------------------------------
"""
x = np. linspace(-3,3,100)

def ebm(x):

    x_tensor = torch.tensor(x, dtype=torch.float32)
    energy = model.energy(x_tensor).detach().numpy()

    return np.exp(-(energy))

Z = np.trapz(ebm(x), dx=x[1]-x[0])
print(Z)
#
def pdf(x):
    return ebm(x)/Z


"""
Generate Samples
-------------------------------------------------------------------------------------------------------------------------------------------
"""  
model_samples = likelihood.gen_model_samples(
    batch_size = batch_size,
    burnin_offset = burnin_offset,
)

samples = model_samples.numpy()


"""
Plot
-------------------------------------------------------------------------------------------------------------------------------------------
"""  
plt.plot(x, pdf(x), label='pdf')
#plt.scatter(yt, len(yt)*[1], c=np.arange(len(yt)), cmap='coolwarm', marker='I', alpha=1.0)
plt.hist(samples, bins=50, density=True, alpha=0.5, label='samples')
plt.legend()
plt.gcf().set_size_inches(6,3)
plt.title(f'{sampler_class} - burnin: {burnin_offset} - samples: {batch_size}')
#plt.savefig(f'Figures/{sampler_class}_bi{burnin_offset}_size{batch_size}.png')

plt.show()
