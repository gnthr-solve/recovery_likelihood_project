import torch
import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from hydra import initialize, compose
from hydra.utils import instantiate
from pathlib import Path

from test_models import UnivPolynomial, UnivModeratedCosine
from recovery_adapter import RecoveryAdapter
from experiment_params import SamplingParameters
from experiment import ExperimentBuilder

# check computation backend to use
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("-device:", device)

### Set Paths ###
result_directory = Path('./Experiment_Results')
experiment_name = 'COS_RL_ML'
experiment_dir = result_directory / experiment_name

config_name = 'recovery_config.yaml'
start_batch_name = 'start_batch.pt'
dataset_name = 'dataset.pt'

print(experiment_dir)
### Load from directory ###
start_batch = torch.load(experiment_dir.joinpath(start_batch_name))
dataset = torch.load(experiment_dir.joinpath(dataset_name))

random_indices = torch.randperm(dataset.shape[0])
selected_indices = random_indices[:start_batch.shape[0]]
dataset_batch = dataset[selected_indices].unsqueeze(-1)

initialize(config_path= str(experiment_dir), version_base = None)
cfg = compose(config_name = config_name)


"""
Setup
-------------------------------------------------------------------------------------------------------------------------------------------
"""  
hyper_parameters = instantiate(cfg.HyperParameters)

### Sampler ###
epsilon = torch.tensor(1e-2, dtype = torch.float32)
M = 1*torch.eye(n = 1)
sampler_class = 'ULASampler'
sampling_parameters = SamplingParameters(
    sampler_class = sampler_class,
    epsilon = epsilon,
    L = 3,
    M = M,
)

### Model ###
#W = torch.tensor([-1.2, -0.7, 2, 1], dtype = torch.float32)
#model = UnivPolynomial(W)

W = torch.tensor(-0.5, dtype = torch.float32)
mu = torch.tensor(-2, dtype = torch.float32)
model = UnivModeratedCosine(W = W, mu = mu)
model = RecoveryAdapter(model, 0.5)
model.set_perturbed_samples(dataset_batch)

### Build ###
builder = ExperimentBuilder()
sampler = builder.setup_sampler(model, start_batch, sampling_parameters)
likelihood = builder.setup_likelihood(model, sampler, hyper_parameters)


batch_size = 1e+4
burnin_offset = 10000

"""
Create pdf to plot
-------------------------------------------------------------------------------------------------------------------------------------------
"""
x = np.linspace(-10,10,200)

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
    data_samples = dataset_batch
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
#plt.savefig(f'Figures/{sampler_class}_bi{burnin_offset}.png')

plt.show()
