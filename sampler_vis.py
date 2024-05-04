import torch
import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from hydra import initialize, compose
from hydra.utils import instantiate
from pathlib import Path

from test_models import UnivPolynomial, UnivModeratedCosine
from experiment_params import SamplingParameters
from experiment import ExperimentBuilder

"""
Sampler Setup Functions
-------------------------------------------------------------------------------------------------------------------------------------------
"""
def setup_sampler_params(sampler_classes, epsilons):
    
    M = 1*torch.eye(n = 1)
    
    sampling_parameters_dict = {
        f'{sampler_class}, $\epsilon = {round(float(epsilon), 3)}$': SamplingParameters(
            sampler_class = sampler_class,
            epsilon = epsilon,
            L = 3,
            M = M,
        )
        for sampler_class in sampler_classes
        for epsilon in epsilons
    }

    return sampling_parameters_dict


def setup_samplers(model, start_batch, sampling_parameters_dict):

    builder = ExperimentBuilder()
    sampler_dict = {
        key: builder.setup_sampler(model, start_batch, sampling_parameters)
        for key, sampling_parameters in sampling_parameters_dict.items()
    }
    return sampler_dict


def sample(num_samples, burnin_offset, sampler_dict):

    torch_samples_dict = {
        key: sampler.sample(
            num_samples = num_samples,
            burnin_offset = burnin_offset,
        )
        for key, sampler in sampler_dict.items()
    }

    return {key: samples.numpy() for key, samples in torch_samples_dict.items()}


"""
Load
-------------------------------------------------------------------------------------------------------------------------------------------
"""  
# check computation backend to use
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("-device:", device)

### Set Paths ###
result_directory = Path('./Experiment_Results')
experiment_name = 'COS_RL_ML'
experiment_dir = result_directory / experiment_name

start_batch_name = 'normal_start_batch.pt'
#start_batch_name = 'zeros_start_batch.pt'
dataset_name = 'dataset.pt'

print(experiment_dir)

### Load from directory ###
dataset = torch.load(experiment_dir.joinpath(dataset_name))

#start_batch = torch.load(experiment_dir.joinpath(start_batch_name))
start_batch = -3.2 * torch.ones(size = (200,1))




"""
Model Setup
-------------------------------------------------------------------------------------------------------------------------------------------
"""
W = torch.tensor([-1.2, -0.7, 2, 1], dtype = torch.float32)
model = UnivPolynomial(W)

W = torch.tensor(1, dtype = torch.float32)
mu = torch.tensor(2, dtype = torch.float32)
#model = UnivModeratedCosine(W = W, mu = mu)


#x = np.linspace(-25,25,2000)
x = np.linspace(-5,5,2000)

def ebm(x):

    x_tensor = torch.tensor(x, dtype=torch.float32)
    energy = model.energy(x_tensor).detach().numpy()

    return np.exp(-(energy))

Z = np.trapz(ebm(x), dx=x[1]-x[0])
#print(Z)

def pdf(x):
    return ebm(x)/Z


"""
Sampler Setup
-------------------------------------------------------------------------------------------------------------------------------------------
"""
sampler_classes = [
    'ULASampler', 
    'MALASampler', 
    'HMCSampler'
]
epsilons = [
    torch.tensor(1e-1, dtype = torch.float32),
    torch.tensor(1e-2, dtype = torch.float32),
    #torch.tensor(1e-3, dtype = torch.float32),
]

sampling_parameters_dict = setup_sampler_params(sampler_classes = sampler_classes, epsilons = epsilons)
print(sampling_parameters_dict.keys())
sampling_parameters_dict.pop('MALASampler, $\epsilon = 0.01$')
sampling_parameters_dict.pop('HMCSampler, $\epsilon = 0.01$')
sampling_parameters_dict.pop('ULASampler, $\epsilon = 0.1$')

sampler_dict = setup_samplers(model = model, start_batch = start_batch, sampling_parameters_dict = sampling_parameters_dict)


"""
Generate Samples
-------------------------------------------------------------------------------------------------------------------------------------------
"""  
batch_size = 2e+2
burnin_offset = 100
samples_dict = sample(num_samples = batch_size, burnin_offset = burnin_offset, sampler_dict = sampler_dict)


"""
Plot
-------------------------------------------------------------------------------------------------------------------------------------------
"""  
plt.plot(x, pdf(x), label='pdf')
#plt.scatter(yt, len(yt)*[1], c=np.arange(len(yt)), cmap='coolwarm', marker='I', alpha=1.0)

for label, samples in samples_dict.items():
    plt.hist(samples, bins=50, density=True, alpha=0.5, label = label)

plt.legend()
plt.gcf().set_size_inches(6,3)
plt.title(f'Model Samples for burnin = {burnin_offset} and batch size: {batch_size}')
#plt.savefig(f'Figures/{sampler_class}_bi{burnin_offset}.png')

plt.show()





"""
Residue
-------------------------------------------------------------------------------------------------------------------------------------------
### Sampler ###
epsilon = torch.tensor(1e-2, dtype = torch.float32)
M = 1*torch.eye(n = 1)
sampler_class = 'HMCSampler'
sampling_parameters = SamplingParameters(
    sampler_class = sampler_class,
    epsilon = epsilon,
    L = 3,
    M = M,
)


### Build ###
builder = ExperimentBuilder()
sampler = builder.setup_sampler(model, start_batch, sampling_parameters)


model_samples = sampler.sample(
    num_samples = batch_size,
    burnin_offset = burnin_offset,
)

"""  