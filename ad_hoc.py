
from pathlib import Path

import torch
from torch.distributions import MultivariateNormal
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim import Adam, Optimizer
from torch.optim.lr_scheduler import LRScheduler, ExponentialLR
from tqdm import trange, tqdm

from test_models import MultivariateGaussianModel
from mc_samplers import ULASampler, MALASampler, HMCSampler
from recovery_adapter import RecoveryAdapter
from likelihood import RecoveryLikelihood, Likelihood

from training_observer import TimingObserver, ParameterObserver, LikelihoodObserver
from training_procedure import TrainingProcedure
from exporter import ResultExporter
import matplotlib.pyplot as plt

# check computation backend to use
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("-device:", device)


### Create Data ###
data_mv_normal = MultivariateNormal(
    loc = torch.tensor([3, 3], dtype = torch.float32), 
    covariance_matrix = 2* torch.diag(torch.ones(size = (2,), dtype=torch.float32)),
)
dataset = data_mv_normal.sample(sample_shape = (10000,))

### Instantiate Model with initial Parameters ###
mu_0 = torch.tensor([2, 2], dtype = torch.float32)
Sigma_0 = torch.tensor(
        [[2, 0],
         [0, 1],],
        dtype=torch.float32,
)
org_model = MultivariateGaussianModel(mu_0 = mu_0, Sigma_0 = Sigma_0)

perturbation_var = torch.tensor(1, dtype = torch.float32)
model = RecoveryAdapter(energy_model = org_model, perturbation_var = perturbation_var)

batch_size = 500
### Instantiate Sampler with initial Parameters ###
x_0_batch = torch.zeros(size = (batch_size, 2))

epsilon = torch.tensor(1e-1, dtype = torch.float32)
#sampler = ULASampler(epsilon = epsilon, energy_model = model, x_0_batch = x_0_batch)
sampler = MALASampler(epsilon = epsilon, energy_model = model, x_0_batch = x_0_batch)
#sampler = HMCSampler(epsilon = epsilon, L = 3, M = torch.eye(n = 2), energy_model = model, x_0_batch = x_0_batch)


### Instantiate Standard Likelihood ###
likelihood = RecoveryLikelihood(adapted_model = model, sampler = sampler)

### Training ###
optimizer = Adam(model.parameters(), lr=1e-1)
scheduler = ExponentialLR(optimizer, gamma=0.9)

training_procedure = TrainingProcedure(
    dataset = dataset,
    batch_size = batch_size,
    model = model, 
    likelihood = likelihood,
    optimizer = optimizer,
    scheduler = scheduler,
)

training_observers = [
    TimingObserver(),
    LikelihoodObserver(),
    ParameterObserver(),
]

training_procedure.register_observers(training_observers)

training_procedure(epochs = 10, model_batch_size = batch_size, burnin_offset = int(batch_size/4))

observation_dfs = [
    observer.return_observations()
    for observer in training_observers
]

exporter = ResultExporter(
    export_name = 'test.csv',
    export_folder_path = Path('/Users/gnthr/Desktop/Studium/CMS/Research_Project/recovery_likelihood_project/Experiment_Results')
)

exporter.export_observations(training_run_id = 'test_id_123', observation_dfs = observation_dfs)

#plt.plot()
#plt.show()