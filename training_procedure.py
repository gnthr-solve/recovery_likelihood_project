
import torch

from torch.distributions import MultivariateNormal
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from tqdm import trange, tqdm

from ebm import EnergyModel
from test_models import MultivariateGaussianModel
from likelihood import Likelihood

from training_observer import Subject



"""
TrainingProcedure Blueprint
-------------------------------------------------------------------------------------------------------------------------------------------
"""
class TrainingProcedure(Subject):

    def __init__(
            self,
            dataset: torch.Tensor,
            model: EnergyModel, 
            likelihood: Likelihood, 
            optimizer: Optimizer,
            scheduler: LRScheduler,
            epochs: int,
            batch_size: int,
            model_batch_size: int = None,
            burnin_offset: int = 0,
            **kwargs
        ):

        super().__init__()

        self.model = model
        self.likelihood = likelihood
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.dataset = dataset
        self.epochs = epochs
        self.batch_size = batch_size
        self.model_batch_size = model_batch_size if model_batch_size else batch_size
        self.burnin_offset = burnin_offset
        self.train_loader = DataLoader(dataset, batch_size = batch_size, shuffle=True)


    def __call__(self):

        epoch_progress_bar = tqdm(range(self.epochs))

        for epoch in epoch_progress_bar:

            self.training_epoch(curr_epoch = epoch, model_batch_size = self.model_batch_size, burnin_offset = self.burnin_offset)

            self.scheduler.step()


    def training_loop(self, X_batch: torch.Tensor, model_batch_size: int, burnin_offset: int):

        # reset gradients 
        self.optimizer.zero_grad()
        
        model_samples = self.likelihood.gen_model_samples(
            batch_size = model_batch_size,
            burnin_offset = burnin_offset,
            data_samples = X_batch,
        )

        self.likelihood.gradient(data_samples = X_batch, model_samples = model_samples)
        
        # perform gradient descent step along model.theta.grad
        self.optimizer.step()
        self.notify_observers()

    
    def training_epoch(self, curr_epoch: int, model_batch_size: int, burnin_offset: int):

        for batch_ind, X_batch in enumerate(self.train_loader):
            
            self.training_loop(X_batch = X_batch, model_batch_size = model_batch_size, burnin_offset = burnin_offset)

            #self.
            #print(f"{curr_epoch}_{batch_ind+1}/{self.epochs} Parameters:")
            #for param_name, value in self.model.params.items():
            #    print(f'{param_name}:\n {value.data}')







if __name__=="__main__":

    from torch.optim import Adam
    from torch.optim.lr_scheduler import ExponentialLR

    from test_models import MultivariateGaussianModel
    from mc_samplers import ULASampler, MALASampler, HMCSampler
    from recovery_adapter import RecoveryAdapter
    from likelihood import RecoveryLikelihood
    from timing_decorators import timing_decorator
    from training_observer import TrainingObserver

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

    batch_size = 200
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
        model = model, 
        likelihood = likelihood,
        optimizer = optimizer,
        scheduler = scheduler,
        batch_size = batch_size,
        model_batch_size = batch_size,
        epochs = 10, 
        burnin_offset = int(batch_size/4)
    )
    #observer = TrainingObserver()
    #training_procedure.register_observer()
    training_procedure()

    print(timing_decorator.return_average_times())