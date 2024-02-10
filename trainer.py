
import torch

from torch.distributions import MultivariateNormal
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
from tqdm import trange, tqdm

from test_models import MultivariateGaussianModel
from basic_samplers import ULASampler, MALASampler, HMCSampler
from likelihood import Likelihood

class Trainer:

    def __init__(self, train_dataset, model, likelihood, optimizer):
        self.train_dataset = train_dataset
        self.model = model
        self.likelihood = likelihood
        self.optimizer = optimizer


    def train(self, epochs, batch_size):
        model = self.model
        optimizer = self.optimizer
        likelihood = self.likelihood

        train_loader = DataLoader(self.train_dataset, batch_size = batch_size, shuffle=True)

        scheduler = ExponentialLR(optimizer, gamma=0.9)

        pbar = tqdm(range(epochs))
        for it in pbar:
            for b, X_batch in enumerate(train_loader):
                
                # reset gradients 
                optimizer.zero_grad()
                
                model_samples = likelihood.gen_model_samples(
                    x_0 = x_0,
                    batch_size = 10*batch_size,
                )
                x_0 = model_samples[-1]
                #print(x_0)

                likelihood.gradient(data_samples = X_batch, model_samples = model_samples[batch_size:])
                
                print(f"{it}_{b+1}/{epochs} Parameters:")
                for param_name, value in model.params.items():
                    print(f'{param_name}:\n {value.data}')
                
                # perform gradient descent step along model.theta.grad
                optimizer.step()
            
            scheduler.step()