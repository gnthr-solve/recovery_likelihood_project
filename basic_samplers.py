
import numpy as np
import numpy.linalg as npl

import torch
import torch.linalg as tla

from abc import ABC, abstractmethod
from test_distributions import Distribution, EnergyDistribution
from helper_tools import MultivariateNormalIterator, no_grad_decorator

"""
Samplers
-------------------------------------------------------------------------------------------------------------------------------------------
Use torch and are meant to use the  distribution classes, 
for which a analytical expression of the energy function and its gradient is hardcoded or calculated via autograd
"""

class EnergySampler(ABC):

    @abstractmethod
    def sample(self, x_0: torch.tensor, N: int, energy_distribution: EnergyDistribution, **kwargs):
        pass


"""
Unadjusted Langevin Algorithm
-------------------------------------------------------------------------------------------------------------------------------------------
"""

class ULASampler(EnergySampler):

    def __init__(self, epsilon):
        self.epsilon = epsilon


    def _gen_z_iterator(self, x: torch.tensor, N: int):

        #print(self.x)

        mean = torch.zeros_like(x)

        cov = torch.diag(torch.ones_like(x))

        self._z_iterator = MultivariateNormalIterator(mean, cov, N)


    def _iterate(self, x: torch.tensor):
        
        # Compute the gradient of the energy function at x
        grad = self.energy_dist.energy_grad(x)
        
        # Retrieve generated z
        z = next(self._z_iterator)
        
        # Compute x_new without tracking gradients
        x_new = x - self.epsilon * grad + torch.sqrt(2*self.epsilon) * z
        
        return x_new


    @no_grad_decorator
    def sample(self, x_0: torch.tensor, N: int, energy_distribution: EnergyDistribution, **kwargs):
        
        x = torch.atleast_1d(x_0)

        self.energy_dist = energy_distribution

        self._gen_z_iterator(x, N)

        sample_list = []

        for _ in range(N):

            x_new = self._iterate(x, **kwargs)
            sample_list.append(x_new)
            x = x_new

        #print(self.sample_list)
        #print(self.sample_list[0].shape)
        
        del self.energy_dist

        return torch.stack(sample_list, axis = 0)
        #return torch.concatenate(self.sample_list, axis = 0)


    


"""
Metropolis Adjusted Langevin Algorithm
-------------------------------------------------------------------------------------------------------------------------------------------
"""
class MALASampler(EnergySampler):

    def __init__(self, epsilon):
        self.epsilon = epsilon


    def _gen_z_iterator(self, x: torch.tensor, N: int):

        #print(self.x)

        mean = torch.zeros_like(x)

        cov = torch.diag(torch.ones_like(x))

        self._z_iterator = MultivariateNormalIterator(mean, cov, N)


    def _iterate(self, x: torch.tensor):
        
        # Compute the gradient of the energy function at x
        grad = self.energy_dist.energy_grad(x)
        
        # Retrieve generated z
        z = next(self._z_iterator)
        
        # Compute x_new without tracking gradients
        x_hat = x - self.epsilon * grad + torch.sqrt(2*self.epsilon) * z

        return self._accept_reject(x = x, x_hat= x_hat)
    

    def _accept_reject(self, x: torch.tensor, x_hat: torch.tensor):
        
        x_energy = self.energy_dist.energy(x)
        x_hat_energy = self.energy_dist.energy(x_hat)

        grad_x_energy = self.energy_dist.energy_grad(x)
        grad_x_hat_energy = self.energy_dist.energy_grad(x_hat)

        log_x_proposal = tla.multi_dot([(x - x_hat + self.epsilon * grad_x_hat_energy), (x - x_hat + self.epsilon * grad_x_hat_energy)])
        log_x_hat_proposal = tla.multi_dot([(x_hat - x + self.epsilon * grad_x_energy), (x_hat - x + self.epsilon * grad_x_energy)])
        log_proposal = -(log_x_proposal - log_x_hat_proposal) / (4*self.epsilon)

        alpha = min(1, torch.exp(x_energy - x_hat_energy + log_proposal))

        if torch.rand(1) <= alpha:
            return x_hat
        else:
            return x
        

    @no_grad_decorator
    def sample(self, x_0: torch.tensor, N: int, energy_distribution: EnergyDistribution, **kwargs):
        
        x = torch.atleast_1d(x_0)

        self.energy_dist = energy_distribution

        self._gen_z_iterator(x, N)

        sample_list = []

        for _ in range(N):

            x_new = self._iterate(x, **kwargs)
            sample_list.append(x_new)
            x = x_new

        #print(self.sample_list)
        #print(self.sample_list[0].shape)
        
        del self.energy_dist
            
        return torch.stack(sample_list, axis = 0)
        #return torch.concatenate(self.sample_list, axis = 0)


"""
Hamiltonian Monte Carlo
-------------------------------------------------------------------------------------------------------------------------------------------
"""
class HMCSampler(EnergySampler):

    def __init__(self, epsilon, L, M):
        self.epsilon = epsilon
        self.L = L
        self.M = M


    def _gen_momentum_iterator(self, x: torch.tensor, N: int):

        mean = torch.zeros_like(x)

        #shape = torch.shape(x)
        #cov = torch.diag(torch.ones(shape = shape))

        self._momentum_iterator = MultivariateNormalIterator(mean, self.M, N)


    def _leapfrog_trajectory(self, x: torch.tensor):
        
        p = next(self._momentum_iterator)
        x_hat = x
        
        grad = self.energy_dist.energy_grad(x_hat)
        #half step
        p_hat = p - self.epsilon * (grad / 2)
        
        for _ in range(self.L-1):

            x_hat = x_hat + self.epsilon * p_hat

            grad = self.energy_dist.energy_grad(x_hat)
            #full step for p
            p_hat = p_hat - self.epsilon * grad
        
        #last full step for x
        x_hat = x_hat + self.epsilon * p_hat

        grad = self.energy_dist.energy_grad(x_hat)
        #last half step for p
        p_hat = p_hat - self.epsilon * (grad / 2)

        p_hat = - p_hat
        #print('Current x:', x)
        #print('Current x_hat:', x_hat)
        return self._accept_reject(x = x, p = p, x_hat= x_hat, p_hat=p_hat)


    def _accept_reject(self, x, p, x_hat, p_hat):

        U = self.energy_dist.energy(x)
        U_hat = self.energy_dist.energy(x_hat)
        K = tla.multi_dot([p, self.M, p])/2
        K_hat = tla.multi_dot([p_hat, self.M, p_hat])/2

        alpha = min(1, torch.exp(U + K - U_hat - K_hat))

        if torch.rand(1) <= alpha:
            return x_hat
        else:
            return x


    @no_grad_decorator
    def sample(self, x_0: torch.tensor, N: int, energy_distribution: EnergyDistribution, **kwargs):
        
        x = torch.atleast_1d(x_0)

        self.energy_dist = energy_distribution

        self._gen_momentum_iterator(x, N)

        sample_list = []
        for _ in range(N):
            x_new = self._leapfrog_trajectory(x, **kwargs)
            sample_list.append(x_new)
            x = x_new

        #print(self.sample_list)
        #print(self.sample_list[0].shape)
        
        del self.energy_dist

        return torch.stack(sample_list, axis = 0)
        #return torch.concatenate(self.sample_list, axis = 0)






if __name__=="__main__":
    
    from test_models import MultivariateGaussianModel

    ### Instantiate Model with initial Parameters ###
    mu_0 = torch.tensor([2, 2], dtype = torch.float32)
    Sigma_0 = torch.tensor(
        [[2, 0],
         [0, 1],],
        dtype=torch.float32,
    )
    model = MultivariateGaussianModel(mu_0 = mu_0, Sigma_0 = Sigma_0)


    ### Instantiate Sampler with initial Parameters ###
    epsilon = torch.tensor(0.5, dtype = torch.float32)
    #sampler = ULASampler(epsilon = epsilon)
    sampler = MALASampler(epsilon = epsilon)
    #sampler = HMCSampler(epsilon = epsilon, L = 3, M = torch.eye(n = 2))

    num = 20
    x_0 = torch.tensor([0, 0], dtype = torch.float32)

    model_samples = sampler.sample(x_0, num, model)
    print(model_samples)