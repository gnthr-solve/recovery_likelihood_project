
import torch
import torch.linalg as tla

from torch.distributions import MultivariateNormal
from abc import ABC, abstractmethod

from ebm import EnergyModel
from helper_tools import no_grad_decorator


"""
Multi-Chain Energy Sampler Base Class
-------------------------------------------------------------------------------------------------------------------------------------------
Aspects TODO and to decide:
- What should the chain length be? (currently N/chain_num i.e. all samples are taken into account)
- Should the number of chains continue to be determined by the setup batch?
- How should the next sample sets start batches be determined?
"""
class EnergySamplerMC(ABC):

    def setup(self, x_0_batch: torch.tensor, energy_model: EnergyModel, **kwargs):

        self.energy_model = energy_model

        shape = x_0_batch.shape
        self.data_dim = shape[-1]
        self.chain_num = shape[0]

        self.curr_state_batch = x_0_batch


    def _gen_z_iterator(self, shape):

        mean = torch.zeros(size = (self.data_dim,), dtype = torch.float32)
        cov = torch.eye(self.data_dim)

        mv_normal = MultivariateNormal(loc = mean, covariance_matrix = cov)

        self.z_iterator = iter(mv_normal.sample(sample_shape = shape[:-1]))


    @abstractmethod
    def _iterate(self, x_batch: torch.tensor):
        pass


    @no_grad_decorator
    def sample(self, N: int):

        chain_length = torch.ceil( N / self.chain_num )
        shape = (chain_length, self.chain_num, self.data_dim)

        self._gen_z_iterator(shape)

        raw_samples = torch.empty(size = shape)
        curr_state_batch = self.curr_state_batch

        for j in range(chain_length):

            new_state_batch = self._iterate(curr_state_batch)
            raw_samples[j, :, :] = new_state_batch
            curr_state_batch = new_state_batch

        #take last sample batch as start batch for next call
        self.curr_state_batch = raw_samples[-1, :, :]

        return raw_samples.reshape(shape = (chain_length * self.chain_num, self.data_dim))


"""
Unadjusted Langevin Algorithm
-------------------------------------------------------------------------------------------------------------------------------------------
"""

class ULASamplerMC(EnergySamplerMC):

    def __init__(self, epsilon):
        self.epsilon = epsilon


    def _iterate(self, x_batch: torch.tensor):
        
        # Compute the gradient of the energy function at x_batch
        grad_batch = self.energy_model.energy_grad(x_batch)
        
        # Retrieve generated z_batch
        z_batch = next(self.z_iterator)
        
        # Compute new_state_batch without tracking gradients
        new_state_batch = x_batch - self.epsilon * grad_batch + torch.sqrt(2*self.epsilon) * z_batch
        
        return new_state_batch


    

    


"""
Metropolis Adjusted Langevin Algorithm
-------------------------------------------------------------------------------------------------------------------------------------------
"""
class MALASamplerMC(EnergySamplerMC):

    def __init__(self, epsilon):
        self.epsilon = epsilon


    def _iterate(self, x_batch: torch.tensor):
        
        # Compute the gradient of the energy function at x
        grad = self.energy_model.energy_grad(x_batch)
        
        # Retrieve generated z
        z = next(self.z_iterator)
        
        # Compute new_state_batch without tracking gradients
        x_batch_hat = x_batch - self.epsilon * grad + torch.sqrt(2*self.epsilon) * z

        return self._accept_reject(x = x_batch, x_batch_hat= x_batch_hat)
    

    def _accept_reject(self, x_batch: torch.tensor, x_batch_hat: torch.tensor):
    
        x_energy = self.energy_model.energy(x_batch)
        x_hat_energy = self.energy_model.energy(x_batch_hat)

        grad_x_energy = self.energy_model.energy_grad(x_batch)
        grad_x_hat_energy = self.energy_model.energy_grad(x_batch_hat)

        log_x_proposal = tla.norm((x_batch - x_batch_hat + self.epsilon * grad_x_hat_energy))**2
        log_x_hat_proposal = tla.norm((x_batch_hat - x_batch + self.epsilon * grad_x_energy))**2
        log_proposal = -(log_x_proposal - log_x_hat_proposal) / (4*self.epsilon)

        alpha = torch.exp(x_energy - x_hat_energy + log_proposal)

        u_batch = torch.rand(alpha.shape)
        accept_mask = u_batch <= alpha
        
        new_state_batch = torch.where(accept_mask.unsqueeze(1), x_batch_hat, x_batch)
        return new_state_batch
                

