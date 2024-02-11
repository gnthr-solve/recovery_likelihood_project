
import torch
import torch.linalg as tla

from torch.distributions import MultivariateNormal
from abc import ABC, abstractmethod

from ebm import EnergyModel
from gaussian_iter_strategies import IterStrategy, StdIterStrategy, MomentumIterStrategy
from helper_tools import no_grad_decorator, quadratic_form_batch


"""
Multi-Chain Energy Sampler Base Class
-------------------------------------------------------------------------------------------------------------------------------------------
Aspects TODO and to decide:
- What should the chain length be? (currently N/chain_num i.e. all samples are taken into account)
- Should the number of chains continue to be determined by the setup batch?
- How should the next sample sets start batches be determined?
"""
class EnergySamplerMC(ABC):

    def __init__(self, epsilon: torch.tensor, energy_model: EnergyModel, x_0_batch: torch.tensor, **kwargs):

        self.epsilon = epsilon
        self.energy_model = energy_model

        shape = x_0_batch.shape
        self.data_dim = shape[-1]
        self.chain_num = shape[0]

        self.curr_state_batch = x_0_batch
        self.z_iterator: IterStrategy


    @abstractmethod
    def _iterate(self, x_batch: torch.tensor):
        ### Hook Method ###
        pass


    @no_grad_decorator
    def sample(self, N: int): 
        ### Template Method ###

        chain_length = int(torch.ceil( N / self.chain_num ))
        shape = (chain_length, self.chain_num, self.data_dim)

        self.z_iterator.generate(chain_length)

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

    def __init__(self, epsilon: torch.tensor, energy_model: EnergyModel, x_0_batch: torch.tensor, **kwargs):
        super().__init__(epsilon, energy_model, x_0_batch)

        self.z_iterator = StdIterStrategy(self.data_dim, self.chain_num)


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

    def __init__(self, epsilon: torch.tensor, energy_model: EnergyModel, x_0_batch: torch.tensor, **kwargs):
        super().__init__(epsilon, energy_model, x_0_batch)

        self.z_iterator = StdIterStrategy(self.data_dim, self.chain_num)


    def _iterate(self, x_batch: torch.tensor):
        
        # Compute the gradient of the energy function at x
        grad_batch = self.energy_model.energy_grad(x_batch)
        
        # Retrieve generated z
        z = next(self.z_iterator)
        
        # Compute new_state_batch without tracking gradients
        x_hat_batch = x_batch - self.epsilon * grad_batch + torch.sqrt(2*self.epsilon) * z

        return self._accept_reject(x_batch = x_batch, x_hat_batch= x_hat_batch)
    

    def _accept_reject(self, x_batch: torch.tensor, x_hat_batch: torch.tensor):
    
        x_batch_energy = self.energy_model.energy(x_batch)
        x_hat_batch_energy = self.energy_model.energy(x_hat_batch)

        grad_x_batch = self.energy_model.energy_grad(x_batch)
        grad_x_hat_batch = self.energy_model.energy_grad(x_hat_batch)

        log_x_proposal = tla.norm((x_batch - x_hat_batch + self.epsilon * grad_x_hat_batch))**2
        log_x_hat_proposal = tla.norm((x_hat_batch - x_batch + self.epsilon * grad_x_batch))**2
        log_proposal = -(log_x_proposal - log_x_hat_proposal) / (4*self.epsilon)

        alpha = torch.exp(x_batch_energy - x_hat_batch_energy + log_proposal)

        u_batch = torch.rand(alpha.shape)
        accept_mask = u_batch <= alpha
        
        new_state_batch = torch.where(accept_mask.unsqueeze(1), x_hat_batch, x_batch)
        return new_state_batch
                



"""
Hamiltonian Monte Carlo
-------------------------------------------------------------------------------------------------------------------------------------------
"""
class HMCSamplerMC(EnergySamplerMC):

    def __init__(self, epsilon: torch.tensor, L: int, M: torch.tensor, energy_model: EnergyModel, x_0_batch: torch.tensor, **kwargs):
        super().__init__(epsilon, energy_model, x_0_batch)
        self.L = L
        self.M = M

        self.z_iterator = MomentumIterStrategy(self.data_dim, M, self.chain_num)


    def _iterate(self, x_batch: torch.tensor):
        
        p_batch = next(self.z_iterator)
        x_hat_batch = x_batch
        
        grad_batch = self.energy_model.energy_grad(x_hat_batch)
        #half step
        p_hat_batch = p_batch - self.epsilon * (grad_batch / 2)
        
        for _ in range(self.L-1):

            x_hat_batch = x_hat_batch + self.epsilon * p_hat_batch

            grad_batch = self.energy_model.energy_grad(x_hat_batch)
            #full step for p
            p_hat_batch = p_hat_batch - self.epsilon * grad_batch
        
        #last full step for x_batch
        x_hat_batch = x_hat_batch + self.epsilon * p_hat_batch

        grad_batch = self.energy_model.energy_grad(x_hat_batch)
        #last half step for p
        p_hat_batch = p_hat_batch - self.epsilon * (grad_batch / 2)

        p_hat_batch = - p_hat_batch
        #print('Current x_batch:', x_batch)
        #print('Current x_hat_batch:', x_hat_batch)
        return self._accept_reject(x_batch = x_batch, p_batch = p_batch, x_hat_batch = x_hat_batch, p_hat_batch = p_hat_batch)


    def _accept_reject(self, x_batch, p_batch, x_hat_batch, p_hat_batch):

        U_batch = self.energy_model.energy(x_batch)
        U_hat_batch = self.energy_model.energy(x_hat_batch)
        K_batch = quadratic_form_batch(p_batch, self.M)/2
        K_hat_batch = quadratic_form_batch(p_hat_batch, self.M)/2

        alpha = torch.exp(U_batch + K_batch - U_hat_batch - K_hat_batch)

        u_batch = torch.rand(alpha.shape)
        accept_mask = u_batch <= alpha
        
        new_state_batch = torch.where(accept_mask.unsqueeze(1), x_hat_batch, x_batch)
        return new_state_batch







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

    x_0_batch = torch.tensor(
        [[2, 3],
         [1, 0],
         [0, 0],
         [2, 1],],
        dtype=torch.float32,
    )
    x_0_batch = torch.tensor([0, 0], dtype = torch.float32)

    ### Instantiate Sampler with initial Parameters ###
    epsilon = torch.tensor(0.5, dtype = torch.float32)
    #sampler = ULASamplerMC(epsilon = epsilon, energy_model = model, x_0_batch = x_0_batch)
    #sampler = MALASamplerMC(epsilon = epsilon, energy_model = model, x_0_batch = x_0_batch)
    sampler = HMCSamplerMC(epsilon = epsilon, L = 3, M = torch.eye(n = 2), energy_model = model, x_0_batch = x_0_batch)

    num = torch.tensor(16, dtype = torch.float32)
    
    model_samples = sampler.sample(num)
    print(model_samples)