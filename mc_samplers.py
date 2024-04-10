
import torch
import torch.linalg as tla

from abc import ABC, abstractmethod

from ebm import EnergyModel
from gaussian_iter_strategies import IterStrategy, StdIterStrategy, MomentumIterStrategy
from helper_tools import no_grad_decorator, quadratic_form_batch
from timing_decorators import timing_decorator


"""
Multi-Chain Energy Sampler Abstract Base Class
-------------------------------------------------------------------------------------------------------------------------------------------
The EnergySampler base class provides the 'sample' method and the main part of the __init__ method for its subclasses.

'sample' follows the Template Method pattern. 
It defines the structure of the overall sampling process, relaying the specific iteration procedure to the concrete subclasses.
Subclasses must implement the _iterate method and can use super().__init__ to set up the step size, start batch, number of chains
and data dimension for the IterStrategy.
"""
class EnergySampler(ABC):

    def __init__(self, epsilon: torch.Tensor, energy_model: EnergyModel, start_batch: torch.Tensor, **kwargs):

        self.epsilon = epsilon
        self.energy_model = energy_model

        shape = start_batch.shape
        self.data_dim = shape[-1] if start_batch.dim() > 1 else 1
        self.chain_num = shape[0]

        self.curr_state_batch = start_batch
        self.z_iterator: IterStrategy


    @abstractmethod
    def _iterate(self, x_batch: torch.Tensor):
        ### Hook Method ###
        pass

    
    def set_current_states(self, state_batch: torch.Tensor):
        self.curr_state_batch = state_batch


    @no_grad_decorator
    def sample(self, num_samples: int, burnin_offset: int = 0): 
        ### Template Method ###
        
        #convert num_samples to tensor to use torch.ceil
        num_samples = torch.tensor(num_samples, dtype=torch.float32) if isinstance(num_samples, (int, float)) else num_samples

        chain_length = int(torch.ceil( num_samples / self.chain_num )) + int(burnin_offset)
        #print(chain_length)
        shape = (chain_length, self.chain_num, self.data_dim)
        #print(shape)

        self.z_iterator.generate(chain_length)

        raw_samples = torch.empty(size = shape)
        curr_state_batch = self.curr_state_batch

        for j in range(chain_length):

            new_state_batch = self._iterate(curr_state_batch)
            raw_samples[j, :, :] = new_state_batch
            curr_state_batch = new_state_batch

        #take last sample batch as start batch for next call
        self.curr_state_batch = raw_samples[-1, :, :]
        raw_samples = raw_samples[burnin_offset:, :, :]

        return raw_samples.reshape(shape = ((chain_length-burnin_offset) * self.chain_num, self.data_dim))


"""
Unadjusted Langevin Algorithm
-------------------------------------------------------------------------------------------------------------------------------------------
ULA does not use any checks and the implementation thus consists only of the implementation for the _iterate method
and the instantiation of the StdIterStrategy.
"""

class ULASampler(EnergySampler):

    def __init__(self, epsilon: torch.Tensor, energy_model: EnergyModel, start_batch: torch.Tensor, **kwargs):
        super().__init__(epsilon, energy_model, start_batch)

        self.z_iterator = StdIterStrategy(self.data_dim, self.chain_num)


    @timing_decorator
    def _iterate(self, x_batch: torch.Tensor):
        
        # Compute the gradient of the energy function at x_batch
        grad_batch = self.energy_model.energy_grad(x_batch)
        
        # Retrieve generated z_batch
        z_batch = next(self.z_iterator)
        #print(z_batch[:-10])
        
        # Compute new_state_batch without tracking gradients
        new_state_batch = x_batch - self.epsilon * grad_batch + torch.sqrt(2*self.epsilon) * z_batch
        
        return new_state_batch


    

"""
Metropolis Adjusted Langevin Algorithm
-------------------------------------------------------------------------------------------------------------------------------------------
MALA implements the _iterate and the _accept_reject method.
Before _iterate returns it calls _accept_reject to conduct the Metropolis-Hastings check.
"""

class MALASampler(EnergySampler):

    def __init__(self, epsilon: torch.Tensor, energy_model: EnergyModel, start_batch: torch.Tensor, **kwargs):
        super().__init__(epsilon, energy_model, start_batch)

        self.z_iterator = StdIterStrategy(self.data_dim, self.chain_num)


    @timing_decorator
    def _iterate(self, x_batch: torch.Tensor):
        
        # Compute the gradient of the energy function at x
        grad_batch = self.energy_model.energy_grad(x_batch)
        
        # Retrieve generated z
        z = next(self.z_iterator)
        
        # Compute new_state_batch without tracking gradients
        x_hat_batch = x_batch - self.epsilon * grad_batch + torch.sqrt(2*self.epsilon) * z

        return self._accept_reject(x_batch = x_batch, x_hat_batch = x_hat_batch)
    

    #@timing_decorator
    def _accept_reject(self, x_batch: torch.Tensor, x_hat_batch: torch.Tensor):
        
        x_batch_energy = self.energy_model.energy(x_batch)
        x_hat_batch_energy = self.energy_model.energy(x_hat_batch)
        
        grad_x_batch = self.energy_model.energy_grad(x_batch)
        grad_x_hat_batch = self.energy_model.energy_grad(x_hat_batch)
        
        log_x_proposal = tla.norm((x_batch - x_hat_batch + self.epsilon * grad_x_hat_batch), dim = 1) ** 2
        log_x_hat_proposal = tla.norm((x_hat_batch - x_batch + self.epsilon * grad_x_batch), dim = 1) ** 2
        log_proposal = - (log_x_proposal - log_x_hat_proposal) / (4*self.epsilon)
        
        alpha = torch.exp(x_batch_energy - x_hat_batch_energy + log_proposal)

        u_batch = torch.rand(alpha.shape)
        accept_mask = (u_batch <= alpha)

        new_state_batch = torch.where(accept_mask.unsqueeze(1), x_hat_batch, x_batch)
        return new_state_batch
                



"""
Hamiltonian Monte Carlo
-------------------------------------------------------------------------------------------------------------------------------------------
HMC implements its own versions of _iterate and _accept_reject.
It requires two extra parameters:
The number of steps to take before applying the Metropolis-Hastings check, 'L'
and the momentum matrix 'M' that defines the momentum with which sample particles move.
"""
class HMCSampler(EnergySampler):

    def __init__(self, epsilon: torch.Tensor, L: int, M: torch.Tensor, energy_model: EnergyModel, start_batch: torch.Tensor, **kwargs):
        super().__init__(epsilon, energy_model, start_batch)
        self.L = L
        self.M = M

        self.z_iterator = MomentumIterStrategy(self.data_dim, M, self.chain_num)


    @timing_decorator
    def _iterate(self, x_batch: torch.Tensor):
        
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
        
        return self._accept_reject(x_batch = x_batch, p_batch = p_batch, x_hat_batch = x_hat_batch, p_hat_batch = p_hat_batch)


    #@timing_decorator
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
    """
    Implementation test example for verification.
    """
    from test_models import MultivariateGaussianModel

    ### Instantiate Model with initial Parameters ###
    mu_0 = torch.tensor([2, 2], dtype = torch.float32)
    Sigma_0 = torch.tensor(
        [[2, 0],
         [0, 1],],
        dtype=torch.float32,
    )
    model = MultivariateGaussianModel(mu_0 = mu_0, Sigma_0 = Sigma_0)

    start_batch = torch.tensor(
        [[2, 3],
         [1, 0],
         [0, 0],
         [2, 1],],
        dtype=torch.float32,
    )
    #start_batch = torch.tensor([0, 0], dtype = torch.float32)

    ### Instantiate Sampler with initial Parameters ###
    epsilon = torch.tensor(0.5, dtype = torch.float32)
    #sampler = ULASampler(epsilon = epsilon, energy_model = model, start_batch = start_batch)
    sampler = MALASampler(epsilon = epsilon, energy_model = model, start_batch = start_batch)
    #sampler = HMCSampler(epsilon = epsilon, L = 3, M = torch.eye(n = 2), energy_model = model, start_batch = start_batch)

    num = torch.tensor(16, dtype = torch.float32)
    
    model_samples = sampler.sample(num)
    print(model_samples)