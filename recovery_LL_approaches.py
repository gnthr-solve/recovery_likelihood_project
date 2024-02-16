
import torch

from likelihood import Likelihood
from ebm import EnergyModel
from basic_samplers import EnergySampler

"""
Inheritance Approach
-------------------------------------------------------------------------------------------------------------------------------------------
"""
class RecoveryEBM(EnergyModel):

    def __init__(self, sigma: torch.tensor, energy_model: EnergyModel):
        super().__init__()
        self.sigma = sigma
        self.energy_model = energy_model
        self.params = energy_model.params  # Reference original parameters


    def energy(self, x):
        # Perturb data sample
        tilde_x = x + self.sigma * torch.randn_like(x)
        # Compute energy of perturbed sample with original model
        energy = self.energy_model.energy(tilde_x)
        # Add conditional term
        energy += torch.sum((tilde_x - x)**2, dim=1) / (2 * self.sigma**2)
        return energy


    def energy_grad(self, x):
        # Perturb data sample
        tilde_x = x + self.sigma * torch.randn_like(x)
        # Gradient of original energy w.r.t. perturbed sample
        grad = self.energy_model.energy_grad(tilde_x)
        # Gradient of conditional term
        conditional_grad = (tilde_x - x) / self.sigma**2
        # Combine gradients
        grad += conditional_grad
        return grad




"""
Decorator Approach
-------------------------------------------------------------------------------------------------------------------------------------------
"""

class RecoveryDecorator:
    def __init__(self, energy_model, sigma):
        self.sigma = sigma
        self.energy_model = energy_model
        self.perturbed_samples = None

    def energy(self, x):
        if self.perturbed_samples is None:
            # Perturb data sample only once
            self.perturbed_samples = x + self.sigma * torch.randn_like(x)
        # Use perturbed sample and original model energy
        return self.energy_model.energy(self.perturbed_samples) + self.conditional_term(x)

    def energy_grad(self, x):
        if self.perturbed_samples is None:
            self.perturbed_samples = x + self.sigma * torch.randn_like(x)
        # Use perturbed sample, original model gradient, and conditional gradient
        grad = self.energy_model.energy_grad(self.perturbed_samples)
        grad += self.conditional_grad(x)
        return grad

    def conditional_term(self, x):
        # Calculate and return conditional term based on self.perturbed_samples
        return torch.sum((self.perturbed_samples - x)**2, dim=1) / (2 * self.sigma**2)

    def conditional_grad(self, x):
        # Calculate and return conditional gradient based on self.perturbed_samples
        return (self.perturbed_samples - x) / self.sigma**2

    def clear_perturbed_samples(self):
        # Reset perturbed samples for next data sample
        self.perturbed_samples = None



"""
Object Adapter Approach?
-------------------------------------------------------------------------------------------------------------------------------------------
Would conform to the Object Adapter Pattern if it inherited from EnergyModel
Goal Interface ~ non existant RecoveryEnergyModel class?
Adapter uses Delegation 
"""

class EnergyModelAdapter:
    def __init__(self, energy_model: EnergyModel):
        self.energy_model = energy_model

    def energy(self, x):
        return self.energy_model.energy(x)

    def energy_grad(self, x):
        return self.energy_model.energy_grad(x)

    def avg_param_grad(self, data_samples):
        return self.energy_model.avg_param_grad(data_samples)



class RecoveryAdapter(EnergyModelAdapter):
    def __init__(self, energy_model: EnergyModel, sigma):
        super().__init__(energy_model)
        self.sigma = sigma

    def energy(self, x):
        if self.perturbed_samples is None:
            self.perturbed_samples = x + self.sigma * torch.randn_like(x)
        energy = self.energy_model.energy(self.perturbed_samples)
        conditional_term = torch.sum((self.perturbed_samples - x)**2, dim=1) / (2 * self.sigma**2)
        return energy + conditional_term

    def energy_grad(self, x):
        if self.perturbed_samples is None:
            self.perturbed_samples = x + self.sigma * torch.randn_like(x)
        grad = self.energy_model.energy_grad(self.perturbed_samples)
        conditional_grad = (self.perturbed_samples - x) / self.sigma**2
        return grad + conditional_grad

    def clear_perturbed_samples(self):
        self.perturbed_samples = None