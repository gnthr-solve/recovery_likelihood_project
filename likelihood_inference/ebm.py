
import torch
import torch.nn as nn
import typing

from abc import ABC, abstractmethod
from typing import Any
from .helper_tools import enable_grad_decorator

"""
Energy Model Base Class
-------------------------------------------------------------------------------------------------------------------------------------------
The base class for all test models that is meant to be subclassed.
Child classes need only insert their parameters in the params dict in the __init__ method 
and implement the energy functional. 
Gradients with respect to the input or the parameters can then be calculated automatically using the inherited functions,
or implemented directly if a gradient is known analytically to save computation resources.
energy_grad is predominantly used in the sampling algorithms, while avg_param_grad computes the MC estimates for training.

ATTENTION: Here the convention is that the density of an energy model is given as
p(x) = exp(-U(x))/Z
where U is the energy functional. I.e. the energy method is understood to be negated when producing a density.

For the sampling process gradient tracking is disabled to avoid maintaining the computation graph.
If a test distribution does not implement energy_grad itself however, 
gradient tracking is enabled locally with the enable_grad_decorator.
"""

class EnergyModel(nn.Module):

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.params = nn.ParameterDict()


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute the energy given input x
        energy = self.energy(x)
        return energy


    def energy(self, x: torch.Tensor) -> torch.Tensor:
        pass
    
    
    @enable_grad_decorator
    def energy_grad(self, x: torch.Tensor) -> torch.Tensor:
        """
        Gradient of energy w.r.t. inputs at x for Langevin/HMC type sampling.
        Default implementation for complex energy functions without analytical gradients
        """
        x.requires_grad_(True)

        # Compute the energy
        energy = self.energy(x)

        out = torch.ones_like(energy)
        # Compute gradients with respect to x
        gradient = torch.autograd.grad(outputs = energy, inputs = x, grad_outputs = out)[0]

        return gradient
    

    def avg_param_grad(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Gradient of energy w.r.t. parameters at x for training
        Default implementation for complex energy functions without analytical gradients

        ATTENTION: 
        This implementation aggregates the parameter gradients of a batch by summing them up.
        I.e. the Monte Carlo estimate is already computed in this method, which may be counterintuitive.
        """
        
        #Unsqueeze for if just one datapoint is passed
        x = torch.unsqueeze(x, dim=1) if x.dim() == 1 else x

        with torch.enable_grad():
            
            batch_size = x.shape[0]
            
            # Compute the energy
            energy = self.energy(x)

            # Compute gradients with respect to parameters
            energy.backward(torch.ones_like(energy), retain_graph = False) 

            # Extract gradients from the parameters
            param_grads = {name: param.grad/batch_size for name, param in self.params.items()}
            # Clear the gradients for the next computation
            self.zero_grad()
            
            return param_grads
        



