
import torch
import torch.nn as nn
import typing

from abc import ABC, abstractmethod
from typing import Any




class EnergyModel(nn.Module):

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.params = nn.ParameterDict()


    def forward(self, x: torch.tensor):
        # Compute the energy given input x
        energy = self.energy(x)
        return energy


    def energy(self, x: torch.tensor):
        pass
    

    def energy_grad(self, x: torch.tensor):
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
    

    def avg_param_grad(self, x: torch.tensor):
        """
        Gradient of energy w.r.t. parameters at x for training
        Default implementation for complex energy functions without analytical gradients

        ATTENTION: 
        This implementation aggregates the parameter gradients of a batch by summing them up.
        I.e. the Monte Carlo estimate is already computed in this method, which may be counterintuitive.
        """
        
        #Unsqueeze for if just one datapoint is passed
        x = torch.unsqueeze(x, dim=0) if x.dim() == 1 else x

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

            #for param_name, value in param_grads.items():
            #    print(f'{param_name} grad:\n {value}')
            
            return param_grads
        



