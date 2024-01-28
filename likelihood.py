
import torch

from ebm import EnergyModel
from torch_samplers import EnergySampler


"""
Standard Maximum Likelihood
-------------------------------------------------------------------------------------------------------------------------------------------
"""
class Likelihood:

    def __init__(self, data_samples: torch.tensor, energy_model: EnergyModel, sampler: EnergySampler):
        
        self.data_samples = data_samples
        self.energy_model = energy_model
        self.sampler = sampler
        

    def gradient(self, model_samples):

        grads_data_component = self.energy_model.avg_param_grad(self.data_samples)
        grads_model_component = self.energy_model.avg_param_grad(model_samples)

        for name, param in self.energy_model.params.items():
            param.grad = grads_data_component[name] - grads_model_component[name]
        

    def gen_model_samples(self, x_0: torch.tensor, batch_size: int):

        model_samples = self.sampler.sample(x_0, batch_size, self.energy_model)
        
        return model_samples