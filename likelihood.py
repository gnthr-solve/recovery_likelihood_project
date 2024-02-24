
import torch

from ebm import EnergyModel
from mc_samplers import EnergySampler
from recovery_adapter import RecoveryAdapter



"""
Standard Maximum Likelihood
-------------------------------------------------------------------------------------------------------------------------------------------
"""
class Likelihood:

    def __init__(self, energy_model: EnergyModel, sampler: EnergySampler):
        
        self.energy_model = energy_model
        self.sampler = sampler
        

    def gradient(self, data_samples: torch.tensor, model_samples: torch.tensor):

        grads_data_component = self.energy_model.avg_param_grad(data_samples)
        grads_model_component = self.energy_model.avg_param_grad(model_samples)

        for name, param in self.energy_model.params.items():
            param.grad = grads_data_component[name] - grads_model_component[name]
            print(f'{name} LL grad: \n', param.grad)


    def gen_model_samples(self, batch_size: int, burnin_offset: int):

        num_samples = torch.tensor(batch_size, dtype = torch.float32)
        model_samples = self.sampler.sample(num_samples = num_samples, burnin_offset = burnin_offset)
        
        return model_samples
    


"""
Standard Maximum Likelihood
-------------------------------------------------------------------------------------------------------------------------------------------
"""
class RecoveryLikelihood(Likelihood):

    def __init__(self, adapted_model: RecoveryAdapter, sampler: EnergySampler):

        self.adapted_model = adapted_model
        self.sampler = sampler
        

    def gradient(self, data_samples: torch.tensor, model_samples):

        grads_data_component = self.adapted_model.avg_param_grad(data_samples)
        grads_model_component = self.adapted_model.avg_param_grad(model_samples)

        for name, param in self.adapted_model.params.items():
            param.grad = grads_data_component[name] - grads_model_component[name]
            print(f'{name} LL grad: \n', param.grad)
        

    def gen_model_samples(self, data_samples: torch.tensor, burnin_offset: int):

        perturbed_data_samples = data_samples + self.adapted_model.sigma * torch.randn_like(data_samples)
        self.adapted_model.set_perturbed_samples(perturbed_samples = perturbed_data_samples)

        num_samples = torch.tensor(perturbed_data_samples.shape[0], dtype = torch.float32)
        model_samples = self.sampler.sample(num_samples = num_samples, burnin_offset= burnin_offset)
        
        return model_samples