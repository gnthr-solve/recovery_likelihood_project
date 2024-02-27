
import torch

from ebm import EnergyModel
from helper_tools import check_nan



"""
Recovery Adapter
-------------------------------------------------------------------------------------------------------------------------------------------
Following the Object Adapter design pattern.
The adapter inherits from EnergyModel and wraps an existing EnergyModel instance.
Calls to the EnergyModel methods are delegated to the wrapped instance and extended with the conditional terms.
Keeps a reference to the wrapped instance params attribute so that the parameters of this are updated correctly.
Thus, after training, one can simply use the (trained) original model instance.
"""

class RecoveryAdapter(EnergyModel):

    def __init__(self, energy_model: EnergyModel, perturbation_var: torch.tensor):
        super().__init__()

        self.sigma = perturbation_var

        self.energy_model = energy_model
        self.params = energy_model.params


    def energy(self, x: torch.tensor):

        energy = self.energy_model.energy(x)
        
        conditional_term = torch.sum((self.perturbed_samples - x)**2, dim=1) / (2 * self.sigma**2)
        
        return energy + conditional_term


    def energy_grad(self, x: torch.tensor):
        
        grad = self.energy_model.energy_grad(x)
        
        conditional_grad = (self.perturbed_samples - x) / self.sigma**2
        
        return grad - conditional_grad


    def avg_param_grad(self, x: torch.tensor):
        return self.energy_model.avg_param_grad(x)


    def set_perturbed_samples(self, perturbed_samples: torch.tensor):
        self.perturbed_samples = perturbed_samples





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

    adapted_model = RecoveryAdapter(model, 1)

    x_batch = torch.tensor(
        [[2, 3],
         [1, 0],
         [0, 0],
         [2, 1],],
        dtype=torch.float32,
    )
    #x_batch = torch.tensor([0, 0], dtype = torch.float32)

    perturbed_data_samples = x_batch + adapted_model.sigma * torch.randn_like(x_batch)
    adapted_model.set_perturbed_samples(perturbed_samples = perturbed_data_samples)

    ### Return Grads function ###
    return_param_grads = (lambda param_dict: {
            key: param.grad
            for key, param in param_dict.items()
        }
    )

    avg_param_grads = adapted_model.avg_param_grad(x_batch)
    #print("Model mean parameter grad: \n", model.avg_param_grad(x_batch))
    print("Adapted Model mean parameter grad: \n", avg_param_grads)

    print(return_param_grads(adapted_model.params))

    for name, param in adapted_model.params.items():
            param.grad = avg_param_grads[name]

    print(return_param_grads(adapted_model.params))
    print(return_param_grads(model.params))