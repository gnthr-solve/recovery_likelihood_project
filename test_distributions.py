

import torch
import torch.linalg as tla
import typing

from abc import ABC, abstractmethod
from ebm import EnergyModel
from helper_tools import enable_grad_decorator



"""
Base Classes
-------------------------------------------------------------------------------------------------------------------------------------------
"""
class Distribution(ABC):

    def density(self, x: torch.Tensor, *args, **kwargs):

        density_value = self.kernel(x = x) / self._norm_const

        return density_value
    

    @abstractmethod
    def kernel(self, x: torch.Tensor, *args, **kwargs):

        pass
    
    
   
class EnergyDistribution(Distribution):


    def kernel(self, x: torch.Tensor, *args, **kwargs):
        return torch.exp( - self.energy(x = x))


    @abstractmethod
    def energy(self, x: torch.Tensor, *args, **kwargs):

        pass

    
    def energy_grad(self, x: torch.Tensor, *args, **kwargs):

        x.requires_grad_(True)

        # Compute the energy
        energy = self.energy(x)

        out = torch.ones_like(energy)
        # Compute gradients with respect to x
        gradient = torch.autograd.grad(outputs = energy, inputs = x, grad_outputs = out)[0]

        return gradient



"""
Distribution Adapter
-------------------------------------------------------------------------------------------------------------------------------------------
"""
class DistributionAdapter(EnergyDistribution):

    def __init__(self, energy_model: EnergyModel, norm_const: torch.Tensor = None):
        self.energy_model = energy_model
        self._norm_const = norm_const

    def energy(self, x: torch.Tensor, *args, **kwargs):
        return self.energy_model(x)


"""
Multivariate Gaussian
-------------------------------------------------------------------------------------------------------------------------------------------
"""
class MultivariateGaussian(EnergyDistribution):

    def __init__(self, mu: torch.Tensor, Sigma: torch.Tensor):

        self.mu = mu
        self.Sigma = Sigma
        self.Sigma_inv = torch.inverse(Sigma)


    def density(self, x: torch.Tensor):

        dim = x.size(-1)

        kernel_value = self.kernel(x)
        
        density_value = kernel_value / torch.sqrt((2 * torch.pi)**dim * tla.det(self.Sigma))

        return density_value


    def kernel(self, x: torch.Tensor):

        # x can be of shape (d,) or (n, d)
        x = torch.unsqueeze(x, dim=0) if x.dim() == 1 else x # shape (1, d) or (n, d)

        # Reshape mu and Sigma_inv to support broadcasting
        mu = self.params['mu'].unsqueeze(0)  # Shape (1, d)
        Sigma_inv = self.Sigma_inv()

        diff = x - mu # shape (1, d) or (n, d)
        exponent = -0.5 * tla.vecdot(
            torch.matmul(diff, Sigma_inv),
            diff,
        )

        kernel_value = torch.exp(exponent)
    
        return kernel_value


    def energy(self, x: torch.Tensor):

        energy = -torch.log(self.kernel(x))
        #energy = npl.multi_dot([(x - self.mu), self.Sigma_inv, (x - self.mu)])/2

        return energy
    

    def energy_grad(self, x: torch.Tensor):

        grad = tla.multi_dot([(x - self.mu), self.Sigma_inv])

        return grad



"""
Moderated Cosine
-------------------------------------------------------------------------------------------------------------------------------------------
Becomes more multimodal the larger W_cos gets.
"""

class ModeratedCosine(EnergyDistribution):

    def __init__(self, W_cos: torch.Tensor, mu: torch.Tensor):
        
        self.params = {
            'W_cos': W_cos.clone(),
            'mu': mu.clone(),
        }


    def energy(self, x: torch.Tensor):

        W_cos = self.params['W_cos']
        mu = self.params['mu']

        #Make x a tensor with dim = 2, if mu is scalar and x a batch the x values need to be stacked.
        x = torch.atleast_1d(x)
        x = x.unsqueeze(1) if mu.dim() == 0 else torch.atleast_2d(x)

        diff = x - mu

        cos_term = W_cos * torch.cos(x)
        log_norm_term = torch.log(torch.norm(diff, p = 2, dim = 1)**2 + 1)

        return cos_term + log_norm_term
    


"""
LogSum
-------------------------------------------------------------------------------------------------------------------------------------------
"""

class TwoLogSum(EnergyDistribution):

    def __init__(self, W: torch.Tensor, mu_1: torch.Tensor, mu_2: torch.Tensor):
        """
        args:
            W: tensor([w_1, w_2])
            Weights for modes

            mu_1: tensor of shape (d,)
            mu_2: tensor of shape (d,)
            Locations of the modes
        """
        self.params = {
            'W': W.clone(),
            'mu_1': mu_1.clone(),
            'mu_2': mu_2.clone(),
        }


    def energy(self, x: torch.Tensor):

        W = self.params['W_cos']
        mu_1 = self.params['mu_1']
        mu_2 = self.params['mu_2']

        #Make x a tensor with dim = 2, if mu is scalar and x a batch the x values need to be stacked.
        x = torch.atleast_1d(x)
        x = x.unsqueeze(1) if mu.dim() == 0 else torch.atleast_2d(x)

        diff_1 = x - mu_1
        diff_2 = x - mu_2

        log_norm_1 = torch.log(torch.norm(diff_1, p = 2, dim = 1)**2 + 1)
        log_norm_2 = torch.log(torch.norm(diff_2, p = 2, dim = 1)**2 + 1)

        return W[0]*log_norm_1 + W[1]*log_norm_2
    


"""
Univariate Polynomial Distribution
-------------------------------------------------------------------------------------------------------------------------------------------
"""
class UnivPolynomial(EnergyDistribution):

    def __init__(self, W_0: torch.Tensor):
        """
        Univariate Polynomial Energy
        Weight associated powers are interpreted like their index i.e. W[i] -> W[i] * x**i
        """
        self.params = {}
        self.params['W'] = W_0.clone()

    
    def energy(self, x: torch.Tensor):

        W = self.params['W']
        x = torch.atleast_1d(x).squeeze()
        
        # Use torchs method to create a matching Vandermonde Matrix
        vander = torch.vander(x, W.shape[0], increasing = True)
        
        result = torch.matmul(vander, W)
        
        return result
    

    @enable_grad_decorator
    def energy_grad(self, x: torch.Tensor):
        
        W_prime = self.params['W'][1:]
        x = torch.atleast_1d(x).squeeze()
        
        coeff = torch.arange(W_prime.shape[0], dtype=x.dtype) + 1
        W_prime = W_prime * coeff

        vander = torch.vander(x, W_prime.shape[0], increasing = True)
        
        grad = torch.matmul(vander, W_prime)
        grad = grad.unsqueeze(dim = -1)
        
        return grad






"""
Testing
-------------------------------------------------------------------------------------------------------------------------------------------
"""
if __name__ == "__main__":
    
    mu = torch.tensor([3, 3], dtype = torch.float32)
    Sigma = 2* torch.diag(torch.ones(size = (2,), dtype=torch.float32))
    print(Sigma)
    distribution = MultivariateGaussian(mu, Sigma)

    x_0 = torch.tensor([0, 0], dtype=torch.float32)

    density_value = distribution.density(x = x_0)
    energy_value = distribution.energy(x = x_0)
    energy_grad = distribution.energy_grad(x = x_0)

    print(f'Density value for {x_0}: \n', density_value)

    print(f'Energy value for {x_0}: \n', energy_value)

    print(f'Energy Gradient for {x_0}: \n', energy_grad)
