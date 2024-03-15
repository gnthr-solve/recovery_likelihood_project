

import torch
import torch.linalg as tla
from abc import ABC, abstractmethod
import typing

class EnergyDistribution(ABC):

    @abstractmethod
    def energy(self, x, *args, **kwargs):

        pass

    @abstractmethod
    def energy_grad(self, x, *args, **kwargs):

        pass



class Distribution(EnergyDistribution):

    def density(self, x, *args, **kwargs):

        density_value = self.kernel(x = x) / self._norm_const

        return density_value
    

    @abstractmethod
    def kernel(self, x, *args, **kwargs):

        pass
    
    
    def MC_expectation(self, samples, transform = (lambda x: x), *args, **kwargs):

        transformed_samples = transform(samples)

        MC_mean = torch.mean(transformed_samples)

        return MC_mean




"""
Multivariate Gaussian
-------------------------------------------------------------------------------------------------------------------------------------------
"""



class MultivariateGaussian(Distribution):

    def __init__(self, mu: torch.Tensor, Sigma: torch.Tensor):
        #super().__init__()
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
