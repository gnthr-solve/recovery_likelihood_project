

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
Gaussians
-------------------------------------------------------------------------------------------------------------------------------------------
"""
class UnivariateGaussian(Distribution):

    def __init__(self, mu, sigma):
        #super().__init__()
        self.mu = mu
        self.sigma = sigma


    def density(self, x):
        
        density_value = torch.exp(-(x - self.mu)**2 /(2*self.sigma**2)) / torch.sqrt(2 * torch.pi * self.sigma**2)

        return density_value


    def kernel(self, x):

        kernel_value = torch.exp(-(x - self.mu)**2 /(2*self.sigma**2))

        return kernel_value


    def energy(self, x):

        energy = -torch.log(self.kernel(x))
        #energy = torch.log(self.density(x))
        #energy = -((x/self.sigma - self.mu/self.sigma)**2 - torch.log(2 * torch.pi * self.sigma**2))/2

        return energy
    

    def energy_grad(self, x):

        #grad = (self.mu - x) / self.sigma**2
        grad = (x - self.mu) / self.sigma**2

        return grad
        


class MultivariateGaussian(Distribution):

    def __init__(self, mu, Sigma):
        #super().__init__()
        self.mu = mu
        self.Sigma = Sigma
        self.Sigma_inv = torch.inverse(Sigma)


    def density(self, x):

        dim = x.size(-1)

        kernel_value = self.kernel(x)
        
        density_value = kernel_value / torch.sqrt((2 * torch.pi)**dim * tla.det(self.Sigma))

        return density_value


    def kernel(self, x):

        kernel_value = torch.exp(-tla.multi_dot([(x - self.mu), self.Sigma_inv, (x - self.mu)])/2)

        return kernel_value


    def energy(self, x):

        energy = -torch.log(self.kernel(x))
        #energy = npl.multi_dot([(x - self.mu), self.Sigma_inv, (x - self.mu)])/2

        return energy
    

    def energy_grad(self, x):

        grad = tla.multi_dot([self.Sigma_inv, (x - self.mu)])

        return grad




class MultivariateGaussianB(Distribution):

    def __init__(self, mu, Sigma):
        #super().__init__()
        self.mu = mu
        self.Sigma = Sigma
        self.Sigma_inv = torch.inverse(Sigma)


    def density(self, x):

        dim = x.size(-1)

        kernel_value = self.kernel(x)
        
        density_value = kernel_value / torch.sqrt((2 * torch.pi)**dim * tla.det(self.Sigma))

        return density_value


    def kernel(self, x):

        # x can be of shape (d,) or (n, d)
        # Reshape mu and Sigma_inv to support broadcasting
        mu = self.mu.unsqueeze(0)  # Shape (1, d)
        Sigma_inv = self.Sigma_inv.unsqueeze(0)  # Shape (1, d, d)

        # Perform batch matrix multiplication for the quadratic form
        # (x - mu) is of shape (n, d), unsqueeze to (n, 1, d) for bmm
        # (x - mu).unsqueeze(1) @ Sigma_inv performs (n, 1, d) @ (1, d, d) -> (n, 1, d)
        # The result is then multiplied by (x - mu).unsqueeze(-1) of shape (n, d, 1)
        # Resulting in a shape of (n, 1, 1), which we squeeze back to (n,)
        diff = x - mu
        exponent = -0.5 * diff.unsqueeze(1).bmm(Sigma_inv).bmm(diff.unsqueeze(-1)).squeeze()

        kernel_value = torch.exp(exponent)

        return kernel_value


    def energy(self, x):

        energy = -torch.log(self.kernel(x))
        #energy = npl.multi_dot([(x - self.mu), self.Sigma_inv, (x - self.mu)])/2

        return energy
    

    def energy_grad(self, x):

        grad = tla.multi_dot([self.Sigma_inv, (x - self.mu)])

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
