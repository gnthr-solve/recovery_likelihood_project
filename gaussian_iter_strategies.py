
import torch
import torch.linalg as tla

from torch.distributions import MultivariateNormal
from abc import ABC, abstractmethod



"""
Gaussian Iteration Strategy
-------------------------------------------------------------------------------------------------------------------------------------------
"""
class IterStrategy(ABC):

    @abstractmethod
    def generate(self, chain_length: int):
        pass

    @abstractmethod
    def __iter__(self):
        pass
    
    @abstractmethod
    def __next__(self):
        pass

"""
Standard Gaussian Iteration Strategy
-------------------------------------------------------------------------------------------------------------------------------------------
"""
class StdIterStrategy(IterStrategy):

    def __init__(self, data_dim, chain_num):

        self.chain_num = chain_num

        mean = torch.zeros(size = (data_dim,), dtype = torch.float32)
        cov = torch.eye(data_dim)

        self.dist = MultivariateNormal(loc = mean, covariance_matrix = cov)


    def generate(self, chain_length: int):

        samples = self.dist.sample(sample_shape = (chain_length, self.chain_num))

        self.sample_iterator = iter(samples)


    def __iter__(self):
        
        return self


    def __next__(self):

        return next(self.sample_iterator)
    


"""
Standard Gaussian Iteration Strategy
-------------------------------------------------------------------------------------------------------------------------------------------
"""
class MomentumIterStrategy(IterStrategy):

    def __init__(self, data_dim, M, chain_num):

        self.chain_num = chain_num

        mean = torch.zeros(size = (data_dim,), dtype = torch.float32)

        self.dist = MultivariateNormal(loc = mean, covariance_matrix = M)


    def generate(self, chain_length: int):

        samples = self.dist.sample(sample_shape = (chain_length, self.chain_num))
        
        self.sample_iterator = iter(samples)


    def __iter__(self):
        
        return self


    def __next__(self):

        return next(self.sample_iterator)