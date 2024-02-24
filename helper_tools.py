import torch
import torch.linalg as tla
import numpy as np
import scipy.stats as st
import re

from itertools import product
from functools import wraps
from torch.distributions import MultivariateNormal


"""
No-Grad Decorator
-------------------------------------------------------------------------------------------------------------------------------------------
"""
def no_grad_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        with torch.no_grad():
            result = func(*args, **kwargs)
        return result
    return wrapper



"""
check_nan Decorator
-------------------------------------------------------------------------------------------------------------------------------------------
"""
def check_nan(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        if torch.isnan(result).any():
            print(f"NaN detected in result of {func.__name__}")
            print(*args)
        return result
    return wrapper


"""
Input - Output Decorator
-------------------------------------------------------------------------------------------------------------------------------------------
"""
def in_out_logger(func):
    @wraps(func)
    def wrapper(*args, **kwargs):

        print(f'Inputs {func.__name__}:')
        for arg in args[1:]:
            print(arg[-10:])

        result = func(*args, **kwargs)

        print(f'Output {func.__name__}:')
        print(result[-10:])

        return result
    
    return wrapper

"""
Helper functions
-------------------------------------------------------------------------------------------------------------------------------------------
"""
def quadratic_form_batch(x_batch: torch.tensor, matrix: torch.tensor):
    
    result = tla.vecdot(
        torch.matmul(x_batch, matrix),
        x_batch
    )
    
    return result

"""
Helper Classes
-------------------------------------------------------------------------------------------------------------------------------------------
"""
class MultivariateNormalIterator:

    def __init__(self, mu, Sigma, num_samples):

        self.num_samples = num_samples

        mv_normal = MultivariateNormal(loc = mu, covariance_matrix = Sigma)
        self.samples = mv_normal.sample_n(n = num_samples)
        #self.samples = np.random.multivariate_normal(mean = mu, cov = Sigma, size = num_samples)
        
        self.current_sample = 0


    def __iter__(self):
        
        return self


    def __next__(self):

        if self.current_sample < self.num_samples:

            sample = self.samples[self.current_sample]
            self.current_sample += 1
            return sample
        
        else:

            raise StopIteration



if __name__ == "__main__":
    # Example usage:
    mean_vector = torch.tensor([0, 0], dtype = torch.float32)
    covariance_matrix = torch.tensor([[1, 0.5], [0.5, 1]], dtype = torch.float32)
    num_samples = 5

    mvn_iterator = MultivariateNormalIterator(mean_vector, covariance_matrix, num_samples)

    for sample in mvn_iterator:
        print(sample)


