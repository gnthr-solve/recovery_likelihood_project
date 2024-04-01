
import torch
import numpy as np
import pandas as pd
import scipy.stats as st
import matplotlib.pyplot as plt

from scipy.stats.sampling import NumericalInversePolynomial
from scipy.integrate import trapezoid, quad

from ebm import EnergyModel

"""
File to generate the training datasets for test models.
The partition is numerically approximated with numpy or scipy methods 
and NumericalInversePolynomial is used to sample from the resulting densities.
"""


"""
Sample adapter Blueprint
-------------------------------------------------------------------------------------------------------------------------------------------
"""
class SampleAdapter:

    def __init__(self, model: EnergyModel, domain_limits):
        self.model = model
        self.domain_limits = domain_limits

    def kernel(self, x):
        return np.exp(-self.energy(x))
    
    def energy(self, x):
        return self.model.energy(x)
    
    def calc_norm_const(self):
        self.norm_const = quad(self.kernel, *self.domain_limits)


"""
Univariate Polynomial
-------------------------------------------------------------------------------------------------------------------------------------------


def poly_kernel(x):
    return np.exp(-( x**4 + 2*(x**3) - 0.7*(x**2) - 1.2*x))


domain = np.linspace(-5, 5, num = int(1e+4))
kernel_values = poly_kernel(domain)
print(kernel_values[:10])
norm_const = trapezoid(kernel_values, domain)
print(norm_const)


class UnivPolynomial:
    
    def pdf(self, x):
        return poly_kernel(x)/norm_const
    


dist = UnivPolynomial()
urng = np.random.default_rng()
rng = NumericalInversePolynomial(dist, random_state=urng)
poly_dataset = rng.rvs(10000)
print(poly_dataset[:10])
'''
x = np.linspace(rvs.min()-0.1, rvs.max()+0.1, num=10000)
fx = dist.pdf(x)
plt.plot(x, fx, "r-", label="pdf")
plt.hist(rvs, bins=50, density=True, alpha=0.8, label="rvs")
plt.xlabel("x")
plt.ylabel("PDF(x)")
plt.title("Samples drawn using PINV method.")
plt.legend()
plt.show()
'''
"""

"""
Univ Cosine Model
-------------------------------------------------------------------------------------------------------------------------------------------
"""

def cos_kernel(x):
    diff = x - 2
    
    energy = 1 * np.cos(diff**2) + np.log(diff**2 + 1)
    #energy = 1 * np.cos(diff) + np.log(diff**2 + 1)
    return np.exp(-energy)


domain = np.linspace(-28, 32, num = int(1e+5))
kernel_values = cos_kernel(domain)
#print(kernel_values[:10])
norm_const = trapezoid(kernel_values, domain)
#print(norm_const)


class UnivCos:
    
    def pdf(self, x):
        return cos_kernel(x)/norm_const
    


dist = UnivCos()
urng = np.random.default_rng()
rng = NumericalInversePolynomial(dist, domain = [-28, 32], random_state=urng)
cos_dataset = rng.rvs(10000)
print(cos_dataset[:10])


x = np.linspace(cos_dataset.min()-0.1, cos_dataset.max()+0.1, num=10000)
fx = dist.pdf(x)
plt.plot(x, fx, "r-", label="pdf")
plt.hist(cos_dataset, bins=500, density=True, alpha=0.8, label="rvs")
plt.xlabel("x")
plt.ylabel("PDF(x)")
plt.title("Samples drawn using PINV method.")
plt.legend()
plt.show()
'''
'''