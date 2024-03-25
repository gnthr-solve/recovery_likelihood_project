
import torch
import numpy as np
import pandas as pd
import scipy.stats as st
import matplotlib.pyplot as plt

from scipy.stats.sampling import NumericalInversePolynomial
from scipy.integrate import trapezoid, quad

from ebm import EnergyModel
"""
sample adapter
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
"""

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