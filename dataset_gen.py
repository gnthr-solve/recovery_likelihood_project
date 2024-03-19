
import torch
import numpy as np
import pandas as pd
import scipy.stats as st
import matplotlib.pyplot as plt

from scipy.stats import norm
from scipy.stats.sampling import NumericalInversePolynomial
"""
Univariate Polynomial
-------------------------------------------------------------------------------------------------------------------------------------------
"""
class UnivPolynomial:
    
    def pdf(self, x):
        return np.exp(-( x**4 + 2*(x**3) - 0.7*(x**2) - 1.2*x))/12.165

dist = UnivPolynomial()
urng = np.random.default_rng()
rng = NumericalInversePolynomial(dist, random_state=urng)
rvs = rng.rvs(10000)
x = np.linspace(rvs.min()-0.1, rvs.max()+0.1, num=10000)
fx = dist.pdf(x)
plt.plot(x, fx, "r-", label="pdf")
plt.hist(rvs, bins=50, density=True, alpha=0.8, label="rvs")
plt.xlabel("x")
plt.ylabel("PDF(x)")
plt.title("Samples drawn using PINV method.")
plt.legend()
plt.show()