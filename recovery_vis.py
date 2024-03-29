import torch
import numpy as np
import matplotlib.pyplot as plt

from numpy.random import normal
from abc import ABC, abstractmethod
from scipy.integrate import trapezoid, quad
"""
Classes
-------------------------------------------------------------------------------------------------------------------------------------------
"""

class EnergyDist(ABC):
    
    def energy(self, x):
        pass

    def kernel(self, x):
        return np.exp(-self.energy(x))
    
    def pdf(self, x):
        return self.kernel(x)/self.norm_const


class Poly(EnergyDist):

    def __init__(self, W, mu, domain):
        self.W = W
        self.mu = mu
        self.norm_const = np.trapz(self.kernel(domain), dx = domain[1]- domain[0])

    def energy(self, x: np.ndarray):
        W = self.W
        mu = self.mu
        
        diff = x - mu
        diff = diff[:, np.newaxis]
        
        energy = np.sum(W * diff ** np.arange(1, len(W) + 1), axis=1)

        return energy


class Cos(EnergyDist):

    def __init__(self, W, mu, domain):
        self.W = W
        self.mu = mu
        self.norm_const = np.trapz(self.kernel(domain), dx = domain[1]- domain[0])
        #self.norm_const = trapezoid(self.kernel(domain), domain)
        #print(self.norm_const)


    def energy(self, x: np.ndarray):
        W = self.W
        mu = self.mu
        
        diff = x - mu
        diff = diff[:, np.newaxis]
        
        energy = W * np.cos(diff**2) + np.log(diff**2 + 1)
        #print(x.shape)
        #print(diff.shape)
        #print(energy.shape)
        return energy.squeeze()


class RecoveryAdapter(EnergyDist):

    def __init__(self, dist: EnergyDist, sigma, domain):
        self.dist = dist
        self.sigma = sigma
        self.domain = domain

    def energy(self, x: np.ndarray):

        x_tilde = self.sample + normal(loc = 0, scale=self.sigma)

        cond_term = 1/(2*self.sigma**2) * (x_tilde - x)**2

        energy = self.dist.energy(x) + cond_term
        
        return energy

    def set_sample(self, sample):
        self.sample = sample
        self.norm_const = np.trapz(self.kernel(self.domain), dx = self.domain[1]- self.domain[0])
        #self.norm_const = trapezoid(self.kernel(self.domain), self.domain)
        print(self.norm_const)

"""
Create pdf to plot
-------------------------------------------------------------------------------------------------------------------------------------------
"""
x = np. linspace(-10,10,1000)

target_w = [-1.2, -0.7, 2, 1]
poly = Poly(W = target_w, mu = -4, domain = x)

sigma = 0.3
adapted_poly = RecoveryAdapter(poly, sigma, x)
adapted_poly.set_sample(-4)

adapted_poly_2 = RecoveryAdapter(poly, sigma, x)
adapted_poly_2.set_sample(-7)

target_cos = Cos(W = 1, mu = 2, domain = x)
start_cos = Cos(W = -0.5, mu = -2, domain = x)

adapted_start_cos = RecoveryAdapter(start_cos, sigma, x)
adapted_start_cos.set_sample(2)

#print(poly.kernel(x)[:10])
#print(target_cos.kernel(x)[:10])
#print(adapted_start_cos.pdf(x)[:10])
"""
Plot
-------------------------------------------------------------------------------------------------------------------------------------------
"""  

plt.plot(x, target_cos.pdf(x), label='target_cos pdf')
plt.plot(x, start_cos.pdf(x), label='start pdf')
plt.plot(x, adapted_start_cos.pdf(x), label='recovery pdf')
#plt.plot(x, poly.pdf(x), label='poly pdf')
#plt.plot(x, adapted_poly.pdf(x), label='recovery pdf -4')
#plt.plot(x, adapted_poly_2.pdf(x), label='recovery pdf -7')
#plt.plot(x, approx_pdf(x), label='approx pdf')

plt.legend()
plt.gcf().set_size_inches(6,3)
#plt.title(f'{sampler_class} - burnin: {burnin_offset} - samples: {batch_size}')
#plt.savefig(f'Figures/{sampler_class}_bi{burnin_offset}.png')

plt.show()






"""

normal_sample = normal(loc = 0, scale = sigma)


def approx_pdf(x):
    x_tilde = -4 + normal_sample
    mu = x_tilde - sigma**2 * (4*x_tilde**3 + 3*x_tilde**2 + 2*x_tilde +1)
    return np.sqrt(1/(2 *np.pi * sigma**2))* np.exp(- (x - mu)**2 /(2*sigma**2))


"""