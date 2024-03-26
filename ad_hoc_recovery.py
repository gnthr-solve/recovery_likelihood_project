import torch
import numpy as np
import matplotlib.pyplot as plt

from numpy.random import normal

from test_models import UnivPolynomial
from experiment_params import SamplingParameters
from experiment import ExperimentBuilder



"""
Create pdf to plot
-------------------------------------------------------------------------------------------------------------------------------------------
"""
x = np. linspace(-10,10,1000)

def target_closure(mu):

    def target_kernel(x):
        poly = (x-mu)**4 + 2 * (x-mu)**3 - 0.7 * (x-mu)**2 - 1.2 * (x-mu)
        return np.exp(-(poly))
    
    return target_kernel

target_kernel = target_closure(-4)

def target_pdf(x):
    return target_kernel(x)/np.trapz(target_kernel(x), dx=x[1]-x[0])



def start_kernel(x):
    poly = x**4 + x**3 + x**2 + x
    return np.exp(-(poly))

def start_pdf(x):
    return start_kernel(x)/np.trapz(start_kernel(x), dx=x[1]-x[0])


sigma = 0.5
normal_sample = normal(loc = 0, scale = sigma)

def target_recovery_closure(target_sample, sigma=0.5):

    x_tilde = target_sample + normal_sample
    
    def recovery_kernel(x):
        poly = (x)**4 + 2 * (x)**3 - 0.7 * (x)**2 - 1.2 * (x)
        cond_term = 1/(2*sigma**2) * (x_tilde - x)**2
        return np.exp(-(poly + cond_term))
    
    return recovery_kernel


def start_recovery_closure(target_sample, sigma=0.5):

    x_tilde = target_sample + normal_sample
    
    def recovery_kernel(x):
        poly = x**4 + x**3 + x**2 + x
        cond_term = 1/(2 * sigma**2) * (x_tilde - x)**2
        return np.exp(-(poly + cond_term))
    
    return recovery_kernel


recovery_kernel = start_recovery_closure(-6, sigma)
#recovery_kernel = target_recovery_closure(0, sigma)

def recovery_pdf(x):
    return recovery_kernel(x)/np.trapz(recovery_kernel(x), dx=x[1]-x[0])


def approx_pdf(x):
    x_tilde = -4 + normal_sample
    mu = x_tilde - sigma**2 * (4*x_tilde**3 + 3*x_tilde**2 + 2*x_tilde +1)
    return np.sqrt(1/(2 *np.pi * sigma**2))* np.exp(- (x - mu)**2 /(2*sigma**2))
"""
Plot
-------------------------------------------------------------------------------------------------------------------------------------------
"""  
plt.plot(x, target_pdf(x), label='target pdf')
plt.plot(x, start_pdf(x), label='start pdf')
plt.plot(x, recovery_pdf(x), label='recovery pdf')
#plt.plot(x, approx_pdf(x), label='approx pdf')

plt.legend()
plt.gcf().set_size_inches(6,3)
#plt.title(f'{sampler_class} - burnin: {burnin_offset} - samples: {batch_size}')
#plt.savefig(f'Figures/{sampler_class}_bi{burnin_offset}.png')

plt.show()
