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
x = np. linspace(-5,5,100)

def target_closure(mu):

    def target_kernel(x):
        poly = (x-mu)**4 + 2 * (x-mu)**3 - 0.7 * (x-mu)**2 - 1.2 * (x-mu)
        return np.exp(-(poly))
    
    return target_kernel

target_kernel = target_closure(-2)

def target_pdf(x):
    return target_kernel(x)/np.trapz(target_kernel(x), dx=x[1]-x[0])



def start_kernel(x):
    poly = x**4 + x**3 + x**2 + x
    return np.exp(-(poly))

def start_pdf(x):
    return start_kernel(x)/np.trapz(start_kernel(x), dx=x[1]-x[0])



def recovery_closure(target_sample, sigma=1):

    x_tilde = target_sample + normal(loc = 0, scale = sigma)
    
    def recovery_kernel(x):
        poly = x**4 + x**3 + x**2 + x
        cond_term = 1/(2*sigma) * (x_tilde - x)**2
        return np.exp(-(poly - cond_term))
    
    return recovery_kernel

recovery_kernel = recovery_closure(-4, 1)

def recovery_pdf(x):
    return recovery_kernel(x)/np.trapz(recovery_kernel(x), dx=x[1]-x[0])

"""
Plot
-------------------------------------------------------------------------------------------------------------------------------------------
"""  
plt.plot(x, target_pdf(x), label='target pdf')
plt.plot(x, start_pdf(x), label='start pdf')
plt.plot(x, recovery_pdf(x), label='recovery pdf')

plt.legend()
plt.gcf().set_size_inches(6,3)
#plt.title(f'{sampler_class} - burnin: {burnin_offset} - samples: {batch_size}')
#plt.savefig(f'Figures/{sampler_class}_bi{burnin_offset}.png')

plt.show()
