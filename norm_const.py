
import torch
import torch.linalg as tla
import typing

from abc import ABC, abstractmethod
from ebm import EnergyModel
from mc_samplers import EnergySampler
from experiment_params import SamplingParameters
from helper_tools import retrieve_class


class NormConstant:

    def __init__(self, sampler_params: SamplingParameters, start_batch_func, burn_in, num_samples):
    
        self.sampler_type = retrieve_class('mc_samplers', sampler_params['sampler_class'])
        self.sampler_params = sampler_params
        self.start_batch_func = start_batch_func
        self.burn_in = burn_in
        self.num_samples = num_samples


    def __call__(self, distribution_instance, dim) -> torch.Tensor:

        start_batch = self.start_batch_func(dim)

        sampler = self.sampler_type(
            energy_model = distribution_instance, 
            start_batch = start_batch, 
            **self.sampler_params
        )

        sample_batch = sampler.sample(num_samples = self.num_samples, burnin_offset = self.burn_in)
        kernel_values = distribution_instance.kernel(sample_batch)

        norm_constant = kernel_values.mean()

        return norm_constant
    


sampling_params = SamplingParameters(
    sampler_class = 'MALASampler',
    epsilon = torch.tensor(1e-1, dtype = torch.float32),
)
start_batch_func = lambda dim: torch.zeros(size = (500, dim))#.squeeze()
burn_in = 1e+4
num_samples = torch.tensor(1e+5, dtype = torch.float32)

norm_const = NormConstant(
    sampler_params = sampling_params,
    start_batch_func = start_batch_func,
    burn_in = burn_in,
    num_samples= num_samples
)