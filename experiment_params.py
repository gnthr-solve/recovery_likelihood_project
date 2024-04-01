
import torch

from abc import ABC, abstractmethod
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
from omegaconf import DictConfig, OmegaConf

from helper_tools import param_dict_tolist, param_dict_to_hydra


"""
This file contains custom dictionary-like implementations for the involved parameter sets.
The classes allow organising the parameters in semantic units, and pass them with the **kwargs syntax.
This is particularly useful for component instantiation in the ExperimentBuilder.

All Parameterset subclasses also inherit the as_config method.
as_config creates the structured DictConfig form of the parameter sets 
that allow the setup files to create the configs necessary for hydra to instantiate classes and torch.Tensors.
It uses the param_dict_to_hydra function defined in helper_tools.
"""


"""
Parameterset interface
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
"""
class Parameterset(ABC):
    __slots__ = ()  # To be overridden by subclasses

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def __delitem__(self, key):
        delattr(self, key)

    def keys(self):
        return self.__slots__

    def values(self):
        return [getattr(self, key) for key in self.__slots__]

    def items(self):
        return [(key, getattr(self, key)) for key in self.__slots__]


    def as_config(self):
        
        #content_dict = param_dict_tolist(self)
        content_dict = param_dict_to_hydra(self)

        #content = {self.__class__.__name__: content_dict}
        content = {self.__class__.__name__: {
                '_target_': f'experiment_params.{self.__class__.__name__}',
                **content_dict
            }
        }
        config = DictConfig(content = content)
        #config = OmegaConf.create(content)
        return config


"""
SamplingParameters
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
"""

class SamplingParameters(Parameterset):

    __slots__ = (
        'sampler_class',
        'epsilon',
        #'start_batch',
        'L',
        'M',
    )

    def __init__(self, **kwargs):

        for key in self.__slots__:
            
            setattr(self, key, kwargs.get(key, None))
    

    
"""
ModelParameters
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
"""
class ModelParameters(Parameterset):

    __slots__ = (
        'model_class',
        'target_params',
        'start_params',
        'requires_adapter',
        'perturbation_var',
    )

    def __init__(self, **kwargs):

        for key in self.__slots__:
            
            setattr(self, key, kwargs.get(key, None))  



"""
HyperParameters
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
"""
class HyperParameters(Parameterset):

    __slots__ = (
        'batch_size',
        'epochs',
        'burnin_offset',
        'model_batch_size',
        'likelihood_class',
        'optimizer_class',
        'optimizer_params',
        'scheduler_class',
        'scheduler_params',
    )

    def __init__(self, **kwargs):

        for key in self.__slots__:
            
            setattr(self, key, kwargs.get(key, None))  


"""
LikelihoodParameters
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
"""
class LikelihoodParameters(Parameterset):

    __slots__ = (
        
    )

    def __init__(self, **kwargs):

        for key in self.__slots__:
            
            setattr(self, key, kwargs.get(key, None)) 





if __name__ == "__main__":
    """
    Implementation test example for verification.
    """
    import yaml

    hyper_params = HyperParameters(
        batch_size = 200,
        epochs = 10,
        burnin_offset = 50,
        model_batch_size = 200,
        likelihood_class = 'RecoveryLikelihood',
        optimizer_class = 'Adam',
        optimizer_params = {
            'lr': 1e-1,
        },
        scheduler_class = 'ExponentialLR',
        scheduler_params = {
            'gamma': 0.9
        },
    )

    sampling_params = SamplingParameters(
        sampler_class = 'MALASampler',
        #start_batch = torch.zeros(size = (5, 2)),
        epsilon = torch.tensor(1e-1, dtype = torch.float32),
        L = 3,
        M = torch.eye(n = 2),
    )

    model_params = ModelParameters(
        model_class = 'MultivariateGaussianModel',
        start_params = {
            'mu_0': torch.tensor([2, 2], dtype = torch.float32),
            'Sigma_0': torch.tensor(
                [[2, 0],
                 [0, 1],],
                dtype=torch.float32,
            )
        },
        requires_adapter = True,
        perturbation_var = torch.tensor(1, dtype = torch.float32),
    )


    paramsets = [
        hyper_params, 
        model_params, 
        sampling_params, 
    ]

    paramsets_config = [params.as_config() for params in paramsets]
    
    params_config = OmegaConf.merge(*paramsets_config)
    print(params_config)
    OmegaConf.save(config = params_config, f="config.yaml")
