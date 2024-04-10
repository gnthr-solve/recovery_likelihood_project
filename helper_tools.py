
import torch
import torch.linalg as tla
import numpy as np
import pandas as pd
import re
import importlib

from itertools import product
from functools import wraps
from torch.distributions import MultivariateNormal


"""
No-Grad Decorator
-------------------------------------------------------------------------------------------------------------------------------------------
A wrapper decorator that enables the torch.no_grad() context to avoid gradient tracking.
Used in the samplers, for whose iterations we do not want to build a computational graph.
"""
def no_grad_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        with torch.no_grad():
            result = func(*args, **kwargs)
        return result
    return wrapper


class NoGradDescriptor:

    def __init__(self, func):
        self.func = func

    def __get__(self, instance, owner):

        def wrapper(*args, **kwargs):

            with torch.no_grad():
                result = self.func(instance, *args, **kwargs)

            return result
        
        return wrapper
    
"""
Enable-Grad Decorator
-------------------------------------------------------------------------------------------------------------------------------------------
Wraps a function or method in torch.enable_grad context manager.
When a test model does not have an implementation for the energy_grad method it is done via torchs autograd.
energy_grad is called repeatedly in the samplers however which are in a no_grad context.
Decorating EnergyModel's energy_grad method with this decorator 
allows to temporarily enable gradient tracking within the samplers general no_grad context.
"""
def enable_grad_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        with torch.enable_grad():
            result = func(*args, **kwargs)
        return result
    return wrapper


"""
Numpy Decorator
-------------------------------------------------------------------------------------------------------------------------------------------
Small helper designed to use numpy/scipy quadrature for methods that expect a torch.Tensor
"""
def numpy_adapter(func):

    @wraps(func)
    def wrapper(self, x):
        
        x = torch.tensor(data = x, dtype=torch.float32)
        
        result = func(self, x).numpy()
        
        return result
    
    return wrapper



"""
Torch Functions
-------------------------------------------------------------------------------------------------------------------------------------------
"""
def quadratic_form_batch(x_batch: torch.Tensor, matrix: torch.Tensor):
    """
    Calculates the quadratic form x^T M x for a batch of tensors x_batch.
    """
    result = tla.vecdot(
        torch.matmul(x_batch, matrix),
        x_batch
    )
    
    return result


def non_batch_dims(tensor_batch: torch.Tensor):
    return tuple(range(1, tensor_batch.dim()))



"""
Config Setup Functions
-------------------------------------------------------------------------------------------------------------------------------------------
Instantiating classes with hydra requires a specific form (see the param_dict_to_hydra function).
The _target_ keyword that is stored in a config.yaml accepts either a type to instantiate or a function that handles the instantiation.
Since it is important for torch that the datatypes of tensors align, the _target_ for tensors is set to the convert_to_tensor function.

To avoid having to import all possible models, optimisers, schedulers and so on the retrieve_class function is defined.
This is used in the ExperimentBuilder.
"""

def retrieve_class(module_name, class_name):
    return importlib.import_module(module_name).__dict__[class_name]



def param_dict_tolist(param_dict: dict):
    output_dict = {}
    for name, value in param_dict.items():
        #print(name, type(value))
        if isinstance(value, torch.Tensor):
            output_dict[name] = value.tolist()
        elif isinstance(value, dict):
            output_dict[name] = param_dict_tolist(value)
        else:
            output_dict[name] = value
    
    #print(output_dict)
    return output_dict


def param_dict_to_hydra(param_dict: dict):
    output_dict = {}
    for name, value in param_dict.items():
        #print(name, type(value))
        if isinstance(value, torch.Tensor):
            output_dict[name] = {
                '_target_': 'helper_tools.convert_to_tensor',
                'data': value.tolist(),
            }
        elif isinstance(value, dict):
            output_dict[name] = param_dict_to_hydra(value)
        else:
            output_dict[name] = value
    
    #print(output_dict)
    return output_dict


def convert_to_tensor(data):
    return torch.tensor(data=data, dtype= torch.float32)



"""
Pandas to torch
-------------------------------------------------------------------------------------------------------------------------------------------
The ParameterAssessor in the metrics module calculates metrics for the parameter columns.
Here the problem is that pandas stores tensors and arrays as strings.
This function evaluates such strings, converts them to numpy arrays, stacks them along the batch dimension 
and returns them as a torch.Tensor.
"""
def param_record_to_torch(param_column):
    
    func = lambda s: np.array(eval(s))

    param_array_column = param_column.map(func) if param_column.dtype == object else param_column

    params_array = np.stack(param_array_column, axis = 0)
    # Check if the array is 1D
    if params_array.ndim == 1:
        params_array = np.expand_dims(params_array, axis=1)

    params_tensor = torch.tensor(params_array, dtype= torch.float32)
    
    return params_tensor


"""
Torch Functions
-------------------------------------------------------------------------------------------------------------------------------------------
"""






if __name__ == "__main__":
    """
    Implementation test example for verification.
    """
    test_model_dict = {
        'model_class': 'MultivariateGaussianModel',
        'start_params': {
            'mu_0': torch.tensor([2, 2], dtype = torch.float32),
            'Sigma_0': torch.tensor(
                [[2, 0],
                 [0, 1],],
                dtype=torch.float32,
            )
        }
    }

    #print(param_dict_tolist(test_model_dict))
    print(param_dict_to_hydra(test_model_dict))

