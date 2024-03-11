import torch
import torch.linalg as tla
import re
import importlib

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
def quadratic_form_batch(x_batch: torch.Tensor, matrix: torch.Tensor):
    
    result = tla.vecdot(
        torch.matmul(x_batch, matrix),
        x_batch
    )
    
    return result



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




if __name__ == "__main__":
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

