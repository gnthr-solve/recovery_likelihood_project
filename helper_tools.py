
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
"""
def no_grad_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        with torch.no_grad():
            result = func(*args, **kwargs)
        return result
    return wrapper


"""
Enable-Grad Decorator
-------------------------------------------------------------------------------------------------------------------------------------------
"""
def enable_grad_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        with torch.enable_grad():
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
Torch Functions
-------------------------------------------------------------------------------------------------------------------------------------------
"""
def quadratic_form_batch(x_batch: torch.Tensor, matrix: torch.Tensor):
    
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
"""
def param_record_to_torch(param_column):
    
    func = lambda s: np.array(eval(s))

    param_array_column = param_column.map(func)

    params_array = np.stack(param_array_column, axis = 0)

    params_tensor = torch.tensor(params_array, dtype= torch.float32)
    
    return params_tensor



"""
Plotting Helper Functions
-------------------------------------------------------------------------------------------------------------------------------------------
"""

def prepare_sub_dfs(result_df: pd.DataFrame, comparison_column: str, filter_cols: dict = None)->dict[str, pd.DataFrame]:
    """
        Filters a DataFrame and splits it based on unique values in a comparison column.
        
        Args:
        - df: Pandas DataFrame
        - comparison_column: Name of the column to use for splitting
        - filter_cols: Dictionary of column, value pairs to filter before splitting
        
        Returns:
        - Dictionary where keys are unique values in comparison column and values are corresponding DataFrame subsets
    """
    split_dict = {}

    # Filter the dataframe by comparison columns having specific value like sampler = MALASampler
    if filter_cols:
        for col, val in filter_cols.items():
            filter_mask = result_df[col] == val
            result_df = result_df.loc[filter_mask]
    
    unique_col_values = result_df[comparison_column].unique()
    matching_mask = lambda value: result_df[comparison_column] == value

    for col_value in unique_col_values:
        entry_name = comparison_column + f': {col_value}'
        split_dict[entry_name] = result_df.loc[matching_mask(col_value)].copy()
    
    return split_dict



def remove_duplicate_plot_descriptors(array: np.ndarray, axis: int, inverse: bool):

    shape = array.shape
    axis_iter = [
        array[k,:] if axis == 0 else array[:, k]
        for k in range(shape[axis])
    ]

    for k, slice in enumerate(axis_iter):

        if inverse:
            mask_slice = slice[::-1]
        else:
            mask_slice = slice
        
        _, unique_slice_inds = np.unique(mask_slice, return_index = True)
        
        mask = np.zeros_like(slice, dtype=bool)
        mask[unique_slice_inds] = True

        if inverse:
            slice = np.where(mask[::-1], slice, '')
        else:
            slice = np.where(mask, slice, '')

        if axis == 0:
            array[k, :] = slice
        else:
            array[:, k] = slice
        
    return array
        









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

