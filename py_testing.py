import numpy as np
import scipy.stats as st
import re
import torch

from dataclasses import dataclass, field
from itertools import product

"""
Test
-------------------------------------------------------------------------------------------------------------------------------------------
"""

def copy_reference_test():

    some_list = []
    some_dict = {}
    
    x = 13

    some_list.append(x)
    some_dict['x'] = x

    x = 0

    #immutable objects like integers will not be changed in the container
    print('List preserves immutable:', some_list[0])
    print('Dict preserves immutable:',some_dict['x'])

    original_list = [1, 2, 3]

    some_list.append(original_list)
    some_dict['original_list'] = original_list

    original_list.append(4)
    
    #mutable objects like will also be changed in the container
    print('List tracks mutable:', some_list[1])
    print('Dict tracks mutable:',some_dict['original_list'])




"""
Test
-------------------------------------------------------------------------------------------------------------------------------------------
"""

def dataclasses_test():

    @dataclass(slots=True)
    class Parameters:

        name: str
        purpose: str
        param_list: list = field(default_factory = list)
        x: torch.tensor = torch.tensor(2, dtype = torch.float32)

    
    init_name = 'test_name'
    init_purpose = 'testing'
    init_param_list = [1]
    init_x = torch.tensor(1, dtype = torch.float32)

    params = Parameters(
        init_name, 
        init_purpose, 
        init_param_list, 
        init_x
    )

    print('Parameters after init:', params)
    
    params.name = 'new_test_name'
    params.param_list.append(2)
    
    print('Parameters after modification attempt:', params)







if __name__=="__main__":

    #copy_reference_test()
    dataclasses_test()