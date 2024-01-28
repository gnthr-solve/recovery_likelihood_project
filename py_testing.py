import numpy as np
import scipy.stats as st
import re
import torch_distributions

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








if __name__=="__main__":

    copy_reference_test()