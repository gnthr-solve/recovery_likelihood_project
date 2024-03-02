

import time
import numpy as np

from functools import wraps


"""
Timing Info Descriptor
-----------------------------------------------------------------------------------------------------------------------------------------------
"""
class MethodTimingDescriptor:

    def __init__(self, func):
        self.func = func
        self.execution_times = []


    def __get__(self, instance, owner):

        self.class_name = owner.__name__

        def wrapper(*args, **kwargs):

            start_time = time.time()
            
            result = self.func(instance, *args, **kwargs)
            
            end_time = time.time()
            exec_time = end_time - start_time
            
            self.execution_times.append(exec_time)
            
            return result
        
        return wrapper


"""
Timing Info Descriptor-Decorator
-----------------------------------------------------------------------------------------------------------------------------------------------
"""
class TimingDecorator:

    def __init__(self):
        self._timed_methods = {}


    def __call__(self, func):

        descriptor = MethodTimingDescriptor(func)

        self._timed_methods[id(descriptor)] = descriptor
        
        return descriptor


    def return_average_times(self):

        avg_exec_times = {
            f'{descriptor.class_name}.{descriptor.func.__name__}': np.mean(descriptor.execution_times)
            for descriptor in self._timed_methods.values()
            if descriptor.execution_times != []
        }

        return avg_exec_times


# Instantiate the decorator
timing_decorator = TimingDecorator()



