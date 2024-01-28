
import time
from functools import wraps
import statistics


"""
BaseTimingDecorator
-------------------------------------------------------------------------------------------------------------------------------------------
"""

class BaseTimingDecorator:
    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.elapsed_time = 0

    def __call__(self, func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            self.start_time = time.time()
            result = func(*args, **kwargs)
            self.end_time = time.time()
            self.elapsed_time += self.end_time - self.start_time
            return result
        return wrapper

    def get_elapsed_time(self):
        return self.elapsed_time




"""
ProgramTimingDecorator
-------------------------------------------------------------------------------------------------------------------------------------------
"""

class ProgramTimingDecorator(BaseTimingDecorator):
    def __init__(self):
        super().__init__()
        self.method_times = {}

    def __call__(self, func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            method_start_time = time.time()
            result = func(*args, **kwargs)
            method_end_time = time.time()
            method_elapsed = method_end_time - method_start_time
            self.method_times[func.__name__] = self.method_times.get(func.__name__, 0) + method_elapsed
            self.elapsed_time += method_elapsed
            return result
        return wrapper

    def get_method_times(self):
        return self.method_times


"""
IterativeStepTimingDecorator
-------------------------------------------------------------------------------------------------------------------------------------------
"""
class IterativeStepTimingDecorator(BaseTimingDecorator):
    def __init__(self):
        super().__init__()
        self.iteration_times = []

    def __call__(self, func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            iteration_start_time = time.time()
            result = func(*args, **kwargs)
            iteration_end_time = time.time()
            iteration_elapsed = iteration_end_time - iteration_start_time
            self.iteration_times.append(iteration_elapsed)
            return result
        return wrapper

    def get_iteration_statistics(self):
        return {
            'mean': statistics.mean(self.iteration_times),
            'median': statistics.median(self.iteration_times),
            'std_dev': statistics.stdev(self.iteration_times),
            'total_iterations': len(self.iteration_times),
        }
    



"""
Descriptor-Decorator Timing
-------------------------------------------------------------------------------------------------------------------------------------------
"""
class LogAccess:
    def __init__(self, method):
        self.method = method

    def __get__(self, instance, owner):
        
        if instance is None:
            return self
        
        def wrapper(*args, **kwargs):
            print(f"Accessing method {self.method.__name__}")
            return self.method(instance, *args, **kwargs)
        
        return wrapper

class MyClass:
    @LogAccess
    def my_method(self):
        print("Doing something important.")

instance = MyClass()
instance.my_method()
