
import torch
import time

from abc import ABC, abstractmethod


"""
Subject Role Interface
-------------------------------------------------------------------------------------------------------------------------------------------
"""
class Subject:
    def __init__(self):
        self.observers = []

    def register_observer(self, observer):
        self.observers.append(observer)

    def notify_observers(self):
        for observer in self.observers:
            observer.update(self)

"""
Observer ABC
-------------------------------------------------------------------------------------------------------------------------------------------
"""
class Observer(ABC):

    @abstractmethod
    def update(subject_instance, *args, **kwargs):
        pass



"""
TrainingObserver Class Approach following Observer Pattern
-------------------------------------------------------------------------------------------------------------------------------------------
Make TrainingProcedure incorporate a Subject role and update the observer at every Training Loop completion.
Benefits:
    Modular
    Can add more than one observer and vary observer and subject independently
Downsides:
    Requires altering TrainingProcedure class.
    More Boilerplate.
    Potentially larger overhead.
"""
class TrainingObserver(Observer):

    def __init__(self, start_time, start_params):

        self.prev_time = start_time
        self.iteration_times = []
        
        self.param_values = {
            name: [param.cpu().detach().numpy()]
            for name, param in start_params
        }


    def update(self, train_instance, *args, **kwargs):
        
        curr_time = time.time()
        self.iteration_times.append(curr_time - self.prev_time)
        self.prev_time = curr_time

        for name, param in train_instance.model.params.items():
            self.param_values[name].append(
                param.cpu().detach().numpy()
            )



"""
Descriptor-Decorator Approach
-------------------------------------------------------------------------------------------------------------------------------------------
Use a descriptor decorator to wrap the training loop method and track parameters and execution times like this.
Benefits: 
    No need to change TrainingProcedure class itself.
    Decorator can be turned on and off or replaced easily
    
Problems: 
    How to access these times, parameters afterwards?
Potential Solutions: 
    Have decorator take an observer?
    Use closure mechanism to create the Descriptor before decorating with it.
    Track epoch iterations and save parameters when all iterations complete
"""

class TrainingLoopDecorator:

    def __init__(self, method):

        self.method = method

        self.iteration_times = []
        self.model_params = []


    def __get__(self, instance, owner):
        
        self.prev_time = time.time()
        # Append starting parameters, i.e. theta_0
        self.model_params.append(
                {
                    name: param.cpu().detach().numpy()
                    for name, param in instance.model.params.items()
                }
        )
        
        def wrapper(*args, **kwargs):

            self.method(instance, *args, **kwargs)

            self.model_params.append(
                {
                    name: param.cpu().detach().numpy()
                    for name, param in instance.model.params.items()
                }
            )

            curr_time = time.time()
            self.iteration_times.append(curr_time - self.prev_time)
            self.prev_time = curr_time

            return None
        
        return wrapper
