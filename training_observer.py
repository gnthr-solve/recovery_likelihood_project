
import torch
import time
import pandas as pd

from abc import ABC, abstractmethod

from recovery_adapter import RecoveryAdapter

"""
Subject Role Interface
-------------------------------------------------------------------------------------------------------------------------------------------
"""
class Subject:
    def __init__(self):
        self.observers = []

    def register_observers(self, observers):
        self.observers.extend(observers)

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

    @abstractmethod
    def return_observations(self):
        pass

"""
TrainingObserver Class Approach following Observer Pattern
-------------------------------------------------------------------------------------------------------------------------------------------
"""
class ParameterObserver(Observer):

    def __init__(self):

        self.param_values = []


    def update(self, train_instance, *args, **kwargs):
        
        param_dict = {
            name: param.cpu().detach().clone().tolist()
            for name, param in train_instance.model.params.items()
        }

        self.param_values.append(param_dict)


    def return_observations(self, as_df: bool = True):

        param_values = self.param_values

        #reset
        self.param_values = []
        if as_df:
            return pd.DataFrame(param_values)
        else:
            return param_values



class TimingObserver(Observer):

    def __init__(self):

        self.iteration_times = []


    def update(self, train_instance, *args, **kwargs):
        
        self.iteration_times.append(time.time())


    def return_observations(self, as_df: bool = True):

        iteration_times = [
            time_stamp - self.iteration_times[0]
            for time_stamp in self.iteration_times
        ]

        #reset
        self.iteration_times = []

        if as_df:
            return pd.DataFrame({'Iteration Timestamp': iteration_times})
        else:
            return iteration_times 



class LikelihoodObserver(Observer):

    def __init__(self):

        self.likelihood_values = []


    def update(self, train_instance, *args, **kwargs):

        dataset = train_instance.dataset

        if type(train_instance.model) == RecoveryAdapter:
    
            train_instance.model.set_perturbed_samples(dataset)

        likelihood_value = train_instance.likelihood.unnormalised_log_likelihood(dataset)

        self.likelihood_values.append(
            likelihood_value.cpu().detach()
        )
    

    def return_observations(self, as_df: bool = True):
        
        likelihood_values = torch.atleast_1d(self.likelihood_values)
        likelihood_values = torch.cat(likelihood_values).numpy()

        #reset
        self.likelihood_values = []

        if as_df:
            return pd.DataFrame({'Unnorm. Likelihood Values': likelihood_values})
        else:
            return likelihood_values 





"""
Descriptor-Decorator Approach
-------------------------------------------------------------------------------------------------------------------------------------------
"""
class ObservationDescriptor:

    def __init__(self, method):

        self.method = method


    def __get__(self, instance: Subject, owner):
        
        if instance.observers != []:
        
            def wrapper(*args, **kwargs):

                self.method(instance, *args, **kwargs)

                instance.notify_observers()

                return None
        
        else:

            wrapper = self.method
            
        return wrapper
