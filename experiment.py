
import torch
import json
import hashlib

from ebm import EnergyModel
from mc_samplers import EnergySampler
from recovery_adapter import RecoveryAdapter
from likelihood import RecoveryLikelihood
from experiment_params import ModelParameters, SamplingParameters, LikelihoodParameters, HyperParameters
from helper_tools import retrieve_class
from training_procedure import TrainingProcedure
from training_observer import Observer

"""
ExperimentBuilder Idea TODO
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
"""
class ExperimentBuilder:

    def setup_model(self, model_parameters: ModelParameters):

        model_type = retrieve_class('test_models', model_parameters['model_class'])

        start_params = model_parameters['start_params']

        return model_type(**start_params)

    
    def setup_sampler(self, model: EnergyModel, sampling_parameters: SamplingParameters):

        sampler_type = retrieve_class('mc_samplers', sampling_parameters['sampler_class'])
        
        return sampler_type(energy_model = model, **sampling_parameters)
    

    def setup_likelihood(self, model: EnergyModel, sampler: EnergySampler, likelihood_parameters: LikelihoodParameters):
        
        likelihood_type = retrieve_class('likelihood', likelihood_parameters['likelihood_class'])

        if likelihood_type == RecoveryLikelihood:

            perturbation_var = likelihood_parameters['perturbation_var']

            model = RecoveryAdapter(energy_model = model, perturbation_var = perturbation_var)

        return model, likelihood_type(model, sampler)
    

    def setup_train_components(self, model: EnergyModel, hyper_parameters: HyperParameters):

        optimizer_type = retrieve_class('torch.optim', hyper_parameters['optimizer_class'])
        scheduler_type = retrieve_class('torch.optim.lr_scheduler', hyper_parameters['scheduler_class'])

        optimizer = optimizer_type(model.parameters(), **hyper_parameters['optimizer_params'])
        scheduler = scheduler_type(optimizer, **hyper_parameters['scheduler_params'])

        return optimizer, scheduler
    


"""
Experiment Idea TODO
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
"""
class Experiment:
    def __init__(
            self,
            dataset: torch.Tensor,
            model_parameters: ModelParameters,
            sampling_parameters: SamplingParameters,
            likelihood_parameters: LikelihoodParameters,
            hyper_parameters: HyperParameters
        ):

        self.builder = ExperimentBuilder()
        self.dataset = dataset
        self.hyper_parameters = hyper_parameters
        self.model_parameters = model_parameters
        self.sampling_parameters = sampling_parameters
        self.likelihood_parameters = likelihood_parameters


    def build_components(self, observers: list[Observer]):

        builder = self.builder

        model = builder.setup_model(self.model_parameters)
        sampler = builder.setup_sampler(model, self.sampling_parameters)
        model, likelihood = builder.setup_likelihood(model, sampler, self.likelihood_parameters)

        optimizer, scheduler = builder.setup_train_components(model, self.hyper_parameters)

        self.training_procedure = TrainingProcedure(
            dataset = self.dataset,
            model = model,
            likelihood = likelihood,
            optimizer = optimizer,
            scheduler = scheduler,
            **self.hyper_parameters
        )

        self.training_procedure.register_observers(observers=observers)


    def run(self, num_trials):

        for _ in range(num_trials):

            self.training_procedure()

