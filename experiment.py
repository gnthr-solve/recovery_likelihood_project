
import torch
import json
import hashlib

from ebm import EnergyModel
from mc_samplers import EnergySampler
from recovery_adapter import RecoveryAdapter
from likelihood import RecoveryLikelihood
from experiment_params import ModelParameters, SamplingParameters, HyperParameters
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

        model = model_type(**start_params)
        if model_parameters['requires_adapter']:

            perturbation_var = model_parameters['perturbation_var']

            model = RecoveryAdapter(energy_model = model, perturbation_var = perturbation_var)

        return model

    
    def setup_sampler(self, model: EnergyModel, start_batch: torch.Tensor, sampling_parameters: SamplingParameters):

        sampler_type = retrieve_class('mc_samplers', sampling_parameters['sampler_class'])
        
        return sampler_type(energy_model = model, start_batch = start_batch, **sampling_parameters)
    

    def setup_likelihood(self, model: EnergyModel, sampler: EnergySampler, hyper_parameters: HyperParameters):
        
        likelihood_type = retrieve_class('likelihood', hyper_parameters['likelihood_class'])

        return likelihood_type(model, sampler)
    

    def setup_train_components(self, model: EnergyModel, hyper_parameters: HyperParameters):

        optimizer_type = retrieve_class('torch.optim', hyper_parameters['optimizer_class'])
        optimizer = optimizer_type(model.parameters(), **hyper_parameters['optimizer_params'])

        scheduler_class = hyper_parameters['scheduler_class']
        if scheduler_class:
            scheduler_type = retrieve_class('torch.optim.lr_scheduler', scheduler_class)
            scheduler = scheduler_type(optimizer, **hyper_parameters['scheduler_params'])
        else:
            scheduler = None

        return optimizer, scheduler
    


"""
Experiment Idea TODO
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
"""
class Experiment:
    def __init__(
            self,
            dataset: torch.Tensor,
            sampler_start_batch: torch.Tensor,
            model_parameters: ModelParameters,
            sampling_parameters: SamplingParameters,
            hyper_parameters: HyperParameters
        ):

        self.builder = ExperimentBuilder()
        self.dataset = dataset
        self.sampler_start_batch = sampler_start_batch
        self.hyper_parameters = hyper_parameters
        self.model_parameters = model_parameters
        self.sampling_parameters = sampling_parameters

        if hyper_parameters['likelihood_class'] == 'RecoveryLikelihood' and sampler_start_batch.shape[0] != hyper_parameters['batch_size']:
            raise Exception("The batch dimension of the sampler starting batch must coincide with the batch size for RecoveryLikelihood.")


    def build_components(self, observers: list[Observer]):

        builder = self.builder

        model = builder.setup_model(self.model_parameters)
        sampler = builder.setup_sampler(model, self.sampler_start_batch, self.sampling_parameters)
        likelihood = builder.setup_likelihood(model, sampler, self.hyper_parameters)

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


    def run(self, num_trials, exporter, observers: list[Observer]):
        
        for i in range(num_trials):

            self.build_components(observers = observers)
            self.training_procedure()

            observation_dfs = [
                observer.return_observations()
                for observer in self.training_procedure.observers
            ]

            exporter.export_observations(training_run_id = f'run_{i+1}', observation_dfs = observation_dfs)
