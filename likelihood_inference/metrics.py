
import torch
import torch.linalg as tla
import pandas as pd

from abc import ABC, abstractmethod

from .helper_tools import param_record_to_torch


"""
Parameter Error Metrics
-------------------------------------------------------------------------------------------------------------------------------------------
Classes for calculating error metrics for the deviation between estimated parameter and target parameters for a model.

The __call__method expects the true parameter of the target distribution and a batch of estimates.
These classes are used by the ParameterAssessor to create the error columns in the csv files the Experiment class produces.
"""

class ParameterMetric(ABC):

    def __init__(self):
        self.name = None

    @abstractmethod
    def __call__(self, *args, **kwargs) -> torch.Tensor:
        pass



class FrobeniusError(ParameterMetric):

    def __init__(self):
        self.name = 'Frob. Error'

    def __call__(self, target_matrix: torch.Tensor, model_matrix_batch: torch.Tensor) -> torch.Tensor:
        """
        Compute the Frobenius norm between a target matrix and a batch of model matrices.

        Args:
            target_matrix (torch.Tensor): The target matrix.
            model_matrix_batch (torch.Tensor): A batch of model matrices.

        Returns:
            torch.Tensor: Frobenius norm for each matrix in the batch.
        """

        # Compute the difference matrix for each model matrix in the batch
        difference_matrices = target_matrix.unsqueeze(0) - model_matrix_batch

        # Compute the Frobenius norm along the appropriate dimension
        frobenius_norms = torch.norm(difference_matrices, p='fro', dim=(1, 2))
        return frobenius_norms



class LpError(ParameterMetric):
    def __init__(self, p):
        self.p = p
        self.name = f'L{p}-Error'

    def __call__(self, target_vector: torch.Tensor, model_vector_batch: torch.Tensor) -> torch.Tensor:

        difference_vector = target_vector.unsqueeze(0) - model_vector_batch

        return tla.norm(difference_vector, ord = self.p, dim = 1)



class SimplexLpError(ParameterMetric):
    def __init__(self, p):
        self.p = p
        self.name = f'L{p}-Error'

    def __call__(self, target_vector: torch.Tensor, model_vector_batch: torch.Tensor) -> torch.Tensor:

        difference_vector = target_vector.unsqueeze(0) - model_vector_batch.softmax(dim = -1)

        return tla.norm(difference_vector, ord = self.p, dim = 1)
    

"""
ParameterAssessor
-------------------------------------------------------------------------------------------------------------------------------------------
Uses ParameterMetric's to update a result dataframe with assigned error metrics.
It is instantiated with the models target parameters, to which metrics can be assigned in assign_metric.
apply_param_metrics_to_df then takes a dataframe, calculates the assigned parameter metrics and adds them to the dataframe as columns.
"""
class ParameterAssessor:

    def __init__(self, target_params: dict[str, torch.Tensor]):

        self.target_params = {param_name: param.clone() for param_name, param in target_params.items()}
        self.param_metrics = {}


    def assign_metric(self, param_name: str, metric: ParameterMetric):
        
        if param_name in self.target_params.keys():

            self.param_metrics[param_name] = metric


    def apply_param_metrics_to_df(self, df: pd.DataFrame) -> pd.DataFrame:

        df = df.copy()
        for param_name, metric in self.param_metrics.items():

            target_param = self.target_params[param_name]
            metric_col_name = f'{param_name} {metric.name}'

            if metric_col_name in df.columns:
                df_slice = df.loc[df[metric_col_name].isna()]
            else:
                df_slice = df

            for training_run_id in df_slice['training_run_id'].unique():

                id_mask = df['training_run_id'] == training_run_id

                df_subset = df[id_mask]
                param_column = df_subset[param_name]
                param_estimates = param_record_to_torch(param_column)

                metric_evals = metric(target_param, param_estimates)

                df.loc[id_mask, metric_col_name] = metric_evals.tolist()

        return df


