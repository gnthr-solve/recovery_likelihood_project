
import torch
import torch.linalg as tla
import pandas as pd

from abc import ABC, abstractmethod

from helper_tools import param_record_to_torch


"""
Parameter Error Metrics
-------------------------------------------------------------------------------------------------------------------------------------------
"""
class ParameterMetric(ABC):

    def __init__(self):
        self.name = None

    @abstractmethod
    def __call__(self, *args, **kwargs):
        pass


class FrobeniusError(ParameterMetric):

    def __init__(self):
        self.name = 'frob_error'

    def __call__(self, target_matrix: torch.Tensor, model_matrix_batch: torch.Tensor):
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
        self.name = f'L{p}_error'

    def __call__(self, target_vector: torch.Tensor, model_vector_batch: torch.Tensor):

        difference_vector = target_vector.unsqueeze(0) - model_vector_batch

        return tla.norm(difference_vector, ord = self.p, dim = 1)



"""
Apply Metrics
-------------------------------------------------------------------------------------------------------------------------------------------
"""
class ParameterAssessor:

    def __init__(self, target_params: dict[str, torch.Tensor]):

        self.target_params = {param_name: param.clone() for param_name, param in target_params.items()}
        self.param_metrics = {}


    def assign_metric(self, param_name: str, metric: ParameterMetric):
        
        if param_name in self.target_params.keys():

            self.param_metrics[param_name] = metric


    def apply_param_metrics_to_df(self, df: pd.DataFrame):

        df = df.copy()
        for param_name, metric in self.param_metrics.items():

            target_param = self.target_params[param_name]

            for training_run_id in df['training_run_id'].unique():

                id_mask = df['training_run_id'] == training_run_id

                df_subset = df[id_mask]
                param_column = df_subset[param_name]
                param_estimates = param_record_to_torch(param_column)

                metric_evals = metric(target_param, param_estimates)

                df.loc[id_mask, f'{param_name}_{metric.name}'] = metric_evals.tolist()

        return df






"""
Apply Metrics
-------------------------------------------------------------------------------------------------------------------------------------------
"""
def apply_param_metric_to_df(df: pd.DataFrame, target_param: torch.Tensor, param_name: str, metric: ParameterMetric):

    df = df.copy()
    for training_run_id in df['training_run_id'].unique():

        id_mask = df['training_run_id'] == training_run_id

        df_subset = df[id_mask]
        param_column = df_subset[param_name]
        param_estimates = param_record_to_torch(param_column)

        frob_norms = metric(target_param, param_estimates)

        df.loc[id_mask, f'{param_name}_{metric.name}'] = frob_norms.tolist()

    return df


"""
Simple Metric Functions
-------------------------------------------------------------------------------------------------------------------------------------------
"""
def vector_p_norm(target_vector: torch.Tensor, model_vector: torch.Tensor, p):

    difference_vector = target_vector - model_vector

    return tla.norm(difference_vector, ord = p)



def frobenius_norm(target_matrix: torch.Tensor, model_matrix: torch.Tensor):

    difference_matrix = target_matrix - model_matrix

    return tla.norm(difference_matrix, ord = 'fro')



def batch_frobenius_norm(target_matrix: torch.Tensor, model_matrix_batch: torch.Tensor):
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
    #return difference_matrices
    #print(difference_matrices.dim())

    # Compute the Frobenius norm along the appropriate dimension
    frobenius_norms = torch.norm(difference_matrices, p='fro', dim=(1, 2))
    return frobenius_norms

    return frobenius_norms.reshape(-1, 1)  # Reshape to (n, 1)?
