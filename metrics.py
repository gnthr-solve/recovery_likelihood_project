
import torch
import torch.linalg as tla

import pandas as pd
from helper_tools import param_record_to_torch

"""
Apply Metrics
-------------------------------------------------------------------------------------------------------------------------------------------
"""
def apply_param_metric_to_df(df: pd.DataFrame, target_param: torch.Tensor, param_name: str, metric):

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
Parameter Error Metrics
-------------------------------------------------------------------------------------------------------------------------------------------
"""
class FrobeniusError:
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


class LpError:
    def __init__(self, p):
        self.p = p
        self.name = f'L{p}_error'

    def __call__(self, target_vector: torch.Tensor, model_vector_batch: torch.Tensor):

        difference_vector = target_vector.unsqueeze(0) - model_vector_batch

        return tla.norm(difference_vector, ord = self.p, dim = 1)

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
