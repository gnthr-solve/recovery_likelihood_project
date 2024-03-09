
import torch
import torch.linalg as tla



def vector_p_norm(target_vector: torch.Tensor, model_vector: torch.Tensor, p):

    difference_vector = target_vector - model_vector

    return tla.norm(difference_vector, ord = p)



def frobenius_norm(target_matrix: torch.Tensor, model_matrix: torch.Tensor):

    difference_matrix = target_matrix - model_matrix

    return tla.norm(difference_matrix, ord = 'fro')