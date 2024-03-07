
import torch
import torch.linalg as tla



def vector_p_norm(target_vector: torch.tensor, model_vector: torch.tensor, p):

    difference_vector = target_vector - model_vector

    return tla.norm(difference_vector, ord = p)



def frobenius_norm(target_matrix: torch.tensor, model_matrix: torch.tensor):

    difference_matrix = target_matrix - model_matrix

    return tla.norm(difference_matrix, ord = 'fro')