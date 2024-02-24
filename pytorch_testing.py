
import torch
import torch.nn as nn
import torch.linalg as tla

from torch.distributions import MultivariateNormal
from itertools import product



"""
Test Functions
-------------------------------------------------------------------------------------------------------------------------------------------
"""
def properties_test():
    
    x = torch.tensor([2, 3, 4], dtype=torch.float32)
    y = torch.tensor(
        [[2, 3, 4],
         [1, 9, 7],
         [0, 8, 3],
         [2, 1, 1]],
        dtype=torch.float32,
    )

    print(x.shape)
    #print(x.size())
    print(x.shape[0])
    print(x.dim())

    z = torch.atleast_2d(x)
    print(z.shape)
    print(z.shape[0])
    print(z.dim())

    print(y.shape)
    #print(y.size())
    print(y.shape[0])


def squeeze_test():
    
    x = torch.tensor([2, 3, 4], dtype=torch.float32)
    y = torch.tensor(
        [[2, 3, 4],
         [1, 9, 7],
         [0, 8, 3],
         [2, 1, 1]], 
        dtype=torch.float32,
    )
    
    print('For', x)
    print(x.unsqueeze(0))
    print(x.unsqueeze(0).shape[0])
    print(x.unsqueeze(1))
    print(x.unsqueeze(1).shape)

    print('For', y)
    print(y.unsqueeze(0))
    print(y.unsqueeze(0).shape)
    print(y.unsqueeze(1))
    print(y.unsqueeze(1).shape)
    print(y.squeeze().shape)


def matrix_multiplication_test():
    
    #constants meant to be multiplied
    mu = torch.tensor([2, 2, 2], dtype=torch.float32) #shape should be (3,)
    Sigma = torch.tensor(
        [[2, 0, 1],
         [0, 1, 0],
         [1, 0, 3]],
        dtype=torch.float32,
    ) # shape should be (3,3)

    #input, either as a single vector or dataset
    x = torch.tensor([2, 3, 4], dtype=torch.float32)
    x_batch = torch.tensor(
        [[2, 3, 4],
         [1, 9, 7],
         [0, 8, 3],
         [2, 1, 1],
         [1, 1, 1],],
        dtype=torch.float32,
    )
    print("Shape of x: \t", x.shape)
    #print("Shape of x_batch: \t", x_batch.shape)
    
    #diff = x - mu
    #diff_batch = x_batch - mu
    #print(diff.shape)            # (3,) - (3,) -> (3,)
    #print(diff_batch.shape)      # (5, 3) - (3,) -> (5,3)
    
    mu_2d = mu.unsqueeze(dim = 0) #shape = (1, 3)
    diff_batch = x - mu_2d
    diff_batch = x_batch - mu_2d
    print(diff_batch.shape)             # (3,) - (1, 3) -> (1, 3)
    print(diff_batch)                   # (5, 3) - (1, 3) -> (5, 3)

    # (n, d) x (d, d) x (d, n) = (n, n)
    # that matrix would have all possible combinations x_i Sigma x_j, but we just want x_i Sigma x_i i.e. the diagonal
    intermediate = torch.matmul(diff_batch, Sigma)
    print(intermediate)
    print(intermediate.shape)
    intermediate_2 = torch.matmul(intermediate, diff_batch.T)
    print(intermediate_2)
    print(intermediate_2.shape)

    result_expensive = torch.diag(intermediate_2)
    print(result_expensive)

    result_direct = tla.vecdot(intermediate, diff_batch)
    print(result_direct)



def torch_mv_normal_test():
    
    mu = torch.tensor([2, 2, 2], dtype=torch.float32)
    Sigma = torch.tensor(
        [[2, 0, 1],
         [0, 1, 0],
         [1, 0, 3]],
        dtype=torch.float32,
    )

    mv_normal = MultivariateNormal(loc = mu, covariance_matrix = Sigma)

    x_0_batch = torch.tensor(
        [[2, 3, 4],
         [1, 9, 7],
         [0, 8, 3],
         [2, 1, 1],
         [1, 1, 1],],
        dtype=torch.float32,
    )
    shape = x_0_batch.shape
    print("Shape of x_0_batch: \t", shape)
    
    sample_dim = shape[-1]
    print("x_0 dimension: \t", sample_dim)

    add_dim = 10
    sample_shape = (add_dim,) + shape[:-1]
    print(sample_shape)

    samples = mv_normal.sample(sample_shape = sample_shape)
    print(samples[:2])
    print(samples.reshape(shape = (add_dim*shape[0], sample_dim)))

    sample_iterator = iter(samples)
    #print(next(sample_iterator))



def slicing_assignment_test():

    torch.manual_seed(123)
    mu = torch.tensor([2, 2, 2], dtype=torch.float32)
    Sigma = torch.tensor(
        [[2, 0, 1],
         [0, 1, 0],
         [1, 0, 3]],
        dtype=torch.float32,
    )

    mv_normal = MultivariateNormal(loc = mu, covariance_matrix = Sigma)

    x_0_batch = torch.tensor(
        [[2, 3, 4],
         [1, 9, 7],
         [0, 8, 3],
         [2, 1, 1],],
        dtype=torch.float32,
    )
    shape = x_0_batch.shape
    print("Shape of x_0_batch: \t", shape)

    add_dim = 2
    sample_shape = (add_dim,) + shape[:-1]
    empty_shape = (add_dim,) + shape
    #print(sample_shape)

    empty_tensor = torch.empty(size = empty_shape, dtype=torch.float32)
    samples = mv_normal.sample(sample_shape = sample_shape)
    print(samples[0])
    empty_tensor[0] = samples[0]
    print(empty_tensor)
    empty_tensor[0, :, :] = samples[0]
    print(empty_tensor)



def operations_broadcasting_test():
    
    torch.manual_seed(123)
    mu = torch.tensor([2, 2, 2], dtype=torch.float32)
    Sigma = torch.tensor(
        [[2, 0, 1],
         [0, 1, 0],
         [1, 0, 3]],
        dtype=torch.float32,
    )

    mv_normal = MultivariateNormal(loc = mu, covariance_matrix = Sigma)

    x_0_batch = torch.tensor(
        [[2, 3, 4],
         [1, 9, 7],
         [0, 8, 3],
         [2, 1, 1],],
        dtype=torch.float32,
    )
    shape = x_0_batch.shape
    #print("Shape of x_0_batch: \t", shape)

    add_dim = 2
    sample_shape = (add_dim,) + shape[:-1]
    #print(sample_shape)

    samples = mv_normal.sample(sample_shape = sample_shape)
    print(samples)
    transformed = 2* samples[0]
    print(transformed)
    transformed = transformed + samples[1]
    print(transformed)



def squared_norm_test():
    
    x_batch = torch.tensor(
        [[2, 3, 4],
         [1, 9, 7],
         [0, 8, 3],
         [2, 1, 1],],
        dtype=torch.float32,
    )
    
    squared_norms_sum = torch.sum(x_batch ** 2, dim=1)
    print(squared_norms_sum)
    squared_norms = tla.norm(x_batch, dim=1) ** 2
    print(squared_norms)



def torch_where_test():
    
    x_batch = torch.tensor(
        [[2, 3, 4],
         [1, 9, 7],
         [0, 8, 3],
         [2, 1, 1],],
        dtype=torch.float32,
    )

    squared_norms_by_sum = torch.sum((x_batch)**2, dim=1)
    print(squared_norms_by_sum)

    squared_norms = tla.norm(x_batch, dim=1) ** 2
    print(squared_norms)

    squared_norms_unsqueezed = squared_norms.unsqueeze(1)
    print(squared_norms_unsqueezed)
    
    bounded_norm_x_batch = torch.where(squared_norms_unsqueezed < 50, x_batch, torch.zeros_like(x_batch))
    print(bounded_norm_x_batch)






if __name__=="__main__":

    #properties_test()
    #squeeze_test()
    #matrix_multiplication_test()
    #torch_mv_normal_test()
    #slicing_assignment_test()
    #operations_broadcasting_test()
    #squared_norm_test()
    torch_where_test()