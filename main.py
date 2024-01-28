


def main():

    import numpy as np
    import torch
    import torch.nn as nn

    from torch.optim import Adam
    from sklearn.model_selection import train_test_split
    #from tqdm import tqdm_notebook as tqdm

    from test_models import MultivariateGaussianModel
    from torch_samplers import ULASampler, MALASampler, HMCSampler

    # check computation backend to use
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("-device:", device)

    

def gaussian_test():

    import torch

    from torch.distributions import MultivariateNormal

    from test_models import MultivariateGaussianModel
    from torch_samplers import ULASampler, MALASampler, HMCSampler
    from likelihood import Likelihood

    # check computation backend to use
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("-device:", device)

    ### Create Data ###
    data_mv_normal = MultivariateNormal(
        loc = torch.tensor([3, 3], dtype = torch.float32), 
        covariance_matrix = 2* torch.diag(torch.ones(size = (2,), dtype=torch.float32)),
    )
    data_samples = data_mv_normal.sample_n(n = 100)
    
    ### Instantiate Model with initial Parameters ###
    mu_0 = torch.tensor([2, 2], dtype = torch.float32)
    Sigma_0 = torch.tensor(
        [[2, 0],
         [0, 1],],
        dtype=torch.float32,
    )
    model = MultivariateGaussianModel(mu_0 = mu_0, Sigma_0 = Sigma_0)

    ### Instantiate Sampler with initial Parameters ###
    sampler = ULASampler(epsilon = 0.2)
    #sampler = MALASampler(epsilon = 0.2)
    #sampler = HMCSampler(epsilon = 0.2, L = 3, M = torch.eye(n = 2))

    ### Instantiate Standard Likelihood ###
    likelihood = Likelihood(data_samples = data_samples, energy_model = model, sampler = sampler)

    model_samples = likelihood.gen_model_samples(
        x_0 = torch.tensor([0, 0], dtype = torch.float32),
        batch_size = 100,
    )
    print(model_samples)







if __name__=="__main__":

    #main()
    gaussian_test()
    