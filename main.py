


def main():

    import numpy as np
    import torch
    import torch.nn as nn

    from torch.optim import Adam
    from sklearn.model_selection import train_test_split
    from tqdm import trange, tqdm

    from test_models import MultivariateGaussianModel
    from torch_samplers import ULASampler, MALASampler, HMCSampler

    # check computation backend to use
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("-device:", device)

    

def gaussian_test():

    import torch

    from torch.distributions import MultivariateNormal
    from torch.utils.data import Dataset, DataLoader, random_split
    from torch.optim import Adam
    from torch.optim.lr_scheduler import ExponentialLR
    from tqdm import trange, tqdm

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
    dataset = data_mv_normal.sample_n(n = 10000)
    # Define the sizes of your training and validation sets
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size

    # Use random_split to create training and validation datasets
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])


    ### Instantiate Model with initial Parameters ###
    mu_0 = torch.tensor([2, 2], dtype = torch.float32)
    Sigma_0 = torch.tensor(
        [[2, 0],
         [0, 1],],
        dtype=torch.float32,
    )
    model = MultivariateGaussianModel(mu_0 = mu_0, Sigma_0 = Sigma_0)


    ### Instantiate Sampler with initial Parameters ###
    epsilon = torch.tensor(0.5, dtype = torch.float32)
    sampler = ULASampler(epsilon = epsilon)
    #sampler = MALASampler(epsilon = epsilon)
    #sampler = HMCSampler(epsilon = epsilon, L = 3, M = torch.eye(n = 2))


    ### Instantiate Standard Likelihood ###
    likelihood = Likelihood(energy_model = model, sampler = sampler)

    ### Training ###
    batch_size = 200
    train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle=True)

    optimizer = Adam(model.parameters(), lr=1e-1)
    scheduler = ExponentialLR(optimizer, gamma=0.9)
    
    x_0 = torch.tensor([0, 0], dtype = torch.float32)
    epochs = 10
    pbar   = tqdm(range(epochs))
    for it in pbar:
        for b, X_batch in enumerate(train_loader):
            
            # reset gradients 
            optimizer.zero_grad()
            
            '''
            model_samples = likelihood.gen_model_samples(
                x_0 = x_0,
                batch_size = 10*batch_size,
            )
            x_0 = model_samples[-1]
            #print(x_0)
            '''
            model_mv_normal = MultivariateNormal(
                loc = model.params['mu'], 
                covariance_matrix = model.params['Sigma'],
            )
            model_samples = model_mv_normal.sample(sample_shape= (2*batch_size, ))
            
            likelihood.gradient(data_samples = X_batch, model_samples = model_samples[batch_size:])
            
            print(f"{it}_{b+1}/{epochs} Parameters:")
            for param_name, value in model.params.items():
                print(f'{param_name}:\n {value.data}')
            
            # perform gradient descent step along model.theta.grad
            optimizer.step()
        
        scheduler.step()






if __name__=="__main__":

    #main()
    gaussian_test()
    