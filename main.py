


def main():

    import torch

    from torch.distributions import MultivariateNormal
    from hydra import initialize, compose
    from hydra.utils import instantiate

    from experiment import Experiment
    from training_observer import TimingObserver, ParameterObserver, LikelihoodObserver
    from exporter import ResultExporter

    # check computation backend to use
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("-device:", device)

    ### Create Data ###
    data_mv_normal = MultivariateNormal(
        loc = torch.tensor([3, 3], dtype = torch.float32), 
        covariance_matrix = 2* torch.diag(torch.ones(size = (2,), dtype=torch.float32)),
    )
    dataset = data_mv_normal.sample(sample_shape = (10000,))

    initialize(config_path= '.')
    cfg = compose(config_name="config.yaml")

    experiment = Experiment(
        dataset = dataset,
        model_parameters = instantiate(cfg.ModelParameters),
        sampling_parameters = instantiate(cfg.SamplingParameters),
        likelihood_parameters = instantiate(cfg.LikelihoodParameters),
        hyper_parameters = instantiate(cfg.HyperParameters),
    )
    #print(dict(experiment.hyper_parameters))

    training_observers = [
        TimingObserver(),
        LikelihoodObserver(),
        ParameterObserver(),
    ]

    experiment.build_components(observers = training_observers)

    experiment.run(num_trials = 1)


    

def gaussian_test_ML():

    import torch

    from torch.distributions import MultivariateNormal
    from torch.utils.data import Dataset, DataLoader, random_split
    from torch.optim import Adam
    from torch.optim.lr_scheduler import ExponentialLR
    from tqdm import trange, tqdm

    from test_models import MultivariateGaussianModel
    from mc_samplers import ULASampler, MALASampler, HMCSampler
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
    x_0_batch = torch.zeros(size = (200,2))

    epsilon = torch.tensor(1e-3, dtype = torch.float32)
    sampler = ULASampler(epsilon = epsilon, energy_model = model, x_0_batch = x_0_batch)
    #sampler = MALASampler(epsilon = epsilon, energy_model = model, x_0_batch = x_0_batch)
    #sampler = HMCSampler(epsilon = epsilon, L = 3, M = torch.eye(n = 2), energy_model = model, x_0_batch = x_0_batch)


    ### Instantiate Standard Likelihood ###
    likelihood = Likelihood(energy_model = model, sampler = sampler)

    ### Training ###
    batch_size = 200
    train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle=True)

    optimizer = Adam(model.parameters(), lr=1e-1)
    scheduler = ExponentialLR(optimizer, gamma=0.9)
    
    epochs = 10
    pbar   = tqdm(range(epochs))

    for it in pbar:
        for b, X_batch in enumerate(train_loader):
            
            # reset gradients 
            optimizer.zero_grad()
            
            
            model_samples = likelihood.gen_model_samples(
                batch_size = batch_size,
                burnin_offset = 2*batch_size,
            )
            #print(model_samples[:10])
            #print(X_batch[:10])
            
            '''
            model_mv_normal = MultivariateNormal(
                loc = model.params['mu'], 
                covariance_matrix = model.params['Sigma'],
            )
            model_samples = model_mv_normal.sample(sample_shape= (2*batch_size, ))
            '''

            likelihood.gradient(data_samples = X_batch, model_samples = model_samples)
            
            # perform gradient descent step along model.theta.grad
            optimizer.step()

            print(f"{it}_{b+1}/{epochs} Parameters:")
            for param_name, value in model.params.items():
                print(f'{param_name}:\n {value.data}')
                if torch.isnan(value.data).any():
                    raise StopIteration
            
            
        
        scheduler.step()




def gaussian_test_RL():

    import torch

    from torch.distributions import MultivariateNormal
    from torch.utils.data import Dataset, DataLoader, random_split
    from torch.optim import Adam
    from torch.optim.lr_scheduler import ExponentialLR
    from tqdm import trange, tqdm

    from test_models import MultivariateGaussianModel
    from mc_samplers import ULASampler, MALASampler, HMCSampler
    from recovery_adapter import RecoveryAdapter
    from likelihood import RecoveryLikelihood

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
    org_model = MultivariateGaussianModel(mu_0 = mu_0, Sigma_0 = Sigma_0)

    perturbation_var = torch.tensor(1, dtype = torch.float32)
    model = RecoveryAdapter(energy_model = org_model, perturbation_var = perturbation_var)

    batch_size = 200
    ### Instantiate Sampler with initial Parameters ###
    x_0_batch = torch.zeros(size = (batch_size, 2))

    epsilon = torch.tensor(1e-1, dtype = torch.float32)
    sampler = ULASampler(epsilon = epsilon, energy_model = model, x_0_batch = x_0_batch)
    #sampler = MALASampler(epsilon = epsilon, energy_model = model, x_0_batch = x_0_batch)
    #sampler = HMCSampler(epsilon = epsilon, L = 3, M = torch.eye(n = 2), energy_model = model, x_0_batch = x_0_batch)


    ### Instantiate Standard Likelihood ###
    likelihood = RecoveryLikelihood(adapted_model = model, sampler = sampler)

    ### Training ###
    
    train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle=True)

    optimizer = Adam(model.parameters(), lr=1e-1)
    scheduler = ExponentialLR(optimizer, gamma=0.9)
    
    epochs = 10
    pbar   = tqdm(range(epochs))

    for it in pbar:
        for b, X_batch in enumerate(train_loader):
            
            # reset gradients 
            optimizer.zero_grad()
            
            
            model_samples = likelihood.gen_model_samples(
                data_samples = X_batch,
                burnin_offset = int(batch_size/4),
            )
            #print(model_samples[:10])
            #print(X_batch[:10])
            '''
            model_mv_normal = MultivariateNormal(
                loc = model.params['mu'], 
                covariance_matrix = model.params['Sigma'],
            )
            model_samples = model_mv_normal.sample(sample_shape= (2*batch_size, ))
            '''

            likelihood.gradient(data_samples = X_batch, model_samples = model_samples)
            
            print(f"{it}_{b+1}/{epochs} Parameters:")
            for param_name, value in model.params.items():
                print(f'{param_name}:\n {value.data}')
                if torch.isnan(value.data).any():
                    raise StopIteration
            
            # perform gradient descent step along model.theta.grad
            optimizer.step()
        
        scheduler.step()





if __name__=="__main__":

    main()
    #gaussian_test_ML()
    #gaussian_test_RL()

    