


def main():

    import torch

    from hydra import initialize, compose
    from hydra.utils import instantiate
    from pathlib import Path
    from experiment import Experiment
    from training_observer import TimingObserver, ParameterObserver, LikelihoodObserver
    from result_manager import ResultManager
    from metrics import FrobeniusError, LpError, ParameterAssessor

    # check computation backend to use
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("-device:", device)

    ### Set Paths ###
    result_directory = Path('./Experiment_Results')
    experiment_name = 'GMM_RL_ML'
    experiment_dir = result_directory / experiment_name

    config_name = 'recovery_config.yaml'
    dataset_name = 'dataset.pt'
    start_batch_name = 'start_batch.pt'
    result_name = 'recovery_wo_Scheduler_lr2.csv'

    print(experiment_dir)
    ### Load from directory ###
    dataset = torch.load(experiment_dir.joinpath(dataset_name))
    start_batch = torch.load(experiment_dir.joinpath(start_batch_name))

    initialize(config_path= str(experiment_dir), version_base = None)
    cfg = compose(config_name = config_name)


    ### Experiment Setup ###
    hyper_parameters = instantiate(cfg.HyperParameters)
    model_parameters = instantiate(cfg.ModelParameters)
    sampling_parameters = instantiate(cfg.SamplingParameters)
    
    experiment = Experiment(
        dataset = dataset,
        sampler_start_batch = start_batch,
        model_parameters = model_parameters,
        sampling_parameters = sampling_parameters,
        hyper_parameters = hyper_parameters,
    )
    #print(dict(experiment.hyper_parameters))

    training_observers = [
        TimingObserver(),
        LikelihoodObserver(),
        ParameterObserver(),
    ]

    #experiment.build_components(observers = training_observers)

    exporter = ResultManager(
        file_name = result_name,
        file_folder_path = experiment_dir,
    )
    experiment.run(num_trials = 10, exporter = exporter, observers = training_observers)

    exporter.load_results_df()
    df = exporter.results_df.copy()
    #print(df.info())

    assessor = ParameterAssessor(
        target_params = model_parameters['target_params']
    )
    assessor.assign_metric('W', LpError(p = 2))
    assessor.assign_metric('mu_1', LpError(p = 2))
    assessor.assign_metric('Sigma_1', FrobeniusError())
    assessor.assign_metric('mu_2', LpError(p = 2))
    assessor.assign_metric('Sigma_2', FrobeniusError())

    update_df = assessor.apply_param_metrics_to_df(df)
    exporter.update_results(update_df)
    

    

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
    start_batch = torch.zeros(size = (200,2))

    epsilon = torch.tensor(1e-3, dtype = torch.float32)
    sampler = ULASampler(epsilon = epsilon, energy_model = model, start_batch = start_batch)
    #sampler = MALASampler(epsilon = epsilon, energy_model = model, start_batch = start_batch)
    #sampler = HMCSampler(epsilon = epsilon, L = 3, M = torch.eye(n = 2), energy_model = model, start_batch = start_batch)


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

    #print(mu_0, Sigma_0)



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
    model_batch_size = 200
    
    start_batch_size = 200
    ### Instantiate Sampler with initial Parameters ###
    start_batch = torch.zeros(size = (start_batch_size, 2))

    epsilon = torch.tensor(1e-1, dtype = torch.float32)
    sampler = ULASampler(epsilon = epsilon, energy_model = model, start_batch = start_batch)
    #sampler = MALASampler(epsilon = epsilon, energy_model = model, start_batch = start_batch)
    #sampler = HMCSampler(epsilon = epsilon, L = 3, M = torch.eye(n = 2), energy_model = model, start_batch = start_batch)

    if start_batch.shape[0] != batch_size:
            raise Exception("The batch dimension of the sampler starting batch doesn't coincide with the batch size.")

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
                batch_size = model_batch_size,
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

    