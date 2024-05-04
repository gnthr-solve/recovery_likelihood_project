


def main():
    """
    Main experiment routine. 
    Instantiates the involved components from a given config, 
    conducts a specified number of training procedure runs (num_trials) 
    and stores the observed results in the specified result csv.
    """
    import torch

    from hydra import initialize, compose
    from hydra.utils import instantiate
    from pathlib import Path
    from experiment import Experiment
    from training_observer import TimingObserver, ParameterObserver, LikelihoodObserver
    from result_manager import ResultManager
    from metrics import FrobeniusError, LpError, SimplexLpError, ParameterAssessor
    from timing_decorators import timing_decorator

    # check computation backend to use
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("-device:", device)

    ### Set Paths ###
    result_directory = Path('./Experiment_Results')
    experiment_name = 'COS_RL_ML'
    experiment_dir = result_directory / experiment_name

    config_name = 'recovery_config.yaml'
    #config_name = 'marginal_config.yaml'

    dataset_name = 'dataset.pt'
    #start_batch_name = 'start_batch.pt'
    #start_batch_name = 'zeros_start_batch.pt'
    start_batch_name = 'normal_start_batch.pt'

    result_name = 'results_RL_MALA.csv'
    #result_name = 'results_ML_MALA.csv'

    print(experiment_dir)
    print(config_name)
    ### Load from directory ###
    dataset = torch.load(experiment_dir.joinpath(dataset_name))
    start_batch = torch.load(experiment_dir.joinpath(start_batch_name))

    initialize(config_path= str(experiment_dir), version_base = None)
    cfg = compose(config_name = config_name)


    ### Experiment Setup ###
    hyper_parameters = instantiate(cfg.HyperParameters)
    model_parameters = instantiate(cfg.ModelParameters)
    sampling_parameters = instantiate(cfg.SamplingParameters)
    
    ### ATTENTION: ONLY UNIVARIATE ###
    dataset = dataset.unsqueeze(-1)

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

    exporter = ResultManager(
        file_name = result_name,
        file_folder_path = experiment_dir,
    )

    experiment.run(num_trials = 100, exporter = exporter, observers = training_observers)

    exporter.save_results()

    ### Update with Metrics ###
    exporter.load_results_df()
    df = exporter.results_df.copy()
    #print(df.info())

    assessor = ParameterAssessor(
        target_params = model_parameters['target_params']
    )
    
    assessor.assign_metric('W', LpError(p = 2))
    assessor.assign_metric('mu', LpError(p = 2))
    '''
    assessor.assign_metric('W', SimplexLpError(p = 2))
    assessor.assign_metric('mu_1', LpError(p = 2))
    assessor.assign_metric('Sigma_1', FrobeniusError())
    assessor.assign_metric('mu_2', LpError(p = 2))
    assessor.assign_metric('Sigma_2', FrobeniusError())
    
    
    assessor.assign_metric('mu', LpError(p = 2))
    assessor.assign_metric('Sigma', FrobeniusError())
    '''
    

    update_df = assessor.apply_param_metrics_to_df(df)
    
    exporter.update_by_replacement(update_df)
    #exporter.update_results(update_df)
    #print(timing_decorator.return_average_times())
    

    



def unit_test():
    """
    Unit test routine. 
    Instantiates the involved components from a given config like in main, 
    and conducts a single training procedure for small scale experiments and testing purposes.
    """
    import torch

    from torch.utils.data import DataLoader
    from hydra import initialize, compose
    from hydra.utils import instantiate
    from pathlib import Path
    from tqdm import tqdm
    
    from timing_decorators import timing_decorator
    from experiment import ExperimentBuilder

    # check computation backend to use
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("-device:", device)

    ### Set Paths ###
    result_directory = Path('./Experiment_Results')
    experiment_name = 'COS_RL_ML'
    experiment_dir = result_directory / experiment_name

    config_name = 'recovery_config.yaml'
    #config_name = 'marginal_config.yaml'
    dataset_name = 'dataset.pt'
    #start_batch_name = 'start_batch.pt'
    #start_batch_name = 'zeros_start_batch.pt'
    start_batch_name = 'normal_start_batch.pt'

    print(experiment_dir)

    ### Load from directory ###
    dataset = torch.load(experiment_dir.joinpath(dataset_name))
    start_batch = torch.load(experiment_dir.joinpath(start_batch_name))
    #print(start_batch.shape)

    initialize(config_path= str(experiment_dir), version_base = None)
    cfg = compose(config_name = config_name)


    ### Experiment Setup ###
    hyper_parameters = instantiate(cfg.HyperParameters)
    model_parameters = instantiate(cfg.ModelParameters)
    sampling_parameters = instantiate(cfg.SamplingParameters)

    builder = ExperimentBuilder()

    model = builder.setup_model(model_parameters)
    sampler = builder.setup_sampler(model, start_batch, sampling_parameters)
    likelihood = builder.setup_likelihood(model, sampler, hyper_parameters)

    optimizer, scheduler = builder.setup_train_components(model, hyper_parameters)

    ### ATTENTION: ONLY UNIVARIATE ###
    dataset = dataset.unsqueeze(-1)

    ### Training ###
    batch_size = hyper_parameters['batch_size']
    model_batch_size = hyper_parameters['model_batch_size']
    burnin_offset = hyper_parameters['burnin_offset']
    train_loader = DataLoader(dataset, batch_size = batch_size, shuffle=True)

    epochs = hyper_parameters['epochs']
    #epochs = 100
    pbar   = tqdm(range(epochs))

    for it in pbar:
        for b, X_batch in enumerate(train_loader):
            
            # reset gradients 
            optimizer.zero_grad()
            
            model_samples = likelihood.gen_model_samples(
                batch_size = model_batch_size,
                burnin_offset = burnin_offset,
                data_samples = X_batch,
            )
            #print(model_samples.shape)
            
            ### Calculate and print energy, energy_grad and samples extreme values to investigate failure ###
            #test_energy = model.energy(model_samples)
            #test_grad = model.energy_grad(model_samples)

            print('Samples (components) max/min: ', float(torch.max(model_samples)), float(torch.min(model_samples)))
            #print('Energy max/min: ', float(torch.max(test_energy)), float(torch.min(test_energy)))
            #print('Gradients (components) max/min: ', float(torch.max(test_grad)), float(torch.min(test_grad)))


            likelihood.gradient(data_samples = X_batch, model_samples = model_samples)

            print(f"{it}_{b+1}/{epochs} Parameters:")
            for param_name, value in model.params.items():
                print(f'{param_name} value:\n {value.data}')
                print(f'{param_name} grad:\n {value.grad}')
                
                if torch.isnan(value.data).any():
                    raise StopIteration
                
            # perform gradient descent step along model.theta.grad
            optimizer.step()
        
        #break
            
        if scheduler:    
            scheduler.step()







if __name__=="__main__":

    #main()
    unit_test()
   

    