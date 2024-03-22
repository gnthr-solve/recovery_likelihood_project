
import torch
import numpy as np
import pandas as pd

from pathlib import Path

from plotting_components import PlotMatrix, TimeSeriesPlot, ProcessPlot, HistogramPlot, StackedHistogramPlot
from helper_tools import prepare_sub_dfs, prepare_sub_dfs_alt

### Set Paths ###-------------------------------------------------------
result_directory = Path('./Experiment_Results')
experiment_name = 'POLY_RL_ML'
experiment_dir = result_directory / experiment_name

result_name = 'results.csv'

result_file_path = experiment_dir.joinpath(result_name)

results_df = pd.read_csv(result_file_path)

### Select Columns ###-------------------------------------------------------
data_columns = ['Unnorm. Likelihood Values', 'W L2-Error', 'mu_1_L2_error', 'Sigma_1_frob_error', 'mu_2_L2_error', 'Sigma_2_frob_error']

plot_index = 'Iteration Timestamp'
#column_to_plot = 'mu_L2_error'


### Split dataframe ###-------------------------------------------------------
filter_cols = {
    'Sampler': 'HMCSampler'
}
sub_result_dfs = prepare_sub_dfs(
    result_df = results_df,
    comparison_column = 'Likelihood',
    #filter_cols = filter_cols,
)

sub_result_dfs = prepare_sub_dfs_alt(
    result_df = results_df,
    comparison_columns = ['Likelihood', 'Sampler'],
    #filter_cols = filter_cols,
)
#for name, sub_df in sub_result_dfs.items():
#    print(name)
#    print(sub_df[-10:])

### Create Plot Dict ###-------------------------------------------------------
'''
plot_dict = {
    (i, j): TimeSeriesPlot(results_df, plot_index, column_to_plot)
    if j == 0 
    else ProcessPlot(results_df = results_df, column_to_plot = column_to_plot)
    for i, column_to_plot in enumerate(data_columns[:2])
    for j in range(2)
}


process_plot_dict = {
    (i, j): ProcessPlot(results_df = sub_result_dfs['Likelihood: Marginal'], column_to_plot = column_to_plot, title='Marginal')
    if j == 0 
    else ProcessPlot(results_df = sub_result_dfs['Likelihood: Recovery'], column_to_plot = column_to_plot, title='Recovery')
    for i, column_to_plot in enumerate(data_columns[:2])
    for j in range(2)
}
'''
process_plot_dict = {
    (0,0): ProcessPlot(results_df= sub_result_dfs['Likelihood: Marginal, Sampler: MALASampler'], column_to_plot=data_columns[1], title = r'Marginal MALA $\varepsilon = 0.1$'),
    (0,1): ProcessPlot(results_df= sub_result_dfs['Likelihood: Recovery, Sampler: MALASampler'], column_to_plot=data_columns[1], title = r'Recovery MALA $\varepsilon = 0.1$'),
    (1,0): ProcessPlot(results_df= sub_result_dfs['Likelihood: Marginal, Sampler: ULASampler'], column_to_plot=data_columns[1], title = r'Marginal ULA $\varepsilon = 0.01$'),
    (1,1): ProcessPlot(results_df= sub_result_dfs['Likelihood: Recovery, Sampler: ULASampler'], column_to_plot=data_columns[1], title = r'Recovery ULA $\varepsilon = 0.01$'),
    (2,0): ProcessPlot(results_df= sub_result_dfs['Likelihood: Marginal, Sampler: HMCSampler'], column_to_plot=data_columns[1], title = r'Marginal HMC $\varepsilon = 0.1$'),
    (2,1): ProcessPlot(results_df= sub_result_dfs['Likelihood: Recovery, Sampler: HMCSampler'], column_to_plot=data_columns[1], title = r'Recovery HMC $\varepsilon = 0.1$'),
}
### Histogram plots ###

hist_plot_dict = {
    (0,0): HistogramPlot(results_df= sub_result_dfs['Likelihood: Marginal, Sampler: MALASampler'], column_to_plot=data_columns[1], bins=20, title = r'Marginal MALA $\varepsilon = 0.1$'),
    (0,1): HistogramPlot(results_df= sub_result_dfs['Likelihood: Recovery, Sampler: MALASampler'], column_to_plot=data_columns[1], bins=20, title = r'Recovery MALA $\varepsilon = 0.1$'),
    (1,0): HistogramPlot(results_df= sub_result_dfs['Likelihood: Marginal, Sampler: ULASampler'], column_to_plot=data_columns[1], bins=20, title = r'Marginal ULA $\varepsilon = 0.01$'),
    (1,1): HistogramPlot(results_df= sub_result_dfs['Likelihood: Recovery, Sampler: ULASampler'], column_to_plot=data_columns[1], bins=20, title = r'Recovery ULA $\varepsilon = 0.01$'),
    (2,0): HistogramPlot(results_df= sub_result_dfs['Likelihood: Marginal, Sampler: HMCSampler'], column_to_plot=data_columns[1], bins=20, title = r'Marginal HMC $\varepsilon = 0.1$'),
    (2,1): HistogramPlot(results_df= sub_result_dfs['Likelihood: Recovery, Sampler: HMCSampler'], column_to_plot=data_columns[1], bins=20, title = r'Recovery HMC $\varepsilon = 0.1$'),
}

stacked_hist_plot_dict = {(0,0): StackedHistogramPlot(results_df_dict = sub_result_dfs, column_to_plot=data_columns[1], bins=20)}

### Create Plot ###-------------------------------------------------------
plotter = PlotMatrix(
    #title='Gaussian Mixture Model',
    title='UnivariatePolynomial', 
    sharex = 'col',
    sharey = 'row',
)
plotter.add_plot_dict(plot_dict = hist_plot_dict)

plotter.draw(fontsize=10)
#plotter.draw(fontsize=10)
#plotter.draw(fontsize=9)
#plotter.draw(fontsize=8)