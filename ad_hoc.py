
import torch
import numpy as np
import pandas as pd

from pathlib import Path

import plotly.express as px
import plotly.graph_objects as go

import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D


def plot_run_metric(results_df: pd.DataFrame, plot_index: str, column_to_plot: str, save: bool = False):

    for run_id, data in results_df.groupby('training_run_id'):
        plt.plot(data[plot_index], data[column_to_plot], label=f"{run_id}")

    title = column_to_plot
    # Add labels and title
    plt.xlabel(plot_index)
    plt.ylabel("Values")
    plt.title(title)

    plt.legend()  
    plt.grid(True)
    if save:
        plt.savefig(f'Figures/{title}')
    plt.show()


def process_plot(df, column_to_plot, save: bool = False):
    # Calculate mean and standard deviation at each iteration
    mean_values = df.groupby('iteration')[column_to_plot].mean()
    std_values = df.groupby('iteration')[column_to_plot].std()

    # Plot the mean line
    plt.plot(
        mean_values.index, 
        mean_values.values, 
        label="Mean", 
        #marker='o', 
        linewidth=1
    )

    # Shade the standard deviation area
    plt.fill_between(
        std_values.index,
        mean_values.values - std_values.values,
        mean_values.values + std_values.values,
        alpha=0.2, 
        color='gray',
        label="± Standard Deviation"
    )

    title = f"{column_to_plot} over Iterations"
    
    plt.xlabel("Iteration")
    plt.ylabel(column_to_plot)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    if save:
        plt.savefig(f'Figures/{title}')

    plt.show()



if __name__=="__main__":
    ### Set Paths ###
    result_directory = Path('./Experiment_Results')
    experiment_name = 'MVG_RL_ML'
    experiment_dir = result_directory / experiment_name

    result_name = 'recovery.csv'

    result_file_path = experiment_dir.joinpath(result_name)

    results_df = pd.read_csv(result_file_path)


    data_columns = ['Likelihood Values', 'mu_L2_error', 'Sigma_frob_error']

    plot_index = 'Iteration Timestamp'
    column_to_plot = 'Likelihood Values'

    plot_run_metric(results_df, plot_index, column_to_plot, save = True)
    process_plot(df = results_df, column_to_plot = column_to_plot, save = True)