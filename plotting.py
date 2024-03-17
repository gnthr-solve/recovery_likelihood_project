
import torch
import numpy as np
import pandas as pd

from pathlib import Path
from abc import ABC, abstractmethod
from typing import Any

import plotly.express as px
import plotly.graph_objects as go

import matplotlib.pyplot as plt

from matplotlib.axes import Axes
from matplotlib.figure import Figure

from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

"""
Plot Component Base Class
-------------------------------------------------------------------------------------------------------------------------------------------
"""
class PlotComponent:
  @abstractmethod
  def get_data(self):
    pass

  @abstractmethod
  def draw(self, ax: Axes):
    pass



"""
Plot Matrix 
-------------------------------------------------------------------------------------------------------------------------------------------
"""

class PlotMatrix:

    def __init__(self):
        
        self.plots = {}


    def add_plot(self, row, col, plot: PlotComponent):

        if not isinstance(plot, PlotComponent):
            raise TypeError("Plot must be an PlotComponent subclass")
        
        self.plots[(row, col)] = plot

    
    def add_plot_dict(self, plot_dict: dict[tuple, PlotComponent]):
       self.plots.update(plot_dict)


    def setup_fig(self):

        key_tuples = self.plots.keys()
        rows = max([key[0] for key in key_tuples]) + 1
        cols = max([key[1] for key in key_tuples]) + 1
        #figsize=(15,0.4*number_diagrams), sharex=True
        self.fig, self.axes = plt.subplots(
           rows, 
           cols, 
           sharex = 'col', 
           squeeze = False,
        )


    def draw(self):

        self.setup_fig()

        for (row, col), plot in self.plots.items():
            #print(row, col, plot)
            ax = self.axes[row, col]  
            plot.draw(ax)
        
        #.get_current_fig_manager().set_window_title("Trends Vorbehandlung " + 'name_entdata')
        #.subplots_adjust(wspace=0.4)
        plt.show()



"""
Concrete Plot Components - To be inserted in Plot Matrix
-------------------------------------------------------------------------------------------------------------------------------------------
"""
class TimeSeriesPlot(PlotComponent):
  
  def __init__(self, results_df: pd.DataFrame, plot_index: str, column_to_plot: str, save: bool = False):
    
    self.results_df = results_df
    self.column_to_plot = column_to_plot
    self.plot_index = plot_index


  def draw(self, ax: Axes):

    for run_id, data in results_df.groupby('training_run_id'):

        ax.plot(data[self.plot_index], data[self.column_to_plot], label=f"{run_id}")

    ax.set_xlabel(self.plot_index, fontsize = 10)
    ax.set_ylabel(self.column_to_plot, fontsize = 10)
    ax.grid(True)
    #ax.tick_params(labelsize=10)
    #ax.set_title()



class ProcessPlot(PlotComponent):
  
  def __init__(self, results_df: pd.DataFrame, column_to_plot: str, save: bool = False):
    
    self.results_df = results_df
    self.column_to_plot = column_to_plot
    self.prepare_data()

  def prepare_data(self):
    self.mean_values = self.results_df.groupby('iteration')[self.column_to_plot].mean()
    self.std_values = self.results_df.groupby('iteration')[self.column_to_plot].std()


  def draw(self, ax: Axes):
    
    ax.plot(
        self.mean_values.index, 
        self.mean_values.values, 
        label="Mean", 
        #marker='o', 
        linewidth=1
    )

    ax.fill_between(
        self.std_values.index,
        self.mean_values.values - self.std_values.values,
        self.mean_values.values + self.std_values.values,
        alpha=0.2, 
        color='gray',
        label="± Standard Deviation"
    )

    ax.set_xlabel("Iteration", fontsize = 10)
    ax.set_ylabel(self.column_to_plot, fontsize = 10)
    ax.grid(True)
    #ax.tick_params(axis = 'y', labelsize=10)
    #ax.set_title()
    #ax.yaxis.label.set_color(color)



"""
Plotting Functions
-------------------------------------------------------------------------------------------------------------------------------------------
"""
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

    result_name = 'recovery_wo_Scheduler_lr2.csv'

    result_file_path = experiment_dir.joinpath(result_name)

    results_df = pd.read_csv(result_file_path)


    data_columns = ['Likelihood Values', 'mu_L2_error', 'Sigma_frob_error']

    plot_index = 'Iteration Timestamp'
    column_to_plot = 'mu_L2_error'

    ''''''
    plot_dict = {
        (i, j): TimeSeriesPlot(results_df, plot_index, column_to_plot)
        if j == 0 
        else ProcessPlot(results_df = results_df, column_to_plot = column_to_plot)
        for i, column_to_plot in enumerate(data_columns)
        for j in range(2)
    }

    
    plotter = PlotMatrix()
    plotter.add_plot_dict(plot_dict=plot_dict)
    plotter.draw()