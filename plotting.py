
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

    def __init__(self, title: str, sharex: str|bool = False, save: bool = False ):
        
        self.plots = {}
        self.title = title
        self.sharex = sharex
        self.save = save


    def add_plot(self, row, col, plot: PlotComponent):

        if not isinstance(plot, PlotComponent):
            raise TypeError("Plot must be a PlotComponent subclass")
        
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
           sharex = self.sharex, 
           squeeze = False,
        )


    def draw(self):

        self.setup_fig()

        for (row, col), plot in self.plots.items():
            #print(row, col, plot)
            ax = self.axes[row, col]  
            plot.draw(ax)
        
        #.get_current_fig_manager().set_window_title()
        #.subplots_adjust(wspace=0.4)
        self.fig.suptitle(self.title)
        
        if self.save:
            filename = self.title.lower().replace(" ", "_")
            plt.savefig(f'Figures/{filename}')

        plt.show()



"""
Concrete Plot Components - To be inserted in Plot Matrix
-------------------------------------------------------------------------------------------------------------------------------------------
"""
class TimeSeriesPlot(PlotComponent):
  
  def __init__(self, results_df: pd.DataFrame, plot_index: str, column_to_plot: str):
    
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
  
    def __init__(self, results_df: pd.DataFrame, column_to_plot: str):
    
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
        #plt.legend()
        #ax.tick_params(axis = 'y', labelsize=10)
        #ax.set_title()
        #ax.yaxis.label.set_color(color)



class HistogramPlot(PlotComponent):
  
    def __init__(self, results_df: pd.DataFrame, column_to_plot: str, bins: int):
    
        self.results_df = results_df
        self.column_to_plot = column_to_plot
        self.bins = bins


    def draw(self, ax: Axes):

        error_samples = [
            data[self.column_to_plot].iloc[-1]
            for run_id, data in results_df.groupby('training_run_id')
        ]
        ax.hist(
            x = error_samples,
            bins = self.bins,
            density = True,
        )
        ax.set_xlabel('Parameter Error', fontsize = 10)
        ax.set_ylabel('Frequency', fontsize = 10)
        ax.grid(True)
        #ax.tick_params(labelsize=10)
        ax.set_title(f'{self.column_to_plot}')



class StackedHistogramPlot(PlotComponent):
  
    def __init__(self, results_df_dict: dict[str, pd.DataFrame], column_to_plot: str):
        
        self.results_df_dict = results_df_dict
        self.column_to_plot = column_to_plot


    def draw(self, ax: Axes):

        error_sample_dict = { 
            name: [
                    data[self.column_to_plot].iloc[-1]
                    for run_id, data in sub_df.groupby('training_run_id')
                ]
            for name, sub_df in self.results_df_dict.items()
        }

        for name, error_samples in error_sample_dict.items():
            ax.hist(
                x = error_samples, 
                density = True,
                label = name,
            )

        ax.set_xlabel('Parameter Error', fontsize = 10)
        ax.set_ylabel('Frequency', fontsize = 10)
        ax.grid(True)
        #ax.tick_params(labelsize=10)
        ax.set_title(f'{self.column_to_plot}')



"""
Plotting Preparation
-------------------------------------------------------------------------------------------------------------------------------------------
"""

def prepare_sub_dfs(result_df: pd.DataFrame, comparison_column: str)->dict[str, pd.DataFrame]:
    """
        Split a DataFrame based on unique values in a comparison column.
        
        Args:
        - df: Pandas DataFrame
        - comparison_column: Name of the column to use for splitting
        
        Returns:
        - Dictionary where keys are unique values in comparison column and values are corresponding DataFrame subsets
    """
    split_dict = {}
    
    unique_col_values = result_df[comparison_column].unique()
    matching_mask = lambda value: result_df[comparison_column] == value

    for col_value in unique_col_values:
        entry_name = comparison_column + f': {col_value}'
        split_dict[entry_name] = result_df.loc[matching_mask(col_value)].copy()
    
    return split_dict




if __name__=="__main__":
    ### Set Paths ###
    result_directory = Path('./Experiment_Results')
    experiment_name = 'POLY_RL_ML'
    experiment_dir = result_directory / experiment_name

    result_name = 'results.csv'

    result_file_path = experiment_dir.joinpath(result_name)

    results_df = pd.read_csv(result_file_path)


    data_columns = ['Likelihood Values', 'W L2-Error', 'mu_1_L2_error', 'Sigma_1_frob_error', 'mu_2_L2_error', 'Sigma_2_frob_error']

    plot_index = 'Iteration Timestamp'
    #column_to_plot = 'mu_L2_error'

    ''''''
    plot_dict = {
        (i, j): TimeSeriesPlot(results_df, plot_index, column_to_plot)
        if j == 0 
        else ProcessPlot(results_df = results_df, column_to_plot = column_to_plot)
        for i, column_to_plot in enumerate(data_columns[:2])
        for j in range(2)
    }
    

    plot_dict = {(0,0): HistogramPlot(results_df=results_df, column_to_plot=data_columns[1], bins=20)}

    plotter = PlotMatrix(
       #title='Gaussian Mixture Model',
       title='UnivariatePolynomial', 
       #sharex = 'col'
    )
    plotter.add_plot_dict(plot_dict=plot_dict)
    plotter.draw()