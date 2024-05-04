
import os

import torch
import pandas as pd
import numpy as np

from pathlib import Path


"""
ResultManager
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Simple class to manage the results csv files produced during experiment runs.
Opens file as pandas.Dataframe and appends the data the obervers accumulate during a training procedure.
It also sets an iteration number and the training_run_id to distinguish individual experiment runs.
"""

class ResultManager:

    def __init__(self, file_name: str, file_folder_path: Path):

        self.file_path = file_folder_path.joinpath(file_name)
        

    def export_observations(self, training_run_id: str, observation_df: pd.DataFrame):

        if not hasattr(self, 'results_df'):
            print('Loading df')
            self.load_results_df()

        new_results_df = observation_df
        new_results_df['training_run_id'] = training_run_id
        new_results_df['iteration'] = range(1, len(new_results_df)+1)

        self.results_df = pd.concat([self.results_df, new_results_df], axis=0)
        self.results_df.reset_index(inplace = True, drop = True)

        #self.results_df.to_csv(self.file_path, index = False)

    
    def update_by_replacement(self, update_df: pd.DataFrame):
        """
        Assumes that update_df is a copy of current results_df with extra metric column.
        This is a primitive solution, essentially calculating all the metrics again.
        """
        
        update_df.to_csv(self.file_path, index = False)

    
    def load_results_df(self):

        if os.path.isfile(self.file_path):
            # If the file exists, read it into a DataFrame
            self.results_df = pd.read_csv(self.file_path)
        else:
            # If the file doesn't exist, create an empty DataFrame with the identifier
            self.results_df = pd.DataFrame(columns = ['training_run_id'])


    def save_results(self):
        self.results_df.to_csv(self.file_path, index = False)




'''
def update_results(self, update_df: pd.DataFrame):

        self.load_results_df()
        
        extra_columns = [col for col in update_df.columns if col not in self.results_df.columns]
        merge_df = update_df[['training_run_id', 'iteration', *extra_columns]]

        self.results_df = self.results_df.merge(merge_df, how='left', on=['training_run_id', 'iteration'])
        
        self.results_df.to_csv(self.file_path, index = False)
'''