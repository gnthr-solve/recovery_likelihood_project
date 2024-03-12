
import os

import torch
import pandas as pd
import numpy as np

from pathlib import Path
"""
Result Exporter Blueprint TODO
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
"""
class ResultManager:

    def __init__(self, file_name: str, file_folder_path: Path):

        self.file_path = file_folder_path.joinpath(file_name)
        

    def export_observations(self, training_run_id: str, observation_dfs: list[pd.DataFrame]):

        self.load_results_df()

        new_results_df = pd.concat(observation_dfs, axis=1)
        new_results_df['training_run_id'] = training_run_id
        new_results_df['iteration'] = range(1, len(new_results_df)+1)

        self.results_df = pd.concat([self.results_df, new_results_df], axis=0)
        self.results_df.reset_index(inplace = True, drop = True)

        self.results_df.to_csv(self.file_path, index = False)

    
    def update_results(self, update_df: pd.DataFrame):

        self.load_results_df()
        
        extra_columns = [col for col in update_df.columns if col not in self.results_df.columns]
        merge_df = update_df[['training_run_id', 'iteration', *extra_columns]]

        self.results_df = self.results_df.merge(merge_df, how='left', on=['training_run_id', 'iteration'])
        
        self.results_df.to_csv(self.file_path, index = False)

    
    def load_results_df(self):

        if os.path.isfile(self.file_path):
            # If the file exists, read it into a DataFrame
            self.results_df = pd.read_csv(self.file_path)
        else:
            # If the file doesn't exist, create an empty DataFrame with the identifier
            self.results_df = pd.DataFrame(columns = ['training_run_id'])


