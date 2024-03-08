
import os

import torch
import pandas as pd
import numpy as np

from pathlib import Path
"""
Result Exporter Blueprint TODO
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
"""
class ResultExporter:

    def __init__(self, export_name: str, export_folder_path: Path,):

        self.export_path = export_folder_path.joinpath(export_name)

        if os.path.isfile(self.export_path):
            # If the file exists, read it into a DataFrame
            self.results_df = pd.read_csv(self.export_path)
        else:
            # If the file doesn't exist, create an empty DataFrame
            self.results_df = pd.DataFrame(columns = ['training_run_id'])
        

    def export_observations(self, training_run_id: str, observation_dfs: list[pd.DataFrame]):

        new_results_df = pd.concat(observation_dfs, axis=1)
        new_results_df['training_run_id'] = training_run_id

        #set the index to unique identifier to merge and update
        new_results_df.set_index('training_run_id', inplace = True)
        self.results_df.set_index('training_run_id', inplace = True)

        #combine the previous and new data in case we add information
        self.results_df = self.results_df.combine_first(new_results_df)
        self.results_df.reset_index(inplace = True)

        self.results_df.to_csv(self.export_path)