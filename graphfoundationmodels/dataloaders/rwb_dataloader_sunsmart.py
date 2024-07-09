# %%
import sys
import os
import re
import glob
from functools import reduce
import math
import numpy as np
import pandas as pd
from datetime import datetime
import scipy.sparse as sp
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torch_geometric.nn import ChebConv
from tqdm import tqdm
import random as random
import matplotlib.pyplot as plt

class stDataloader(Dataset):
    
    def __init__(self, parent_directory, raw_parquet_path, node_parquet_path, channel_parquet_path, adjacency_parquet_path,
                    meta_parquet_path, num_nodes,
                    traintestsplit = True, splittype = 'space', normalize = True,
                    fillnan = True, fillnan_value = 0,
                    model_output_directory = '',
                    test_name = 'test_1'):

        self.raw_parquet_path = raw_parquet_path
        self.node_parquet_path = node_parquet_path
        self.channel_parquet_path = channel_parquet_path
        self.adjacency_parquet_path = adjacency_parquet_path
        self.meta_parquet_path = meta_parquet_path
        self.model_output_directory = model_output_directory
        self.test_name = test_name
        self.num_nodes = num_nodes
        self.traintestsplit = traintestsplit
        self.splittype = splittype
        self.intialize_directories(parent_directory)
        self.adjacency_paths = glob.glob(self.adjacency_parquet_path + "*.parquet")
        self.channel_paths = glob.glob(self.channel_parquet_path + "*.parquet")
        self.node_paths = glob.glob(self.node_parquet_path + "*.parquet")
        self.meta_df = pd.read_parquet(self.meta_parquet_path)
        self.node_dataframes = []
        self.channel_dataframes = [pd.read_parquet(p) for p in self.channel_paths]
        self.adjacency_dataframes = [pd.read_parquet(p) for p in self.adjacency_paths]
        self.channel_names = [os.path.basename(p).split(".parquet")[0] for p in self.channel_paths]
        self.adjacency_names = [os.path.basename(p).split(".parquet")[0] for p in self.adjacency_paths]
        self.normalize = normalize
        self.channel_indices = {name: int(idx) for idx, name in enumerate(self.channel_names)}
        self.adjacency_indices = {name: int(idx) for idx, name in enumerate(self.adjacency_names)}
        self.fillnan = fillnan
        self.fillnan_value = fillnan_value
        self.window_size = None  # Placeholder for window size
        self.intialize_directories(parent_directory)

    def __len__(self):
        
        return len(self.channel_dataframes[0])

    def __getitem__(self, idx):
        
        if self.window_size is None:
            raise ValueError("Window size is not set. Call set_window_size() method first.")
        
        windows = self.get_windows()
        if idx >= len(windows):
            raise IndexError("Index out of range.")
        
        window_data = windows[idx]

        return window_data['channels'], self.weight_dataframes, self.meta_df, {
            "datetimes": window_data["datetimes"],
            "channel_names": self.channel_names,
            "weight_names": self.weight_names,
        }

    def intialize_directories(self, parent_directory):

        raw_parquet_path = parent_directory + 'raw_data/' 
        if not os.path.exists(raw_parquet_path):
            os.mkdir(raw_parquet_path)
            print("Creating raw_data directory")
        self.raw_parquet_path = raw_parquet_path

        node_directory = parent_directory + 'nodes/' 
        if not os.path.exists(node_directory):
            os.mkdir(node_directory)
            print("Creating nodes directory")
            self.node_parquet_path = node_directory 
        else:       
            self.node_parquet_path = node_directory

        channel_directory = parent_directory + 'channels/' 
        if not os.path.exists(channel_directory):
            os.mkdir(channel_directory)
            print("Creating channels directory")
            self.channel_parquet_path = channel_directory
        else:
            self.channel_parquet_path = channel_directory

        adjacency_parquet_path = parent_directory + 'adjacency_matrices/' 
        if not os.path.exists(adjacency_parquet_path):
            os.mkdir(adjacency_parquet_path)
            print("Creating adjacency_matrices directory")
        else:
            self.adjacency_parquet_path = adjacency_parquet_path

        model_output_directory = parent_directory + 'tests/'
        if not os.path.exists(model_output_directory):
            os.mkdir(model_output_directory)
            print("Creating tests directory")
        self.model_output_directory = model_output_directory

        test_output_directory = self.model_output_directory + self.test_name + '/'
        if not os.path.exists(test_output_directory):
            os.mkdir(test_output_directory)
            print("Creating tests directory")
        self.test_output_directory = test_output_directory

        model_path = self.test_output_directory + 'models/' 
        if not os.path.exists(model_path):
            os.mkdir(model_path)
            print("Creating models directory")
        self.model_path = model_path

        model_results_path = self.test_output_directory + 'model_results/' 
        if not os.path.exists(model_results_path):
            os.mkdir(model_results_path)
            print("Creating model results directory")
        self.model_results_path = model_results_path

        pred_test_path = self.test_output_directory + 'pred_tests/' 
        if not os.path.exists(pred_test_path):
            os.mkdir(pred_test_path)
            print("Creating pred_tests directory")
        self.pred_test_path = pred_test_path

        pred_path = self.test_output_directory + 'pred/'
        if not os.path.exists(pred_path):
            os.mkdir(pred_path)
            print("Creating pred directory")
        self.pred_path = pred_path
        

    def get_windows(self):

        if self.window_size is None:
            raise ValueError("Window size is not set. Call set_window_size() method first.")
        
        total_length = self.__len__()
        windows = []

        for idx in range(0, total_length, self.window_size):
            window_data = []

            for channel_df in self.channel_dataframes:
                sample = channel_df.iloc[idx:idx+self.window_size]
                window_data.append(np.array(sample))

            datetimes = [str(t) for t in list(set(sample.index))]
            windows.append({"channels": np.array(window_data), "datetimes": datetimes})

        return windows


    def set_date_range(self, start, stop, freq, tz = 'UTC'):

        self.start = pd.Timestamp(start)
        self.stop = pd.Timestamp(stop)
        self.freq = freq
        self.tz = tz
        self.time_index = pd.date_range(self.start, end=stop, freq=freq, tz=tz)

    def read_files_with_nodeID(self):
        """
        Reads a list of CSV files and adds a 'nodeID' column to each DataFrame with the name of the file (without .csv).
        
        Parameters:
        filepaths (list): A list of file paths to the CSV files.
        
        Returns:
        list: A list of DataFrames with the 'nodeID' column added.
        """
        dataframes = []

        for f in tqdm(self.node_paths, total = len(self.node_paths)):

            df = pd.read_parquet(f)
            
            filename = os.path.basename(f)
            node_id = os.path.splitext(filename)[0]

            df['nodeID'] = node_id

            self.node_dataframes.append(df)

        return dataframes

    def read_from_tabular(self, raw_parquet_path = None, df_split_col = None, df_id_col = None, write = False):

        #Add tabular data manulupation function 
        # that applies on all dataframes at once 
        # i,e rename column, trim etc...

        if raw_parquet_path == None:
            filepaths = os.listdir(self.raw_parquet_path)
        else:
            filepaths = os.listdir(raw_parquet_path)

        if df_id_col == None: 

            if filepaths[0].split(".")[-1] == "csv":
                list_dfs = [pd.read_csv(os.path.join(self.raw_parquet_path, filepath)).assign(filename=os.path.basename(filepath).split('.')[0])
                                for filepath in filepaths
                                ]

            elif filepaths[0].split(".")[-1] == "parquet":
                list_dfs = [pd.read_parquet(os.path.join(self.raw_parquet_path, filepath)).assign(filename=os.path.basename(filepath).split('.')[0])
                                for filepath in filepaths
                                ]

            else:
                 raise ValueError("File Formats Accepted: .csv , .parquet")

        if df_split_col != None:

            split_dfs = []

            for df in list_dfs:
                split_df = self.split_nodes_from_rowwise(df = df, df_split_col=df_split_col, df_id_col=df_id_col,write=write)
                [split_dfs.append(df) for df in split_df]

            if write == True:
                #Iterate over each DataFrame in the list
                for df in split_dfs:
                    # Define the filename for the CSV file
                    filename = f"{df['nodeID'].iloc[0]}.parquet"
                    filepath = os.path.join(self.node_parquet_path, filename)
                    df.to_parquet(filepath, index=True)
                self.node_dataframes = split_dfs

            else:
                self.node_dataframes = split_dfs

            return split_dfs

        else:

            if write == True:
                #Iterate over each DataFrame in the list
                for i,df in enumerate(list_dfs):
                    #Define the filename for the CSV file
                    filename = f"{df['nodeID'].iloc[0]}.parquet"
                    filepath = os.path.join(self.node_parquet_path, filename)
                    df.to_parquet(filepath, index=True)
                self.node_dataframes = list_dfs

            else: 
                self.node_dataframes = list_dfs
        
            return list_dfs

    def split_nodes_from_rowwise(self, df, df_split_col, df_id_col = None, write = False):
        
        if df_id_col == None:
            df_id_col = 'filename'

        groups = df.groupby(df_split_col).groups

        # Split the DataFrame into groups based on 'invt'
        grouped_dfs = [df.loc[group] for group in groups.values()]
        grouped_dfs = [grouped_df.assign(nodeID=grouped_df[df_id_col].iloc[0] + '_' + grouped_df[df_split_col].iloc[0]) for grouped_df in grouped_dfs]

        return grouped_dfs
    
    def get_longest_timespan(self, list_dfs = None, freq = '15min', time_var = 'tmst', tz = 'EST', set_params = True):

        if list_dfs == None:
            list_dfs = self.node_dataframes
            
        set_params = set_params
        longest_start = None
        longest_end = None
        max_overlap = None
        df_timespans = []

        for df in tqdm(list_dfs, total = len(list_dfs)):

            #Read the CSV file into a DataFrame
            # df[time_var] = pd.to_datetime(df[time_var])
            # df = df.set_index(time_var)
            
            # # Get the index range of the DataFrame
            start_index = df.index.min()
            end_index = df.index.max()
            df_index = pd.date_range(start_index, end=end_index, freq=freq)


            if longest_start is None:
                    longest_start = start_index
                    longest_end = end_index
            
            # Compare the index range with the longest overlapping range
            overlap = min(end_index, longest_end) - max(start_index, longest_start)
            
            if max_overlap is None:
                    max_overlap = overlap

            if overlap > max_overlap:
                max_overlap = overlap
                longest_start = max(start_index, longest_start)
                longest_end = min(end_index, longest_end)

            elif overlap == max_overlap and overlap != 0:
                longest_start = min(start_index, longest_start)
                longest_end = max(end_index, longest_end)
            
            df_timespans.append(df_index)

        print('Start:' + str(longest_start)  + '\n' + 'End:' + str(longest_end))

        if set_params == True:
            self.set_date_range(longest_start, longest_end, df.index.freq, tz = df.index.tz)

        return longest_start,longest_end

    def tabular_to_matrix(self, measurement_cols, list_dfs = None, time_var = 'tmst', write = False):

        if list_dfs == None:
            list_dfs = self.node_dataframes
        
        if self.time_index is None:
            raise ValueError("Time Index is not set. Call set_date_range() or get_longest_timespan() method first.")
        
        measurement_list = []

        for measurement in measurement_cols: 

            col_list = []

            for df in tqdm(list_dfs, total = len(list_dfs)):
                
                column_name = str(df['nodeID'].iloc[0]) + '_' + str(measurement)

                # df[time_var] = pd.to_datetime(df[time_var])
                # df = df.set_index(time_var).tz_localize(self.tz)
                df = df[~df.index.duplicated()]
                df = df.reindex(self.time_index)
                
                # Extract the relevant column within the largest overlapping time window
                relevant_column = df[measurement]
                relevant_column = relevant_column.rename(column_name)
                col_list.append(relevant_column)

            if len(col_list) == 1:
                measurement_matrix = col_list
            else:
                measurement_matrix = pd.concat(col_list, axis=1)

            if write == True:
                filename = measurement + '.parquet'
                filepath = os.path.join(self.channel_parquet_path, filename)
                measurement_matrix.to_parquet(filepath)

                self.channel_dataframes.append(measurement_matrix)
                self.channel_names.append(measurement)

            else : 
                self.channel_dataframes.append(measurement_matrix)
                self.channel_names.append(measurement)

        measurement_list.append(measurement_matrix)
        self.channel_indices = {name: int(idx) for idx, name in enumerate(self.channel_names)}

        return measurement_list

    def data_availibility(self, measurement_cols, list_dfs = None, time_var = 'tmst', write = False):

        if list_dfs == None:
            list_dfs = self.node_dataframes

        if self.time_index is None:
            raise ValueError("Time Index is not set. Call set_date_range() or get_longest_timespan() method first.")
        
        measurement_list = []

        for measurement in measurement_cols: 

            presence_list = []

            for df in tqdm(list_dfs, total = len(list_dfs)):

                column_name = str(df['nodeID'].iloc[0]) + '_' + str(measurement)

                # df[time_var] = pd.to_datetime(df[time_var])
                # df = df.set_index(time_var).tz_localize(self.tz)
                df = df[~df.index.duplicated()]
                df = df.reindex(self.time_index)
                
                # Extract the relevant column within the largest overlapping time window
                relevant_column = df[measurement]
                relevant_column = relevant_column.rename(column_name)
                
                # Check if the column has a value at each index
                presence = relevant_column.notnull().astype(int)
                
                # Append the presence information to the list
                presence_list.append(presence)

            # Concatenate the presence information for all files into a single DataFrame
            if len(presence_list) == 1:
                presence_df = presence_list
            else:
                presence_df = pd.concat(presence_list, axis=1)

            if write == True:
                filename = measurement + '_' + 'missing_values'+ '.parquet'
                filepath = os.path.join(self.channel_parquet_path, filename)
                presence_df.to_parquet(filepath)

                self.channel_dataframes.append(presence_df)
                self.channel_names.append(measurement + '_missingvalues')

            else : 
                self.channel_dataframes.append(presence_df)
                self.channel_names.append(measurement + '_missingvalues')
            
        self.channel_indices = {name: int(idx) for idx, name in enumerate(self.channel_names)}
        measurement_list.append(presence_df)
        
        return measurement_list

    def add_channel(self, df, df_name = 'new_df', write = False):

        if write == True:
            filename = df_name + '.parquet'
            filepath = os.path.join(self.channel_parquet_path, filename)
            df.to_parquet(filepath)
            self.channel_dataframes.append(df)
            self.channel_names.append(df_name)
            self.channel_indices = {name: int(idx) for idx, name in enumerate(self.channel_names)}
        
        else:
            self.channel_dataframes.append(df)
            self.channel_names.append(df_name)
            self.channel_indices = {name: int(idx) for idx, name in enumerate(self.channel_names)}

    def ts_reindex_with_interpolation(self, df_to_reindex, start_freq = '1h', chunk_size = None, method = 'linear', order = 3):
        """
        Reindex a DataFrame to a new index range and perform spline interpolation.

        Parameters:
            df_to_reindex (pd.DataFrame): Input DataFrame.
        Returns:
            pd.DataFrame: DataFrame with reindexed index and interpolated values.
        """
        if self.time_index is None:
            raise ValueError("Time Index is not set. Call set_date_range() or get_longest_timespan() method first.")

        start_freq = start_freq
        df = self.channel_dataframes[self.channel_indices[df_to_reindex]]
        df = df[~df.index.duplicated()]
        df.asfreq(start_freq)

        # Create a new index with the specified range and frequency
        padded_index = pd.date_range(start=self.start,
                                        end=self.stop,
                                        freq=self.freq, tz=self.tz
                                        )

        # Reindex DataFrame to the new index
        df.index = df.index.tz_convert(None)
        df_padded = df.reindex(padded_index)

        if chunk_size == None:
            chunk_size = len(df_padded)

        # Perform spline interp - 1) // chunk_sizeolation
        num_chunks = (len(df_padded) + chunk_size) // chunk_size 
        chunks = []

        for i in range(num_chunks):

            start_idx = i * chunk_size
            end_idx = start_idx + chunk_size
            chunk = df_padded.iloc[start_idx:end_idx].copy()
            
            if method == 'spline' and order is not None:
                chunk_interpolated = chunk.interpolate(method=method, order=order, limit_direction='both')
            
            else:
                chunk_interpolated = chunk.interpolate(method=method, limit_direction='both')

            chunks.append(chunk_interpolated)

        df_interpolated = pd.concat(chunks, axis = 0)

        #Overwrites Previous Entry
        self.channel_dataframes[self.channel_indices[df_to_reindex]] = df_interpolated

        return df_interpolated

    def apply_conditions(self, conditions_list, write = False):

        """
        Apply a list of conditions to the specified DataFrame in a list using conditions based on other DataFrames.

        Parameters:
            conditions_list (list of tuples): Each tuple contains (target_index, conditions, fill_df_index, fallback_index).
                - conditions: A list of tuples, each containing (channel_x, operator, channel_y, scalar).
                - conditions_op: Operator to act on conditional statments.
                - channel_if_true: Name of DataFrame from self.channel_names to get the value_if_true from. If None, use scalar_values.
                - channel_if_false: Name of DataFrame self.channel_names to return values from if conditions are not met.
                - scalar_if_true: Value used for filling if fill_df_index = None.
                - scalar_if_true: Value used for returning if df_df_index = None.
                - condition_name: Name corresponding to this condition.

        Returns:
            pd.DataFrame: The modified DataFrame list after applying all conditions.
        """
        
        mod_dfs = []
        mod_names = []
        
        # Process each condition in the conditions list
        for conditions, conditions_op, fill_df, return_df, fill_scalar, return_scalar, condition_name in conditions_list:

            if fill_df is not None:
                fill_df_index = self.channel_indices[fill_df]

            if return_df is not None:
                return_df_index = self.channel_indices[return_df]

            # Evaluate each condition in the list
            condition_results = []

            for df1, operator, df2, scalar in conditions: 

                if df2 is not None: 

                    df1_index = self.channel_indices[df1]
                    df2_index = self.channel_indices[df2]

                    if df1.shape == df2.shape:
                        df1_values = self.channel_dataframes[df1_index].values
                        df2_values = self.channel_dataframes[df2_index].values
                        condition = operator(df1_values, df2_values)                        
                        condition_results.append(condition)

                    else : 
                        raise ValueError("All DataFrames distance_ecutoff_1must have the same dimensions")
                else:

                    df1_index = self.channel_indices[df1]

                    if operator is np.isnan or operator is pd.isna:        
                        df1_values = self.channel_dataframes[df1_index].values        
                        condition = operator(df1_values)
                
                    else:
                        df1_values = self.channel_dataframes[df1_index]
                        condition = operator(df1_values, scalar)
                
                    condition_results.append(condition)

            # Combine condition results with logical AND
            all_conditions_true = conditions_op(condition_results, axis=0)

            # Get value_if_true from value_df
            value_if_true = self.channel_dataframes[fill_df_index].values if fill_df is not None else fill_scalar
            value_if_false = self.channel_dataframes[return_df_index].values if return_df is not None else return_scalar

            # Apply np.where
            modified_values = np.where(all_conditions_true, value_if_true, value_if_false)
            
            # Append the modified DataFrame along with its name to self.channel_dataframes and self.channel_names
            modified_df_name = condition_name if condition_name else f"Condition {i}"
            self.channel_names.append(modified_df_name)
            self.channel_indices = {name: int(idx) for idx, name in enumerate(self.channel_names)}
            
            if return_df is None and fill_df is None:
                df = pd.DataFrame(modified_values, dtype=np.float64)
                self.channel_dataframes.append(df)

            elif fill_df is not None and return_df is None:
                df = pd.DataFrame(modified_values, columns=self.channel_dataframes[fill_df_index].columns, index=self.channel_dataframes[fill_df_index].index, dtype=np.float64)
                self.channel_dataframes.append(df)
            else: 
                df = pd.DataFrame(modified_values, columns=self.channel_dataframes[return_df_index].columns, index=self.channel_dataframes[return_df_index].index, dtype=np.float64)
                self.channel_dataframes.append(df)
            
            if write == True:
                modified_df_savepath = self.channel_parquet_path + modified_df_name + '.parquet'
                df.to_parquet(modified_df_savepath)

            mod_dfs.append(df)
            mod_names.append(modified_df_name)

        return mod_dfs, mod_names

    def generate_distance_adjacency_matrix(self, latitude = 'latd', longitude = 'longd', epsilons = [0.75], weighted = True, nodeID_col = 'nodeID', write = False):

        nodeID_col = nodeID_col
        latitude = latitude 
        longitude = longitude
        write = write
        distances_df = pd.DataFrame(index = self.meta_df[nodeID_col], columns = self.meta_df[nodeID_col], dtype=np.float64)

        # Calculate distances
        for i in range(len(self.meta_df)):
            for j in range(len(self.meta_df)):
                distances_df.iloc[i, j] = self.haversine(self.meta_df[longitude][i], self.meta_df[latitude][i], self.meta_df[longitude][j], self.meta_df[latitude][j])

        sigma = np.std(distances_df.to_numpy().flatten())
        w_list = []
        w_names = []

        for e in tqdm(epsilons, total = len(epsilons)):

            w = np.zeros(shape=(len(distances_df), len(distances_df)))

            for i in range(len(distances_df)):
                for j in range(len(distances_df)):
                    if i == j: 
                        w[i][j] = 1
                    else:
                        # Computed distance between stations
                        d_ij = distances_df.iloc[i, j]
                        # Compute weight w_ij
                        w_ij = np.exp(-d_ij**2 / sigma**2)
                        if w_ij >= e:
                            if weighted == True:
                                w[i, j] = w_ij
                            else:
                                w[i, j] = 1

            adjacency_matrix = pd.DataFrame(w, index = self.meta_df[nodeID_col], columns = self.meta_df[nodeID_col], dtype=np.float64)
            
            if weighted == True:
                w_name = 'epsilon_weighted_' + str(e)
            else: 
                w_name = 'epsilon_' + str(e)

            w_names.append(w_name)
            w_list.append(adjacency_matrix)

            self.adjacency_dataframes.append(adjacency_matrix)
            self.adjacency_names.append(w_name)

            if write == True:
                adjacency_savepath = self.adjacency_parquet_path + w_name + '.parquet'   
                adjacency_matrix.to_parquet(adjacency_savepath)

        self.adjacency_indices = {name: int(idx) for idx, name in enumerate(self.adjacency_names)}

        return(w_list, w_names)

    def haversine(self,lon1, lat1, lon2, lat2):
        """
        Calculate the great circle distance between two points
        on the earth (specified in decimal degrees)
        """
        # Convert decimal degrees to radians
        lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

        # Haversine formula
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        
        # Radius of Earth in kilometers is 6371
        km = 6371 * c
        return km

    def generate_feature_adjacency_matrix(self, feature_cols = [], nodeID_col = 'nodeID', write = False):
        
        nodeID_col = nodeID_col
        feature_cols = feature_cols
        write = write 
        feature_list = []
        feature_names = []

        for feature_col in tqdm(feature_cols, total = len(feature_cols)):

            # Extract the column as a numpy array
            values = self.meta_df[feature_col].values

            # Create a boolean matrix using broadcasting
            bool_matrix = values[:, None] == values

            # Convert the boolean matrix into a DataFrame
            bool_df = pd.DataFrame(bool_matrix, index=self.meta_df[nodeID_col], columns=self.meta_df[nodeID_col])
            
            feature_names.append(feature_col)
            feature_list.append(bool_df)
            
            if write == True:
                adjacency_savepath = self.adjacency_parquet_path + feature_col + '.parquet'   
                bool_df.to_parquet(adjacency_savepath)    
        
            self.adjacency_dataframes.append(feature_list)
            self.adjacency_names.append(feature_col)
            
        self.adjacency_indices = {name: int(idx) for idx, name in enumerate(self.adjacency_names)}

        return(feature_list, feature_names)

    def generate_datasetgrade_adjacency_matrix(self, grade_col = '', min_grade_threshold = [0.70], max_diff_thresholds = [0.75], weighted = True, nodeID_col = 'nodeID', write = False):

        nodeID_col = nodeID_col
        write = write
        grades_df = pd.DataFrame(index = self.meta_df[nodeID_col], columns = self.meta_df[nodeID_col], dtype=np.float64)

        # Calculate Connetivity
        w_list = []
        w_names = []

        for threshold_value in tqdm(max_diff_thresholds, total = len(max_diff_thresholds)):

            w = np.zeros(shape=(len(grades_df), len(grades_df)))

            for i in range(len(grades_df)):

                for j in range(len(grades_df)):

                    if i == j: 
                        w[i][j] = 1

                    else:
                        # Computed distance between stations
                        if threshold_value > abs(self.meta_df[grade_col].iloc[i] - self.meta_df[grade_col].iloc[j]) or self.meta_df[grade_col].iloc[i] > self.meta_df[grade_col].iloc[j]:
                            
                            if weighted == True:
                                if self.meta_df[grade_col].iloc[i] > min_grade_threshold:
                                    w[i, j]  = self.meta_df[grade_col].iloc[i]
                                else:
                                    w[i, j]  = 0 
                            else:
                                if self.meta_df[grade_col].iloc[i] > min_grade_threshold:
                                    w[i, j]  = 1
                                else:
                                    w[i, j]  = 0 
                                    
                        elif self.meta_df[grade_col].iloc[i]  < self.meta_df[grade_col].iloc[j]:
                             w[i, j]  = 0

            adjacency_matrix = pd.DataFrame(w, index = self.meta_df[nodeID_col], columns = self.meta_df[nodeID_col], dtype=np.float64)
            
            if weighted == True:
                w_name = 'datasetgrade_weighted_mingrade_' + str(min_grade_threshold) + 'maxdiff_' + str(threshold_value) 
            else: 
                w_name = 'datasetgrade_mingrade_' + str(min_grade_threshold) + 'maxdiff_' + str(threshold_value) 

            w_names.append(w_name)
            w_list.append(adjacency_matrix)

            self.adjacency_dataframes.append(adjacency_matrix)
            self.adjacency_names.append(w_name)

            if write == True:
                adjacency_savepath = self.adjacency_parquet_path + w_name + '.parquet'   
                adjacency_matrix.to_parquet(adjacency_savepath)

        self.adjacency_indices = {name: int(idx) for idx, name in enumerate(self.adjacency_names)}

        return(w_list, w_names)

    def set_train_test_splits(self, splits):
        
        self.train_percentage, self.test_percentage, self.val_percentage = splits
        self.train_cols, self.test_cols, self.val_cols = self.split_columns_randomly()
        self.train_windows, self.test_windows, self.val_windows = self.split_windows_randomly()
        self.train_indices = self.find_indices_within_bins(self.train_windows) 
        self.test_indices = self.find_indices_within_bins(self.test_windows)
        self.val_indices = self.find_indices_within_bins(self.val_windows)

    def set_window_size(self, window_size):
        
        self.window_size = window_size

    def set_model_inputs(self, input_channels, output_channels, mask_channels, adjacency_matrices):
        
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.mask_channels = mask_channels
        self.adjacency_matrices = adjacency_matrices

    def get_model_inputs(self, input_channels, output_channels, mask_channels):
        
        if self.traintestsplit == True:
            self.train_tensor, self.test_tensor, self.val_tensor = self.data_to_model_input_transform()
        
        self.data_tensor = data_to_model_input_transform()

    def split_columns_randomly(self):

        # Get the total number of columns
        total_columns = self.num_nodes
        # Create a list of all column indices
        all_columns = list(range(total_columns))
        
        # Calculate the number of columns for each dataset split
        train_columns_count = int(total_columns * self.train_percentage)
        test_columns_count = int(total_columns * self.test_percentage)
        val_columns_count = total_columns - train_columns_count - test_columns_count
        
        # Randomly shuffle the list of column indices
        np.random.shuffle(all_columns)
        
        # Split the shuffled list of column indices into training, testing, and validation sets
        train_columns = all_columns[:train_columns_count]
        test_columns = all_columns[train_columns_count:train_columns_count + test_columns_count]
        val_columns = all_columns[train_columns_count + test_columns_count:]
        
        return np.sort(train_columns), np.sort(test_columns), np.sort(val_columns)

    def split_windows_randomly(self):

        # Get the total number of columns
        total_windows = int(np.ceil(len(self.time_index) / self.window_size))
        # Create a list of all column indices
        all_windows = list(range(total_windows))
        
        # Calculate the number of columns for each dataset split
        train_windows_count = int(total_windows * self.train_percentage)
        test_windows_count = int(total_windows* self.test_percentage)
        val_windows_count = total_windows - train_windows_count - test_windows_count
        
        # Randomly shuffle the list of column indices
        np.random.shuffle(all_windows)
        
        # Split the shuffled list of column indices into training, testing, and validation sets
        train_windows = all_windows[:train_windows_count]
        test_windows = all_windows[train_windows_count:train_windows_count + test_windows_count]
        val_windows = all_windows[train_windows_count + test_windows_count:]
        
        return np.sort(train_windows), np.sort(test_windows), np.sort(val_windows)

    def find_indices_within_bins(self, bins):
        indices_within_bins = []
        for bin_start in bins:
            indices_within_bin = list(range(bin_start, bin_start + self.window_size))
            indices_within_bins.append(indices_within_bin)
        return indices_within_bins

    def col_normalization(self, channel, norm_scale = 100):

        channel_max = channel.max(axis = 0)
        channel_norm = channel / channel_max

        return channel_norm * norm_scale, channel_max

    def subset(self, data, dim):
        """
        Splits the array along a specified dimension using predefined column indices.

        Parameters:
            array (np.array): The input array to be split.
            dim (int): The dimension along which to split the array.

        Returns:
            tuple: A tuple containing the subsets of the array for training, testing, and validation.
        """
        if self.splittype == "space":
            # Create slices for all dimensions as ':' (slice(None))
            slices = [slice(None)] * len(data.shape)
            
            # Replace the slice for the specified dimension with the indices for training
            slices[dim] = self.train_cols
            train = data[tuple(slices)]

            # Replace the slice for the specified dimension with the indices for testing
            slices[dim] = self.test_cols
            test = data[tuple(slices)]

            # Replace the slice for the specified dimension with the indices for validation
            slices[dim] = self.val_cols
            val = data[tuple(slices)]

            return train, test, val
        
        elif self.splittype == 'time':

            slices = [slice(None)] * len(data.shape)
            
            # Replace the slice for the specified dimension with the indices for training
            slices[dim] = self.train_windows
            train = data[tuple(slices)]

            # Replace the slice for the specified dimension with the indices for testing
            slices[dim] = self.test_windows
            test = data[tuple(slices)]

            # Replace the slice for the specified dimension with the indices for validation
            slices[dim] = self.val_windows
            val = data[tuple(slices)]

            return train, test, val

        # elif self.splittype == 'both':

        #     slices_time = [slice(None)] * len(data.shape)
            
        #     # Replace the slice for the specified dimension with the indices for training
        #     slices_time[dim] = self.train_windows
        #     train_time = data[tuple(slices_time)]
        #     slices_sp1 = [slice(None)] * len(train_time.shape)
        #     slices_sp1[dim + 2] = self.train_cols
        #     train = train_time[tuple(slices_sp1)]

        #     # Replace the slice for the specified dimension with the indices for testing
        #     slices_time[dim] = self.test_windows
        #     test_time = data[tuple(slices_time)]
        #     slices_sp2 = [slice(None)] * len(test_time.shape)
        #     slices_sp2[dim + 2] = self.test_cols
        #     test = test_time[tuple(slices_sp2)]

        #     # Replace the slice for the specified dimension with the indices for validation
        #     slices_time[dim] = self.val_windows
        #     val_time = data[tuple(slices_time)]
        #     slices_sp3 = [slice(None)] * len(val_time.shape)
        #     slices_sp3[dim + 2] = self.val_cols
        #     val = val_time[tuple(slices_sp3)]

        #     return train, test, val

        else:
            raise ValueError("splittype not recognized, please select one of ['space','time','both]")


    def weight_subset(self, data, dim):
        """
        Splits the data along a specified dimension using predefined column indices,
        while also dropping the corresponding row indices.

        Parameters:
            data (np.array): The input array to be split.
            dim (int): The dimension along which to split the array.

        Returns:
            tuple: A tuple containing the subsets of the array for training, testing, and validation.
        """
        # Create slices for all dimensions as ':' (slice(None))
        slices = [slice(None)] * len(data.shape)
        
        # Replace the slice for the specified dimension with the indices for training
        slices[dim] = self.train_cols
        train = data[tuple(slices)]
        train_indices = np.arange(data.shape[dim])
        train_indices = np.delete(train_indices, self.train_cols)
        train = np.delete(train, train_indices, axis=(dim-1))

        # Replace the slice for the specified dimension with the indices for testing
        slices[dim] = self.test_cols
        test = data[tuple(slices)]
        test_indices = np.arange(data.shape[dim])
        test_indices = np.delete(test_indices, self.test_cols)
        test = np.delete(test, test_indices, axis=(dim-1))

        # Replace the slice for the specified dimension with the indices for validation
        slices[dim] = self.val_cols
        val = data[tuple(slices)]
        val_indices = np.arange(data.shape[dim])
        val_indices = np.delete(val_indices, self.val_cols)
        val = np.delete(val, val_indices, axis=(dim-1))

        return train, test, val

    def get_channel_indices(self):
        
        if self.input_channels is None:
            raise ValueError("Input channel is not set. Call set_model_inputs() method first.")
        
        # Select corresponding channels based on column indices
        selected_input_channels = [self.channel_indices[name] for name in self.input_channels if name in self.channel_indices]
        selected_output_channels = [self.channel_indices[name] for name in self.output_channels if name in self.channel_indices]
        selected_mask_channels = [self.channel_indices[name] for name in self.mask_channels if name in self.channel_indices]
        selected_adjacency_channels = [self.adjacency_indices[name] for name in self.adjacency_matrices if name in self.adjacency_indices]

        return selected_input_channels, selected_output_channels, selected_mask_channels, selected_adjacency_channels

    def weight_to_model_input_transform(self, datasplit = "training"):

        #operator = operator
        _,_,_,adjacency_indices= self.get_channel_indices()

        adjacency_matrices = []

        for adj_idx in adjacency_indices:
            adjacency_matrices.append(self.adjacency_dataframes[adj_idx])

        weight = np.nanmean(adjacency_matrices, axis = 0)

        w_full = weight
        G_full = sp.coo_matrix(weight)

        edge_index_full = torch.tensor(np.array([G_full.row, G_full.col]), dtype=torch.int64)
        edge_weight_full = torch.tensor(G_full.data).float()

        if self.traintestsplit == True:

            if self.splittype == "space" or self.splittype == 'both':

                if self.train_cols is None:
                    raise ValueError("Training Testing Split is not set. Call set_train_test_splits() method first.")
                
                train_w, test_w, val_w = self.weight_subset(weight,1)

                if datasplit == 'training': 
                    w = train_w
                    G = sp.coo_matrix(train_w)
                    edge_index = torch.tensor(np.array([G.row, G.col]), dtype=torch.int64)
                    edge_weight = torch.tensor(G.data).float()

                elif datasplit == 'testing': 
                    w = test_w
                    G = sp.coo_matrix(test_w)
                    edge_index = torch.tensor(np.array([G.row, G.col]), dtype=torch.int64)
                    edge_weight = torch.tensor(G.data).float()

                else : 
                    w = val_w
                    G = sp.coo_matrix(val_w)
                    edge_index = torch.tensor(np.array([G.row, G.col]), dtype=torch.int64)
                    edge_weight = torch.tensor(G.data).float()

                return w, edge_index, edge_weight, w_full, edge_index_full, edge_weight_full

            else:
                return weight, edge_index_full, edge_weight_full, weight, edge_index_full, edge_weight_full

        return weight, edge_index_full, edge_weight_full, weight, edge_index_full, edge_weight_full
            
    def data_to_model_input_transform(self, normalize = True):

        # Initialize lists to store arrays for input, output, and mask channels
        input_list = []
        output_list = []
        mask_list = []
        
        input_channels, output_channels, mask_channels, _= self.get_channel_indices()
        # Get the window size
        window_size = self.window_size
        
        # Process each channel separately
        for channel_idx in input_channels:
            data = self.channel_dataframes[channel_idx].values

            if self.normalize == True:
                data, _ = self.col_normalization(data)

            if self.fillnan == True:
                data[np.isnan(data)] = self.fillnan_value

            # Create bins and pad if necessary
            binned_data = [data[i:i + window_size] for i in range(0, len(data), window_size)]
            padded_data = [np.pad(bin_data, pad_width=((0, window_size - bin_data.shape[0]), (0, 0)), mode='constant') for bin_data in binned_data if bin_data.shape[0] < window_size]
            binned_data[-1] = padded_data[-1] if padded_data else binned_data[-1]
            input_list.append(np.stack(binned_data))
        
        for channel_idx in output_channels:
            data = self.channel_dataframes[channel_idx].values

            if normalize == True:
                data, _ = self.col_normalization(data)

            if self.fillnan == True:
                data[np.isnan(data)] = self.fillnan_value

            binned_data = [data[i:i + window_size] for i in range(0, len(data), window_size)]
            padded_data = [np.pad(bin_data, pad_width=((0, window_size - bin_data.shape[0]), (0, 0)), mode='constant') for bin_data in binned_data if bin_data.shape[0] < window_size]
            binned_data[-1] = padded_data[-1] if padded_data else binned_data[-1]
            output_list.append(np.stack(binned_data))
        
        for channel_idx in mask_channels:
            data = self.channel_dataframes[channel_idx].values

            if self.fillnan == True:
                data[np.isnan(data)] = self.fillnan_value

            binned_data = [data[i:i + window_size] for i in range(0, len(data), window_size)]
            padded_data = [np.pad(bin_data, pad_width=((0, window_size - bin_data.shape[0]), (0, 0)), mode='constant') for bin_data in binned_data if bin_data.shape[0] < window_size]
            binned_data[-1] = padded_data[-1] if padded_data else binned_data[-1]
            mask_list.append(np.stack(binned_data))
        
        x = np.array(input_list).transpose(1,2,3,0)
        y = np.array(output_list)
        
        # Combine masks using logical AND
        if mask_list:
            # combined_mask = np.logical_and.reduce(np.array(mask_list), axis=0).reshape(y.shape[0],y.shape[1],y.shape[2],y.shape[3])
            combined_mask = np.logical_and.reduce(np.array(mask_list), axis=0).reshape(1,y.shape[1],y.shape[2],y.shape[3])
        else:
            combined_mask = np.zeros_like(y)
        
        y = y.transpose(1,2,3,0)
        mask = combined_mask.transpose(1,2,3,0)

        if self.traintestsplit == True:

            if self.splittype == "space":
            
                if self.train_cols is None:
                    raise ValueError("Training Testing Split is not set. Call set_train_test_splits() method first.")
                
                train_x, test_x, val_x = self.subset(x,2)
                train_y, test_y, val_y = self.subset(y,2)
                train_mask, test_mask, val_mask = self.subset(mask,2)

            elif self.splittype == "time":

                if self.train_windows is None:
                    raise ValueError("Training Testing Split is not set. Call set_train_test_splits() method first.")
                
                train_x, test_x, val_x = self.subset(x,0)
                train_y, test_y, val_y = self.subset(y,0)
                train_mask, test_mask, val_mask = self.subset(mask,0)

            # elif self.splittype == 'both':

            #     if self.train_cols is None:
            #         raise ValueError("Training Testing Split is not set. Call set_train_test_splits() method first.")
                
            #     train_x, test_x, val_x = self.subset(x,0)
            #     train_y, test_y, val_y = self.subset(y,0)
            #     train_mask, test_mask, val_mask = self.subset(mask,0)
                
            else:
                raise ValueError("Split Type is not set. Specify the type of validation in the dataloader")

            train = torch.utils.data.TensorDataset(torch.Tensor(train_x), torch.Tensor(train_y),torch.Tensor(train_mask))
            test = torch.utils.data.TensorDataset(torch.Tensor(test_x),torch.Tensor(test_y),torch.Tensor(test_mask))
            val = torch.utils.data.TensorDataset(torch.Tensor(val_x),torch.Tensor(val_y),torch.Tensor(val_mask))
            full = torch.utils.data.TensorDataset(torch.Tensor(x),torch.Tensor(y),torch.Tensor(mask)) 

            return train, test, val, full

        else:
            data = torch.utils.data.TensorDataset(torch.Tensor(x), torch.Tensor(y), torch.Tensor(mask))
            
            return data, data, data, data

    def get_modelprediction_stats(self, model_list = [''], write_name = ''): 

        filepath = self.model_output_directory + write_name
        test_filepath = self.model_output_directory + write_name + 'mean.parquet'

        if os.path.exists(test_filepath):
            # File exists, load the Parquet file
            mean_path = filepath + 'mean.parquet'
            var_path = filepath + 'var.parquet'

            mean = pd.read_parquet(mean_path)
            var = pd.read_parquet(var_path)

        else:
            # File does not exist, perform operations and save as Parquet
            print("File does not exist. Calculating Mean and Variance of Model Predictions")

            # Example operation: Creating a new DataFrame
            # You should replace this with your actual data processing or DataFrame creation logic
            paths = [glob.glob(self.model_output_directory + 'predictions/' + 'model_' + str(model) + '.parquet') for model in model_list]

            dfs = [pd.read_parquet(p) for p in paths]
            arrays = [df.to_numpy() for df in dfs]

            array_list = []
            for array in arrays:
                if array.shape[0] != 0:
                    array_list.append(array)
                else:
                    print("Zero Array")

            tensor = np.stack(array_list)

            mean = np.nanmean(tensor, axis = 0)
            var = np.nanvar(tensor, axis = 0)

            mean_path = filepath + 'mean.parquet'
            var_path = filepath + 'var.parquet'

            # Save the DataFrame to a Parquet file
            pd.DataFrame(mean, dtype=np.float64).to_parquet(mean_path)
            pd.DataFrame(var, dtype=np.float64).to_parquet(var_path)
            print("New Parquet file saved.")

        return mean, var

    def get_model_parameters(self):

        models_params = pd.read_parquet(self.model_output_directory + 'scripts/params.parquet')
        
        subfolders = ['nodes', 'windows', 'test_loss', 'train_loss', 'errors', 'errors_test']

        # Create an empty list to store DataFrames
        dfs = []

        # Loop through each subfolder
        for subfolder in subfolders:

            subfolder_path = os.path.join(self.model_output_directory,
                                                subfolder)
            
            # Get a list of CSV files in the subfolder
            csv_files = [file for file in os.listdir(subfolder_path) if file.endswith('.csv')]
            folder_dfs = []
            file_df = pd.DataFrame()

            # Read each CSV file into a DataFrame
            for i, file in zip(range(0, len(csv_files)), csv_files):

                file_path = os.path.join(subfolder_path, file)
                model_number = file.split('_')[1].split('.')[0]

                # Create a DataFrame with a single element containing the entire CSV dat
                values = (pd.read_csv(file_path, index_col=0).T).values.flatten()

                if len(file.split('_')) > 2: 
                    split = file.split('_')[2].split('.')[0]
                    file_df['split'] = split

                file_df = pd.DataFrame(data=[[values]], columns=[subfolder])
                
                # Add additional column for the file path
                file_df['model_num'] = int(model_number)
                
                # Add the DataFrame to the list
                folder_dfs.append(file_df)

            # Concatenate all files into a single DataFrame
            folder_df = pd.concat(folder_dfs, axis = 0)

            dfs.append(folder_df)

        #Concatenate final df
        common_column = 'model_num'
        model_paths = glob.glob(self.model_output_directory + 'models/' + '*.pt')
        model_paths_df = pd.DataFrame(model_paths, columns = ['model_path'])
        model_paths_df['model_num'] = [int(file.split('_')[1].split('.')[0]) for file in os.listdir(self.model_output_directory + 'models/')]
        
        # Merge all DataFrames in the list on the common column
        merged_df = reduce(lambda left, right: pd.merge(left, right, on=common_column, how='outer'), dfs)
        merged_df = pd.merge(merged_df, model_paths_df, on = "model_num", how = "outer")
        final_df = pd.merge(models_params, merged_df, left_index = True, right_on = merged_df['model_num'])
        
        return final_df

    def batch_plot(self, x, mask, y, y_pred, plot_nodes = 5, num_plots_per_node = 3, plot_range = 200, plot_type = 'sunangle'):
        
        plot_nodes = plot_nodes
        num_plots_per_node = num_plots_per_node
        plot_range = plot_range
        plot_type = plot_type
        y_plot = y.cpu().flatten(start_dim = 0, end_dim = 1)[:,:,0].detach().numpy()
        y_pred_plot =  y_pred.cpu().flatten(start_dim = 0, end_dim = 1)[:,:,0].detach().numpy()
        emp_plot = y.cpu().flatten(start_dim = 0, end_dim = 1)[:,:,1].detach().numpy()
        #y_pred_plot =  y_pred.flatten(start_dim = 0, end_dim = 1)[:,:,0].detach().numpy()
        mask_plot = mask.cpu().flatten(start_dim = 0, end_dim = 1)[:,:,0].detach().numpy()
        dhi_plot = x.cpu().flatten(start_dim = 0, end_dim = 1)[:,:,0].detach().numpy()
        dni_plot = x.cpu().flatten(start_dim = 0, end_dim = 1)[:,:,1].detach().numpy()
        ghi_plot = x.cpu().flatten(start_dim = 0, end_dim = 1)[:,:,2].detach().numpy()
        temp_plot = x.cpu().flatten(start_dim = 0, end_dim = 1)[:,:,3].detach().numpy()
        wspa_plot = x.cpu().flatten(start_dim = 0, end_dim = 1)[:,:,4].detach().numpy()
        sun_plot = x.cpu().flatten(start_dim = 0, end_dim = 1)[:,:,5].detach().numpy()

        for i in [random.randint(0, y_plot.shape[1]-1) for _ in range(plot_nodes)]:
            
            for start in range(0,y_plot.shape[0], y_plot.shape[0] // num_plots_per_node):
                
                end = start + plot_range
                subset_y = y_plot[start:end,i]
                subset_mask = mask_plot[start:end,i]
                subset_y_pred = y_pred_plot[start:end,i]
                subset_emp = emp_plot[start:end,i]
                subset_dni = dni_plot[start:end,i]
                subset_dhi = dhi_plot[start:end,i]
                subset_ghi = ghi_plot[start:end,i]
                subset_temp = temp_plot[start:end,i]
                subset_wspa = wspa_plot[start:end,i]
                subset_sun = sun_plot[start:end,i]

                if plot_type == 'irrad':

                    plt.plot(subset_dni, '-', label='DNI')
                    plt.plot(subset_dhi, '-', label='DHI')                    
                    plt.plot(subset_ghi, '-', label='GHI')                        
                    plt.plot(subset_y_pred, '.', label='Prediction')

                    plt.xlabel('Index')
                    plt.ylabel('Values')
                    plt.title(f'Node: {i}, Time: {start}')
                    plt.legend()
                    plt.show()                    
                
                elif plot_type == 'temp':

                    plt.plot(subset_temp, '-', label='TEMP')
                    plt.plot(np.where(subset_mask == False,subset_y,0), label='Real Data')
                    plt.plot(subset_y_pred, '.', label='Prediction')
                    
                    plt.xlabel('Index')
                    plt.ylabel('Values')
                    plt.title(f'Node: {i}, Time: {start}')
                    plt.legend()
                    plt.show()  

                elif plot_type == 'emp':

                    plt.plot(subset_emp, '-', label='EMP')
                    plt.plot(np.where(subset_mask == False,subset_y,0), label='Real Data')
                    plt.plot(subset_y_pred, '.', label='Prediction')
                    
                    plt.xlabel('Index')
                    plt.ylabel('Values')
                    plt.title(f'Node: {i}, Time: {start}')
                    plt.legend()
                    plt.show()

                elif plot_type == 'input/output':

                    plt.plot(np.where(subset_mask == False,subset_y,0), label='Real Data')
                    plt.plot(subset_y_pred, '.', label='Prediction')

                    plt.plot(subset_dni, '-', label='DNI')
                    plt.plot(subset_dhi, '-', label='DHI')                    
                    plt.plot(subset_ghi, '-', label='GHI')                    
                    plt.plot(subset_temp, '-', label='TEMP')
                    plt.plot(subset_wspa, '-', label='WSPA')
                    plt.plot(subset_sun, '-', label='SUN')  

                    plt.xlabel('Index')
                    plt.ylabel('Values')
                    plt.title(f'Node: {i}, Time: {start}')
                    plt.legend()

                else:

                    plt.plot(np.where(subset_mask == False,subset_y,0), label='Real Data')
                    plt.plot(subset_y_pred, '.', label='Prediction')

                    plt.xlabel('Index')
                    plt.ylabel('Values')
                    plt.title(f'Node: {i}, Time: {start}')
                    plt.legend()
                    plt.show()

def extract_numbers(file_paths):

    numbers = []
    pattern = re.compile(r'_(\d+)\.pt$')

    for path in file_paths:
        # Find all matches in the file name
        match = pattern.search(path)
        if match:
            numbers.append(int(match.group(1)))  # Append the captured number to the list

    return numbers

def within_n_std_dev(mean_df, var_df, input_df, n=1):

    #Reset Col Names
    input_df.columns = range(0,mean_df.shape[1])
    # Calculate upper and lower bounds based on mean and variance
    lower_bound = mean_df - n * np.sqrt(var_df)
    upper_bound = mean_df + n * np.sqrt(var_df)
    
    # Pad
    padded_input_df = input_df.reindex(mean_df.index).fillna(0)

    # Generate mask DataFrame
    mask_df = (padded_input_df >= lower_bound) & (padded_input_df <= upper_bound)
    
    return mask_df.to_numpy(), upper_bound.to_numpy(), lower_bound.to_numpy()
# %%
