#%%
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
from tsl.engines import Predictor
from typing import Callable, Mapping, Optional, Tuple, Union
from tsl.utils.casting import torch_to_numpy
from tsl.ops.connectivity import adj_to_edge_index
from tsl.metrics.torch import MaskedMSE, MaskedMAE, MaskedMAPE
from tsl.data.preprocessing import MinMaxScaler
from tsl.data.datamodule import SpatioTemporalDataModule, TemporalSplitter
import pytorch_lightning as pl
from tsl.typing import DataArray, SparseTensArray, TemporalIndex
from pytorch_lightning.callbacks import Callback, ModelCheckpoint

# out of the box data loader
from tsl.data import ImputationDataset

class DIWDataset(ImputationDataset):
    def __init__(self,
                 target_path: str,
                 connectivity = None, 
                 eval_mask = None,
                 index= None,
                 mask = None,
                 covariates= None,
                 input_map = None,
                 target_map = None,
                 auxiliary_map = None,
                 scalers = None,
                 trend = None,
                 transform = None,
                 window: int = 12,
                 stride: int = 1,
                 window_lag: int = 1,
                 precision: Union[int, str] = 32,
                 name: Optional[str] = None):
        target, connectivity_default, scaler = self._load_from_parquet(target_path)
        self.scaler = scaler
        if eval_mask is None:
            eval_mask = torch.zeros_like(torch.from_numpy(target))
        if connectivity is None:
            connectivity = connectivity_default
        super(DIWDataset, self).__init__(target,
                                        eval_mask,
                                        index=index,
                                        mask=None,
                                        connectivity=connectivity,
                                        covariates=covariates,
                                        input_map=input_map,
                                        target_map=target_map,
                                        auxiliary_map=auxiliary_map,
                                        trend=trend,
                                        transform=transform,
                                        scalers=scalers,
                                        window=window,
                                        stride=stride,
                                        precision=precision,
                                        name=name)
        


    def _load_from_parquet(self, target_path):
        df = pd.read_parquet(target_path)
        df['node_id'] = df['node_id'].apply(lambda x: '_'.join(x.split('_')[0:2]))
        df = df.sort_values(by=['node_id', 'Time']).reset_index(drop=True)

        target_cols = ['X_Pos_Error', 'Y_Pos_Error', 'Z_Pos_Error']

        all_cols = df.columns.tolist()
        #remove_cols = ['Time', 'node_id', 'delta_X_Pos', 'delta_Y_Pos', 'delta_Z_Pos']
        remove_cols = ['Time', 'node_id', 'delta_X_Pos', 'delta_Y_Pos',
                       'delta_Z_Pos', 'folder.id', 'layer.num', 'shape', 
                       'on.off', 'velocity', 'z_height', 'acceleration']
        predictor_cols = [col for col in all_cols if col not in target_cols + remove_cols]

        ids = df['node_id'].unique()

        id_prefixes = [id.split('_')[0] for id in ids]
        id_layer = [int(id.split('_')[1]) for id in ids]

        df['Timestep'] = df.groupby('node_id').cumcount() + 1

        df_max_timesteps = df.groupby('node_id')['Timestep'].max().reset_index(drop=True)
        min_max_timestep = df_max_timesteps.min()

        df = df[df['Timestep'] <= min_max_timestep]
        df = df.drop(columns=['Time'])

        all_cols = target_cols + predictor_cols

        df_pivot = df.pivot_table(index='Timestep', columns='node_id', values=all_cols)
        df_pivot = df_pivot[all_cols]
        df_pivot = df_pivot.reorder_levels([1, 0], axis=1)
        values = df_pivot.values

        num_timesteps = df['Timestep'].nunique()
        num_ids = df['node_id'].nunique()
        num_features = len(all_cols)

        node_ftr = values.reshape(num_timesteps, num_features, num_ids)
        node_ftr = np.moveaxis(node_ftr, 1, 2)

        adj_matrix = np.zeros((len(ids), len(ids)))

        for i, id1 in enumerate(ids):
            for j, id2 in enumerate(ids):
                # if id_prefixes[i] == id_prefixes[j] and i != j and id_layer[i] % 2 == id_layer[j] % 2:
                if id_prefixes[i] == id_prefixes[j] and i != j:
                    adj_matrix[i, j] = 1
        
        scaler = MinMaxScaler(axis=(0))
        scaler.fit(node_ftr)
        node_ftr = scaler.transform(node_ftr)

        return node_ftr, adj_matrix, scaler
