import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
# from tsl.engines import Predictor
# from tsl.data import ImputationDataset
# from tsl.utils.casting import torch_to_numpy
# from tsl.ops.connectivity import adj_to_edge_index
# from tsl.metrics.torch import MaskedMSE, MaskedMAE, MaskedMAPE
# from tsl.data.preprocessing import MinMaxScaler
# from tsl.data.datamodule import SpatioTemporalDataModule, TemporalSplitter

#from model import STConvAE
#from graphfoundationmodels.models.stGAE import STConvAE
#from graphfoundationmodels.dataloaders.dataloader_DIW import DIWDataset

import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback, ModelCheckpoint



class LossPlotCallback(Callback):
    def __init__(self):
        super().__init__()
        self.train_losses = []
        self.val_losses = []
    
    def on_train_epoch_end(self, trainer, pl_module):
        train_loss = trainer.logged_metrics.get('train_loss') 
        if train_loss is not None:
            self.train_losses.append(train_loss.item())
        
    def on_validation_epoch_end(self, training, pl_module):
        val_loss = trainer.logged_metrics.get('val_loss')
        if val_loss is not None:
            self.val_losses.append(val_loss.item())
    
    def plot_losses(self):
        plt.plot(self.train_losses, label='Training Loss')
        plt.plot(self.val_losses, label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Loss Curves')
        plt.legend()
        plt.show()