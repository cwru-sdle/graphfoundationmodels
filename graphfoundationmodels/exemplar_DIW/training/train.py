# %%
''' load libraries '''

import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
from tsl.engines import Predictor
from tsl.data import ImputationDataset
from tsl.utils.casting import torch_to_numpy
from tsl.ops.connectivity import adj_to_edge_index
from tsl.metrics.torch import MaskedMSE, MaskedMAE, MaskedMAPE
from tsl.data.preprocessing import MinMaxScaler
from tsl.data.datamodule import SpatioTemporalDataModule, TemporalSplitter

#from model import STConvAE
from graphfoundationmodels.models.stGAE import STConvAE
from graphfoundationmodels.dataloaders.dataloader_DIW import DIWDataset

import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback, ModelCheckpoint

device = torch.device('cuda')

# %%
''' load data and basic preprocessing '''

#ADJ_MATRIX_PATH = '/mnt/vstor/CSE_MSE_RXF131/staging/mds3/llnl/DIW/2304giera-hdf5_cwru/adj_matrix_part_layer.npy'
#NODE_FTR_PATH = '/mnt/vstor/CSE_MSE_RXF131/staging/mds3/llnl/DIW/2304giera-hdf5_cwru/node_ftr_all.npy'

ADJ_MATRIX_PATH = '../adjacency_matrix/adj_matrix_part_layer.npy'
NODE_FTR_PATH = '../data/node_ftr_all.npy'


adj_matrix = np.load(ADJ_MATRIX_PATH)
adj_matrix = adj_to_edge_index(adj_matrix)

node_ftr = np.load(NODE_FTR_PATH)[:, :, :3]

scaler = MinMaxScaler(axis=(0))
scaler.fit(node_ftr)
node_ftr = scaler.transform(node_ftr)

# %%
''' dataset creation and preprocessing '''

torch_dataset = DIWDataset(target=node_ftr,
                                  eval_mask=torch.zeros_like(torch.from_numpy(node_ftr), dtype=torch.bool),
                                  connectivity=adj_matrix,
                                  window=250,
                                  stride=25)

splitter = TemporalSplitter(val_len=0.25, test_len=0.05)

dm = SpatioTemporalDataModule(
    dataset=torch_dataset,
    splitter=splitter,
    batch_size=8,
)

dm.setup()

# %%
''' instantiate model '''

stgnn = STConvAE(device=device, 
                 num_nodes=torch_dataset.n_nodes, 
                 channel_size_list=np.array([[3, 8, 16], [16, 8, 3]]), 
                 num_layers=2,
                 kernel_size=4, 
                 K=2,
                 kernel_size_de=2,
                 stride=1,
                 padding=1,
                 normalization='sym',
                 bias=True)



loss_fn = MaskedMAE()

metrics = {'mse': MaskedMSE(),
           'mae': MaskedMAE(),
           'mape': MaskedMAPE(),
           }

predictor = Predictor(
    model=stgnn,
    optim_class=torch.optim.Adam,
    optim_kwargs={'lr': 0.01},
    loss_fn=loss_fn,
    metrics=metrics,
    scheduler_class=torch.optim.lr_scheduler.StepLR,
    scheduler_kwargs={'step_size':15}
)

# %%
''' train model '''

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

loss_plot_callback = LossPlotCallback()

trainer = pl.Trainer(max_epochs=25,
                     accelerator='gpu',
                     callbacks=[loss_plot_callback])

trainer.fit(predictor, datamodule=dm)

loss_plot_callback.plot_losses()

# %%
''' predict on test set '''

predictor.freeze()
trainer.test(predictor, datamodule=dm)

# generate predictions
output = trainer.predict(predictor, dataloaders=dm.test_dataloader())
output = predictor.collate_prediction_outputs(output)
output = torch_to_numpy(output)

truth = output['y']
pred = output['y_hat']

# %%
''' plot scatters of error '''

num_features = pred.shape[3]
truth_flattened = truth.reshape(-1, num_features)
pred_flattened = pred.reshape(-1, num_features)

fig, axes = plt.subplots(1, num_features, figsize=(15, 5))

feature_names = ['X-error', 'Y-error', 'Z-error']

for feature_idx in range(num_features):
    ax = axes[feature_idx]
    ax.scatter(truth_flattened[:, feature_idx], pred_flattened[:, feature_idx], alpha=0.5, s=1)

    ax.set_title(f'{feature_names[feature_idx]}')
    ax.set_xlabel('Actual Values')
    ax.set_ylabel('Predicted Values')
    ax.grid(True)

plt.tight_layout()
plt.show()

# %%
''' plot reconstructions of error '''

for i in random.sample(range(0, pred.shape[0]), 3):

    node_id = random.sample(range(0, pred.shape[2]), 1)
    feature_names = ['X-error', 'Y-error', 'Z-error']

    fig, axes = plt.subplots(1, num_features, figsize=(15, 5))
    for feature_idx in range(num_features):
        ax = axes[feature_idx]
        ax.plot(truth[i, :, node_id, feature_idx][0])
        ax.plot(pred[i, :, node_id, feature_idx][0])

        ax.set_title(f'{feature_names[feature_idx]}')
        ax.set_xlabel('Timestep')
        ax.set_ylabel('Error')
        ax.grid(True)

    plt.tight_layout()
    plt.show()