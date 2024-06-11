import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
from tsl.engines import Predictor

from tsl.utils.casting import torch_to_numpy
from tsl.ops.connectivity import adj_to_edge_index
from tsl.metrics.torch import MaskedMSE, MaskedMAE, MaskedMAPE
from tsl.data.preprocessing import MinMaxScaler
from tsl.data.datamodule import SpatioTemporalDataModule, TemporalSplitter
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback, ModelCheckpoint

# out of the box data loader
from tsl.data import ImputationDataset as DIWDataset