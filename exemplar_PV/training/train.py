import sys
from datetime import datetime
import pandas as pd
from time import sleep
import os 
# Do not edit the print statements!!!!

try:
    #line contains one line of your input CSV file
    parquet_path = sys.argv[1]
    batch = int(sys.argv[2])
    array_start = int(sys.argv[3])
    start = array_start + int(os.environ['SLURM_ARRAY_TASK_ID'])
    end = start + batch

    print(f'[SLURMSTART] {datetime.now().strftime("%Y-%m-%d-%H:%M:%S")} Job started with args:-{parquet_path} {start} {end}') 
    input_df = pd.read_parquet(parquet_path, engine='pyarrow').iloc[start:end].reset_index()
    
    #Your code here! 
    # %%
    import sys
    import os
    import re
    import glob
    import math
    import numpy as np
    import pandas as pd
    from datetime import datetime
    import scipy.sparse as sp
    from sklearn.preprocessing import StandardScaler
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import multiprocessing
    from torch.utils.data import Dataset, DataLoader
    from torchvision import transforms, utils
    from torch_geometric.nn import ChebConv
    from tqdm import tqdm
    from rwb_stgnn_functions import *
    from rwb_dataloader_sunsmart import *
    import matplotlib.pyplot as plt
    import seaborn as sns

    # %%
    for column in input_df.columns:
        globals()[column] = input_df[column].iloc[0]

    # Directorys
    meta_directory = '/mnt/vstor/CSE_MSE_RXF131/staging/sdle/foundational_models/sunsmart/sunsmart_meta_grade.parquet'

    #Dataloader
    pvts = stDataloader(parent_directory = '/mnt/vstor/CSE_MSE_RXF131/staging/sdle/foundational_models/sunsmart/',
                                raw_parquet_path = '',
                                node_parquet_path ='',
                                channel_parquet_path = '',
                                adjacency_parquet_path = '', 
                                meta_parquet_path = meta_directory,
                                num_nodes = 90, 
                                traintestsplit= traintestsplit,
                                splittype = splittype,
                                normalize = True,
                                model_output_directory='',
                                test_name = test_name)

    #Model Save Paths
    model_name = 'model_' + str(start) + '.pt'
    model_save_path = pvts.model_path + model_name

    node_name = 'model_' + str(start) + '.csv'
    training_node_save_path = pvts.training_nodes_path + node_name

    window_name = 'model_' + str(start) + '.csv'
    training_window_save_path = pvts.training_windows_path + node_name

    train_name = 'model_' + str(start)  + '.csv' 
    train_loss_save_path = pvts.train_loss_path + train_name

    test_name = 'model_' + str(start)  +'.csv'
    test_loss_save_path = pvts.test_loss_path + test_name

    error_test_name = 'model_' + str(start) + '.csv'
    error_test_save_path = pvts.error_path_test + error_test_name

    error_name = 'model_' + str(start) + '.csv'
    error_save_path = pvts.error_path + error_name

    pred_test_name = 'model_' + str(start) + '.parquet'
    pred_test_save_path = pvts.pred_test_path + pred_test_name

    pred_name = 'model_' + str(start) + '.parquet'
    pred_save_path = pvts.pred_path + pred_name

    #Set Timespan
    pvts.set_date_range('2012-09-28 07:00:00', '2015-10-08 05:30:00', '15min', tz = 'UTC')
    # %% Set Model Inputs
    pvts.set_window_size(n_his)
    pvts.set_model_inputs(inputs, outputs, masks, adjacency)
    pvts.set_train_test_splits(splits)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # %% Transform the Datafrom tables to matrices
    train_data, test_data, eval_data, full_data = pvts.data_to_model_input_transform()
    train_iter = DataLoader(train_data, batch_size=int(batch_size),
                            shuffle=shuffle, num_workers=2)
    test_iter = DataLoader(test_data, batch_size=int(batch_size),
                            shuffle=False, num_workers=2)
    full_iter = DataLoader(full_data, batch_size=int(batch_size),
                            shuffle=False, num_workers=2)

    # %% Transform the adjacency matrix tables to matrices
    train_w, train_edge_index, train_edge_weight, full_w, full_edge_index, full_edge_weight = pvts.weight_to_model_input_transform(datasplit = 'training')
    test_w, test_edge_index, test_edge_weight, _, _, _ = pvts.weight_to_model_input_transform(datasplit = 'testing')
    # %% Initializes Model
    model = STConvAE(device, num_nodes, channels, num_layers, 
                        kernel_size, K, n_his, kernel_size_de, stride, 
                        padding, normalization = 'sym', bias = True)
    model = model.to(model.device)
    loss = nn.MSELoss()
    min_test_loss = np.inf
    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate) 

    # Model Training Loop
    train_loss = []
    test_loss = []

    for epoch in tqdm(range(1, num_epochs + 1), desc = 'Epoch', position = 0):
        epoch_train_loss = 0.0
        model.train()
        i = 0
        
        for x, y, mask in tqdm(train_iter, desc = 'Batch', position = 0):

            optimizer.zero_grad()
            y = y.to(model.device)
            mask = mask.to(model.device)
            y_pred = model(x.to(model.device), 
                            train_edge_index.to(model.device), 
                            train_edge_weight.to(model.device)
                            ) 
            loss= torch.nanmean((torch.where(mask == False, 
                                    (y_pred-y)**2, 
                                    torch.tensor(float('nan'))
                                    ))
                                    )
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item() 

        train_loss.append(epoch_train_loss)
        epoch_test_loss = 0.0
        model.eval()

        with torch.no_grad():
            for x, y, mask in tqdm(test_iter, desc = 'Batch', position = 0):
                y = y.to(model.device)
                mask = mask.to(model.device)
                y_pred = model(x.to(model.device),
                                    test_edge_index.to(model.device),
                                    test_edge_weight.to(model.device)
                                    ) 
                loss = torch.nanmean((torch.where(mask == False,
                                        (y_pred-y)**2,
                                        torch.tensor(float('nan'))
                                        ))
                                        )
                epoch_test_loss += loss.item()
            test_loss.append(epoch_test_loss)

        print(f'Epoch: {epoch} \n Training Loss: {epoch_train_loss / len(train_iter)} \n Evaulation Loss: {epoch_test_loss / len(test_iter)}')
        if min_test_loss > epoch_test_loss:
            min_test_loss = epoch_test_loss
            torch.save(model.state_dict(), model_save_path)
    
    pd.DataFrame(pvts.train_cols).to_csv(training_node_save_path)
    pd.DataFrame(pvts.train_windows).to_csv(training_window_save_path)
    pd.DataFrame(train_loss).to_csv(train_loss_save_path)
    pd.DataFrame(test_loss).to_csv(test_loss_save_path)
    # Model Evaluation
    
    #Intialize Model
    eval_model = STConvAE(device, num_nodes, channels, num_layers, kernel_size, K, n_his, kernel_size_de, stride, padding, normalization = 'sym', bias = True).to(device)
    eval_model.load_state_dict(torch.load(model_save_path))
    eval_model = eval_model.to(eval_model.device)
    eval_loss = nn.MSELoss()
    first_eval = 1
    eval_model.eval()

    with torch.no_grad():
        for x, y, mask in tqdm(test_iter, desc = 'Batch', position = 0):
            torch.cuda.empty_cache()
            # get model predictions and compute loss
            y_pred = eval_model(x.to(device), test_edge_index.to(device), test_edge_weight.to(device))
            if first_eval == 1:
                y_complete_test = y
                mask_complete_test = mask
                y_pred_complete_test = y_pred
            else:
                y_complete_test = torch.cat((y_complete_test, y))
                mask_complete_test = torch.cat((mask_complete_test, mask))
                y_pred_complete_test = torch.cat((y_pred_complete_test, y_pred))
            first_eval+=1

    first_full = 1

    with torch.no_grad():
        for x, y, mask in tqdm(full_iter, desc = 'Batch', position = 0):
            torch.cuda.empty_cache()
            # get model predictions and compute loss
            y_pred = eval_model(x.to(device), full_edge_index.to(device), full_edge_weight.to(device))
            if first_full == 1:
                y_complete = y
                mask_complete = mask
                y_pred_complete = y_pred
            else:
                y_complete = torch.cat((y_complete, y))
                mask_complete = torch.cat((mask_complete, mask))
                y_pred_complete = torch.cat((y_pred_complete, y_pred))
            first_full+=1

    raw_test = (y_complete_test).cpu().flatten(start_dim = 0, end_dim = 1).flatten(start_dim = 1, end_dim = 2).detach()
    mask_test = (mask_complete_test).cpu().flatten(start_dim = 0, end_dim = 1).flatten(start_dim = 1, end_dim = 2).detach()
    pred_test = (y_pred_complete_test).cpu().flatten(start_dim = 0, end_dim = 1).flatten(start_dim = 1, end_dim = 2).detach()
    raw_test = torch.where(raw_test == 0, torch.tensor(float(.1)), raw_test)
    
    raw = (y_complete).cpu().flatten(start_dim = 0, end_dim = 1).flatten(start_dim = 1, end_dim = 2).detach()
    mask = (mask_complete).cpu().flatten(start_dim = 0, end_dim = 1).flatten(start_dim = 1, end_dim = 2).detach()
    pred = (y_pred_complete).cpu().flatten(start_dim = 0, end_dim = 1).flatten(start_dim = 1, end_dim = 2).detach()
    raw = torch.where(raw == 0, torch.tensor(float(.1)), raw)

    #Calculate Model Error
    MAE_test = torch.nanmean((torch.where(mask_test == False, torch.abs(pred_test - raw_test), torch.tensor(float('nan'))))).detach().numpy()
    MAPE_test = torch.nanmean((torch.where(mask_test == False, torch.abs((pred_test - raw_test) / raw_test), torch.tensor(float('nan'))))).detach().numpy()
    MSE_test = torch.nanmean((torch.where(mask_test == False, (pred_test - raw_test)**2, torch.tensor(float('nan'))))).detach().numpy()
    RMSE_test = ((torch.nanmean((torch.where(mask_test == False, (pred_test - raw_test)**2, torch.tensor(float('nan'))))))**.5).detach().numpy()
    errors_test = [MAE_test,MAPE_test,MSE_test,RMSE_test]

    MAE = torch.nanmean((torch.where(mask == False, torch.abs(pred - raw), torch.tensor(float('nan'))))).detach().numpy()
    MAPE = torch.nanmean((torch.where(mask == False, torch.abs((pred - raw) / raw), torch.tensor(float('nan'))))).detach().numpy()
    MSE = torch.nanmean((torch.where(mask == False, (pred - raw)**2, torch.tensor(float('nan'))))).detach().numpy()
    RMSE = ((torch.nanmean((torch.where(mask == False, (pred - raw)**2, torch.tensor(float('nan'))))))**.5).detach().numpy()
    errors = [MAE,MAPE,MSE,RMSE]

    pred_test = pred_test.numpy()
    pred = pred.numpy()

    pd.DataFrame(errors).to_csv(error_save_path)
    pd.DataFrame(errors_test).to_csv(error_test_save_path)
    pd.DataFrame(pred_test).to_parquet(pred_test_save_path)
    pd.DataFrame(pred).to_parquet(pred_save_path)

    #Fail the job if desired output is not vaiable
    if not os.path.exists(model_save_path) or pd.read_parquet(prediction_save_path, engine='pyarrow').shape[0] == 0:
        print(f'[SLURMCHECK] {datetime.now().strftime("%Y-%m-%d-%H:%M:%S")} Job failed, Did not generate desired output')
        sys.exit(os.EX_SOFTWARE)

    #Your code ends here
    print(f'[SLURMEND] {datetime.now().strftime("%Y-%m-%d-%H:%M:%S")} Job finished successfully')
except Exception as e:
    print(f'[SLURMFAIL] {datetime.now().strftime("%Y-%m-%d-%H:%M:%S")} Job failed, see error below')
    print(repr(e))
    sys.exit(os.EX_SOFTWARE)