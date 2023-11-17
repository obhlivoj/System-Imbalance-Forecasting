import os
import warnings
from pathlib import Path
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV

from transformer_dataset import TSDataset, prepare_time_series_data, scale_data_seq, create_lags, get_train_test_set
import xgboost as xgb

from config import get_config

import pandas as pd
from tqdm import tqdm
import pickle

from torch.utils.tensorboard import SummaryWriter

def run_validation():
    pass

def write_loss(run, train_loss: list, val_loss: list):
    folder_path = f'./loss/run{run}/'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    train_file = f'training_loss.txt'
    val_file = f'val_loss.txt'

    paths = {f'{folder_path}{train_file}': train_loss, f'{folder_path}{val_file}': val_loss}
    for path, loss in paths.items():
        with open(path, 'a') as file:
            for loss_item in loss:
                file.write(str(f'{loss_item:.2f}')+"\n")

def get_ds(config):
    with open(os.path.join(config["path_pickle"], config["data_pickle_name"]), 'rb') as file:
        ds_raw =  pickle.load(file)
    ds_lags, new_vars = create_lags(ds_raw, config["lags"], config["diffs"])

    # add new lagged variables to the exo_vars in config
    config["exo_vars"] += new_vars
    
    # train-val split
    training_split = pd.Timestamp('2023-04-01 00:00:00')
    data_train = ds_lags[lambda x: x.datetime_utc <= training_split].copy()
    data_train = data_train.fillna(data_train.median(numeric_only=True))

    data_val = ds_lags[lambda x: x.datetime_utc > training_split].copy()
    data_val = data_val.fillna(data_train.median(numeric_only=True))

    train_data_raw, train_data_tensor, train_label_tensor = prepare_time_series_data(data_train, config["exo_vars"], config["target"], config['tgt_step'], config['src_seq_len'], config['tgt_seq_len'])
    val_data_raw, _, _ = prepare_time_series_data(data_val, config["exo_vars"], config["target"], config['tgt_step'], config['src_seq_len'], config['tgt_seq_len'])

    data_to_scale = {
        "train": train_data_raw,
        "val": val_data_raw,
    }
    
    data_scaled = scale_data_seq(train_data_tensor, train_label_tensor, data_to_scale)
    train_scl, val_scl = data_scaled

    train_ds = TSDataset(train_scl, config['src_seq_len'], config['tgt_seq_len'])
    val_ds = TSDataset(val_scl, config['src_seq_len'], config['tgt_seq_len'])
    
    x_train, y_train, hist_train = get_train_test_set(train_ds)
    x_test, y_test, hist_test = get_train_test_set(val_ds)

    # return train_sets, val_sets
    return x_train, y_train, x_test, y_test, hist_train, hist_test

def get_model(cfg, device):
    params = {
    'objective': 'reg:squarederror',  
    'max_depth': cfg["max_depth"],
    'learning_rate': cfg["lr"],
    'n_estimators': cfg["n_estimators"],
    'early_stopping_rounds': cfg['early_stopping_rounds'],
    'eval_metric': ["mae", "rmse"]
    }
    if device.type == 'cuda':
        model = xgb.XGBRegressor(**params, tree_method="hist", device="cuda")
    else:
        model = xgb.XGBRegressor(**params)
    return model

def train_model(cfg):
    # define the device on which we train the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Make sure the weights folder exists
    Path(cfg['model_folder']).mkdir(parents=True, exist_ok=True)

    x_train, y_train, x_test, y_test, _, _ = get_ds(cfg)
    model = get_model(cfg, device)

    # If the user specified a model to preload before training, load it
    model_name = f"xgb_model_{cfg['tgt_step']+1}.json"
    if cfg['preload']:
        model.load_model(model_name)
        print(f'Preloading model: {model_name}')

    model.fit(x_train, y_train, 
          eval_set=[(x_train, y_train), (x_test, y_test)], verbose=True)

    model.save_model(f'{cfg["model_folder"]}/{model_name}')
    write_loss(cfg['run'], model.evals_result()['validation_0']['rmse'], model.evals_result()['validation_1']['rmse'])

    return model

def grid_search(cfg, device, lr_cv: float, n_cv: int, param_grid: dict, n_iter: int=20, n_split: int=4, cv_dic: int=5):  
    x_train, y_train, _, _, _, _ = get_ds(cfg)
    x_train = x_train[:100]
    y_train = y_train[:100]
    cfg['lr'] = lr_cv
    cfg['n_estimators'] = n_cv
    cfg['early_stopping_rounds'] = None
    model = get_model(cfg, device)

    tscv = TimeSeriesSplit(n_splits=n_split, test_size=int(len(x_train)/cv_dic))
    grid_search = RandomizedSearchCV(estimator=model, cv=tscv,
                                scoring='neg_mean_squared_error',
                                param_distributions=param_grid,
                                n_iter=n_iter, error_score='raise',
                                verbose=0)
    grid_search.fit(x_train, y_train)

    return grid_search

if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    config = get_config()

    train_model(config)