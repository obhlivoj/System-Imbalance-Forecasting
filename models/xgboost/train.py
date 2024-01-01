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

    paths = {f'{folder_path}{train_file}': train_loss,
             f'{folder_path}{val_file}': val_loss}
    for path, loss in paths.items():
        with open(path, 'a') as file:
            for loss_item in loss:
                file.write(str(f'{loss_item:.2f}')+"\n")


def get_ds(config, return_raw=False):
    with open(os.path.join(config["path_pickle"], config["data_pickle_name"]), 'rb') as file:
        ds_raw = pickle.load(file)
    ds_lags, new_vars = create_lags(ds_raw, config["lags"], config["diffs"])

    # add new lagged variables to the exo_vars in config
    config["exo_vars"] += new_vars
    if config["forward_vars"]:
        config["forward_vars"] += new_vars

    # train-val split
    data_train = ds_lags[lambda x: x.datetime_utc <=
                         pd.Timestamp(config["train_split"])].copy()
    data_train = data_train.fillna(data_train.median(numeric_only=True))

    data_val = ds_lags[lambda x: (pd.Timestamp(config["train_split"]) < x.datetime_utc) & (
        x.datetime_utc <= pd.Timestamp(config["test_split"]))].copy()
    data_val = data_val.fillna(data_train.median(numeric_only=True))

    data_test = ds_lags[lambda x: x.datetime_utc >
                        pd.Timestamp(config["test_split"])].copy()
    data_test = data_test.fillna(data_train.median(numeric_only=True))

    train_data_raw, train_data_tensor = prepare_time_series_data(
        data_train, config)
    val_data_raw, _ = prepare_time_series_data(data_val, config)
    test_data_raw, _ = prepare_time_series_data(data_test, config)

    data_to_scale = {
        "train": train_data_raw,
        "val": val_data_raw,
        "test": test_data_raw,
    }

    data_scaled = scale_data_seq(config, train_data_tensor, data_to_scale)
    train_scl, val_scl, test_scl = data_scaled

    if return_raw:
        return train_scl, val_scl, test_scl

    train_ds = TSDataset(
        train_scl, config['src_seq_len'], config['tgt_seq_len'])
    val_ds = TSDataset(val_scl, config['src_seq_len'], config['tgt_seq_len'])
    test_ds = TSDataset(test_scl, config['src_seq_len'], config['tgt_seq_len'])

    train = get_train_test_set(train_ds)
    val = get_train_test_set(val_ds)
    test = get_train_test_set(test_ds)

    # return train_sets, val_sets
    return train, val, test


def get_model(cfg, device):
    params = {
        'objective': 'reg:squarederror',
        'max_depth': cfg["max_depth"],
        'learning_rate': cfg["lr"],
        'n_estimators': cfg["n_estimators"],
        'reg_alpha': cfg['reg_alpha'],
        'reg_lambda': cfg['reg_lambda'],
        'subsample': cfg['subsample'],
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

    train, val, _ = get_ds(cfg)
    x_train, y_train, _ = train
    x_val, y_val, _ = val

    model = get_model(cfg, device)

    # If the user specified a model to preload before training, load it
    model_name = f"xgb_model_{cfg['tgt_step']+1}.json"
    if cfg['preload']:
        model.load_model(model_name)
        print(f'Preloading model: {model_name}')

    model.fit(x_train, y_train,
              eval_set=[(x_train, y_train), (x_val, y_val)], verbose=True)

    model.save_model(f'{cfg["model_folder"]}/{model_name}')
    write_loss(cfg['run'], model.evals_result()['validation_0']
               ['rmse'], model.evals_result()['validation_1']['rmse'])

    return model


def grid_search(cfg, device, lr_cv, n_cv: int, param_grid: dict, n_iter: int = 20, n_split: int = 4, cv_dic: int = 5):
    train, val, _ = get_ds(cfg)
    x_train0, y_train0, _ = train
    x_val0, y_val0, _ = val
    x_train = np.concatenate((x_train0, x_val0))
    y_train = np.concatenate((y_train0, y_val0))
    if lr_cv:
        cfg['lr'] = lr_cv
    cfg['n_estimators'] = n_cv
    cfg['early_stopping_rounds'] = None
    model = get_model(cfg, device)

    tscv = TimeSeriesSplit(
        n_splits=n_split, test_size=int(len(x_train)/cv_dic))
    grid_search = RandomizedSearchCV(estimator=model, cv=tscv,
                                     scoring='neg_mean_squared_error',
                                     param_distributions=param_grid,
                                     n_iter=n_iter, error_score='raise',
                                     verbose=10)
    grid_search.fit(x_train, y_train)

    return grid_search


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    config = get_config()

    train_model(config)
