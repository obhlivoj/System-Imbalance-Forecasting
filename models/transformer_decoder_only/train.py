import os
import warnings
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from transformer_dataset import TSDataset, prepare_time_series_data, scale_data_seq, create_lags
from model import TransformerDecoderOnly

from config import get_config, get_weights_file_path

import pandas as pd
from tqdm import tqdm
import pickle

import numpy as np
import itertools
from sklearn.model_selection import TimeSeriesSplit

from torch.utils.tensorboard import SummaryWriter

def write_loss(run: int, train_loss: float, val_loss: float):
    folder_path = f'./loss/run{run}/'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    train_file = f'training_loss.txt'
    val_file = f'val_loss.txt'

    paths = {f'{folder_path}{train_file}': train_loss, f'{folder_path}{val_file}': val_loss}
    for path, loss in paths.items():
        with open(path, 'a') as file:
            file.write(str(f'{loss:.2f}')+"\n")

def run_validation(model, device, validation_dataloader):
    model.eval()

    src_input = []
    ground_truth = []
    predicted = []

    with torch.no_grad():
        batch_iterator_val = tqdm(validation_dataloader)
        for batch in batch_iterator_val:
            encoder_input = batch['encoder_input'].to(device)
            model_out = model(encoder_input, device)

            src_data = batch["x_orig"].to(device)
            label = batch["label"].to(device)

            src_input.append(src_data)
            ground_truth.append(label)
            predicted.append(model_out)
            
        gt_torch = torch.cat(ground_truth)
        pred_torch = torch.cat(predicted)
        src_torch = torch.cat(src_input)

        loss_fn = nn.MSELoss()
        loss = loss_fn(pred_torch.view(-1), gt_torch.view(-1))

    return float(loss), gt_torch, pred_torch, src_torch


def get_ds(config, train_bs = None, return_raw=False):
    with open(os.path.join(config["path_pickle"], config["data_pickle_name"]), 'rb') as file:
        ds_raw =  pickle.load(file)
    ds_lags, new_vars = create_lags(ds_raw, config["lags"], config["diffs"])

    # add new lagged variables to the exo_vars in config
    config["exo_vars"] += new_vars
    
    # train-val split
    data_train = ds_lags[lambda x: x.datetime_utc <=
                         pd.Timestamp(config["train_split"])].copy()
    data_train = data_train.fillna(data_train.median(numeric_only=True))

    data_val = ds_lags[lambda x: (pd.Timestamp(config["train_split"]) < x.datetime_utc) & (
        x.datetime_utc <= pd.Timestamp(config["test_split"]))].copy()
    data_val = data_val.fillna(data_train.median(numeric_only=True))

    data_test = ds_lags[lambda x: x.datetime_utc > pd.Timestamp(config["test_split"])].copy()
    data_test = data_test.fillna(data_train.median(numeric_only=True))

    train_data_raw, train_data_tensor, train_label_tensor = prepare_time_series_data(data_train, config["exo_vars"], config["target"], config['tgt_step'], config['src_seq_len'], config['tgt_seq_len'])
    val_data_raw, _, _ = prepare_time_series_data(data_val, config["exo_vars"], config["target"], config['tgt_step'], config['src_seq_len'], config['tgt_seq_len'])
    test_data_raw, _, _ = prepare_time_series_data(data_test, config["exo_vars"], config["target"], config['tgt_step'], config['src_seq_len'], config['tgt_seq_len'])

    data_to_scale = {
        "train": train_data_raw,
        "val": val_data_raw,
        "test": test_data_raw,
    }
    
    data_scaled = scale_data_seq(train_data_tensor, train_label_tensor, data_to_scale)
    train_scl, val_scl, test_scl = data_scaled

    if return_raw:
        return train_scl, val_scl, test_scl

    train_ds = TSDataset(train_scl, config['src_seq_len'], config['tgt_seq_len'])
    val_ds = TSDataset(val_scl, config['src_seq_len'], config['tgt_seq_len'])
    test_ds = TSDataset(test_scl, config['src_seq_len'], config['tgt_seq_len'])
    
    if train_bs:
        train_dataloader = DataLoader(train_ds, batch_size=train_bs, shuffle=True)
    else:   
        train_dataloader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)
    val_dataloader = DataLoader(val_ds, batch_size=config['batch_size']*10, shuffle=False)
    test_dataloader = DataLoader(
        test_ds, batch_size=config['batch_size']*10, shuffle=False)

    # return train_dataloader, val_dataloader
    return train_dataloader, val_dataloader, test_dataloader

def get_model(cfg):
    model = TransformerDecoderOnly(len(cfg["exo_vars"] + cfg["target"]), cfg["src_seq_len"], cfg["tgt_seq_len"],
                                    n_head = cfg['n_head'], Nx = cfg['Nx'], d_ff = cfg['d_ff'], d_d = cfg['d_d'], dropout = cfg['dropout'])
    return model

def train_model(cfg):
    # define the device on which we train the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Make sure the weights folder exists
    Path(cfg['model_folder']).mkdir(parents=True, exist_ok=True)

    train_dataloader, val_dataloader, _ = get_ds(cfg)
    model = get_model(cfg).to(device)

    # Tensorboard
    writer = SummaryWriter(cfg['experiment_name'])

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['lr'], eps=1e-9)

    # If the user specified a model to preload before training, load it
    initial_epoch = 0
    global_step = 0
    if cfg['preload']:
        model_filename = get_weights_file_path(cfg, cfg['preload'])
        print(f'Preloading model {model_filename}')
        state = torch.load(model_filename)
        model.load_state_dict(state['model_state_dict'])
        initial_epoch = state['epoch'] + 1
        optimizer.load_state_dict(state['optimizer_state_dict'])
        global_step = state['global_step']

    loss_fn = nn.MSELoss().to(device)

    for epoch in range(initial_epoch, cfg['num_epochs']):
        epoch_loss = 0
        model.train()
        batch_iterator = tqdm(train_dataloader, desc=f"Processing Epoch {epoch:02d}")
        for batch in batch_iterator:
            encoder_input = batch['encoder_input'].to(device) # (batch, seq_len, n_features)

            # Run the tensor through the encoder and decoder
            output = model(encoder_input, device) # (batch, tgt_seq_len)

            # Compare the output with the label
            label = batch['label'].to(device) # (batch, tgt_seq_len)

            # Compute the loss using MSE
            loss = loss_fn(output.view(-1), label.view(-1))
            epoch_loss += loss.item()
            batch_iterator.set_postfix({"step_loss": f"{loss.item():6.3f}"})

            # Log the loss
            writer.add_scalar('train loss', loss.item(), global_step)
            writer.flush()

            # Backpropagate the loss
            loss.backward()

            # Update the weights
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            global_step += 1
        

        txt_msg = f"Training loss of epoch {epoch}: {epoch_loss/len(train_dataloader)}"
        batch_iterator.write(txt_msg)

        val_loss, _, _, _ = run_validation(model, device, val_dataloader)
        txt_msg = f"Validation loss of epoch {epoch}: {val_loss}"
        batch_iterator.write(txt_msg)

        write_loss(cfg['run'], train_loss=epoch_loss/len(train_dataloader), val_loss=val_loss)


        model_filename = get_weights_file_path(cfg, f"{epoch:02d}")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'global_step': global_step
        }, model_filename)

    return model

def training_loop(cfg, device, train_dataloader, val_dataloader):
    score = []

    model = get_model(cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['lr'], eps=1e-9)
    loss_fn = nn.MSELoss().to(device)

    for _ in range(cfg['num_epochs']):
        epoch_loss = 0
        model.train()
        for batch in train_dataloader:
            encoder_input = batch['encoder_input'].to(device)
            output = model(encoder_input, device)
            label = batch['label'].to(device)

            # Compute the loss using MSE
            loss = loss_fn(output.view(-1), label.view(-1))
            epoch_loss += loss.item()

            # Backpropagate the loss, update the weights
            loss.backward()

            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        val_loss = loop_validation(model, device, val_dataloader)
        score.append(val_loss)

    return score

def loop_validation(model, device, validation_dataloader):
    model.eval()

    ground_truth = []
    predicted = []

    with torch.no_grad():
        for batch in validation_dataloader:
            encoder_input = batch['encoder_input'].to(device)
            model_out = model(encoder_input, device)
            predicted.append(model_out)

            label = batch["label"].to(device)
            ground_truth.append(label)

        gt_torch = torch.cat(ground_truth)
        pred_torch = torch.cat(predicted)

        loss_fn = nn.MSELoss()
        loss = loss_fn(pred_torch.view(-1), gt_torch.view(-1))

    return float(loss)


def grid_search(config, device, lr_cv: float, n_epoch: int, param_grid: dict, n_iter: int = 20, n_split: int = 4, cv_dic: int = 5):
    config["num_epochs"] = n_epoch
    config['lr'] = lr_cv
    train_scl, _, _ = get_ds(config, return_raw=True)

    tscv = TimeSeriesSplit(
        n_splits=n_split, test_size=int(len(train_scl)/cv_dic))
    dataloaders = []
    for i, (train_ind, test_ind) in enumerate(tscv.split(train_scl)):
        train_list = [train_scl[k] for k in train_ind]
        test_list = [train_scl[k] for k in test_ind]

        train_ds = TSDataset(
            train_list, config['src_seq_len'], config['tgt_seq_len'])
        val_ds = TSDataset(
            test_list, config['src_seq_len'], config['tgt_seq_len'])

        train_dataloader = DataLoader(
            train_ds, batch_size=config['batch_size'], shuffle=True)
        val_dataloader = DataLoader(
            val_ds, batch_size=config['batch_size']*10, shuffle=False)

        dataloaders.append({"train": train_dataloader, "test": val_dataloader})

    hyperparameter_combinations = list(itertools.product(*param_grid.values()))
    np.random.shuffle(hyperparameter_combinations)

    param_combinations = [dict(zip(param_grid.keys(), values))
                          for values in hyperparameter_combinations]

    # prepare list to store results, calculate avg score per iter and select the best params
    final_scores = {"avg_score": [], "score_list": [],
                    "params": [], "full_score": []}

    # Number of iterations for random search
    for i in tqdm(range(min(n_iter, len(param_combinations)))):
        selected_hyperparameters = param_combinations[i]
        info = "hyperparams: "
        for k, v in selected_hyperparameters.items():
            config[k] = v
            info += f"{k}: {v}, "

        print(info)
        losses = []
        full_score_list = []
        for i, data in enumerate(dataloaders):
            score = training_loop(config, device, data['train'], data['test'])
            full_score_list.append(score)
            losses.append(np.min(score))

        final_scores['full_score'].append(full_score_list)
        final_scores['avg_score'].append(np.mean(losses))
        final_scores['score_list'].append(losses)
        final_scores['params'].append(selected_hyperparameters)

        print(
            f"Scores: {', '.join([f'{x:.2f}' for x in losses])}; avg score: {np.mean(losses):.2f}")
        print(20*"-")

        best_ind = np.argmin(final_scores['avg_score'])
        best_params, best_score = final_scores['params'][best_ind], final_scores['avg_score'][best_ind]

    return final_scores, best_params, best_score

if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    config = get_config()

    train_model(config)