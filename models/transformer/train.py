import os
import warnings
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr_scheduler


from transformer_dataset import TSDataset, causal_mask, prepare_time_series_data, scale_data_seq, create_lags
from model import build_transformer

from config import get_config, get_weights_file_path

import pandas as pd
from tqdm import tqdm
import pickle

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

def greedy_decode(model, config, source, source_mask, decoder_in, scaler, device):

    # precompute the encoder output and reuse it for every token we get from the decoder
    encoder_output = model.encode(source, source_mask)
    decoder_input = decoder_in[:,0:1,:].type_as(source).to(device)
                                                                                                                                                                                                                                                                                                                               
    for _ in range(config['val_seq_len']):

        # build a mask for the target (decoder input)
        decoder_mask = causal_mask(decoder_input.size(1)).type_as(source_mask).to(device)
        # calculate the output of the decoder
        out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)
        # get the next token
        pred = model.project(out)
        pred_new = pred[:,-1,-1]
        scaled_pred = torch.tensor(scaler.transform(pred_new.view(-1,1)))
        decoder_input = torch.cat([decoder_input, scaled_pred.unsqueeze(2).type_as(source).to(device)], dim=1)

    # prediction "pred" should equal to "decoder_input[1:]", since there is only added the first initialization value
    return pred


def run_validation(model, config, validation_dataloader, scaler, device, print_msg, global_step, writer, epoch):
    model.eval()

    src_input = []
    ground_truth = []
    predicted = []

    with torch.no_grad():
        batch_iterator_val = tqdm(validation_dataloader)
        for batch in batch_iterator_val:
            encoder_input = batch['encoder_input'].to(device)
            encoder_mask = batch['encoder_mask'].to(device)
            decoder_input = batch['decoder_input'].to(device)

            model_out = greedy_decode(model, config, encoder_input, encoder_mask, decoder_input, scaler, device)

            src_data = batch["x_orig"]
            label = batch["label"]
            output = model_out.detach().cpu()

            src_input.append(src_data)
            ground_truth.append(label)
            predicted.append(output)


        gt_torch = torch.cat(ground_truth)
        pred_torch = torch.cat(predicted)

        loss_fn = nn.MSELoss()
        loss = loss_fn(pred_torch.view(-1), gt_torch.view(-1))

    txt_msg = f"Validation loss of epoch {epoch}: {loss}"
    print_msg(txt_msg)

    return float(loss), ground_truth, predicted, src_input


def get_ds(config, train_bs = None):
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

    train_data_raw, train_data_tensor, train_label_tensor = prepare_time_series_data(data_train, config["exo_vars"], config["target"], config['src_seq_len'], config['tgt_seq_len'])
    val_data_raw, _, _ = prepare_time_series_data(data_val, config["exo_vars"], config["target"], config['src_seq_len'], config['tgt_seq_len'])


    data_to_scale = {
        "train": train_data_raw,
        "val": val_data_raw,
    }
    
    data_scaled, label_scaler = scale_data_seq(train_data_tensor, train_label_tensor, data_to_scale)
    train_scl, val_scl = data_scaled

    train_ds = TSDataset(train_scl, config['src_seq_len'], config['tgt_seq_len'])
    val_ds = TSDataset(val_scl, config['src_seq_len'], config['tgt_seq_len'])
    
    if train_bs:
        train_dataloader = DataLoader(train_ds, batch_size=train_bs, shuffle=True)
    else:   
        train_dataloader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)
    val_dataloader = DataLoader(val_ds, batch_size=config['batch_size']*10, shuffle=False)
    val_dataloader_onebatch = DataLoader(val_ds, batch_size=1, shuffle=True)

    # return train_dataloader, val_dataloader
    return train_dataloader, val_dataloader, label_scaler, val_dataloader_onebatch

def get_model(config):
    model = build_transformer(d_model=len(config['exo_vars'])+1, d_ff = config["d_ff"])
    return model

def train_model(config):
    # define the device on which we train the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Make sure the weights folder exists
    Path(config['model_folder']).mkdir(parents=True, exist_ok=True)

    train_dataloader, val_dataloader, scaler, _ = get_ds(config)
    model = get_model(config).to(device)

    # Tensorboard
    writer = SummaryWriter(config['experiment_name'])

    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], eps=1e-9)
    scheduler = lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.01, total_iters=8)

    # If the user specified a model to preload before training, load it
    initial_epoch = 0
    global_step = 0
    if config['preload']:
        model_filename = get_weights_file_path(config, config['preload'])
        print(f'Preloading model {model_filename}')
        state = torch.load(model_filename)
        model.load_state_dict(state['model_state_dict'])
        initial_epoch = state['epoch'] + 1
        optimizer.load_state_dict(state['optimizer_state_dict'])
        global_step = state['global_step']

    loss_fn = nn.MSELoss().to(device)

    for epoch in range(initial_epoch, config['num_epochs']):
        epoch_loss = 0
        model.train()
        batch_iterator = tqdm(train_dataloader, desc=f"Processing Epoch {epoch:02d}")
        for batch in batch_iterator:
            encoder_input = batch['encoder_input'].to(device) # (batch, seq_len)
            decoder_input = batch['decoder_input'].to(device) # (batch, seq_len)
            encoder_mask = batch['encoder_mask'].to(device) # (batch, 1, 1, seq_len)
            decoder_mask = batch['decoder_mask'].to(device) # (batch, 1, seq_len, seq_len)

            # Run the tensors through the encoder, decoder and the projection layer
            encoder_output = model.encode(encoder_input, encoder_mask) # (batch, seq_len, d_model)
            decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask) # (batch, seq_len, d_model)
            proj_output = model.project(decoder_output) # (batch, seq_len, tgt_vocab_size)

            # Compare the output with the label
            label = batch['label'].to(device) # (batch, seq_len)

            # Compute the loss using MSE
            loss = loss_fn(proj_output.view(-1), label.view(-1))
            epoch_loss += loss.item()
            batch_iterator.set_postfix({"loss": f"{loss.item():6.3f}"})

            # Log the loss
            writer.add_scalar('train loss', loss.item(), global_step)
            writer.flush()

            # Backpropagate the loss
            loss.backward()

            # Update the weights
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            global_step += 1
        
        before_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()
        after_lr = optimizer.param_groups[0]["lr"]
        print("Epoch %d: Adam lr %.6f -> %.6f" % (epoch, before_lr, after_lr))


        txt_msg = f"Training loss of epoch {epoch}: {epoch_loss/len(train_dataloader)}"
        batch_iterator.write(txt_msg)

        val_loss, _, _, _ = run_validation(model, config, val_dataloader, scaler, device, lambda msg: batch_iterator.write(msg), global_step, writer, epoch)

        write_loss(config['run'], train_loss=epoch_loss/len(train_dataloader), val_loss=val_loss)

        # Run validation at the end of every epoch
        # run_validation(model, val_dataloader, tokenizer_src, tokenizer_tgt, config['seq_len'], device, lambda msg: batch_iterator.write(msg), global_step, writer)

        # Save the model at the end of every epoch
        #if val_loss < best_loss:
        #    best_loss = val_loss
        if True:
            model_filename = get_weights_file_path(config, f"{epoch:02d}")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'global_step': global_step
            }, model_filename)

if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    config = get_config()

    train_model(config)