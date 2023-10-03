import pandas as pd

from typing import Any, Dict, List, Union
import torch
from torch.utils.data import Dataset

from sklearn.preprocessing import StandardScaler
from copy import deepcopy


class TSDataset(Dataset):

    def __init__(self, ds: List[dict], src_seq_len: int, tgt_seq_len: int) -> None:
        super().__init__()
        self.ds = ds
        self.src_seq_len = src_seq_len
        self.tgt_seq_len = tgt_seq_len

    def __len__(self):
        return len(self.ds)
    
    def __getitem__(self, index: Any) -> Any:
        datapoint = self.ds[index]
        enc_input = datapoint['x_input']
        dec_input = datapoint['target_decoder']
        label = datapoint['y_true']
        x_orig = datapoint["target_history"]

        assert enc_input.size(0) == self.src_seq_len
        assert dec_input.size(0) == self.tgt_seq_len
        assert label.size(0) == self.tgt_seq_len

        return {
            "encoder_input": enc_input,  # (src_seq_len, n_features)
            "decoder_input": dec_input,  # (tgt_seq_len, n_tgt)
            "encoder_mask":  torch.ones(1, self.src_seq_len, self.src_seq_len).bool(), # (1, src_seq_len, src_seq_len), cannot be set to None because it is handled poorly by dataloader
            "decoder_mask": causal_mask(dec_input.size(0)), # (1, tgt_seq_len, tgt_seq_len),
            "label": label, # (tgt_seq_len, n_tgt)
            "x_orig": x_orig # (src_seq_len, n_features)
        }
    
    def collate_fn(self, batch):
        # Handle None values for encoder_mask
        encoder_mask = [item["encoder_mask"] for item in batch]
        encoder_mask = torch.stack(encoder_mask) if None not in encoder_mask else None
        
        # Stack other tensors
        other_tensors = {
            key: torch.stack([item[key] for item in batch]) for key in batch[0].keys() if key != "encoder_mask"
        }
        
        return {
            "encoder_input": other_tensors["encoder_input"],
            "decoder_input": other_tensors["decoder_input"],
            "encoder_mask": encoder_mask,
            "decoder_mask": other_tensors["decoder_mask"],
            "label": other_tensors["label"],
            "x_orig": other_tensors["x_orig"]
        }

# returns a triangular mask to protect the decoder to get information from future
def causal_mask(size):
    mask = torch.triu(torch.ones(1, size, size), diagonal=1).type(torch.int)
    return mask == 0

def prepare_time_series_data(data: pd.DataFrame, exo_vars: list, target: list, input_seq_len: int, target_seq_len: int):
    data_array = data[target + exo_vars].values
    data_label = data[target].values

    data_tensor = torch.tensor(data_array, dtype=torch.float32)
    label_tensor = torch.tensor(data_label, dtype=torch.float32)

    data_seq = []

    num_obs = len(data_tensor)
    max_start_idx = num_obs - input_seq_len - target_seq_len

    for start_idx in range(max_start_idx):
        end_idx = start_idx + input_seq_len
        x_input_seq = data_tensor[start_idx:end_idx]
        label_input_seq = label_tensor[start_idx:end_idx]
        
        target_start_idx = end_idx - 1
        target_end_idx = target_start_idx + target_seq_len
        target_dec_seq = label_tensor[target_start_idx:target_end_idx]
        ground_truth = label_tensor[target_start_idx+1:target_end_idx+1]

        data_seq.append({
        "x_input" : x_input_seq,
        "target_decoder" : target_dec_seq,
        "y_true" : ground_truth,
        "target_history" : label_input_seq,
        })

    return data_seq, data_tensor, label_tensor

def scale_data_seq(data_tensor: torch.Tensor, label_tensor: torch.Tensor, train_ds_index: int, data_to_scale: dict):
    scaled_data = deepcopy(data_to_scale)
    
    enc_scaler = StandardScaler()
    dec_scaler = StandardScaler()

    enc_scaler.fit(data_tensor[:train_ds_index])
    dec_scaler.fit(label_tensor[:train_ds_index])

    for name, dt in data_to_scale.items():
        for ind, obs in enumerate(dt):
            scaled_data[name][ind]['x_input'] = torch.tensor(enc_scaler.transform(obs['x_input']), dtype=torch.float32)
            scaled_data[name][ind]['target_decoder'] = torch.tensor(dec_scaler.transform(obs['target_decoder']), dtype=torch.float32)

    return scaled_data['train'], scaled_data['val'], scaled_data['test'], dec_scaler

# create lags and lagged diffs
def create_lags(df: pd.DataFrame, lags_dict: Union[None, Dict[str, List[int]]] = None, lagged_difs: Union[None, Dict[str, List[int]]] = None):
    """
    Create lagged variables and lagged differences for specified variables.

    Parameters:
    - df: Pandas DataFrame containing the data.
    - lags_dict: A dictionary where keys are variable names, and values are lists of integer lag values.
      If lags_dict is provided, lagged variables will be created for the specified variables.
    - lagged_difs: A dictionary where keys are variable names, and values are lists of integer lag values.
      If lagged_difs is provided, lagged differences will be created for the specified variables.

    Returns:
    - Pandas DataFrame with lagged variables and lagged differences added as new columns.
    """
    new_vars = []
    data = df.copy()
    if lags_dict:
        for variable_name, lag_values in lags_dict.items():
            for num_lag in lag_values:
                new_column_name = f"{variable_name}_lag{num_lag}"
                new_vars.append(new_column_name)
                data[new_column_name] = data[variable_name].shift(num_lag)

    if lagged_difs:
        for variable_name, lag_values in lagged_difs.items():
            for num_lag in lag_values:
                new_column_name = f"{variable_name}_lag_diff{num_lag}"
                new_vars.append(new_column_name)
                data[new_column_name] = data[variable_name].diff(1).shift(num_lag-1)

    return data.dropna(), new_vars