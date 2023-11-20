import pandas as pd

from typing import Any, Dict, List, Tuple, Union
import torch
from torch.utils.data import Dataset

from sklearn.preprocessing import StandardScaler
from copy import deepcopy


class TSDataset(Dataset):
    """
    Custom PyTorch Dataset for Time Series data.

    Parameters:
    - ds (List[dict]): List of dictionaries containing time series data.
    - src_seq_len (int): Length of the source sequence.
    - tgt_seq_len (int): Length of the target sequence.

    Returns a dictionary with the following keys:
    - "encoder_input": Tensor of encoder input data.
    - "label": Tensor of target labels.
    - "x_orig": Tensor of the original data.
    """

    def __init__(self, ds: List[dict], src_seq_len: int, tgt_seq_len: int) -> None:
        super().__init__()
        self.ds = ds
        self.src_seq_len = src_seq_len
        self.tgt_seq_len = tgt_seq_len

    def __len__(self) -> int:
        return len(self.ds)
    
    def __getitem__(self, index: int) -> dict:
        datapoint = self.ds[index]
        enc_input = datapoint['x_input']
        label = datapoint['y_true']
        x_orig = datapoint["target_history"]

        assert label.size(0) == self.tgt_seq_len

        return {
            "encoder_input": enc_input,  # (src_seq_len, n_features)
            "label": label, # (tgt_seq_len, n_tgt)
            "x_orig": x_orig # (src_seq_len, n_features)
        }
    
    def collate_fn(self, batch: List[dict]) -> dict:
        # Handle None values for encoder_mask
        encoder_mask = [item["encoder_mask"] for item in batch]
        encoder_mask = torch.stack(encoder_mask) if None not in encoder_mask else None
        
        # Stack other tensors
        other_tensors = {
            key: torch.stack([item[key] for item in batch]) for key in batch[0].keys() if key != "encoder_mask"
        }
        
        return {
            "encoder_input": other_tensors["encoder_input"],
            "label": other_tensors["label"],
            "x_orig": other_tensors["x_orig"]
        }

def prepare_time_series_data(data: pd.DataFrame, cfg) -> Tuple[List[dict], torch.Tensor, torch.Tensor]:
    """
    Prepare time series data for modeling.

    Parameters:
    - data (pd.DataFrame): Pandas DataFrame containing the data.
    - exo_vars (list): List of exogenous variables.
    - target (list): List of target variables.
    - input_seq_len (int): Length of the input sequence.
    - target_seq_len (int): Length of the target sequence.

    Returns:
    - data_seq (list): List of dictionaries containing prepared data sequences.
    - data_tensor (Tensor): Tensor of the entire data array.
    - label_tensor (Tensor): Tensor of the target labels.
    """
    data_array = data[cfg['target'] + cfg['exo_vars']].values
    data_label = data[cfg['target']].values

    data_tensor = torch.tensor(data_array, dtype=torch.float32)
    label_tensor = torch.tensor(data_label, dtype=torch.float32)

    if cfg['forward_lags']:
        data_fl = data[cfg['target'] + cfg["forward_vars"]].values
        data_fl_tensor = torch.tensor(data_fl, dtype=torch.float32)

    data_seq = []

    num_obs = len(data_tensor)
    max_start_idx = num_obs - cfg['src_seq_len'] - cfg['tgt_seq_len'] - cfg['tgt_step'] + 1

    for start_idx in range(max_start_idx):
        end_idx = start_idx + cfg['src_seq_len']
        end_label_idx = end_idx + cfg['tgt_seq_len'] + cfg['tgt_step']

        x_input_seq = data_tensor[start_idx:end_idx]
        x_forward_lag = data_fl_tensor[end_idx:end_label_idx]
        label_input_seq = label_tensor[start_idx:end_idx]
        
        ground_truth = label_tensor[end_idx + cfg['tgt_step']:end_label_idx]

        data_seq.append({
        "x_input_raw" : x_input_seq,
        "x_forward_lag" : x_forward_lag,
        "y_true" : ground_truth,
        "target_history" : label_input_seq,
        })

    return data_seq, data_tensor

def scale_data_seq(cfg, data_tensor: torch.Tensor, data_to_scale: Dict[str, List[dict]]) -> Tuple[List[dict], List[dict], List[dict], StandardScaler]:
    """
    Scale time series data sequences.

    Parameters:
    - data_tensor (torch.Tensor): Tensor of the training data array.
    - label_tensor (torch.Tensor): Tensor of the training labels.
    - train_ds_index (int): Index separating training and validation/test data.
    - data_to_scale (dict): Dictionary of data sequences to scale.

    Returns:
    - train_data (list): Scaled training data sequences.
    - val_data (list): Scaled validation data sequences.
    - test_data (list): Scaled test data sequences.
    """
    scaled_data = deepcopy(data_to_scale)
    
    enc_scaler = StandardScaler()
    enc_scaler.fit(data_tensor)

    for name, dt in data_to_scale.items():
        for ind, obs in enumerate(dt):
            x_in = torch.tensor(enc_scaler.transform(obs['x_input_raw']), dtype=torch.float32)
            x_fl = torch.tensor(enc_scaler.transform(obs['x_forward_lag']), dtype=torch.float32)
            scaled_data[name][ind]['x_input'] = torch.concat((x_in.flatten(), x_fl[:,len(cfg['target']):].flatten()))

    return [scaled_data[name] for name in data_to_scale.keys()]

# create lags and lagged diffs
def create_lags(df: pd.DataFrame, lags_dict: Union[None, Dict[str, List[int]]] = None, lagged_difs: Union[None, Dict[str, List[int]]] = None) -> Tuple[pd.DataFrame, List[str]]:
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

    return data.dropna(subset=new_vars), new_vars