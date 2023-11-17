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
    - "decoder_input": Tensor of decoder input data.
    - "encoder_mask": Tensor representing the encoder mask.
    - "decoder_mask": Tensor representing the decoder mask.
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
        dec_input = datapoint['target_decoder']
        label = datapoint['y_true']
        x_orig = datapoint["target_history"]
        enc_input_raw = datapoint['encoder_input_raw']
        label_scaled = datapoint['y_true_scaled']
        dec_input_raw = datapoint["decoder_input_raw"]

        assert enc_input.size(0) == self.src_seq_len
        assert dec_input.size(0) == self.tgt_seq_len
        assert label.size(0) == self.tgt_seq_len

        return {
            "encoder_input": enc_input,
            "decoder_input": dec_input,
            # cannot be set to None because it is handled poorly by dataloader
            "encoder_mask":  torch.ones(1, self.src_seq_len, self.src_seq_len).bool(),
            "decoder_mask": causal_mask(dec_input.size(0)),
            "label": label,
            "x_orig": x_orig,
            "enc_input_raw": enc_input_raw,
            "label_scaled": label_scaled,
            "dec_input_raw": dec_input_raw,
        }

    def collate_fn(self, batch: List[dict]) -> dict:
        # Handle None values for encoder_mask
        encoder_mask = [item["encoder_mask"] for item in batch]
        encoder_mask = torch.stack(
            encoder_mask) if None not in encoder_mask else None

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


def causal_mask(size: int) -> torch.Tensor:
    """
    Returns a triangular mask to protect the decoder from getting information from the future.

    Parameters:
    - size (int): Size of the mask.

    Returns:
    - Tensor: Triangular mask tensor.
    """
    mask = torch.triu(torch.ones(1, size, size), diagonal=1).type(torch.int)
    return mask == 0


def prepare_time_series_data(data: pd.DataFrame, exo_vars_enc: List[str], exo_vars_dec: List[str], target: List[str], input_seq_len: int, target_seq_len: int) -> Tuple[List[dict], torch.Tensor, torch.Tensor]:
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
    data_array_encoder = data[target + exo_vars_enc].values
    data_array_decoder = data[target + exo_vars_dec].values
    data_label = data[target].values

    data_tensor_enc = torch.tensor(data_array_encoder, dtype=torch.float32)
    data_tensor_dec = torch.tensor(data_array_decoder, dtype=torch.float32)
    label_tensor = torch.tensor(data_label, dtype=torch.float32)

    data_seq = []

    num_obs = len(label_tensor)
    max_start_idx = num_obs - input_seq_len - target_seq_len

    for start_idx in range(max_start_idx):
        end_idx = start_idx + input_seq_len
        x_input_seq = data_tensor_enc[start_idx:end_idx]
        label_input_seq = label_tensor[start_idx:end_idx]

        target_start_idx = end_idx - 1
        target_end_idx = target_start_idx + target_seq_len
        target_dec_seq = data_tensor_dec[target_start_idx:target_end_idx]
        ground_truth = label_tensor[target_start_idx+1:target_end_idx+1]

        data_seq.append({
            "encoder_input_raw": x_input_seq,
            "decoder_input_raw": target_dec_seq,
            "y_true": ground_truth,
            "target_history": label_input_seq,
        })

    return data_seq, data_tensor_enc, data_tensor_dec, label_tensor


def scale_data_seq(data_tensor: torch.Tensor, label_tensor: torch.Tensor, data_to_scale: Dict[str, List[dict]]) -> Tuple[List[dict], List[dict], List[dict], StandardScaler]:
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
    - dec_scaler: Scaler used for scaling the target decoder.
    """
    scaled_data = deepcopy(data_to_scale)

    enc_scaler = StandardScaler()
    label_scaler = StandardScaler()

    enc_scaler.fit(data_tensor)
    label_scaler.fit(label_tensor)

    for name, dt in data_to_scale.items():
        for ind, obs in enumerate(dt):
            scaled_data[name][ind]['x_input'] = torch.tensor(
                enc_scaler.transform(obs['encoder_input_raw']), dtype=torch.float32)
            scaled_data[name][ind]['target_decoder'] = torch.tensor(
                enc_scaler.transform(obs['decoder_input_raw']), dtype=torch.float32)
            scaled_data[name][ind]['y_true_scaled'] = torch.tensor(
                label_scaler.transform(obs['y_true']), dtype=torch.float32)

    return [scaled_data[name] for name in data_to_scale.keys()], label_scaler


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
                data[new_column_name] = data[variable_name].diff(
                    1).shift(num_lag-1)

    return data.dropna(subset=new_vars), new_vars
