import os
import torch
import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from train import get_model, get_ds, run_validation
from config import get_weights_file_path, get_config
from train import get_model

from typing import List, Tuple, Dict, Union

from operator import itemgetter


def loss_se(predicted: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """
    Calculate the squared error between two tensors.

    Args:
        predicted (torch.Tensor): The predicted tensor.
        gt (torch.Tensor): The ground truth tensor.

    Returns:
        torch.Tensor: The squared error between the predicted and ground truth tensors.
    """
    if predicted.shape != gt.shape:
        raise ValueError("Tensors must have the same shape.")

    squared_diff = (predicted - gt) ** 2

    return squared_diff

def plot_k_results(cfg: dict, gt: torch.Tensor, pred: torch.Tensor, src_input: torch.Tensor, inds: list) -> None:
    """
    Plot the ground truth and predicted results.

    Args:
        cfg (dict): Configuration dictionary.
        gt (torch.Tensor): Ground truth tensor.
        pred (torch.Tensor): Predicted tensor.
        src_input (torch.Tensor): Source input tensor.
        inds (list): List of indices.

    Returns:
        None
    """
    history = [i for i in range(0, cfg['src_seq_len'])]
    future = [i for i in range(cfg['src_seq_len'], cfg['src_seq_len'] + cfg['tgt_seq_len'])]

    for gt_tensor, pred_tensor, src in zip(gt[inds], pred[inds], src_input[inds]):
        plt.figure()

        plt.plot(history, src, label="History", color='g')
        if cfg['tgt_seq_len'] == 1:
            plt.plot(future, gt_tensor, label="Ground Truth", marker='o', color='b')
            plt.plot(future, pred_tensor, label="Predicted", marker='o', color='r')
        else:
            plt.plot(future, gt_tensor, label="Ground Truth", color='b')
            plt.plot(future, pred_tensor, label="Predicted", color='r')

        # Add labels and legend
        plt.xlabel("Index")
        plt.ylabel("Value")
        plt.legend()

        # Show the plot
        plt.show()

def compute_metrics(model, cfg, train_dataloader, val_dataloader, label_scaler, device, result_dict):
    """
    Compute various metrics for the model's performance on the training and validation datasets.

    Args:
        model: The machine learning model.
        cfg (dict): Configuration dictionary.
        train_dataloader: Dataloader for training data.
        val_dataloader: Dataloader for validation data.
        label_scaler: Scaler for labels.
        device: Device for training (e.g., 'cuda' or 'cpu').
        result_dict (dict): Dictionary to store the computed metrics.

    Returns:
        None
    """
    for tv, dataloader in zip(["train", "val"], [train_dataloader, val_dataloader]):
        # calculate validation data
        loss, ground_truth, predicted, _ = run_validation(model, cfg, dataloader, label_scaler, device, lambda msg: print(msg), 0, None, 0)
        ground_truth_tensor, predicted_tensor = torch.cat(ground_truth), torch.cat(predicted)
        
        result_dict[f"{tv}_total_loss"].append(loss)

        # get loss for each datapoint
        se_loss_raw = loss_se(predicted_tensor, ground_truth_tensor)
        for ind, error in enumerate(se_loss_raw.mean(dim=0)):
            result_dict[f"{tv}_forecast_{f'0{ind+1}'[-2:]}"].append(error)

        # get score metrics
        result_dict[f"{tv}_metrics"].append({
            "r2": r2_score(ground_truth_tensor.squeeze(1,2), predicted_tensor.squeeze(1,2)),
            "rmse": mean_squared_error(ground_truth_tensor.squeeze(1,2), predicted_tensor.squeeze(1,2), squared=False),
            "mae": mean_absolute_error(ground_truth_tensor.squeeze(1,2), predicted_tensor.squeeze(1,2)),
        })

def create_dict(cfg):
    """
    Create a dictionary to store prediction results.

    Args:
        cfg (dict): Configuration dictionary.

    Returns:
        dict: Dictionary with keys for storing prediction results.
    """
    pred_dict = {
        "train_total_loss": [],
        "val_total_loss": [],
        "train_metrics": [],
        "val_metrics": [],
    }
    for i in range(cfg["val_seq_len"]):
        pred_dict[f"train_forecast_{f'0{i+1}'[-2:]}"] = []
        pred_dict[f"val_forecast_{f'0{i+1}'[-2:]}"] = []

    return pred_dict

def timestep_prediction_loss(cfg) -> dict:
    """
    Perform timestep prediction loss calculation.

    Args:
        cfg (dict): Configuration dictionary.

    Returns:
        dict: Dictionary containing prediction results.
    """
    pred_dict = create_dict(cfg)
    model_names = [f'0{i}'[-2:] for i in range(100)]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dataloader, val_dataloader, label_scaler, _ = get_ds(cfg, train_bs=1024)
    model = get_model(cfg).to(device)

    num_models = len(os.listdir(cfg['model_folder']))
    assert num_models <= len(model_names)

    for k in range(num_models):
        model_filename = get_weights_file_path(cfg, model_names[k])
        state = torch.load(model_filename)
        model.load_state_dict(state['model_state_dict'])
    
        compute_metrics(model, cfg, train_dataloader, val_dataloader, label_scaler, device, pred_dict)

    return pred_dict

def get_best_model(cfg, path_pre: str, num_models: int = 8) -> Tuple[List[float], List[int], Dict[str, Dict[str, List[float]]]]:
    """
    Get the best model metrics and data.

    Args:
        path_pre (str): The path prefix for model data.
        num_models (int): The number of models to consider (default is 8).

    Returns:
        Tuple[List[float], List[int], Dict[str, Dict[str, List[float]]]: A tuple containing:
        - A list of best model metrics (minimum values).
        - A list of indices of the best models.
        - A dictionary with model data for each step, including training and validation metrics.
    """
    best_models_inds = []
    best_metrics = []
    data_dict = {}
    for k in range(1, num_models + 1):
        cfg['run'] = k

        gen_path = f'{path_pre}{cfg["run"]}/'
        loss_paths = [f'{gen_path}training_loss.txt', f'{gen_path}val_loss.txt']

        files = {f'train_{k}': None, f'val_{k}': None}
        for file_path, loss_key in zip(loss_paths, files.keys()):
            with open(file_path, 'r') as file:
                lines = file.readlines()
                float_lines = list(map(float, lines))
                if loss_key == f'val_{k}':
                    best_metrics.append(np.min(float_lines))
                    best_models_inds.append(np.argmin(float_lines))
                files[loss_key] = float_lines

        data_dict[f'step_{k}'] = files

    return best_metrics, best_models_inds, data_dict

def validate_n_models(device, path_pre: str, best_models_inds: List[int], num_models: int = 8) -> None: 
    """
    Validate a set of models and print their errors.

    Args:
        path_pre (str): The path prefix for model data.
        best_models_inds (List[int]): List of indices of the best models.
        num_models (int): The number of models to validate (default is 8).

    Returns:
        None
    """
    loss_cat = []
    loss_validation = []
    preds_gt = {"preds": [], "gt": [], "hist": []}
    for k in range(1, num_models + 1):
        cfg = get_config()
        # data updates
        cfg["tgt_step"] = k - 1
        _, val_dataloader = get_ds(cfg, train_bs=1024)
        model = get_model(cfg).to(device)
        # config updates
        cfg['run'] = f"{path_pre}{k}"
        cfg['model_folder'] += cfg['run']
        cfg['experiment_name'] = f"runs{cfg['run']}/tmodel"

        # Load the pretrained weights
        ind = str(best_models_inds[k - 1])
        str_ind = "00"[:-len(ind)] + ind
        model_filename = get_weights_file_path(cfg, str_ind)
        state = torch.load(model_filename)
        model.load_state_dict(state['model_state_dict'])

        print(f"MODEL EVAL - #{k}")
        print(20 * "-")

        # validation
        loss, ground_truth_tensor, predicted_tensor, src_tensor = run_validation(model, device, val_dataloader, None, 0)
        preds_gt["preds"].append(predicted_tensor)
        preds_gt["gt"].append(ground_truth_tensor)
        preds_gt["hist"].append(src_tensor)
        # get loss 
        se_loss_val = loss_se(predicted_tensor, ground_truth_tensor.squeeze(-1))
        loss_validation.append(loss)
        loss_cat.append(se_loss_val)
        print(20 * "-")

    print("Time-step\tError")
    for ind, error in enumerate(loss_validation):
        print(f"{ind + 1}\t\t{float(error):.2f}")

    return loss_validation, loss_cat, preds_gt

def group_data(data, num_models: int = 8):
    seq_data = []
    for ind in range(len(data['gt'][-1])):
        seq_data.append(
            {
                "hist": data['hist'][-1].squeeze(-1).cpu()[ind].numpy(),
                "pred": np.array([data['preds'][i].cpu()[ind].item() for i in range(num_models)]),
                "true": np.array([data['gt'][i].cpu()[ind].item() for i in range(num_models)])
            }
        )

    return seq_data

def plot_results(seq_datapoint: dict, num_models: int, history_len: int) -> plt.figure:
    d0 = [i for i in range(history_len)]
    d = [i for i in range(history_len, history_len+num_models)]

    fig, ax = plt.subplots()
    ax.plot(d0, seq_datapoint['hist'], label="History", color='g')

    if num_models == 1:
        ax.plot(d, seq_datapoint['true'], label="Ground Truth", marker='o', color='b')
        ax.plot(d, seq_datapoint['pred'], label="Predicted", marker='o', color='r')
    else:
        ax.plot(d, seq_datapoint['true'], label="Ground Truth", color='b')
        ax.plot(d, seq_datapoint['pred'], label="Predicted", color='r')

    # Add loss
    rmse_loss = mean_squared_error(seq_datapoint['true'], seq_datapoint['pred'], squared=False)
    mae_loss = mean_absolute_error(seq_datapoint['true'], seq_datapoint['pred'])

    plt.figtext(0.14, 0.93, f'RMSE: {rmse_loss:.2f}', fontsize=12, ha='left')
    plt.figtext(0.14, 0.89, f'MAE: {mae_loss:.2f}', fontsize=12, ha='left')

    # ax.text(-1, 220, f'RMSE: {rmse_loss:.2f}', fontsize=12, ha='left')
    # ax.text(-1, 200, f'MAE: {mae_loss:.2f}', fontsize=12, ha='left')

    # Add labels and legend
    ax.set_xlabel("Index")
    ax.set_ylabel("Value")
    ax.legend()

    # Return the figure
    return fig

def plot_k_results(seq_data: list, inds: list) -> None:
    getter = itemgetter(*inds)
    seq_slice = list(getter(seq_data))
    for dp in seq_slice:
        assert len(dp['pred']) == len(dp['true'])

        num_models = len(dp['pred'])
        history_len = len(dp['hist'])
        
        plot_results(dp, num_models, history_len)
        plt.show()

def compute_val_errors(preds_gt: dict, num_models: int):
    print("Model\tRMSE\tMAE\tR2")

    # create result dict
    result_dict = {}
    for k in range(num_models):
        # get score metrics
        result_dict[f'model_{k+1}'] = {
            "r2": r2_score(preds_gt['gt'][k].squeeze(-1).cpu(), preds_gt['preds'][k].cpu()),
            "rmse": mean_squared_error(preds_gt['gt'][k].squeeze(-1).cpu(), preds_gt['preds'][k].cpu(), squared=False),
            "mae": mean_absolute_error(preds_gt['gt'][k].squeeze(-1).cpu(), preds_gt['preds'][k].cpu()),
            }

        print(f'{k+1}\t{result_dict[f"model_{k+1}"]["rmse"]:.2f}\t{result_dict[f"model_{k+1}"]["mae"]:.2f}\t{result_dict[f"model_{k+1}"]["r2"]:.2f}')

    return result_dict

def plot_loss_curve(prefix: str, ax: Union[plt.axes, None] = None, row_height: float = 3):
    with open(f'{prefix}training_loss.txt', 'r') as train_file, open(f'{prefix}val_loss.txt', 'r') as val_file:
        train_lines = train_file.readlines()
        val_lines = val_file.readlines()

        training_loss = list(map(float, train_lines))
        validation_loss = list(map(float, val_lines))

    assert len(training_loss) == len(validation_loss)

    d = [i for i in range(len(training_loss))]
    
    if not ax:
        _, ax = plt.subplots(figsize=(10, row_height))

    ax.plot(d, training_loss, label="Train", color='b')
    ax.plot(d, validation_loss, label="Val", color='g')

    # Add labels and legend
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()

    return ax

def arrange_figures_in_rows(n_rows: int, prefix: str, row_height: float = 3):
    fig, axs = plt.subplots(n_rows, 1, figsize=(10, row_height * n_rows))
    for i in range(n_rows):
        gen_path = f'./loss/run{prefix}{i+1}/'
        axs[i] = plot_loss_curve(gen_path, axs[i])
    
    fig.tight_layout()
    plt.show()