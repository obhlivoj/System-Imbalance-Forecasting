from pathlib import Path
import pandas as pd

def get_config():
    return {
        "data_pickle_name": "merged_data.pkl",
        "path_pickle": "../../data/data_TS/",
        "exo_vars": [
            "month_sin", "month_cos", "day_sin", "day_cos", "weekday_sin", "weekday_cos",
            "hour_sin", "hour_cos",
            "quarter_hour_sin", "quarter_hour_cos", "measured_&_upscaled_wind",
            "most_recent_forecast_wind", "total_load",
            "most_recent_forecast_load", "measured_&_upscaled_solar",
            "most_recent_forecast_solar", 'year', 'not_working', 'holiday',
        ],
        "forward_vars": [
            "month_sin", "month_cos", "day_sin", "day_cos", "weekday_sin", "weekday_cos",
            "hour_sin", "hour_cos",
            "quarter_hour_sin", "quarter_hour_cos", "day-ahead_6pm_forecast_wind",
            "most_recent_forecast_wind", "day-ahead_6pm_forecast_load",
            "most_recent_forecast_load", "day-ahead_6pm_forecast_solar",
            "most_recent_forecast_solar", 'year', 'not_working', 'holiday',
        ],
        "target": ["system_imbalance"],
        "lags": {
            #"system_imbalance": [i for i in range(4*24-2*4, 4*24+6*4)] + [i for i in range(7*4*24-2*4, 7*4*24+6*4)]
            "system_imbalance": [4*24-2*4, 7*4*24-2*4]
            },  
        "diffs": None,
        "forward_lags": True,
        "train_split": '2023-03-01 00:00:00',
        "test_split": '2023-06-01 00:00:00',
        "batch_size": 64,
        "num_epochs": 15,
        "lr": 1*10**-4,
        "src_seq_len": 32,
        "tgt_seq_len": 8,
        "val_seq_len": 8,
        "d_ff": 256,
        "n_head": 2,
        "Nx": 4,
        "dropout": 0.1,
        "model_folder": "weights",
        "model_basename": "tmodel_",
        "preload": None,
        "experiment_name": "runs/tmodel",
        "run": "final_mod",
    }

def get_weights_file_path(config, epoch: str):
    model_folder = config["model_folder"]
    model_basename = config["model_basename"]
    model_filename = f"{model_basename}{epoch}.pt"
    return str(Path('.') / model_folder / model_filename)

# NOTES:
    # weights, runs/tmodel, run=0: model s tgt/val_seq_len = 1, num_epochs = 15
    # weights1, runs1/tmodel, run=1: model s tgt/val_seq_len = 4, num_epochs = 15
    # weights1, runs2/tmodel, run=2: model s tgt/val_seq_len = 8, num_epochs = 15

    # weightsXY_exact: model s tgt/val_seq_len = 8, num_epochs = 50, lr = 0.0001, bs = 512, dff = 256