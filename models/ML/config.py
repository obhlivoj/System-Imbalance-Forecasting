from pathlib import Path

def get_config():
    return {
        "data_pickle_name": "merged_data.pkl",
        "path_pickle": "../../data/data_TS/",
        "exo_vars": [
                "month_sin", "month_cos", "day_sin", "day_cos", "hour_sin", "hour_cos",
                "quarter_hour_sin", "quarter_hour_cos", "measured_&_upscaled_wind",
                "most_recent_forecast_wind", "total_load",
                "most_recent_forecast_load", "measured_&_upscaled_solar",
                "most_recent_forecast_solar", 'year', 'not_working', 'holiday',
        ],
        "forward_vars": [
                "month_sin", "month_cos", "day_sin", "day_cos", "hour_sin", "hour_cos",
                "quarter_hour_sin", "quarter_hour_cos", "day-ahead_6pm_forecast_wind",
                "most_recent_forecast_wind", "day-ahead_6pm_forecast_load",
                "most_recent_forecast_load", "day-ahead_6pm_forecast_solar",
                "most_recent_forecast_solar", 'year', 'not_working', 'holiday',
        ],
        "target": ["system_imbalance"],
        "lags": {
            "system_imbalance": [4*24-2*4, 7*4*24-2*4]
            },  
        "diffs": None,
        "forward_lags": True,
        "batch_size": 64,
        "num_epochs": 50,
        "lr": 10**-3,
        "src_seq_len": 32,
        "tgt_seq_len": 1,
        "hidden_dim": 512,
        "tgt_step": 0,
        "model_folder": "weights_folder/weights",
        "model_basename": "tmodel_",
        "preload": None,
        "experiment_name": "runs/tmodel",
        "run": "no_forward",
    }

def get_weights_file_path(config, epoch: str):
    model_folder = config["model_folder"]
    model_basename = config["model_basename"]
    model_filename = f"{model_basename}{epoch}.pt"
    return str(Path('.') / model_folder / model_filename)

# NOTES:

    # weightsXY - 8 models, runsXY/tmodel, run=XY: model s tgt_seq_len = 1, num_epochs = 28, lr = 10**-4

    # weightsXY_exact - 8 models, runsXY_exact/tmodel, run=XY_exact: model s tgt_seq_len = 1, num_epochs = 30, lr = 10**-4
    # # # cfg["exo_vars"] = [
    # # #             "month_sin", "month_cos", "day_sin", "day_cos", "hour_sin", "hour_cos",
    # # #             "quarter_hour_sin", "quarter_hour_cos", "measured_&_upscaled_wind",
    # # #             "most_recent_forecast_wind", "total_load",
    # # #             "most_recent_forecast_load", "measured_&_upscaled_solar",
    # # #             "most_recent_forecast_solar", 'year', 'not_working', 'holiday',
    # # # ]