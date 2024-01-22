import os
import pickle
import torch
import io

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots


class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else:
            return super().find_class(module, name)


class PredictionSystem():
    def __init__(self):
        data, df = self.get_data()
        self.data = data
        self.time_index = df
        self.conversion = {
            "vanilla_transformer": 'transformer_nfl_test_res.pkl',
            "mlp": 'mlp_nfl_test_res.pkl',
            "mlp_future_lags": 'mlp_fl_test_res.pkl',
            "xgboost": 'xgboost_nfl_test_res.pkl',
            "xgboost_future_lags": 'xgboost_fl_test_res.pkl',
            "encoder_model": 'encoder_test_res.pkl',
            "TFT": 'tft_test_res.pkl',
            "transformer_future_lags": 'transformer_fl_test_res.pkl',
        }
        self.rev_conversion = {v: k for k, v in self.conversion.items()}
        self.trans_preds = [
            'tft_test_res.pkl',
            'transformer_fl_test_res.pkl',
            'transformer_nfl_test_res.pkl',
        ]
        self.colors = {
            'transformer_nfl_test_res.pkl': 'blue',
            'mlp_nfl_test_res.pkl': 'yellow',
            'mlp_fl_test_res.pkl': 'green',
            'xgboost_nfl_test_res.pkl': 'red',
            'xgboost_fl_test_res.pkl': 'purple',
            'encoder_test_res.pkl': 'orange',
            'tft_test_res.pkl': 'pink',
            'transformer_fl_test_res.pkl': 'magenta'
        }

    def load_pickles(self, folder_path):
        all_data = {}

        for filename in os.listdir(folder_path):
            if filename.endswith('.pkl'):
                file_path = os.path.join(folder_path, filename)
                with open(file_path, 'rb') as file:
                    data = CPU_Unpickler(file).load()
                    # data = pickle.load(file)
                    all_data[filename] = data

        return all_data

    def get_data(self):
        folder_path_nfl = 'models/_results_nfl/'
        folder_path_fl = 'models/_results_fl/'
        data = {**self.load_pickles(folder_path_nfl),
                **self.load_pickles(folder_path_fl)}

        # 2023-06-01T10:15:00+00:00 || ind: 0
        # 2023-08-31T21:45:00+00:00 || ind: 8783
        # Start and end time
        start = '2023-06-01 10:15:00'
        end = '2023-08-31 21:45:00'

        freq = '15T'

        # Creating the time stamp sequence
        x = pd.date_range(start=start, end=end, freq=freq)
        df = pd.DataFrame(index=x, data={'value': range(len(x))})

        return data, df

    def plot_predictions(self, start_time, end_time, models):
        selected_keys = [self.conversion[key]
                         for key in self.conversion.keys() if key in models]

        start_ind = self.time_index.loc[(
            self.time_index.index == start_time), "value"].values[0]
        end_ind = self.time_index.loc[(
            self.time_index.index == end_time), "value"].values[0]
        x = pd.date_range(start=start_time, end=end_time, freq='15T')

        fig = go.Figure()
        add_gt = True
        for name in selected_keys:
            if name in self.trans_preds:
                if add_gt:
                    fig.add_trace(go.Scatter(x=x, y=self.data[name]['gt'][start_ind:end_ind, 0].squeeze(
                    ), mode='lines', name='Ground Truth', line=dict(dash='dash', color='black')))
                    add_gt = False

                fig.add_trace(go.Scatter(x=x, y=self.data[name]['preds'][start_ind:end_ind, 0].squeeze(
                ), mode='lines', name=f'{self.rev_conversion[name]}', line=dict(color=self.colors[name])))

            else:
                if add_gt:
                    fig.add_trace(go.Scatter(x=x, y=self.data[name]['gt'][0][start_ind:end_ind].squeeze(
                    ), mode='lines', name='Ground Truth', line=dict(dash='dash', color='black')))
                    add_gt = False

                fig.add_trace(go.Scatter(x=x, y=self.data[name]['preds'][0][start_ind:end_ind].squeeze(
                ), mode='lines', name=f'{self.rev_conversion[name]}', line=dict(color=self.colors[name])))

        fig.update_layout(
            font=dict(size=20, family="Times New Roman"),
            xaxis_title='Time (h)',
            yaxis_title='System imbalance (MW)',
            bargap=0.05,  # small gap between bars for better visualization
            legend=dict(
                x=0.5,
                y=-0.25,  # You might need to adjust this value based on your specific figure
                xanchor='center',
                yanchor='top',
                orientation='h'  # Horizontal orientation
            ),
            # Adjust bottom margin to ensure legend is visible
            margin=dict(l=20, r=20, t=100, b=20),
            plot_bgcolor='white',
            width=1400,
            height=750,
        )

        # Customize x-axis and y-axis line properties
        fig.update_xaxes(showline=True, linewidth=1,
                         linecolor='black', mirror=True, tickfont=dict(size=16))
        fig.update_yaxes(showline=True, linewidth=1,
                         linecolor='black', mirror=True, tickfont=dict(size=16))
        fig.show()
