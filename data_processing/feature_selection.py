"""
feature_selection - Helper functions for data processing in a Transformer model.

This package contains a collection of utility functions designed to assist with data preprocessing and postprocessing tasks when working with a Transformer-based model.
It provides functions for transforming columns into a wider format, adding various features to the DataFrame, and visualizing correlation matrices.
"""

import itertools
from functools import reduce
from typing import List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def keep_columns(df: pd.DataFrame, cols_to_keep: List[str]) -> pd.DataFrame:
    """
    Keep specified columns in a Pandas DataFrame and drop the rest.

    Parameters:
        df (pd.DataFrame): The DataFrame from which columns will be retained or dropped.
        columns_to_keep (list): A list of column names to retain in the DataFrame.

    Returns:
        pd.DataFrame: A new DataFrame containing only the specified columns.
    """
    # Create a list of columns to drop
    columns_to_drop = [col for col in df.columns if col not in cols_to_keep]

    # Drop the columns not in the list of columns to keep
    df = df.drop(columns_to_drop, axis=1)

    return df


def transform_columns_wider(data: pd.DataFrame, cols: List[str], vars: List[str], contr: List[int]) -> pd.DataFrame:
    """
    Transforms columns into a wider format based on unique combinations of variables.

    Args:
        data (pd.DataFrame): The source DataFrame.
        cols (List[str]): List of columns in `data` to be transformed.
        vars (List[str]): List of variables to be used to create unique combinations.
        contr (List[int]): List of integers specifying the number of characters to use from each variable in suffix.

    Returns:
        pd.DataFrame: The transformed DataFrame.
    """
    un_count = [data[v].unique() for v in vars]
    cartesian_product = list(itertools.product(*un_count))

    data_final_list = []
    for item in cartesian_product:
        conds = [data[v] == it for it, v in zip(item, vars)]
        combined_conditions = reduce(lambda a, b: a & b, conds)

        data_final_list.append(data[combined_conditions])

    combs = cartesian_product.copy()
    length = len(data_final_list)
    for ind, df in enumerate(reversed(data_final_list)):
        f_idx = length - ind - 1
        if df.shape[0] == 0:
            data_final_list.pop(f_idx)
            combs.pop(f_idx)

    new_data_list = []
    for df, c in zip(data_final_list, combs):
        df = df.drop(vars, axis=1)
        suffix = ""
        for ind, short in enumerate(contr):
            suffix += f"_{c[ind][:short]}"
        df = df.rename(columns=lambda x: x + f"_{suffix}" if x in cols else x)
        new_data_list.append(df)

    df_final = reduce(lambda left, right: pd.merge(
        left, right, on='datetime', how='left'), new_data_list)

    return df_final


def transform_columns_sum(data: pd.DataFrame, cols: List[str], vars: List[str]) -> pd.DataFrame:
    """
    Sums the values in columns based on unique combinations of variables.

    Args:
        data (pd.DataFrame): The source DataFrame.
        cols (List[str]): List of columns in `data` to be summed.
        vars (List[str]): List of variables to be used to create unique combinations.
        contr (List[int]): List of integers specifying the number of characters to use from each variable in suffix.

    Returns:
        pd.DataFrame: The DataFrame with summed values.
    """
    un_count = [data[v].unique() for v in vars]
    cartesian_product = list(itertools.product(*un_count))

    data_final_list = []
    for item in cartesian_product:
        conds = [data[v] == it for it, v in zip(item, vars)]
        combined_conditions = reduce(lambda a, b: a & b, conds)

        data_final_list.append(data[combined_conditions])

    combs = cartesian_product.copy()
    length = len(data_final_list)
    for ind, df in enumerate(reversed(data_final_list)):
        f_idx = length - ind - 1
        if df.shape[0] == 0:
            data_final_list.pop(f_idx)
            combs.pop(f_idx)

    new_data_list = []
    datetime_col = data_final_list[0]["datetime"].reset_index(drop=True)
    for df in data_final_list:
        df = df.drop(vars, axis=1)
        new_data_list.append(df[cols].reset_index(drop=True))

    df_final = reduce(lambda left, right: left + right, new_data_list)
    df_final['datetime'] = datetime_col

    return df_final


def additional_features(data: pd.DataFrame, datetime_col: str) -> None:
    """
    Add various features to the DataFrame such as date parts, cyclical encoding of date parts, and categories.

    Args:
        data (pd.DataFrame): The source DataFrame. This DataFrame should have a datetime column.

    Returns:
        None: This function modifies the DataFrame in place.
    """
    data["year"] = data[datetime_col].dt.year
    data["month"] = data[datetime_col].dt.month
    data["day"] = data[datetime_col].dt.dayofyear
    data["weekday"] = data[datetime_col].dt.dayofweek
    data["hour"] = data[datetime_col].dt.hour
    data['quarter_hour'] = data[datetime_col].dt.minute // 15
    data["index"] = list(range(data.shape[0]))
    data["group"] = "timeserie"
    data["holiday"] = data["holiday"].astype(str).astype("category")
    data["not_working"] = data["not_working"].astype(str).astype("category")

    data['month_sin'] = np.sin(2 * np.pi * data['month'] / 12)
    data['month_cos'] = np.cos(2 * np.pi * data['month'] / 12)
    data['day_sin'] = np.sin(2 * np.pi * data['day'] / 365)
    data['day_cos'] = np.cos(2 * np.pi * data['day'] / 365)
    data['weekday_sin'] = np.sin(2 * np.pi * data['day'] / 7)
    data['weekday_cos'] = np.cos(2 * np.pi * data['day'] / 7)
    data['hour_sin'] = np.sin(2 * np.pi * data['hour'] / 24)
    data['hour_cos'] = np.cos(2 * np.pi * data['hour'] / 24)
    data['quarter_hour_sin'] = np.sin(2 * np.pi * data['quarter_hour'] / 4)
    data['quarter_hour_cos'] = np.cos(2 * np.pi * data['quarter_hour'] / 4)

    data["year"] = data["year"].astype(str).astype(
        "category")  # categories have to be strings
    data["month"] = data["month"].astype(str).astype("category")
    data["day"] = data["day"].astype(str).astype("category")
    data["weekday"] = data["weekday"].astype(str).astype("category")
    data["hour"] = data["hour"].astype(str).astype("category")
    data['quarter_hour'] = data['quarter_hour'].astype(str).astype("category")


def plot_corrmat(corr: pd.DataFrame) -> None:
    """
    Create a heatmap to visualize the correlation matrix.

    Args:
        corr (pd.DataFrame): The correlation matrix to be visualized.

    Returns:
        None: This function displays the heatmap but does not return any value.
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=False, cmap='coolwarm', linewidths=0.5)
    plt.title('Correlation matrix heatmap')
    plt.show()
