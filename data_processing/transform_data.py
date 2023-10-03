"""
transform_data - Helper functions for data processing.

This package provides utility functions for data preprocessing and manipulation, including removing non-ASCII characters
from strings, mapping names to lowercase with underscores, dropping columns with a high percentage of missing values
or containing redundant information, adding holiday indicators, and converting date columns to UTC timezone.
"""

import string
from typing import Iterable, List
import holidays
import workalendar
import pandas as pd

from sklearn.preprocessing import LabelEncoder



def remove_non_ascii(a_string: str) -> str:
    """
    Remove non-ASCII characters from a string.

    Args:
        a_string (str): The input string containing ASCII and non-ASCII characters.

    Returns:
        str: The input string with non-ASCII characters removed.
    """
    ascii_chars = set(string.printable)
    return ''.join(filter(lambda x: x in ascii_chars, a_string))

def name_mapper(name: str) -> str:
    """
    Map a name to a lowercase, ASCII-safe string with spaces replaced by underscores.

    Args:
        name (str): The input name.

    Returns:
        str: The mapped name.
    """
    ascii_lit = remove_non_ascii(name)
    space_sep = "_".join(ascii_lit.split(" "))
    lower = space_sep.lower()
    return lower

def drop_nulls_and_redundants(data: pd.DataFrame, th = 0.15) -> pd.DataFrame:
    """
    Drop columns with a high percentage of missing values or containing redundant information.

    Args:
        data (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: The DataFrame with specified columns dropped.
    """
    to_drop = []
    for col in data.columns:
        if sum(data.loc[:, col].isna()) / data.shape[0] > th or data.loc[:, col].nunique() == 1:
            to_drop.append(col)
    return data.drop(to_drop, axis=1)

def add_holidays(data: pd.Series, country: str, years: Iterable[int], sep: str = "T") -> pd.Series:
    """
    Add a Boolean column indicating whether each date is a holiday in a specified country.

    Args:
        data (pd.Series): The input Series containing dates.
        country (str): The country code for holiday lookup (e.g., "US" for the United States).
        years (Iterable[int]): The years to consider for holidays.
        sep (str, optional): The separator used in date strings. Defaults to "T".

    Returns:
        pd.Series: A Boolean Series indicating if each date is a holiday.
    """
    country_holidays = set(holidays.country_holidays(country, years=years).keys())
    country_holidays_str = set(map(str, country_holidays))
    is_country_holidays = data.apply(lambda x: x.split(sep)[0]).isin(country_holidays_str)
    return is_country_holidays

def add_not_working_days(data: pd.Series, cal: workalendar, period: list, sep: str = "T") -> pd.Series:
    """
    Add a Boolean column indicating whether each date is a non-working day within a specified date range.

    Args:
        data (pd.Series): The input Series containing dates.
        cal (workalendar): The workalendar instance for date calculations.
        period (list): A list containing the start and end dates for the period of interest.
        sep (str, optional): The separator used in date strings. Defaults to "T".

    Returns:
        pd.Series: A Boolean Series indicating if each date is a non-working day.
    """
    start_date, end_date = period
    not_working_days = []
    for date in pd.date_range(start=start_date, end=end_date):
        if not cal.is_working_day(date):
            not_working_days.append(date)

    days_str = list(map(str, not_working_days))
    days_str_strip = set(map(lambda x: x.split(" ")[0], days_str))
    is_be_not_working = data.apply(lambda x: x.split(sep)[0]).isin(days_str_strip)
    return is_be_not_working

def date_to_datetime_utc(data: pd.Series, format: str) -> pd.Series:
    """
    Convert date strings to datetime objects in UTC timezone.

    Args:
        data (pd.Series): The input Series containing date strings.
        format (str): The format of the date strings.

    Returns:
        pd.Series: A Series containing datetime objects in UTC timezone.
    """
    datetime_converted = pd.to_datetime(data, format=format, errors="coerce", utc=True)
    datetime_no_tz = datetime_converted.dt.tz_localize(None)
    return datetime_no_tz

def columns_to_utc(data: pd.DataFrame, cols: list, format: str) -> pd.DataFrame:
    """
    Convert specified columns in a DataFrame to datetime objects in UTC timezone.

    Args:
        data (pd.DataFrame): The input DataFrame.
        cols (list): A list of column names to be converted.
        format (str): The format of the date strings.

    Returns:
        pd.DataFrame: The DataFrame with specified columns converted to datetime objects in UTC timezone.
    """
    for col in cols:
        data[col] = date_to_datetime_utc(data[col], format)
    return data

def add_suffix_to_columns(df: pd.DataFrame, columns_to_suffix: List[str], suffix: str):
    """
    Add a suffix to selected columns in a Pandas DataFrame in-place.

    Args:
        df (pd.DataFrame): The DataFrame to modify.
        columns_to_suffix (list): List of column names to which the suffix will be added.
        suffix (str): The suffix to add to the selected columns.

    Returns:
        None: The DataFrame is modified in-place.
    """
    # Create a dictionary mapping original column names to new column names with the suffix
    rename_dict = {col: col + suffix for col in columns_to_suffix}

    # Rename the columns in-place
    df.rename(columns=rename_dict, inplace=True)

def convert_category_to_float(df: pd.DataFrame, columns: List[str], type_format: str = 'float64'):
    """
    Convert categorical columns in a Pandas DataFrame to numerical values.

    This function can handle binary (two unique values) and multi-category columns.
    For binary columns, it uses Label Encoding to convert them to numerical values.
    For multi-category columns, it applies one-hot encoding to create binary columns for each category.

    Parameters:
    - df (pd.DataFrame): The Pandas DataFrame containing the data to be processed.
    - columns (List[str]): A list of column names in the DataFrame to be converted.
    - type_format (str, optional): The data type to which the converted values should be cast (default is 'float64').

    Returns:
    - None: The function modifies the input DataFrame in place by replacing categorical columns with numerical values.
    """
    for col in columns:
        if df[col].nunique() == 2:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col]).astype(type_format)
        elif df[col].nunique() > 2:
            df = pd.get_dummies(df, columns=[col], prefix=[col], dtype=type_format)
        
    return df



# def transform_all(data: pd.DataFrame, date_cols: list, period: list) -> pd.DataFrame:
#     data.datetime = data.datetime.astype(str)
#     data = data[(data.datetime >= '2022-07-01') & (data.datetime <= '2023-03-01')].copy()


# IDEAS

# ----------------------------------------------- ### transform the data chunk by chunk

# import pandas as pd

# # set up a DataFrame to accumulate the chunks
# df_list = []

# # read in the CSV file in chunks and transform the "timestamp" column into a datetime object
# for chunk in pd.read_csv("filename.csv", chunksize=10000):
#     chunk["timestamp"] = pd.to_datetime(chunk["timestamp"], format="%Y-%m-%dT%H:%M:%S%z", errors="coerce")
#     df_list.append(chunk)

# # concatenate the chunks into a single DataFrame
# df = pd.concat(df_list)

# # check the data type of the "timestamp" column
# print(df["timestamp"].dtype)


# ------------------------------------------------ ### transform the datetime while reading the csv 

# import pandas as pd

# # read in the CSV file and transform the "timestamp" column into a datetime object
# df = pd.read_csv("filename.csv", parse_dates=["timestamp"], date_parser=lambda x: pd.to_datetime(x, format="%Y-%m-%dT%H:%M:%S%z", errors="coerce"))

# # check the data type of the "timestamp" column
# print(df["timestamp"].dtype)

# ### convert timezone of the timestamp
# act_price["datetime"] = act_price["datetime"].dt.tz_convert("UTC")
# act_price['datetime'].dt.tz
# act_price['datetime'] = act_price['datetime'].dt.strftime("%Y-%m-%d %H:%M:%S")