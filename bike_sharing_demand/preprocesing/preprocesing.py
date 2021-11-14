from typing import Tuple, List

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from bike_sharing_demand.providers.data_parameters import DataParameters


def perform_preprocessing(df: pd.DataFrame, parameters: DataParameters, log_target: bool = True) -> pd.DataFrame:
    df = _get_time_features(df)
    df['workingday'] = np.where(1 == df['holiday'], 0, df['workingday'])
    df['workinghours'] = np.where(((8 <= df['hour']) & (df['hour'] <= 19)) & (df['workingday'] == 1), 1, 0)
    df['peak_week'] = np.where(((7 <= df['hour']) & (df['hour'] <= 9)) | ((17 <= df['hour']) & (df['hour'] <= 21))
                          & (df['workingday'] == 1), 1, 0)
    df['peak_weekend'] = np.where(((11 <= df['hour']) & (df['hour'] <= 16)) & (df['workingday'] == 0), 1, 0)
    df = _get_dummies(df, parameters)
    df = df.set_index('datetime')
    if log_target:
        for target in parameters.target_columns:
            df[target] = np.log(df[target] + 1)

    return df


def _get_dummies(df: pd.DataFrame, parameters: DataParameters):
    if parameters.dummies:
        for col in parameters.dummies:
            dummies = pd.get_dummies(df[col], prefix=col)
            df = pd.concat([df, dummies], axis=1)
            df.drop(col, inplace=True, axis=1)
    return df


def _get_time_features(df: pd.DataFrame):
    df['year'] = df.datetime.dt.year
    df['month'] = df.datetime.dt.month
    df['hour'] = df.datetime.dt.hour
    df['dayofweek'] = df.datetime.dt.dayofweek
    df['weekend'] = np.where(df.datetime.dt.dayofweek < 5, 0, 1)
    df['dow'] = df.datetime.dt.dayofweek
    df['woy'] = df.datetime.dt.weekofyear

    return df


def split_data(x: pd.DataFrame, y: pd.DataFrame, shuffle: bool = True) -> \
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.05, random_state=42, shuffle=shuffle)

    return x_train, x_val, y_train, y_val


def _add_lag(df, number_of_lag: int, columns: List[str]) -> pd.DataFrame:
    def add_column(data, col, number_obs):
        for num in range(1, number_obs):
            data[f'{col}_-{num}'] = data[col].shift(num)

        return data

    for col in columns:
        df = add_column(df, col, number_of_lag)

    return df
