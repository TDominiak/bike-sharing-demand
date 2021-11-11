from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from bike_sharing_demand.providers.data_parameters import DataParameters


def perform_preprocessing(df: pd.DataFrame, parameters: DataParameters, log_target: bool = True) -> pd.DataFrame:
    df = _get_time_features(df)
    df['workingday_hours'] = np.where((7 <= df['hour']) & (df['hour'] <= 19) & (df['workingday'] == 1), 1, 0)
    df['hour_workingday_peak_1'] = np.where((7 <= df['hour']) & (df['hour'] <= 10) & (df['workingday'] == 1), 1, 0)
    df['hour_workingday_peak_1'] = np.where((16 <= df['hour']) & (df['hour'] <= 19) & (df['workingday'] == 1), 1, 0)
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

    return df


def split_data(x: pd.DataFrame, y: pd.DataFrame, shuffle: bool = True) -> \
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.05, random_state=42, shuffle=shuffle)

    return x_train, x_val, y_train, y_val
