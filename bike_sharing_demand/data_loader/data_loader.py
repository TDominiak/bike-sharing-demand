import os

import pandas as pd


def check_data_folder() -> None:
    for file in ['sampleSubmission', 'test', 'train']:
        if not os.path.isfile(f'./data/bike-sharing-demand/{file}.csv'):
            raise Exception(f'Can not find file: {file}.csv in folder ./data/bike-sharing-demand/.')


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df['datetime'] = pd.to_datetime(df['datetime'])

    return df
