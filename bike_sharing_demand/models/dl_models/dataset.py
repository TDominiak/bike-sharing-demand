from typing import Optional

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from bike_sharing_demand.preprocesing.preprocesing import split_data


class DataModule(pl.LightningDataModule):
    def __init__(self, x: pd.DataFrame, y: pd.DataFrame, batch_size: int = 32):
        super().__init__()
        self.x = x
        self.y = y
        self.batch_size = batch_size
        self.preprocessing: StandardScaler = StandardScaler()

    def setup(self, stage: Optional[str] = None):
        x_train, x_val, self.y_train, self.y_val = split_data(self.x, self.x)
        self.preprocessing.fit(x_train)
        self.x_train = self.preprocessing.transform(x_train)
        self.x_val = self.preprocessing.transform(x_val)

    def train_dataloader(self):
        return DataModule.create_data_loader_from_df(self.x_train, self.y_train, self.batch_size)

    def val_dataloader(self):
        return DataModule.create_data_loader_from_df(self.x_val, self.y_val, self.batch_size)

    def test_dataloader(self):
        return DataModule.create_data_loader_from_df(self.x_val, self.y_val, self.batch_size)

    @staticmethod
    def create_data_loader_from_df(x: np.array, y: pd.DataFrame, batch_size: int, shuffle=True):
        dataset = TensorDataset(
            torch.from_numpy(x).type(torch.FloatTensor),
            torch.from_numpy(y.to_numpy()).type(torch.FloatTensor))

        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
