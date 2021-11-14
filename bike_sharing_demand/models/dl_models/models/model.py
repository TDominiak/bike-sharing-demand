import abc
from abc import ABCMeta
from typing import Any

import torch
import pytorch_lightning as pl
from torch import Tensor, nn

from bike_sharing_demand.models.common import calculate_metric


class BaseModel(pl.LightningModule, metaclass=ABCMeta):

    def __init__(self, learning_rate: float):
        super(BaseModel, self).__init__()
        self.learning_rate = learning_rate
        self.criterion = nn.MSELoss()
        self.device_use = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx) -> float:
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log('val_loss', loss)

        return loss

    def test_step(self, batch, batch_idx) -> float:
        x, y = batch
        y_hat = self(x)
        loss = calculate_metric(y_hat, y)
        self.log('val_loss', loss)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    @abc.abstractmethod
    def forward(self, *args, **kwargs) -> Tensor:
        pass

