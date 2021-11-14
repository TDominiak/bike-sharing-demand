from typing import Any

from torch import nn
import torch.nn.functional as F

from bike_sharing_demand.models.dl_models.models.model import BaseModel


class SimpleMLP(BaseModel):

    def __init__(self, learning_rate: float = 0.001):
        super(SimpleMLP, self).__init__(learning_rate=learning_rate)
        self.fc1 = nn.Linear(19, 64)
        self.fc2 = nn.Linear(64, 32)
        self.out = nn.Linear(32, 1)
        self.save_hyperparameters()

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        out = self.out(x)
        return out

    def __str__(self):
        return 'SimpleMLP'
