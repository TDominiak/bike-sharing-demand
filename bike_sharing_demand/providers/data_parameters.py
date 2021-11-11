from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class DataParameters:
    train_path: str = './data/bike-sharing-demand/train.csv'
    test_path: str = './data/bike-sharing-demand/test.csv'
    #  Use  ['count'] to predict total number
    target_columns: str = field(default_factory=lambda: ['registered', 'casual'])
    dummies: Optional[List] = field(default_factory=lambda: ['season', 'weather', 'workingday'])
