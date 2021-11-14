from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class DataParameters:
    train_path: str = './data/bike-sharing-demand/train.csv'
    test_path: str = './data/bike-sharing-demand/test.csv'
    #  Use  ['casual', 'registered'] to predict total individual
    target_columns: str = field(default_factory=lambda: ['count'])
    dummies: Optional[List] = field(default_factory=lambda: ['season', 'weather', 'workingday', 'holiday',
                                                             'workinghours', 'peak_week', 'peak_weekend'])
    use_voting: bool = True
    use_voting_weights: bool = True
    interpret_model: bool = False
