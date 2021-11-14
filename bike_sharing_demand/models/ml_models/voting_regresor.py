from typing import Optional, List, Tuple

import numpy as np
import pandas as pd
from hyperopt import hp, fmin, tpe, STATUS_OK
from sklearn import clone
from sklearn.ensemble import VotingRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.utils._testing import ignore_warnings

from bike_sharing_demand.log import logger
from bike_sharing_demand.models.model import Model


class VotingReg:
    def __init__(self, use_voting_weights: bool, voting_weights: Tuple = None):
        self.use_voting_weights = use_voting_weights
        self.voting_weights = voting_weights
        self.model: Optional[VotingRegressor] = None
        self.models_list: Optional[List[Model]] = None

    def __str__(self):
        return 'VotingRegressor'

    def initialize(self, models: List, x_train: pd.DataFrame, x_val: pd.DataFrame, y_train: pd.DataFrame,
                   y_val: pd.DataFrame) -> VotingRegressor:
        self.models_list = models

        return self.train(x_train, x_val, y_train, y_val)

    def train(self, x_train: pd.DataFrame, x_val: pd.DataFrame, y_train: pd.DataFrame, y_val: pd.DataFrame) \
            -> VotingRegressor:
        model = self.__create_model(self.models_list, x_train, x_val, y_train, y_val)
        model.fit(x_train, y_train.values.ravel())
        self.model = model

        return model

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        prediction = self.model.predict(df)
        prediction[prediction < 0] = 0

        return prediction

    def __create_model(self, models: List,  x_train: pd.DataFrame, x_val: pd.DataFrame, y_train: pd.DataFrame,
                       y_val: pd.DataFrame) -> VotingRegressor:
        if not self.use_voting_weights:
            return VotingRegressor(models)
        elif self.use_voting_weights and self.voting_weights:
            return VotingRegressor(estimators=models, weights=self.voting_weights)
        else:
            weights = VotingReg.__find_best_weights(models, x_train, x_val, y_train, y_val)
            return VotingRegressor(estimators=models, weights=weights)

    @staticmethod
    def __find_best_weights(models: List, x_train: pd.DataFrame, x_val: pd.DataFrame, y_train: pd.DataFrame,
                            y_val: pd.DataFrame) -> Tuple:
        space = {'w1': hp.quniform('w1', 0, 1, 0.1),
                 'w2': hp.quniform('w2', 0, 1, 0.1),
                 'w3': hp.quniform('w3', 0, 1, 0.1)}

        pipe = Pipeline([["vt", VotingRegressor(estimators=models, weights=None, n_jobs=-1)]])

        @ignore_warnings(category=ConvergenceWarning)
        def objective(weights):
            model = clone(pipe)
            model.set_params(vt__weights=(weights['w1'], weights['w2'], weights['w3']))
            model.fit(x_train, y_train.values.ravel())
            score = mean_squared_error(y_val.values, model.predict(x_val))

            return {'loss': score, 'status': STATUS_OK}

        best = fmin(objective,
                    space=space,
                    algo=tpe.suggest,
                    max_evals=25)

        logger.info(f'Best parameters VotingRegressor: {best}')

        return tuple(best.values())
