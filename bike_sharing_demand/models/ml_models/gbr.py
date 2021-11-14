from typing import Dict, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV

from bike_sharing_demand.log import logger


class GBM:
    def __init__(self, model_parameters: Dict = None):
        self.model_parameters = model_parameters
        self.model: Optional[GradientBoostingRegressor] = None

    def __str__(self):
        return 'GradientBoostingRegressor'

    def train(self, x_train: pd.DataFrame, x_val: pd.DataFrame, y_train: pd.DataFrame, y_val: pd.DataFrame) \
            -> GradientBoostingRegressor:
        model = self.__create_model(x_train, y_train)
        model.fit(x_train, y_train.values.ravel())
        self.model = model

        return model

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        prediction = self.model.predict(df)
        prediction[prediction < 0] = 0

        return prediction

    def __create_model(self, x: pd.DataFrame, y: pd.DataFrame) -> GradientBoostingRegressor:
        if self.model_parameters:
            return GradientBoostingRegressor(**self.model_parameters)
        else:
            return GradientBoostingRegressor(**GBM.__find_best_parameters(x, y))

    @staticmethod
    def __find_best_parameters(x: pd.DataFrame, y: pd.DataFrame) -> Dict:
        gbr = GradientBoostingRegressor()
        search_grid = {'n_estimators': [500, 2000, 4000], 'max_depth': [3, 6, 9, 18],
                       'random_state': [0], 'min_samples_leaf': [2, 4, 8, 16], 'alpha': [0.1, 0.9],
                       'validation_fraction': [0.2], 'n_iter_no_change': [25]}
        grid_search = GridSearchCV(estimator=gbr, param_grid=search_grid, n_jobs=-1, cv=3, scoring='neg_root_mean_squared_error')
        grid_search.fit(x, y.values.ravel())
        logger.info(f'Best parameters {str(GBM.__name__)}: {grid_search.best_params_}')

        return grid_search.best_params_
