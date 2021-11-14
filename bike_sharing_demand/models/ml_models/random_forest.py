from typing import Dict, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

from bike_sharing_demand.log import logger


class RandomForest:
    def __init__(self, model_parameters: Dict = None):
        self.model_parameters = model_parameters
        self.model: Optional[RandomForestRegressor] = None

    def __str__(self):
        return 'RandomForestRegressor'

    def train(self, x_train: pd.DataFrame, x_val: pd.DataFrame, y_train: pd.DataFrame, y_val: pd.DataFrame)\
            -> RandomForestRegressor:
        model = self.__create_model(x_val, y_val)
        model.fit(x_train, y_train.values.ravel())
        self.model = model

        return model

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        prediction = self.model.predict(df)
        prediction[prediction < 0] = 0

        return prediction

    def __create_model(self, x: pd.DataFrame, y: pd.DataFrame) -> RandomForestRegressor:
        if self.model_parameters:
            return RandomForestRegressor(**self.model_parameters)
        else:
            return RandomForestRegressor(**RandomForest.__find_best_parameters(x, y))

    @staticmethod
    def __find_best_parameters(x: pd.DataFrame, y: pd.DataFrame) -> Dict:

        param_grid = {
            'bootstrap': [True],
            'max_features': [2, 3],
            'min_samples_leaf': [1, 3, 4],
            'min_samples_split': [2, 4, 8, 10],
            'n_estimators': [200, 500, 1000, 2000],
            'random_state': [0]
        }

        rf = RandomForestRegressor()

        grid_search = GridSearchCV(estimator=rf, param_grid=param_grid,
                                   cv=3, n_jobs=-1, verbose=2)
        grid_search.fit(x, y.values.ravel())
        logger.info(f'Best parameters {str(RandomForest.__name__)}: {grid_search.best_params_}')

        return grid_search.best_params_
