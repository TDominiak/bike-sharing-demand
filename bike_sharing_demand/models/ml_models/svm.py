from typing import Dict, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR

from bike_sharing_demand.log import logger


class SVM:
    def __init__(self, model_parameters: Dict = None):
        self.model_parameters = model_parameters
        self.model: Optional[SVR] = None

    def __str__(self):
        return 'SVMRegressor'

    def train(self, x_train: pd.DataFrame, x_val: pd.DataFrame, y_train: pd.DataFrame, y_val: pd.DataFrame) -> SVR:
        model = self.__create_model(x_val, y_val)
        model.fit(x_train, y_train.values.ravel())
        self.model = model

        return model

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        prediction = self.model.predict(df)
        prediction[prediction < 0] = 0

        return prediction

    def __create_model(self, x: pd.DataFrame, y: pd.DataFrame) -> SVR:
        if self.model_parameters:
            return SVR(**self.model_parameters)
        else:
            return SVR(**SVM.__find_best_parameters(x, y))

    @staticmethod
    def __find_best_parameters(x: pd.DataFrame, y: pd.DataFrame) -> Dict:

        parameters_bounds = {'kernel': ['rbf'], 'C': [1.5, 10], 'gamma': [1e-7, 1e-4],
                             'epsilon': [0.1, 0.2, 0.5, 0.3]}
        svr = SVR()
        grid_search = GridSearchCV(estimator=svr, param_grid=parameters_bounds,
                                   cv=3, n_jobs=-1, verbose=2)
        grid_search.fit(x, y.values.ravel())
        logger.info(f'Best parameters {str(SVM.__name__)}: {grid_search.best_params_}')

        return grid_search.best_params_
