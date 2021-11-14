from typing import Dict, Optional

from bayes_opt import BayesianOptimization
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error

from bike_sharing_demand.log import logger


class XgbModel:
    def __init__(self, model_parameters: Dict = None):
        self.model_parameters = model_parameters
        self.model: Optional[xgb.XGBRegressor] = None

    def __str__(self):
        return 'XGBRegressor'

    def train(self, x_train: pd.DataFrame, x_val: pd.DataFrame, y_train: pd.DataFrame, y_val: pd.DataFrame)\
            -> xgb.XGBRegressor:
        model = self.__create_model(x_train, x_val, y_train, y_val)
        model.fit(x_train, y_train,
                  eval_set=[(x_train, y_train), (x_val, y_val)],
                  early_stopping_rounds=25,
                  verbose=True)
        xgb.plot_importance(model)
        self.model = model

        return model

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        prediction = self.model.predict(df)
        prediction[prediction < 0] = 0

        return prediction

    def __create_model(self, x_train: pd.DataFrame, x_val: pd.DataFrame, y_train: pd.DataFrame, y_val: pd.DataFrame)\
            -> xgb.XGBRegressor:
        if self.model_parameters:
            return xgb.XGBRegressor(**self.model_parameters)
        else:
            return xgb.XGBRegressor(**XgbModel.__find_best_parameters(x_train, x_val, y_train, y_val))

    @staticmethod
    def __find_best_parameters(x_train: pd.DataFrame, x_val: pd.DataFrame, y_train: pd.DataFrame, y_val: pd.DataFrame) \
            -> Dict:

        xg_train = xgb.DMatrix(x_train, label=y_train)
        xg_val = xgb.DMatrix(x_val, label=y_val)

        def model_bayesian(learning_rate, max_depth, min_child_weight, subsample, colsample_bytree, alpha):

            max_depth = int(max_depth)
            min_child_weight = int(min_child_weight)
            training_params = {
                'n_jobs': -1,
                'learning_rate': learning_rate,
                'max_depth': max_depth,
                'min_child_weight': min_child_weight,
                'subsample': subsample,
                'colsample_bytree': colsample_bytree,
                'alpha': alpha,
                'max_bin': 63,
                'objective': 'reg:squarederror',
                'verbosity': 1}
            clf = xgb.train(training_params, xg_train)
            score = mean_squared_error(xg_val.get_label(), clf.predict(xg_val))

            return -1.0 * score

        parameters_bounds = {
            'learning_rate': (0.001, 0.3),
            'max_depth': (2, 24),
            'min_child_weight': (1, 25),
            'subsample': (0.1, 0.9),
            'colsample_bytree': (0.1, 0.9),
            'alpha': (1e-5, 100),
        }

        optimized_model = BayesianOptimization(model_bayesian, parameters_bounds, random_state=13)
        init_points, n_iter = 25, 1

        optimized_model.maximize(init_points=init_points, n_iter=n_iter, acq='ucb', xi=0.0, alpha=1e-6)
        parameters = XgbModel.__get_parameters(optimized_model)
        logger.info(f'Best parameters {str(XgbModel.__name__)}: {parameters}')

        return parameters

    @staticmethod
    def __get_parameters(optimized_model: BayesianOptimization) -> Dict:

        parameters = {
            'min_child_weight': int(optimized_model.max['params']['min_child_weight']),
            'subsample': optimized_model.max['params']['subsample'],
            'colsample_bytree': optimized_model.max['params']['colsample_bytree'],
            'max_bin': 63,
            'objective': 'count:poisson',
            'verbose': 1,
            'metric': 'rmse',
            'is_unbalance': False,
            'learning_rate': optimized_model.max['params']['learning_rate'],
            'max_depth': int(optimized_model.max['params']['max_depth']),
            'save_binary': True,
            'seed': 1337,
            'alpha': optimized_model.max['params']['alpha']
        }

        return parameters
