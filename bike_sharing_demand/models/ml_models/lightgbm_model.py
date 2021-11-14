from typing import Dict, Optional, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from bayes_opt import BayesianOptimization
from lightgbm import LGBMRegressor
import lightgbm as lgb

from sklearn.metrics import mean_squared_error

from bike_sharing_demand.log import logger


class LightGbm:
    def __init__(self, model_parameters: Dict = None):
        self.model_parameters = model_parameters
        self.model: Optional[lgb] = None

    def __str__(self):
        return 'LightGbm'

    def train(self, x_train: pd.DataFrame, x_val: pd.DataFrame, y_train: pd.DataFrame,
              y_val: pd.DataFrame) -> LGBMRegressor:

        model = self.__create_model(x_train, x_val, y_train, y_val)
        model.fit(x_train, y_train,
                  eval_set=[(x_train, y_train), (x_val, y_val)],
                  early_stopping_rounds=25,
                  verbose=True)
        lgb.plot_importance(model)
        self.model = model

        return model

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        prediction = self.model.predict(df)
        prediction[prediction < 0] = 0

        return prediction

    def __create_model(self, x_train: pd.DataFrame, x_val: pd.DataFrame, y_train: pd.DataFrame, y_val: pd.DataFrame) \
            -> LGBMRegressor:
        if self.model_parameters:
            return LGBMRegressor(**self.model_parameters)
        else:
            return LGBMRegressor(**LightGbm.__find_best_parameters(x_train, x_val, y_train, y_val))

    @staticmethod
    def __find_best_parameters(x_train: pd.DataFrame, x_val: pd.DataFrame, y_train: pd.DataFrame, y_val: pd.DataFrame) \
            -> Dict:
        lgb_train = lgb.Dataset(x_train, label=y_train)

        def model_bayesian(num_leaves, min_data_in_leaf, learning_rate, min_sum_hessian_in_leaf, feature_fraction,
                           lambda_l1, lambda_l2, min_gain_to_split, max_depth, n_estimators):
            num_leaves = int(num_leaves)
            min_data_in_leaf = int(min_data_in_leaf)
            max_depth = int(max_depth)
            n_estimators = int(n_estimators)
            param = {
                'n_estimators': n_estimators,
                'n_jobs': -1,
                'num_leaves': num_leaves,
                'max_bin': 63,
                'min_data_in_leaf': min_data_in_leaf,
                'learning_rate': learning_rate,
                'min_sum_hessian_in_leaf': min_sum_hessian_in_leaf,
                'bagging_fraction': 1.0,
                'bagging_freq': 5,
                'feature_fraction': feature_fraction,
                'lambda_l1': lambda_l1,
                'lambda_l2': lambda_l2,
                'min_gain_to_split': min_gain_to_split,
                'max_depth': max_depth,
                'feature_pre_filter': False,
                'save_binary': True,
                'seed': 1337,
                'feature_fraction_seed': 1337,
                'bagging_seed': 1337,
                'drop_seed': 1337,
                'data_random_seed': 1337,
                'objective': 'regression',
                'boosting_type': 'gbdt',
                'verbose': -1,
                'metric': 'rmse',
                'is_unbalance': True,
                'boost_from_average': False}
            clf = lgb.train(param, lgb_train)
            score = mean_squared_error(y_val.values, clf.predict(x_val))

            return -1.0 * score

        parameters_bounds = {
            'n_estimators': (100, 4000),
            'num_leaves': (2, 10),
            'min_data_in_leaf': (2, 25),
            'learning_rate': (0.01, 0.1),
            'min_sum_hessian_in_leaf': (0.00001, 0.01),
            'feature_fraction': (0.05, 0.5),
            'lambda_l1': (0, 5.0),
            'lambda_l2': (0, 5.0),
            'min_gain_to_split': (0, 1.0),
            'max_depth': (2, 25),
        }

        optimized_model = BayesianOptimization(model_bayesian, parameters_bounds, random_state=13)
        init_points, n_iter = 5, 50
        optimized_model.maximize(init_points=init_points, n_iter=n_iter, acq='ucb', xi=0.0, alpha=1e-6)
        parameters = LightGbm.__get_parameters(optimized_model)
        logger.info(f'Best parameters {str(LightGbm.__name__)}: {parameters}')

        return parameters

    @staticmethod
    def __get_parameters(optimized_model: BayesianOptimization) -> Dict:

        parameters = {
            'n_estimators': int(optimized_model.max['params']['n_estimators']),  # remember to int here
            'num_leaves': int(optimized_model.max['params']['num_leaves']),  # remember to int here
            'max_bin': 63,
            'min_data_in_leaf': int(optimized_model.max['params']['min_data_in_leaf']),  # remember to int here
            'learning_rate': optimized_model.max['params']['learning_rate'],
            'min_sum_hessian_in_leaf': optimized_model.max['params']['min_sum_hessian_in_leaf'],
            'bagging_fraction': 1.0,
            'bagging_freq': 5,
            'feature_fraction': optimized_model.max['params']['feature_fraction'],
            'lambda_l1': optimized_model.max['params']['lambda_l1'],
            'lambda_l2': optimized_model.max['params']['lambda_l2'],
            'min_gain_to_split': optimized_model.max['params']['min_gain_to_split'],
            'max_depth': int(optimized_model.max['params']['max_depth']),  # remember to int here
            'save_binary': True,
            'seed': 1337,
            'feature_fraction_seed': 1337,
            'bagging_seed': 1337,
            'drop_seed': 1337,
            'data_random_seed': 1337,
            'objective': 'regression',
            'boosting_type': 'gbdt',
            'verbose': 1,
            'metric': 'rmse',
            'is_unbalance': True,
            'boost_from_average': False,
        }

        return parameters
