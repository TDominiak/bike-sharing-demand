from typing import Dict, List

import pandas as pd

from bike_sharing_demand.log import logger
from bike_sharing_demand.models.common import calculate_metric
from bike_sharing_demand.models.ml_models.gbr import GBM
from bike_sharing_demand.models.ml_models.lightgbm_model import LightGbm
from bike_sharing_demand.models.ml_models.voting_regresor import VotingReg
from bike_sharing_demand.models.ml_models.xgb_boost import XgbModel
from bike_sharing_demand.models.model import Model
from bike_sharing_demand.models.model_Interpreter import interpret_model
from bike_sharing_demand.preprocesing.preprocesing import split_data
from bike_sharing_demand.providers.data_parameters import DataParameters


def run_ml_models(train_df: pd.DataFrame, parameters: DataParameters) -> Dict:
    models_score: Dict = dict()
    columns = ['count', 'casual', 'registered']
    for target_column in parameters.target_columns:
        models_collections: List[Model] = \
            [XgbModel(), LightGbm(), GBM({'alpha': 0.1, 'max_depth': 6, 'min_samples_leaf': 16, 'n_estimators': 500,
                                          'random_state': 0})]
        models_score[target_column] = {}
        x, y = train_df.drop(columns, axis=1), train_df[[target_column]]
        x_train, x_val, y_train, y_val = split_data(x, y)
        models = []
        for model in models_collections:
            logger.info(f'Start training {str(model)}')
            trained_model = model.train(x_train, x_val, y_train, y_val)
            if parameters.interpret_model:
                interpret_model(trained_model, x_val, y_val, 5)
            predictions = model.predict(x_val)
            metric = calculate_metric(y_val.values, predictions)
            logger.info(f'Target: {target_column}, model: {str(model)}, score: {metric}.')
            models_score[target_column][model] = metric
            models.append((str(model), trained_model))
        if parameters.use_voting:
            vt = VotingReg(parameters.use_voting_weights)
            _ = vt.initialize(models, x_train, x_val, y_train, y_val)
            predictions = vt.predict(x_val)
            metric = calculate_metric(y_val.values, predictions)
            logger.info(f'Target: {target_column}, model: {str(vt)}, score: {metric}.')
            models_score[target_column][vt] = metric

    return models_score
