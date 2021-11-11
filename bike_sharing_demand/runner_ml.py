from typing import Dict, List

import pandas as pd
from sklearn.ensemble import VotingRegressor

from bike_sharing_demand.log import logger
from bike_sharing_demand.models.common import calculate_metric
from bike_sharing_demand.models.ml_models.gbr import GBM
from bike_sharing_demand.models.ml_models.lightgbm_model import LightGbm
from bike_sharing_demand.models.ml_models.xgb_boost import XgbModel
from bike_sharing_demand.models.model import Model
from bike_sharing_demand.preprocesing.preprocesing import split_data
from bike_sharing_demand.providers.data_parameters import DataParameters


def run_ml_models(train_df: pd.DataFrame, parameters: DataParameters) -> Dict:
    models_score: Dict = dict()
    columns = ['count', 'casual', 'registered']
    for target_column in parameters.target_columns:
        models_collections: List[Model] = [LightGbm(), XgbModel(), GBM()]
        models_score[target_column] = {}
        x, y = train_df.drop(columns, axis=1), train_df[[target_column]]
        x_train, x_val, y_train, y_val = split_data(x, y)
        models = []
        for model in models_collections:
            logger.info(f'Start training {str(model)}')
            trained_model = model.train(x_train, x_val, y_train, y_val)
            predictions = model.predict(x_val)
            metric = calculate_metric(y_val.values, predictions)
            logger.info(f'Target: {target_column}, model: {str(model)}, score: {metric}.')
            models_score[target_column][model] = metric
            models.append((str(model), trained_model))
        vt = VotingRegressor(models)
        vt_model = vt.fit(x_train, y_train.values.ravel())
        predictions = vt_model.predict(x_val)
        predictions[predictions < 0] = 0
        models_score[target_column][vt_model] = calculate_metric(y_val.values, predictions)

    return models_score
