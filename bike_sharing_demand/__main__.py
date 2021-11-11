from typing import Dict

import numpy as np

from bike_sharing_demand.data_loader.data_loader import load_data
from bike_sharing_demand.log import logger
from bike_sharing_demand.models.model import get_best_ml_models
from bike_sharing_demand.preprocesing.preprocesing import perform_preprocessing
from bike_sharing_demand.providers.data_parameters import DataParameters
from bike_sharing_demand.runner_ml import run_ml_models


def __save_prediction(data_parameters: DataParameters, best_models: Dict) -> None:
    test_df = perform_preprocessing(load_data(data_parameters.test_path), data_parameters, False)
    count = [0] * len(test_df)
    for target_column in parameters.target_columns:
        model = best_models.get(target_column)
        predictions = np.exp(model.predict(test_df)) - 1
        predictions[predictions < 0] = 0
        count = np.add(count, predictions)
    test_df['count'] = count
    test_df['count'] = round(test_df['count'])
    test_df = test_df.reset_index()
    test_df = test_df[['datetime', 'count']]
    test_df.to_csv('./data/answer.csv', index=False)
    logger.info(f"Saved answer as: './data/answer.csv'")


if __name__ == '__main__':
    logger.info('Initializing parameters...')
    parameters = DataParameters()
    logger.info('Loading data...')
    train_df = load_data(parameters.train_path)
    train_df = perform_preprocessing(train_df, parameters)
    models_score = run_ml_models(train_df, parameters)
    best_models = get_best_ml_models(models_score, parameters)
    __save_prediction(parameters, best_models)
