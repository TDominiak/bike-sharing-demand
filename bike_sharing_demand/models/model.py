from typing import Protocol, Dict, Tuple, Any

import numpy as np
import pandas as pd

from bike_sharing_demand.log import logger
from bike_sharing_demand.providers.data_parameters import DataParameters


class Model(Protocol):

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        ...

    def train(self, x_train: pd.DataFrame, x_val: pd.DataFrame, y_train: pd.DataFrame, y_val: pd.DataFrame) -> Any:
        ...


def get_best_ml_models(models_score: Dict, parameters: DataParameters) -> Dict:
    best_models: Dict = dict()
    for target in parameters.target_columns:
        sorted_models = sorted(models_score.get(target).items(), key=lambda x: x[1], reverse=False)
        best_model = sorted_models[0][0]
        best_models[target] = best_model
        logger.info(f' Target: {target}, Found best model: {str(best_model)}, score: '
                    f'{models_score[target].get(best_model)}')

    return best_models
