from operator import itemgetter

import lime
import lime.lime_tabular
import numpy as np
import pandas as pd

from bike_sharing_demand.log import logger
from bike_sharing_demand.models.model import Model


def interpret_model(model: Model, val_x: pd.DataFrame, val_y: pd.DataFrame, examples: int = 10) -> None:

    predictions = model.predict(val_x)
    errors = (np.exp(predictions) - 1) - (np.exp(val_y.values.ravel()) - 1)
    errors = [(idx, abs(err)) for idx, err in enumerate(errors)]
    sorted_errors = sorted(errors, key=itemgetter(1))
    worse_5 = sorted_errors[-examples:]
    explainer = lime.lime_tabular.LimeTabularExplainer(val_x.values, feature_names=val_x.columns.values.tolist(),
                                                       class_names=['count'], verbose=True, mode='regression')
    for i in range(examples):
        idx = worse_5[i][0]
        logger.info(f'Abs error = {worse_5[i][1]}')
        exp = explainer.explain_instance(val_x.iloc[idx], model.predict, num_features=len(val_x.columns))
        exp.as_pyplot_figure()
