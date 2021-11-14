import numpy as np
from sklearn.metrics import mean_squared_log_error


def calculate_metric(ground_true: np.ndarray, prediction: np.ndarray) -> float:
    return np.sqrt(mean_squared_log_error(np.exp(ground_true) - 1, np.exp(prediction) - 1))
