import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def base_plot(y_val: pd.DataFrame, predictions: np.array):
    fig, ax = plt.subplots()
    y_val['predictions'] = predictions
    y_val = y_val.sort_index()
    ax.plot(y_val.index, y_val['predictions'].values, 'g', label='Predicted values')
    ax.plot(y_val.index, y_val['count'].values, 'b', label='Test values')
    ax.set_ylabel('Count', color='g')
    ax.set_xlabel('Time')
    ax.legend(loc=0)
    plt.show()
