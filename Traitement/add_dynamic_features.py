
import os
from os.path import dirname
import numpy as np
def get_delta_features(array, window=5):
    all_diff = []
    for lag in range(1, window + 1):
        padding = np.ones((lag, array.shape[1]))
        past = np.concatenate([padding * array[0], array[:-lag]])
        future = np.concatenate([array[lag:], padding * array[-1]])
        all_diff.append(future - past)
    tempo =np.array([ all_diff[lag] * lag for lag in range(window)])
    norm = 2 * np.sum(i ** 2 for i in range(1, window + 1))

    delta_features = np.sum(tempo,axis=0)/norm
    return delta_features

