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


def get_speakers_per_corpus(corpus):
    if corpus == "MNGU0":
        speakers = ["MNGU0"]
    elif corpus == "usc":
        speakers = ["F1", "F5", "M1", "M3"]
    elif corpus == "Haskins":
        speakers = ["F01", "F02", "F03", "F04", "M01", "M02", "M03", "M04"]
    elif corpus == "mocha":
        speakers = ["fsew0", "msak0", "faet0", "ffes0", "maps0", "mjjn0", "falh0"]
    else:
        raise NameError("vous navez pas choisi un des corpus")
    return speakers


class petits_traitements():
    def __init__(self,name,sampling_rate_ema,sampling_rate_wav,cutoff):
        super(petits_traitements, self).__init__()
        self.name = name
        self.sampling_rate_ema = sampling_rate_ema
        self.sampling_rate_wav = sampling_rate_wav
        self.speakers = None
        self.get_speakers(name)
        self.cutoff = cutoff
        self.list_EMA_traj = []
        self.list_MFCC_frames = []

