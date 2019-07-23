
import numpy as np
import random
import os
from os.path import dirname
from Traitement.fonctions_utiles import get_speakers_per_corpus
import glob
def normalize_data(speaker):
    """
    calcule puis sauvegarde les données EMA normalisées.
    Norma utilisée : standardisation (standard score) par speaker et par articulateur
    sur l'ensemble des phrases prononcées.


    """
  #  print("normalizing for speaker {}".format(speaker))
    root_path = dirname(dirname(os.path.realpath(__file__)))
    path_speaker = os.path.join(root_path,"Donnees_pretraitees",speaker)
    EMA_files = sorted(
        [name[:-4] for name in os.listdir(os.path.join(path_speaker,"ema_filtered")) if name.endswith(".npy")])

    if not(os.path.exists(os.path.join(path_speaker, "ema_filtered_norma"))):
        os.mkdir(os.path.join(path_speaker, "ema_filtered_norma"))

    if not(os.path.exists(os.path.join(path_speaker, "ema_norma"))):
        os.mkdir(os.path.join(path_speaker, "ema_norma"))

    files = glob.glob(os.path.join(path_speaker, "ema_norma", "*"))
    files += glob.glob(os.path.join(path_speaker, "ema_filtered_norma", "*"))
    for f in files:
        os.remove(f)
    N = len(EMA_files)
    std_ema = np.load(os.path.join(root_path, "Traitement", "norm_values", "std_ema_" + speaker + ".npy"))
    smoothed_moving_average = np.load(
        os.path.join(root_path, "Traitement", "norm_values", "moving_average_ema_" + speaker + ".npy"))
    std_mfcc = np.load(os.path.join(root_path, "Traitement", "norm_values", "std_mfcc_" + speaker + ".npy"))
    mean_mfcc = np.load(os.path.join(root_path, "Traitement", "norm_values", "mean_mfcc_" + speaker + ".npy"))

    for i in range(N):
        ema = np.load(os.path.join(path_speaker, "ema", EMA_files[i] + ".npy"))
        ema_filtered = np.load(os.path.join(path_speaker, "ema_filtered", EMA_files[i] + ".npy"))

        ema = (ema - smoothed_moving_average[i, :])/std_ema
        ema_filtered = (ema_filtered - smoothed_moving_average[i, :]) / std_ema

        mfcc = np.load(os.path.join(path_speaker, "mfcc", EMA_files[i] + ".npy"))
        mfcc = (mfcc-mean_mfcc)/std_mfcc

        np.save(os.path.join(path_speaker, "ema_norma", EMA_files[i]), ema)
        np.save(os.path.join(path_speaker, "ema_filtered_norma", EMA_files[i]), ema_filtered)
        np.save(os.path.join(path_speaker, "mfcc", EMA_files[i]), mfcc)


def normalize_data_per_corpus(corpus) :
    speakers = get_speakers_per_corpus(corpus)
    for sp in speakers :
        normalize_data(sp)

def normalize_phrase(i,speaker):
    """
    calcule puis sauvegarde les données EMA normalisées.
    Norma utilisée : standardisation (standard score) par speaker et par articulateur
    sur l'ensemble des phrases prononcées.


    """
    root_path = os.path.dirname(os.getcwd())
    ema = np.load(os.path.join(root_path,speaker, "ema", EMA_files[i] + ".npy"))
    ema_filtered = np.load(os.path.join(root_path,speaker, "ema_filtered", EMA_files[i] + ".npy"))

    ema = (ema - smoothed_moving_average[i, :])/std_ema
    ema_filtered = (ema_filtered - smoothed_moving_average[i, :]) / std_ema

    mfcc = np.load(os.path.join(path_speaker, "mfcc", EMA_files[i] + ".npy"))
    mfcc = (mfcc-mean_mfcc)/std_mfcc

    np.save(os.path.join(path_speaker, "ema_norma", EMA_files[i]), ema)
    np.save(os.path.join(path_speaker, "ema_filtered_norma", EMA_files[i]), ema_filtered)
    np.save(os.path.join(path_speaker, "mfcc", EMA_files[i]), mfcc)

#normalize_data("falh0")

#orpus =["Haskins"]

#normalize_data_per_corpus("Haskins")
#normalize_data("F02")