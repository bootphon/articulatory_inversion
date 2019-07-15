
import numpy as np
import random
import os
from os.path import dirname

def normalize_data(speaker,max="All"):
    """
    calcule puis sauvegarde les données EMA normalisées.
    Norma utilisée : standardisation (standard score) par speaker et par articulateur
    sur l'ensemble des phrases prononcées.


    """
    print("normalizing for speaker {}".format(speaker))
    if speaker in ["msak0", "fsew0", "maps0", "faet0", "mjjn0", "ffes0"]:
        speaker_2 = "mocha_" + speaker

    elif speaker == ["MNGU0"]:
        speaker_2 = speaker

    elif speaker in ["F1", "F5", "M1", "M3"]:
        speaker_2 = "usc_timit_" + speaker

    elif speaker in ["F01", "F02", "F03", "F04", "M01", "M02", "M03", "M04"]:
        speaker_2 = "Haskins_" + speaker

    root_path = dirname(dirname(os.path.realpath(__file__)))
    path_speaker = os.path.join(root_path,"Donnees_pretraitees",speaker_2)
    EMA_files = sorted(
        [name[:-4] for name in os.listdir(os.path.join(path_speaker,"ema_filtered")) if name.endswith(".npy")])
    if not(os.path.exists(os.path.join(path_speaker, "ema_filtered_norma"))):
        os.mkdir(os.path.join(path_speaker, "ema_filtered_norma"))

    if not(os.path.exists(os.path.join(path_speaker, "ema_norma"))):
        os.mkdir(os.path.join(path_speaker, "ema_norma"))
    N = len(EMA_files)
    if max != "All":
        N = max
    for i in range(N):
        if i % 500 == 0:
            print("{} out of {}".format(i, len(EMA_files)))
        ema = np.load(os.path.join(path_speaker, "ema", EMA_files[i] + ".npy"))
        ema_filtered = np.load(os.path.join(path_speaker, "ema_filtered", EMA_files[i] + ".npy"))
        std_ema= np.load(os.path.join(root_path,"Traitement", "norm_values", "std_ema_" + speaker + ".npy"))
        smoothed_moving_average = np.load(os.path.join(root_path,"Traitement", "norm_values", "moving_average_ema_" + speaker + ".npy"))

        ema = (ema - smoothed_moving_average[i, :])/std_ema
        ema_filtered = (ema_filtered - smoothed_moving_average[i, :]) / std_ema
        np.save(os.path.join(path_speaker, "ema_norma", EMA_files[i]), ema)
        np.save(os.path.join(path_speaker, "ema_filtered_norma", EMA_files[i]), ema_filtered)

def normalize_data_per_corpus(corpus,max="All") :
    if corpus == "MNGU0":
        speakers = ["MNGU0"]
    elif corpus == "usc":
        speakers = ["F1", "F5", "M1","M3"]
    elif corpus == "Haskins":
        speakers= ["F01","F02","F03","F04","M01","M02","M03","M04"]

    elif corpus == "mocha":
        speakers =["fsew0","msak0","faet0","ffes0","maps0","mjjn0"]

    else :
        print("vous navez pas choisi un des corpus")

    for sp in speakers :
        normalize_data(sp,max)


corpus =["mocha","usc","MNGU0","Haskins"]

for cor in corpus :
    normalize_data_per_corpus(cor)