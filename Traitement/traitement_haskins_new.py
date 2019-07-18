""" Lecture des données EMA pour le corpus MNGU0. On ne conserve que les données concernant les articulateurs indiqués
 dans articulators cest a dire 6 articulateurs en 2Dimensions.
 on ajoute une colonne correspondant à l'ouverture des lèvres, cest donc la 13ème colonne
 on ne normalise pas les données mais on conserve la std et mean des mfcc et ema pour pouvoir normaliser par la suiite)

"""
import os
import time
import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)
from os.path import dirname
import numpy as np
import scipy.signal
import matplotlib.pyplot as plt
import scipy.interpolate
from Traitement.add_dynamic_features import get_delta_features
import librosa
from Apprentissage.utils import low_pass_filter_weight
import scipy.io as sio
from Traitement.split_sentences import split_sentences
import shutil
import glob

""" after this script the order of the articulators is the following : """


"""
Module pour traiter les données du corpus mocha, pour les deux locuteurs (fsew0 et msak0)
On filtre sur les 6 articulateurs indiqués .
Enleve les silences en début et fin.
Resample ema pour avoir 1 position par articulateur pour chaque frame mfcc.

Normalisation : on calcule la moyenne glissante des positions moyennes des articulateurs pour chaque trame, puis
on soustrait cette moyenne. A la fin on rajoute à chaque position la position moyenne sur l'ensemble des phrases (et
pas seulement celle autour de la trame en question). Cette moyenne glissante est implémentée avec le même filtre passe
bas que pour filtrer les trajectoires, mais cette fois ci avec une fréquence de coupure de 10Hz.
On normalise ensuite en divisant par l'écart type max (sur lensemble des articulateurs).

"""
import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)

import os
import time
from os.path import dirname
import numpy as np
import scipy.signal
from scipy import stats
import matplotlib.pyplot as plt
import scipy.interpolate
from Traitement.add_dynamic_features import get_delta_features
import librosa
from Apprentissage.utils import low_pass_filter_weight
import shutil
from Traitement.normalization import normalize_data
from Traitement.add_vocal_tract import add_vocal_tract

import glob


def traitement_general_haskins(N_max):

    def traitement_haskins(speaker,N_max=N_max):

        def create_missing_dir():
            if not os.path.exists(os.path.join(path_files_treated, "ema")):
                os.makedirs(os.path.join(path_files_treated, "ema"))
            if not os.path.exists(os.path.join(path_files_treated, "mfcc")):
                os.makedirs(os.path.join(path_files_treated, "mfcc"))
            if not os.path.exists(os.path.join(path_files_treated, "ema_filtered")):
                os.makedirs(os.path.join(path_files_treated, "ema_filtered"))


            files = glob.glob(os.path.join(path_files_treated, "ema", "*"))
            files += glob.glob(os.path.join(path_files_treated, "mfcc", "*"))
            files += glob.glob(os.path.join(path_files_treated, "ema_filtered", "*"))

            for f in files:
                os.remove(f)

        def detect_silence(ma_data):
            try:  # tous les fichiers ne sont pas organisés dans le même ordre dans le dictionnaire, il semble y avoir deux cas
                mon_debut = ma_data[0][5][0][0][1][0][1]
                ma_fin = ma_data[0][5][0][-1][1][0][0]
            except:
                mon_debut = ma_data[0][6][0][0][1][0][1]
                ma_fin = ma_data[0][6][0][-1][1][0][0]
            return [mon_debut, ma_fin]

        def read_ema_and_wav(k):

            order_arti_haskins = ['td_x', 'td_y', 'tb_x', 'tb_y', 'tt_x', 'tt_y', 'ul_x', 'ul_y', "ll_x", "ll_y",
                                  "ml_x", "ml_y", "li_x", "li_y", "jl_x", "jl_y"]

            order_arti = ['tt_x', 'tt_y', 'td_x', 'td_y', 'tb_x', 'tb_y', 'li_x', 'li_y',
                          'ul_x', 'ul_y', 'll_x', 'll_y']

            data = sio.loadmat(os.path.join(path_files_brutes, EMA_files[k] + ".mat"))[EMA_files[k]][0]
            wav = data[0][2]
            np.save(os.path.join(root_path, "Donnees_brutes", "Haskins_IEEE_Rate_Comparison_DB", speaker, "wav",
                                 EMA_files[k]), wav)

            my_ema = np.zeros((len(data[1][2]), len(order_arti_haskins)))
            for arti in range(1, len(data)):  # lecture des trajectoires articulatoires dans le dicionnaire
                my_ema[:, (arti - 1) * 2] = data[arti][2][:, 0]
                my_ema[:, arti * 2 - 1] = data[arti][2][:, 2]

            [debut, fin] = detect_silence(data)
            xtrm_ema = [int(np.floor(debut * sampling_rate_ema)), int(np.floor(fin * sampling_rate_ema) + 1)]
            xtrm_wav = [int(np.floor(debut * sampling_rate_wav)), int(np.floor(fin * sampling_rate_wav) + 1)]
            my_ema = my_ema[xtrm_ema[0]:xtrm_ema[1], :]
            wav = np.reshape(wav[xtrm_wav[0]:xtrm_wav[1]], -1)
            new_order_arti = [order_arti_haskins.index(col) for col in order_arti]
            my_ema = my_ema[:, new_order_arti]

            my_mfcc = librosa.feature.mfcc(y=wav, sr=sampling_rate_wav, n_mfcc=n_coeff,
                                        n_fft=frame_length, hop_length=hop_length).T
            dyna_features = get_delta_features(my_mfcc)
            dyna_features_2 = get_delta_features(dyna_features)
            my_mfcc = np.concatenate((my_mfcc, dyna_features, dyna_features_2), axis=1)
            padding = np.zeros((window, my_mfcc.shape[1]))
            frames = np.concatenate([padding, my_mfcc, padding])
            full_window = 1 + 2 * window
            my_mfcc = np.concatenate([frames[i:i + len(my_mfcc)] for i in range(full_window)], axis=1)

            if np.isnan(my_ema).sum() != 0:
                print("number of nan :", np.isnan(my_ema.sum()))

            n_frames_wanted = my_mfcc.shape[0]

            my_ema = scipy.signal.resample(my_ema, num=n_frames_wanted)

            return my_ema,my_mfcc

        def smooth_data(my_ema):
            pad = 30
            weights = low_pass_filter_weight(cut_off=cutoff, sampling_rate=sampling_rate_ema)

            my_ema_filtered = np.concatenate([np.expand_dims(np.pad(my_ema[:, k], (pad, pad), "symmetric"), 1)
                                              for k in range(my_ema.shape[1])], axis=1)

            my_ema_filtered = np.concatenate([np.expand_dims(np.convolve(channel, weights, mode='same'), 1)
                                              for channel in my_ema_filtered.T], axis=1)
            my_ema_filtered = my_ema_filtered[pad:-pad, :]
            return my_ema_filtered

        def calculate_norm_values(my_list_EMA_traj, my_list_MFCC_frames):
            pad = 30
            all_mean_ema = np.array([np.mean(traj, axis=0) for traj in my_list_EMA_traj])
            weights_moving_average = low_pass_filter_weight(cut_off=10, sampling_rate=sampling_rate_ema)
            moving_average = np.concatenate([np.expand_dims(np.pad(all_mean_ema[:, k], (pad, pad), "symmetric"), 1)
                                             for k in range(all_mean_ema.shape[1])], axis=1)
            smoothed_moving_average = np.concatenate(
                [np.expand_dims(np.convolve(channel, weights_moving_average, mode='same'), 1)
                 for channel in moving_average.T], axis=1)
            smoothed_moving_average = smoothed_moving_average[pad:-pad, :]

            all_EMA_concat = np.concatenate([traj for traj in my_list_EMA_traj], axis=0)
            std_ema = np.std(all_EMA_concat, axis=0)

            mean_ema = np.mean(np.array([np.mean(traj, axis=0) for traj in my_list_EMA_traj]),
                               axis=0)  # apres que chaque phrase soit centrée
            std_mfcc = np.mean(np.array([np.std(frame, axis=0) for frame in my_list_MFCC_frames]), axis=0)
            mean_mfcc = np.mean(np.array([np.mean(frame, axis=0) for frame in my_list_MFCC_frames]), axis=0)
            np.save(os.path.join("norm_values", "moving_average_ema_" + speaker), smoothed_moving_average)
            np.save(os.path.join("norm_values", "moving_average_ema_brute" + speaker), moving_average)
            np.save(os.path.join("norm_values", "std_ema_" + speaker), std_ema)
            np.save(os.path.join("norm_values", "mean_ema_" + speaker), mean_ema)
            np.save(os.path.join("norm_values", "std_mfcc_" + speaker), std_mfcc)
            np.save(os.path.join("norm_values", "mean_mfcc_" + speaker), mean_mfcc)


        root_path = dirname(dirname(os.path.realpath(__file__)))
        path_files_treated = os.path.join(root_path, "Donnees_pretraitees",speaker)
        path_files_brutes = os.path.join(root_path, "Donnees_brutes", "Haskins_IEEE_Rate_Comparison_DB", speaker, "data")
        create_missing_dir()

        EMA_files = sorted(  [name[:-4] for name in os.listdir(path_files_brutes) if "palate" not in name])

        list_EMA_traj = []
        list_MFCC_frames = []

        N = len(EMA_files)
        if N_max != "All":
            N = int(N_max) #on coupe N fichiers

        for i in range(N):
            if i %100==0:
                print("{} out of {}".format(i,N))

            ema,mfcc = read_ema_and_wav(i)
            np.save(os.path.join(path_files_treated, "ema", EMA_files[i]), ema)
            np.save(os.path.join(path_files_treated, "mfcc", EMA_files[i]), mfcc)

            ema_filtered = smooth_data(ema)
            np.save(os.path.join(path_files_treated, "ema_filtered", EMA_files[i]), ema_filtered)
            list_EMA_traj.append(ema_filtered)
            list_MFCC_frames.append(mfcc)

        calculate_norm_values(list_EMA_traj,list_MFCC_frames)
        normalize_data(speaker)
        add_vocal_tract(speaker)

    sampling_rate_ema = 100  # toujours le même, mais lisible directement dans le fichier
    sampling_rate_wav = 44100  # toujours le même, mais lisible directement dans le fichier

    frame_time = 25
    hop_time = 10  # en ms
    hop_length = int((hop_time * sampling_rate_wav) / 1000)
    frame_length = int((frame_time * sampling_rate_wav) / 1000)
    window = 5
    n_coeff = 13
    cutoff = 10
    speakers = ["F01", "F02", "F03", "F04", "M01", "M02", "M03", "M04"]

    for sp in speakers :
        print("speaker ",sp)
        traitement_haskins(sp,N_max = N_max)
        split_sentences(sp)

#traitement_general_haskins(N_max = 30)
