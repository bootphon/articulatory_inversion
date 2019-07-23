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
from Traitement.create_filesets import get_fileset_names

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
from Traitement.class_corpus import Speaker,Corpus
import glob


def traitement_general_haskins(N_max):

    my_corpus_class = Corpus("Haskins")
    sampling_rate_ema = 100  # toujours le même, mais lisible directement dans le fichier
    sampling_rate_wav = 44100  # toujours le même, mais lisible directement dans le fichier
    cutoff = 10

    sampling_rate_ema = my_corpus_class.sampling_rate_ema# toujours le même, mais lisible directement dans le fichier
    sampling_rate_wav = my_corpus_class.sampling_rate_wav  # toujours le même, mais lisible directement dans le fichier

    frame_time = 25
    hop_time = 10  # en ms
    hop_length = int((hop_time * sampling_rate_wav) / 1000)
    frame_length = int((frame_time * sampling_rate_wav) / 1000)
    window = 5
    n_coeff = 13

    def traitement_haskins(speaker,N_max=N_max):
        my_speaker_class = Speaker(speaker)
        root_path = dirname(dirname(os.path.realpath(__file__)))
        path_files_treated = os.path.join(root_path, "Donnees_pretraitees", speaker)
        path_files_brutes = os.path.join(root_path, "Donnees_brutes", "Haskins_IEEE_Rate_Comparison_DB", speaker,   "data")

        def create_missing_dir():
            if not os.path.exists(os.path.join(path_files_treated, "ema")):
                os.makedirs(os.path.join(path_files_treated, "ema"))
            if not os.path.exists(os.path.join(path_files_treated, "mfcc")):
                os.makedirs(os.path.join(path_files_treated, "mfcc"))
            if not os.path.exists(os.path.join(path_files_treated, "ema_filtered")):
                os.makedirs(os.path.join(path_files_treated, "ema_filtered"))
            if not os.path.exists(os.path.join(path_files_brutes, "wav_cut")):
                os.makedirs(os.path.join(path_files_brutes, "wav_cut"))

            files = glob.glob(os.path.join(path_files_treated, "ema", "*"))
            files += glob.glob(os.path.join(path_files_treated, "mfcc", "*"))
            files += glob.glob(os.path.join(path_files_treated, "ema_VT", "*"))
            files += glob.glob(os.path.join(path_files_treated, "ema_filtered_norma", "*"))
            files += glob.glob(os.path.join(path_files_treated, "ema_filtered", "*"))

            for f in files:
                os.remove(f)


        def read_ema_and_wav(k):
            def detect_silence(ma_data):
                try:  # tous les fichiers ne sont pas organisés dans le même ordre dans le dictionnaire, il semble y avoir deux cas
                    mon_debut = ma_data[0][5][0][0][1][0][1]
                    ma_fin = ma_data[0][5][0][-1][1][0][0]
                except:
                    mon_debut = ma_data[0][6][0][0][1][0][1]
                    ma_fin = ma_data[0][6][0][-1][1][0][0]
                return [mon_debut, ma_fin]

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
            librosa.output.write_wav(os.path.join(path_files_brutes, "wav_cut", EMA_files[k] + ".wav"),
                                     wav, sampling_rate_wav)

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

        create_missing_dir()
        EMA_files = sorted(  [name[:-4] for name in os.listdir(path_files_brutes) if "palate" not in name])
        N = len(EMA_files)
        if N_max != 0:
            N = int(N_max) #on coupe N fichiers

        for i in range(N):
            if i+1 %100==0:
                print("{} out of {}".format(i,N))
            ema,mfcc = read_ema_and_wav(i)
            np.save(os.path.join(path_files_treated, "ema", EMA_files[i]), ema)
            np.save(os.path.join(path_files_treated, "mfcc", EMA_files[i]), mfcc)
            ema_filtered = my_speaker_class.smooth_data(ema)
            np.save(os.path.join(path_files_treated, "ema_filtered", EMA_files[i]), ema_filtered)
            my_speaker_class.list_EMA_traj.append(ema_filtered)
            my_speaker_class.list_MFCC_frames.append(mfcc)

        my_speaker_class.calculate_norm_values()

        for i in range(N):
            ema = np.load(os.path.join(root_path, "Donnees_pretraitees", speaker, "ema", EMA_files[i] + ".npy"))
            ema_filtered = np.load(
                os.path.join(root_path, "Donnees_pretraitees", speaker, "ema_filtered", EMA_files[i] + ".npy"))
            mfcc = np.load(os.path.join(root_path, "Donnees_pretraitees", speaker, "mfcc", EMA_files[i] + ".npy"))
            ema_norma, ema_filtered_norma, ema_VT, mfcc = my_speaker_class.traitement_deuxieme_partie(i, ema,
                                                                                                      ema_filtered,
                                                                                                      mfcc)
            np.save(os.path.join(root_path, "Donnees_pretraitees", speaker, "ema_norma", EMA_files[i]), ema)
            np.save(os.path.join(root_path, "Donnees_pretraitees", speaker, "ema_filtered_norma", EMA_files[i]),
                    ema_filtered_norma)
            np.save(os.path.join(root_path, "Donnees_pretraitees", speaker, "mfcc", EMA_files[i]), mfcc)
            np.save(os.path.join(root_path, "Donnees_pretraitees", speaker, "ema_VT", EMA_files[i]), ema_VT)

        split_sentences(speaker)
        get_fileset_names(speaker)

    for sp in my_corpus_class.speakers :
        traitement_haskins(sp,N_max = N_max)
        print("Done for speaker ",sp)

#traitement_general_haskins(N_max = 10)