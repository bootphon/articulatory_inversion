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


def traitement_general_usc(N_max):

    def traitement_usc(speaker,N_max=N_max):

        root_path = dirname(dirname(os.path.realpath(__file__)))
        path_files_treated = os.path.join(root_path, "Donnees_pretraitees",  speaker)
        path_files_brutes = os.path.join(root_path, "Donnees_brutes", "usc_timit", speaker)
        path_files_annotation = os.path.join(root_path, "Donnees_brutes", "usc_timit", speaker, "trans")

        def create_missing_dir():
            if not os.path.exists(os.path.join(path_files_treated, "ema")):
                os.makedirs(os.path.join(path_files_treated, "ema"))
            if not os.path.exists(os.path.join(path_files_treated, "mfcc")):
                os.makedirs(os.path.join(path_files_treated, "mfcc"))
            if not os.path.exists(os.path.join(path_files_treated, "ema_filtered")):
                os.makedirs(os.path.join(path_files_treated, "ema_filtered"))
            if not os.path.exists(os.path.join(path_files_brutes, "mat_cut")):
                os.makedirs(os.path.join(path_files_brutes, "mat_cut"))
            if not os.path.exists(os.path.join(path_files_brutes, "wav_cut")):
                os.makedirs(os.path.join(path_files_brutes, "wav_cut"))

            files = glob.glob(os.path.join(path_files_treated, "ema", "*"))
            files += glob.glob(os.path.join(path_files_treated, "mfcc", "*"))
            files += glob.glob(os.path.join(path_files_treated, "ema_filtered", "*"))
            files += glob.glob(os.path.join(path_files_brutes, "wav_cut","*"))
            files += glob.glob(os.path.join(path_files_brutes, "mat_cut","*"))

            for f in files:
                os.remove(f)

        def cut_all_files():
            for j in range(N):
                path_wav = os.path.join(path_files_brutes, "wav", EMA_files[j] + '.wav')
                wav, sr = librosa.load(path_wav, sr=sampling_rate_wav)  # chargement de données
                # wav = scipy.signal.resample(wav,num=int(len(wav)*sampling_rate_wav/sampling_rate_wav_init))

                my_ema = sio.loadmat(os.path.join(path_files_brutes, "mat", EMA_files[j] + ".mat"))
                my_ema = my_ema[EMA_files[j]][0]  # dictionnaire où les données sont associées à la clé EMA_files[i]pritn()
                my_ema = np.concatenate([my_ema[arti][2][:, [0, 1]] for arti in range(1, 7)], axis=1)

                with open(os.path.join(path_files_annotation, EMA_files[j] + ".trans")) as file:
                    labels = np.array([row.strip("\n").split(",") for row in file])
                    phone_details = labels[:, [0, 1, -1]]
                    id_phrase = set(phone_details[:, 2])
                    id_phrase.remove("")

                    for k in set(id_phrase):
                        temp = phone_details[phone_details[:, 2] == k]
                        xtrm = [float(temp[:, 0][0]), float(temp[:, 1][-1])]
                        xtrm_temp_ema = [int(np.floor(xtrm[0] * sampling_rate_ema)),
                                         int(np.floor(xtrm[1] * sampling_rate_ema) + 1)]
                        xtrm_temp_wav = [int(np.floor(xtrm[0] * sampling_rate_wav)),
                                         int(np.floor(xtrm[1] * sampling_rate_wav) + 1)]

                        ema_temp = my_ema[xtrm_temp_ema[0]:xtrm_temp_ema[1], :]
                        wav_temp = wav[xtrm_temp_wav[0]:xtrm_temp_wav[1]]

                        if os.path.exists(os.path.join(path_files_brutes, "mat_cut", EMA_files[j][:-7] + str(k) + ".npy")):
                            premiere_partie_ema = np.load(
                                os.path.join(path_files_brutes, "mat_cut", EMA_files[j][:-7] + str(k) + ".npy"))
                            ema_temp = np.concatenate((ema_temp, premiere_partie_ema), axis=0)
                            premiere_partie_wav = np.load(
                                os.path.join(path_files_brutes, "wav_cut", EMA_files[j][:-7] + str(k) + ".npy"))
                            wav_temp = np.concatenate((wav_temp, premiere_partie_wav), axis=0)

                        np.save(os.path.join(path_files_brutes, "mat_cut", EMA_files[j][:-7] + str(k)), ema_temp)
                        np.save(os.path.join(path_files_brutes, "wav_cut", EMA_files[j][:-7] + str(k)), wav_temp)



        def read_ema_file(m):
            articulators = ["ul_x", "ul_y", "ll_x", "ll_y", "li_x", "li_y", "td_x", "td_y", "tb_x", "tb_y", "tt_x",
                            "tt_y"]

            order_arti_usctimit = [
                'tt_x', 'tt_y', 'td_x', 'td_y', 'tb_x', 'tb_y', 'li_x', 'li_y',
                'ul_x', 'ul_y', 'll_x', 'll_y']

            new_order_arti = [articulators.index(col) for col in order_arti_usctimit]  # change the order from the initial

            my_ema = np.load(os.path.join(path_files_brutes, "mat_cut", EMA_files_2[m] + ".npy"))

            if np.isnan(my_ema).sum() != 0:
                #        print(np.isnan(ema).sum())
                spline = scipy.interpolate.splrep(np.argwhere(~np.isnan(my_ema).ravel()), my_ema[~np.isnan(my_ema)], k=3)
                for j in np.argwhere(np.isnan(my_ema)).ravel():
                    my_ema[j] = scipy.interpolate.splev(j, spline)
            my_ema = my_ema[:, new_order_arti]  # change order of arti to have the one wanted

            return my_ema

        def from_wav_to_mfcc(k):
            path_wav = os.path.join(path_files_brutes, "wav_cut", EMA_files_2[k] + '.npy')
            data = np.load(path_wav)  # chargement de données
            my_mfcc = librosa.feature.mfcc(y=data, sr=sampling_rate_wav, n_mfcc=n_coeff,
                                        n_fft=frame_length, hop_length=hop_length).T
            dyna_features = get_delta_features(my_mfcc)
            dyna_features_2 = get_delta_features(dyna_features)
            my_mfcc = np.concatenate((my_mfcc, dyna_features, dyna_features_2), axis=1)
            padding = np.zeros((window, my_mfcc.shape[1]))
            frames = np.concatenate([padding, my_mfcc, padding])
            full_window = 1 + 2 * window
            my_mfcc = np.concatenate([frames[j:j+ len(my_mfcc)] for j in range(full_window)], axis=1)
            return my_mfcc

        def synchro_ema_mfcc(k, my_ema, my_mfcc):
            ## zero padding de sorte que l'on intègre les dépendences temporelles : on apprend la trame du milieu
            # mais on ajoute des trames précédent et suivant pour ajouter de l'informatio temporelle
            n_frames_wanted = my_mfcc.shape[0]

            my_ema = scipy.signal.resample(my_ema, num=n_frames_wanted)
            if len(my_ema) != len(my_mfcc):
                print("pbm size", EMA_files_2[k])
            return my_ema, my_mfcc

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


        create_missing_dir()
        path_files = os.path.join(root_path, "Donnees_brutes","usc_timit", speaker)
        EMA_files = sorted([name[:-4] for name in os.listdir(os.path.join(path_files_brutes, "mat")) if name.endswith(".mat")])

        list_EMA_traj = []
        list_MFCC_frames = []

        N = len(EMA_files)
        if N_max != "All":
            N = int(N_max/3) #on coupe N fichiers
        cut_all_files()

        EMA_files_2 = sorted(
        [name[:-4] for name in os.listdir(os.path.join(path_files_brutes, "wav_cut")) if name.endswith(".npy")])
        N_2 = len(EMA_files_2)
        if N_max != "All":
            N_2 = min(N_max,N_2)
        for i in range(N_2):
            if i%100==0:
                print("{} out of {}".format(i,N_2))
            ema = read_ema_file(i)
            mfcc = from_wav_to_mfcc(i)
            ema,mfcc = synchro_ema_mfcc(i,ema,mfcc)
            np.save(os.path.join(root_path, "Donnees_pretraitees",  speaker, "ema", EMA_files_2[i]), ema)
            np.save(os.path.join(root_path, "Donnees_pretraitees",  speaker, "mfcc", EMA_files_2[i]), mfcc)

            ema_filtered = smooth_data(ema)

            np.save(os.path.join(root_path, "Donnees_pretraitees", speaker, "ema_filtered", EMA_files_2[i]), ema_filtered)
            list_EMA_traj.append(ema_filtered)
            list_MFCC_frames.append(mfcc)

        calculate_norm_values(list_EMA_traj,list_MFCC_frames)
        normalize_data(speaker)
        add_vocal_tract(speaker)

    sampling_rate_ema = 100
    sampling_rate_wav = 20000
    frame_time = 25
    hop_time = 10  # en ms
    hop_length = int((hop_time * sampling_rate_wav) / 1000)
    frame_length = int((frame_time * sampling_rate_wav) / 1000)
    window = 5
    n_coeff = 13
    cutoff = 10
    speakers = ["F1","F5","M1","M3"]

    for sp in speakers :
        print("speaker ",sp)
        traitement_usc(sp,N_max = N_max)

traitement_general_usc(N_max = "All")

