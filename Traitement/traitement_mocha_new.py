

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
from Traitement.split_sentences import split_sentences
import glob
import multiprocessing as mp



def traitement_general_mocha(N_max,n_procs=0):

    def traitement_mocha(speaker,N_max):

        root_path = dirname(dirname(os.path.realpath(__file__)))
        path_files_treated = os.path.join(root_path, "Donnees_pretraitees",  speaker)

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

        def read_ema_file(k):
            path_ema_file = os.path.join(path_files, EMA_files[k] + ".ema")
            with open(path_ema_file, 'rb') as ema_annotation:
                column_names = [0] * n_columns
                for line in ema_annotation:
                    line = line.decode('latin-1').strip("\n")
                    if line == 'EST_Header_End':
                        break
                    elif line.startswith('NumFrames'):
                        n_frames = int(line.rsplit(' ', 1)[-1])
                    elif line.startswith('Channel_'):
                        col_id, col_name = line.split(' ', 1)
                        column_names[int(col_id.split('_', 1)[-1])] = col_name.replace(" ",
                                                                                       "")  # v_x has sometimes a space
                ema_data = np.fromfile(ema_annotation, "float32").reshape(n_frames, n_columns + 2)
                cols_index = [column_names.index(col) for col in articulators]
                ema_data = ema_data[:, cols_index]
                ema_data = ema_data / 100  # met en mm, initallement en 10^-1m
                if np.isnan(ema_data).sum() != 0:
                    print("nombre de nan ", np.isnan(ema_data).sum())
                    # Build a cubic spline out of non-NaN values.
                    spline = scipy.interpolate.splrep(np.argwhere(~np.isnan(ema_data).ravel()),
                                                      ema_data[~np.isnan(ema_data)], k=3)
                    # Interpolate missing values and replace them.
                    for j in np.argwhere(np.isnan(ema_data)).ravel():
                        ema_data[j] = scipy.interpolate.splev(j, spline)
                return ema_data

        def from_wav_to_mfcc(k):
            path_wav = os.path.join(path_files, wav_files[k] + '.wav')
            data, sr = librosa.load(path_wav, sr=sampling_rate_wav)  # chargement de données

            my_mfcc = librosa.feature.mfcc(y=data, sr=sampling_rate_wav, n_mfcc=n_coeff,
                                           n_fft=frame_length, hop_length=hop_length
                                           ).T

            dyna_features = get_delta_features(my_mfcc)
            dyna_features_2 = get_delta_features(dyna_features)

            my_mfcc = np.concatenate((my_mfcc, dyna_features, dyna_features_2), axis=1)
            padding = np.zeros((window, my_mfcc.shape[1]))
            frames = np.concatenate([padding, my_mfcc, padding])
            full_window = 1 + 2 * window
            my_mfcc = np.concatenate([frames[j:j + len(my_mfcc)] for j in range(full_window)], axis=1)
            return my_mfcc

        def synchro_ema_mfcc(k, my_ema, my_mfcc):
            if speaker in sp_with_trans:
                path_annotation = os.path.join(path_files, wav_files[k] + '.lab')
                with open(path_annotation) as file:
                    labels = [
                        row.strip('\n').strip('\t').replace(' 26 ', '').split(' ')
                        for row in file
                    ]
                start_time = float(labels[0][1])  # if labels[0][1] == '#' else 0
                end_time = float(labels[-1][0])  # if labels[-1][1] == '#' else labels[-1][0]
                start_frame_mfcc = int(
                    np.floor(
                        start_time * 1000 / hop_time))  # nombre de frame mfcc avant lesquelles il ny a que du silence
                end_frame_mfcc = int(
                    np.ceil(end_time * 1000 / hop_time))  # nombre de frame mfcc apres lesquelles il ny a que du silence
                my_mfcc = np.array(my_mfcc[start_frame_mfcc:end_frame_mfcc])
                start_frame_ema = int(np.floor(start_time * sampling_rate_ema))
                end_frame_ema = int(np.ceil(end_time * sampling_rate_ema))
                my_ema = my_ema[start_frame_ema:end_frame_ema]
                # sous echantillonnage de EMA pour synchro avec WAV
            n_frames_wanted = my_mfcc.shape[0]
            my_ema = scipy.signal.resample(my_ema, num=n_frames_wanted)


            ## zero padding de sorte que l'on intègre les dépendences temporelles : on apprend la trame du milieu
            # mais on ajoute des trames précédent et suivant pour ajouter de l'informatio temporelle

            if len(my_ema) != len(my_mfcc):
                print("pbm size", wav_files[k])
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

            all_EMA_concat = np.concatenate([traj for traj in list_EMA_traj], axis=0)
            std_ema = np.std(all_EMA_concat, axis=0)

            mean_ema = np.mean(np.array([np.mean(traj, axis=0) for traj in my_list_EMA_traj]),
                               axis=0)  # apres que chaque phrase soit centrée
            std_mfcc = np.mean(np.array([np.std(frame, axis=0) for frame in my_list_MFCC_frames]), axis=0)
            mean_mfcc = np.mean(np.array([np.mean(frame, axis=0) for frame in my_list_MFCC_frames]), axis=0)
            np.save(os.path.join("norm_values", "moving_average_ema_" + speaker), smoothed_moving_average)
            np.save(os.path.join("norm_values", "moving_average_ema_brute_" + speaker), moving_average)
            np.save(os.path.join("norm_values", "std_ema_" + speaker), std_ema)
            np.save(os.path.join("norm_values", "mean_ema_" + speaker), mean_ema)
            np.save(os.path.join("norm_values", "std_mfcc_" + speaker), std_mfcc)
            np.save(os.path.join("norm_values", "mean_mfcc_" + speaker), mean_mfcc)

        create_missing_dir()
        path_files = os.path.join(root_path, "Donnees_brutes", "mocha", speaker)
        EMA_files = sorted([name for name in os.listdir(path_files) if "palate" not in name])
        EMA_files = sorted([name[:-4] for name in EMA_files if name.endswith('.ema')])
        n_columns = 20
        wav_files = sorted([name[:-4] for name in os.listdir(path_files) if name.endswith('.wav')])
        N = len(EMA_files)
        if N_max != "All":
            N = N_max
        if speaker in sp_with_velum:  # on ne connait pas le velum
            articulators = [
                'tt_x', 'tt_y', 'td_x', 'td_y', 'tb_x', 'tb_y', 'li_x', 'li_y',
                'ul_x', 'ul_y', 'll_x', 'll_y', 'v_x', 'v_y']
        else:
            articulators = [
                'tt_x', 'tt_y', 'td_x', 'td_y', 'tb_x', 'tb_y', 'li_x', 'li_y',
                'ul_x', 'ul_y', 'll_x', 'll_y']

        list_EMA_traj = []
        list_MFCC_frames = []

        for i in range(N):
            if i+1%50 == 0:
                print("{} out of {}".format(i,N))
            ema = read_ema_file(i)
            mfcc = from_wav_to_mfcc(i)
            ema,mfcc = synchro_ema_mfcc(i,ema,mfcc)
            np.save(os.path.join(root_path, "Donnees_pretraitees",  speaker, "ema", EMA_files[i]), ema)
            np.save(os.path.join(root_path, "Donnees_pretraitees",  speaker, "mfcc", EMA_files[i]), mfcc)
            ema_filtered = smooth_data(ema)
            np.save(os.path.join(root_path, "Donnees_pretraitees", speaker, "ema_filtered", EMA_files[i]), ema_filtered)
            list_EMA_traj.append(ema_filtered)
            list_MFCC_frames.append(mfcc)

        calculate_norm_values(list_EMA_traj,list_MFCC_frames)
        normalize_data(speaker)
        add_vocal_tract(speaker)

    sampling_rate_ema = 500
    sampling_rate_wav = 16000
    frame_time = 25
    hop_time = 10  # en ms
    hop_length = int((hop_time * sampling_rate_wav) / 1000)
    frame_length = int((frame_time * sampling_rate_wav) / 1000)
    window = 5
    n_coeff = 13
    sp_with_velum = ["fsew0", "msak0", "faet0", "falh0", "ffes0"]
    sp_with_trans = ["fsew0", "msak0", "mjjn0", "ffes0"]
    cutoff = 30
    speakers = ["fsew0","msak0","faet0","falh0","ffes0","mjjn0","maps0"]



    for sp in speakers :
        print("speaker ",sp)
        traitement_mocha(sp,N_max = N_max)
        split_sentences(sp)


#traitement_general_mocha(N_max = 3,n_procs = 0)
