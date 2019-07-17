""" Lecture des données EMA pour le corpus MNGU0. On ne conserve que les données concernant les articulateurs indiqués
 dans articulators cest a dire 6 articulateurs en 2Dimensions.
 on ajoute une colonne correspondant à l'ouverture des lèvres, cest donc la 13ème colonne
 on ne normalise pas les données mais on conserve la std et mean des mfcc et ema pour pouvoir normaliser par la suiite)

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
import matplotlib.pyplot as plt
import scipy.interpolate
from Traitement.add_dynamic_features import get_delta_features
import librosa
import shutil
from Traitement.normalization import normalize_data
from Traitement.add_vocal_tract import add_vocal_tract
from Apprentissage.utils import low_pass_filter_weight
import glob


""" after this script the order of the articulators is the following : """
order_arti_MNGU0 = [
        'tt_x','tt_y','td_y','td_y','tb_x','tb_y',
        'li_x','li_y','ul_x','ul_y',
        'll_x','ll_y']

def traitement_general_mngu0(max="All"):
    print("MNGU0")
    speaker = "MNGU0"
    root_path = dirname(dirname(os.path.realpath(__file__)))
    path_files_annotation = os.path.join(root_path, "Donnees_brutes",speaker,"phone_labels")
    sampling_rate_ema = 200
    #articulators in the same order that those of MOCHA
    articulators = [
        'T1_py','T1_pz','T3_py','T3_pz','T2_py','T2_pz',
        'jaw_py','jaw_pz','upperlip_py','upperlip_pz',
        'lowerlip_py','lowerlip_pz']

    path_ema_files = os.path.join(root_path, "Donnees_brutes",speaker,"ema")
    EMA_files = sorted([name[:-4] for name in os.listdir(path_ema_files) if name.endswith('.ema')])
    path_files_treated = os.path.join(root_path, "Donnees_pretraitees",speaker)
    n_columns = 87
    window=5
    path_wav_files = os.path.join(root_path, "Donnees_brutes",speaker,"wav")
    sampling_rate_wav = 16000
    frame_time = 25
    hop_time = 10  # en ms
    hop_length = int((hop_time * sampling_rate_wav) / 1000)
    frame_length = int((frame_time * sampling_rate_wav) / 1000)
    n_coeff = 13

    def create_missing_dir():
        if not os.path.exists(os.path.join(os.path.join(path_files_treated, "ema"))):
            os.makedirs(os.path.join(path_files_treated, "ema"))
        if not os.path.exists(os.path.join(os.path.join(path_files_treated, "ema_filtered"))):
            os.makedirs(os.path.join(path_files_treated, "ema_filtered"))
        if not os.path.exists(os.path.join(os.path.join(path_files_treated, "mfcc"))):
            os.makedirs(os.path.join(path_files_treated, "mfcc"))


        files = glob.glob(os.path.join(path_files_treated, "ema", "*"))
        files += glob.glob(os.path.join(path_files_treated, "ema_filtered", "*"))
        files += glob.glob(os.path.join(path_files_treated, "mfcc", "*"))
        for f in files:
            os.remove(f)
    def read_ema_file(k):
        """
        :param i: index de l'uttérence (ie numero de phrase) dont les données EMA seront extraites
        :return: les données EMA en format npy pour l'utterance i avec les premiers traitements.
        :traitement : lecture du fichier .ema et recup des données, filtre sur les articulateurs qui nous intéressent ,
        ajout du lip aperture, interpolation pour données manquantes
        En sortie nparray de dimension (K,13), où K dépend de la longueur de la phrase
         (fréquence d'échantillonnage de 200Hz donc K = 200*durée_en_sec)
        """
        path_ema_file = os.path.join(path_ema_files, EMA_files[k] + ".ema")
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
                    column_names[int(col_id.split('_', 1)[-1])] = col_name

            ema_data = np.fromfile(ema_annotation, "float32").reshape(n_frames, n_columns + 2)
            cols_index = [column_names.index(col) for col in articulators]
            ema_data = ema_data[:, cols_index]
            ema_data = ema_data*100  #données initiales en 10^-5m on les met en millimètre

            #dabord enlever les nan avant de lisser et sous echantillonner
            if np.isnan(ema_data).sum() != 0:
                # Build a cubic spline out of non-NaN values.
                spline = scipy.interpolate.splrep( np.argwhere(~np.isnan(ema_data).ravel()), ema_data[~np.isnan(ema_data)], k=3)
                # Interpolate missing values and replace them.
                for j in np.argwhere(np.isnan(ema_data)).ravel():
                    ema_data[j] = scipy.interpolate.splev(j, spline)
            return ema_data

    def from_wav_to_mfcc(k): #reste à enlever les blancs et normaliser et ajouter trames passées et futures
        """
           :param i: index de l'uttérence (ie numero de phrase) dont les données WAV seront extraites
           :return: les MFCC en format npy pour l'utterance i avec les premiers traitements.
           :traitement : lecture du fichier .wav, extraction des mfcc avec librosa, ajout des Delta et DeltaDelta
           (features qui représentent les dérivées premières et secondes des mfcc)
           On conserve les 13 plus grands MFCC pour chaque frame de 25ms.
           En sortie nparray de dimension (K',13*3)=(K',39). Ou K' dépend de la longueur de la phrase
           ( Un frame toutes les 10ms, donc K' ~ duree_en_sec/0.01 )
           """
        path_wav = os.path.join(path_wav_files, EMA_files[k] + '.wav')
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
        my_mfcc = np.concatenate([frames[i:i + len(my_mfcc)] for i in range(full_window)], axis=1)
        return my_mfcc

    def synchro_ema_mfcc(k,my_ema,my_mfcc):
        """
        :param i:  index de l'uttérence (ie numero de phrase) pour laquelle on va traiter le fichier EMA et MFCC
        :param ema: Données EMA en format .npy en sortie de la fonction first_step_ema_data(i)
        :param mfcc: Données MFCC en format .npy en sortie de la fonction first_step_wav_data(i)
        :return: les npy EMA et MFCC de taille (K,13) et (K,429) avec le même nombre de lignes
        :traitement lecture du fichier d'annotation .lab , on enlève les frames MFCC et EMA qui correspondent à du silence
        On sous échantillone le nparray EMA pour avoir 1 donnée par frame MFCC.
        On ajoute le 'contexte' aux données MFCC ie les 5 frames précédent et les 5 frames suivant chaque frame,
        d'où la taille de mfcc 429 = 5*39 + 5*39 + 39
        """

        #remove blanks at the beginning and the end, en sortie autant de lignes pour les deux
        path_annotation = os.path.join(path_files_annotation, EMA_files[k] + '.lab')
        with open(path_annotation) as file:
            while next(file) != '#\n':
                pass
            labels = [  row.strip('\n').strip('\t').replace(' 26 ', '').split('\t') for row in file     ]
        labels =  [(round(float(label[0]), 2), label[1]) for label in labels]
        start_time = labels[0][0] if labels[0][1] == '#' else 0
        end_time = labels[-2][0] if labels[-1][1] == '#' else labels[-1][0]
        start_frame_mfcc = int(
            np.floor(start_time * 1000 / hop_time))  # nombre de frame mfcc avant lesquelles il ny a que du silence
        end_frame_mfcc = int(np.ceil(end_time * 1000 / hop_time))  # nombre de frame mfcc apres lesquelles il ny a que du silence
        my_mfcc = np.array(my_mfcc[start_frame_mfcc:end_frame_mfcc])
        start_frame_ema = int(np.floor(start_time * sampling_rate_ema))
        end_frame_ema = int(np.ceil(end_time * sampling_rate_ema))
        my_ema = my_ema[start_frame_ema:end_frame_ema]
        #sous echantillonnage de EMA pour synchro avec WAV
        n_frames_wanted = my_mfcc.shape[0]
        my_ema = scipy.signal.resample(my_ema, num=n_frames_wanted)
        #  padding de sorte que l'on intègre les dépendences temporelles : on apprend la trame du milieu
        # mais on ajoute des trames précédent et suivant pour ajouter de l'informatio temporelle

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

        all_EMA_concat = np.concatenate([traj for traj in list_EMA_traj], axis=0)
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
    list_EMA_traj = []
    list_MFCC_frames = []
    cutoff = 25
    N = len(EMA_files)
    if max != "All":
        N = max

    for i in range(N):
        ema = read_ema_file(i)
        mfcc = from_wav_to_mfcc(i)
        ema, mfcc = synchro_ema_mfcc(i, ema, mfcc)
        np.save(os.path.join(root_path, "Donnees_pretraitees", speaker, "ema", EMA_files[i]), ema)
        np.save(os.path.join(root_path, "Donnees_pretraitees",speaker, "mfcc", EMA_files[i]), mfcc)
        ema_filtered = smooth_data(ema)
        np.save(os.path.join(root_path, "Donnees_pretraitees",speaker, "ema_filtered", EMA_files[i]), ema_filtered)
        list_EMA_traj.append(ema_filtered)
        list_MFCC_frames.append(mfcc)

    calculate_norm_values(list_EMA_traj, list_MFCC_frames)
    normalize_data(speaker)
    add_vocal_tract(speaker)

traitement_general_mngu0(max="All")