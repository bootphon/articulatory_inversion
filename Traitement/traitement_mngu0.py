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
#from Traitement.add_dynamic_features import get_delta_features
import librosa
import shutil

import glob
from Traitement.split_sentences import split_sentences
import multiprocessing as mp

from Traitement.class_corpus import Speaker
from Traitement.fonctions_utiles import get_fileset_names,get_delta_features

""" after this script the order of the articulators is the following : """
order_arti_MNGU0 = [
        'tt_x','tt_y','td_y','td_y','tb_x','tb_y',
        'li_x','li_y','ul_x','ul_y',
        'll_x','ll_y']

def traitement_general_mngu0(N_max="All"):
    speaker = "MNGU0"

    my_speaker_class = Speaker("MNGU0")
    sampling_rate_ema = my_speaker_class.sampling_rate_ema
    sampling_rate_wav = my_speaker_class.sampling_rate_wav

    root_path = dirname(dirname(os.path.realpath(__file__)))
    path_files_annotation = os.path.join(root_path, "Donnees_brutes",speaker,"phone_labels")
   # sampling_rate_ema = 200
    #articulators in the same order that those of MOCHA
    articulators = [
        'T1_py','T1_pz','T3_py','T3_pz','T2_py','T2_pz',
        'jaw_py','jaw_pz','upperlip_py','upperlip_pz',
        'lowerlip_py','lowerlip_pz']

    path_ema_files = os.path.join(root_path, "Donnees_brutes",speaker,"ema")
    EMA_files = sorted([name[:-4] for name in os.listdir(path_ema_files) if name.endswith('.ema')])
    path_files_treated = os.path.join(root_path, "Donnees_pretraitees",speaker)
    path_files_brutes = os.path.join(root_path, "Donnees_brutes",speaker)

    n_columns = 87
    window=5
    path_wav_files = os.path.join(root_path, "Donnees_brutes",speaker,"wav")
    frame_time = 25
    hop_time = 10  # en ms
    hop_length = int((hop_time * sampling_rate_wav) / 1000)
    frame_length = int((frame_time * sampling_rate_wav) / 1000)
    n_coeff = 13

    def create_missing_dir():
        if not os.path.exists(os.path.join(os.path.join(path_files_treated, "ema"))):
            os.makedirs(os.path.join(path_files_treated, "ema"))
        if not os.path.exists(os.path.join(os.path.join(path_files_treated, "ema_final"))):
            os.makedirs(os.path.join(path_files_treated, "ema_final"))
        if not os.path.exists(os.path.join(os.path.join(path_files_treated, "mfcc"))):
            os.makedirs(os.path.join(path_files_treated, "mfcc"))

        files = glob.glob(os.path.join(path_files_treated, "ema", "*"))
        files += glob.glob(os.path.join(path_files_treated, "ema_final", "*"))
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
    def remove_silences(k, my_ema, my_mfcc):
        """
        :param k:  index de l'uttérence (ie numero de phrase) pour laquelle on va traiter le fichier EMA et MFCC
        :param my_ema: Données EMA en format .npy en sortie de la fonction first_step_ema_data(i)
        :param my_mfcc: Données MFCC en format .npy en sortie de la fonction first_step_wav_data(i)
        :return: les npy EMA et MFCC de taille (K,13) et (K,429) avec le même nombre de lignes
        :traitement lecture du fichier d'annotation .lab , on enlève les frames MFCC et EMA qui correspondent à du silence
        On sous échantillone le nparray EMA pour avoir 1 donnée par frame MFCC.
        On ajoute le 'contexte' aux données MFCC ie les 5 frames précédent et les 5 frames suivant chaque frame,
        d'où la taille de mfcc 429 = 5*39 + 5*39 + 39
        """
        # remove blanks at the beginning and the end, en sortie autant de lignes pour les deux
        marge= 0
        path_annotation = os.path.join(path_files_annotation, EMA_files[k] + '.lab')
        with open(path_annotation) as file:
            while next(file) != '#\n':
                pass
            labels = [row.strip('\n').strip('\t').replace(' 26 ', '').split('\t') for row in file]
        labels = [(round(float(label[0]), 2), label[1]) for label in labels]
        start_time = labels[0][0] if labels[0][1] == '#' else 0
        end_time = labels[-2][0] if labels[-1][1] == '#' else labels[-1][0]
        xtrm = [max(start_time-marge,0), end_time+marge]

        xtrm_temp_ema = [int(np.floor(xtrm[0] * sampling_rate_ema)),
                         int(min(np.floor(xtrm[1] * sampling_rate_ema) + 1, len(my_ema)))]

        xtrm_temp_mfcc = [int(np.floor(xtrm[0] * 1000 / hop_time)),
                          int(np.ceil(xtrm[1] * 1000 / hop_time))]

        my_mfcc = my_mfcc[xtrm_temp_mfcc[0]:xtrm_temp_mfcc[1]]
        my_ema = my_ema[xtrm_temp_ema[0]:xtrm_temp_ema[1], :]

        return my_ema, my_mfcc

    def from_wav_to_mfcc(my_wav): #reste à enlever les blancs et normaliser et ajouter trames passées et futures
        """
           :param i: index de l'uttérence (ie numero de phrase) dont les données WAV seront extraites
           :return: les MFCC en format npy pour l'utterance i avec les premiers traitements.
           :traitement : lecture du fichier .wav, extraction des mfcc avec librosa, ajout des Delta et DeltaDelta
           (features qui représentent les dérivées premières et secondes des mfcc)
           On conserve les 13 plus grands MFCC pour chaque frame de 25ms.
           En sortie nparray de dimension (K',13*3)=(K',39). Ou K' dépend de la longueur de la phrase
           ( Un frame toutes les 10ms, donc K' ~ duree_en_sec/0.01 )
           """
        my_mfcc = librosa.feature.mfcc(y=my_wav, sr=sampling_rate_wav, n_mfcc=n_coeff,
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

    def synchro_ema_mfcc(my_ema, my_mfcc):
        n_frames_wanted = my_mfcc.shape[0]
        my_ema = scipy.signal.resample(my_ema, num=n_frames_wanted)
        return my_ema, my_mfcc

    create_missing_dir()
    N = len(EMA_files)
    if N_max != 0:
        N = N_max

    for i in range(N):
      #  if i%50==0:
       #     print("{} out of {}".format(i,N))
        ema = read_ema_file(i)
        ema_VT = my_speaker_class.add_vocal_tract(ema)
        ema_VT_smooth = my_speaker_class.smooth_data(ema_VT)  # filtrage pour meilleur calcul des norm_values
        path_wav = os.path.join(path_wav_files, EMA_files[i] + '.wav')
        wav, sr = librosa.load(path_wav, sr=sampling_rate_wav)  # chargement de données
        wav = 0.5 * wav / np.max(wav)
        mfcc  = from_wav_to_mfcc(wav)
        ema_VT_smooth, mfcc = remove_silences(i,ema_VT_smooth, mfcc)
        ema_VT_smooth, mfcc = synchro_ema_mfcc(ema_VT_smooth, mfcc)
        np.save(os.path.join(root_path, "Donnees_pretraitees", speaker, "ema", EMA_files[i]), ema_VT)
        np.save(os.path.join(root_path, "Donnees_pretraitees", speaker, "mfcc", EMA_files[i]), mfcc)
        np.save(os.path.join(root_path, "Donnees_pretraitees", speaker, "ema_final", EMA_files[i]), ema_VT_smooth)
        my_speaker_class.list_EMA_traj.append(ema_VT_smooth)
        my_speaker_class.list_MFCC_frames.append(mfcc)
    my_speaker_class.calculate_norm_values()

    for i in range(N):
        ema_VT_smooth = np.load(os.path.join(root_path, "Donnees_pretraitees", speaker, "ema_final", EMA_files[i] + ".npy"))
        mfcc = np.load(os.path.join(root_path, "Donnees_pretraitees", speaker, "mfcc", EMA_files[i] + ".npy"))
        ema_VT_smooth_norma, mfcc = my_speaker_class.normalize_phrase(i, ema_VT_smooth, mfcc)
        #ema_VT_smooth_norma = my_speaker_class.smooth_data(ema_VT_smooth_norma)
        np.save(os.path.join(root_path, "Donnees_pretraitees", speaker, "mfcc", EMA_files[i]), mfcc)
        np.save(os.path.join(root_path, "Donnees_pretraitees", speaker, "ema_final", EMA_files[i]), ema_VT_smooth_norma)

    split_sentences(speaker)
    get_fileset_names(speaker)
    print("Done for speaker ",speaker)



traitement_general_mngu0(50)
#print("duree : ",str(t2-t1))