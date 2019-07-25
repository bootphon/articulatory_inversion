

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
from Traitement.create_filesets import get_fileset_names
import glob
import multiprocessing as mp
from Traitement.class_corpus import Corpus,Speaker

np.seterr(all='raise')


def traitement_general_mocha(N_max,n_procs=0):

    my_corpus_class = Corpus("mocha")
    sampling_rate_ema = my_corpus_class.sampling_rate_ema
    sampling_rate_wav = my_corpus_class.sampling_rate_wav

    def traitement_mocha(speaker,N_max):
        my_speaker_class = Speaker(speaker)
        root_path = dirname(dirname(os.path.realpath(__file__)))
        path_files_treated = os.path.join(root_path, "Donnees_pretraitees",  speaker)
        path_files_brutes = os.path.join(root_path, "Donnees_brutes", "mocha", speaker)

        def create_missing_dir():
            if not os.path.exists(os.path.join(path_files_treated, "ema")):
                os.makedirs(os.path.join(path_files_treated, "ema"))
            if not os.path.exists(os.path.join(path_files_treated, "mfcc")):
                os.makedirs(os.path.join(path_files_treated, "mfcc"))
            if not os.path.exists(os.path.join(path_files_treated, "ema_final")):
                os.makedirs(os.path.join(path_files_treated, "ema_final"))

            if not os.path.exists(os.path.join(path_files_brutes, "wav_cut")):
                os.makedirs(os.path.join(path_files_brutes, "wav_cut"))
            files = glob.glob(os.path.join(path_files_treated, "ema", "*"))
            files += glob.glob(os.path.join(path_files_treated, "mfcc", "*"))
            files += glob.glob(os.path.join(path_files_treated, "ema_final", "*"))
            files += glob.glob(os.path.join(path_files_brutes, "wav_cut", "*"))

            for f in files:
                os.remove(f)

        def read_ema_file(k):
            path_ema_file = os.path.join(path_files_brutes, EMA_files[k] + ".ema")
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
                ema_data = np.fromfile(ema_annotation, "float32").reshape(n_frames, -1)
                cols_index = [column_names.index(col) for col in my_speaker_class.articulators]
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

        def remove_silences(my_ema,my_mfcc,k):
            marge=0
            if speaker in sp_with_trans:
                path_annotation = os.path.join(path_files_brutes, wav_files[k] + '.lab')
                with open(path_annotation) as file:
                    labels = [
                        row.strip('\n').strip('\t').replace(' 26 ', '').split(' ')
                        for row in file
                    ]
                xtrm = [max(float(labels[0][1])-marge , 0), float(labels[-1][0])+marge]
                xtrm_temp_ema = [int( xtrm[0] * sampling_rate_ema ),
                                   min(int((xtrm[1] * sampling_rate_ema)+ 1 ) , len(my_ema))   ]

                xtrm_temp_mfcc =[ int( xtrm[0]  / hop_time) ,
                                   int(np.ceil(xtrm[1]  / hop_time))]

                my_mfcc = my_mfcc[xtrm_temp_mfcc[0]:xtrm_temp_mfcc[1]]

                my_ema = my_ema[xtrm_temp_ema[0]:xtrm_temp_ema[1], :]

            return my_ema, my_mfcc

        def from_wav_to_mfcc(my_wav):
            my_mfcc = librosa.feature.mfcc(y=my_wav, sr=sampling_rate_wav, n_mfcc=n_coeff,
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


        create_missing_dir()
        EMA_files = sorted([name for name in os.listdir(path_files_brutes) if "palate" not in name])
        EMA_files = sorted([name[:-4] for name in EMA_files if name.endswith('.ema')])
        n_columns = 20
        wav_files = sorted([name[:-4] for name in os.listdir(path_files_brutes) if name.endswith('.wav')])
        N = len(EMA_files)
        if N_max != 0:
            N = N_max

        for i in range(N):
         #   if i+1%50 == 0:
          #      print("{} out of {}".format(i,N))
            ema = read_ema_file(i)
            ema_VT = my_speaker_class.add_vocal_tract(ema)
            ema_VT_smooth = my_speaker_class.smooth_data(ema_VT) # filtrage pour meilleur calcul des norm_values
           # ema_VT_smooth = ema_VT
            path_wav = os.path.join(path_files_brutes, wav_files[i] + '.wav')
            wav, sr = librosa.load(path_wav, sr=None)  # chargement de données
            wav = 0.5*wav/np.max(wav)
            mfcc = from_wav_to_mfcc(wav)
            ema_VT_smooth,mfcc = remove_silences(ema_VT_smooth,mfcc,i)
            ema_VT_smooth,mfcc = my_speaker_class.synchro_ema_mfcc(ema_VT_smooth,mfcc)

            ema_VT_,rien = remove_silences(ema_VT, mfcc, i)
            ema_VT, rien = my_speaker_class.synchro_ema_mfcc(ema_VT, mfcc)

            np.save(os.path.join(root_path, "Donnees_pretraitees",  speaker, "ema", EMA_files[i]), ema_VT)
            np.save(os.path.join(root_path, "Donnees_pretraitees",  speaker, "mfcc", EMA_files[i]), mfcc)
            np.save(os.path.join(root_path, "Donnees_pretraitees", speaker, "ema_final", EMA_files[i]), ema_VT_smooth)
            my_speaker_class.list_EMA_traj.append(ema_VT_smooth)
            my_speaker_class.list_MFCC_frames.append(mfcc)

        my_speaker_class.calculate_norm_values()
        for i in range(N):
            ema_pas_smooth = np.load(os.path.join(root_path, "Donnees_pretraitees",  speaker, "ema", EMA_files[i]+".npy"))
            ema_VT_smooth = np.load(os.path.join(root_path, "Donnees_pretraitees",  speaker, "ema_final", EMA_files[i]+".npy"))
            mfcc = np.load(os.path.join(root_path, "Donnees_pretraitees",  speaker, "mfcc", EMA_files[i]+".npy"))
            ema_VT_smooth_norma , mfcc = my_speaker_class.normalize_phrase(i, ema_VT_smooth,mfcc)
            ema_pas_smooth_norma , rien =  my_speaker_class.normalize_phrase(i, ema_pas_smooth,mfcc)
            new_sr = 1/hop_time #on a rééchantillonner pour avoir autant de points que dans mfcc : 1 points toutes les 10ms : 100 points par sec

            ema_VT_smooth_norma = my_speaker_class.smooth_data(ema_VT_smooth_norma,new_sr)
            np.save(os.path.join(root_path, "Donnees_pretraitees", speaker, "ema", EMA_files[i]), ema_pas_smooth_norma)
            np.save(os.path.join(root_path, "Donnees_pretraitees", speaker, "mfcc", EMA_files[i]), mfcc)
            np.save(os.path.join(root_path, "Donnees_pretraitees", speaker, "ema_final", EMA_files[i]), ema_VT_smooth_norma)

      #  split_sentences(speaker)
        get_fileset_names(speaker)

    frame_time = 0.025
    hop_time = 0.010  # en ms
    hop_length = int(hop_time * sampling_rate_wav)
    frame_length = int(frame_time * sampling_rate_wav)
    window = 5
    n_coeff = 13
    sp_with_trans = ["fsew0", "msak0", "mjjn0", "ffes0"]

    for sp in my_corpus_class.speakers:
        print("sp ",sp)
        traitement_mocha(sp,N_max = N_max)
        print("Done for speaker ",sp)

#traitement_general_mocha(N_max =0)