### Lecture des données EMA pour le corpus MNGU0. On ne conserve que les données concercnant les articulateurs indiqués
### dans articulators cest a dire 6 articulateurs en 2Dimensions.
### on normalise on soustrayant pour chaque articulateur sa position moyenne et en divisant par sa std
### il semble qu'un articulateur reste à la même position (li_x) voir si on le garde quand meme.
### il n'y au aucune valeur manquante donc pas besoin d'interpolation.
### revoir la longueur de col names
import os
import time
from os.path import dirname
import numpy as np
import scipy.signal
import matplotlib.pyplot as plt
import scipy.interpolate

import sys

print('sys.path ',sys.path)


from Traitement.add_dynamic_features import get_delta_features
import librosa
root_path = dirname(dirname(os.path.realpath(__file__)))

window=5

for time in ["120s"]:
    print("---time :",time)
    path_files_treated = os.path.join(root_path, "Donnees_pretraitees", "donnees_challenge_2017",time)

    path_wav_files = os.path.join(root_path, "Donnees_brutes",time)
    print(path_wav_files,"path wav files")
    wav_files = sorted([name[:-4] for name in os.listdir(path_wav_files) if name.endswith('.wav')])
    print("wav files",wav_files[0:10])
    sampling_rate_mfcc = 16000
    frame_time = 25
    hop_time = 10  # en ms
    hop_length = int((hop_time * sampling_rate_mfcc) / 1000)
    frame_length = int((frame_time * sampling_rate_mfcc) / 1000)
    n_coeff = 13
    n_col_mfcc = n_coeff*(2*window+1)*3
    N=len(wav_files)
    N=5
    def wav_treatment(i): #reste à enlever les blancs et normaliser et ajouter trames passées et futures
        path_wav = os.path.join(path_wav_files, wav_files[i] + '.wav')
        data, sr = librosa.load(path_wav, sr=sampling_rate_mfcc)  # chargement de données
        mfcc = librosa.feature.mfcc(y=data, sr=sampling_rate_mfcc, n_mfcc=n_coeff,
                                    n_fft=frame_length, hop_length=hop_length
                                    ).T
        dyna_features = get_delta_features(mfcc)
        dyna_features_2 = get_delta_features(dyna_features)
        mfcc = np.concatenate((mfcc, dyna_features, dyna_features_2), axis=1)
        ## zero padding de sorte que l'on intègre les dépendences temporelles : on apprend la trame du milieu
        # mais on ajoute des trames précédent et suivant pour ajouter de l'informatio temporelle

        padding = np.zeros((window, mfcc.shape[1]))
        frames = np.concatenate([padding, mfcc, padding])
        full_window = 1 + 2 * window
        mfcc=  np.concatenate(  [frames[i:i + len(mfcc)] for i in range(full_window)], axis=1)
        return mfcc

    ALL_MFCC = np.zeros((1,n_col_mfcc))
    N=len(wav_files)
    for i in range(N):
        if i%5==0:
            print(i," out of ",N)
        mfcc = wav_treatment(i)
        np.save(os.path.join(path_files_treated, wav_files[i]),mfcc)
        ALL_MFCC = np.concatenate((ALL_MFCC,mfcc),axis=0)

    ALL_MFCC = ALL_MFCC[1:]
    mean_mfcc = np.mean(ALL_MFCC,axis=0)
    std_mfcc = np.std(ALL_MFCC,axis=0)

    for i in range(155,N):
        if i%5 ==0:
            print(i," out of ",N)
        mfcc = np.load(os.path.join(path_files_treated,wav_files[i]+".npy"))
        mfcc = (mfcc - mean_mfcc)/std_mfcc
        np.save(os.path.join(path_files_treated, wav_files[i]),mfcc)

    def concat_all_numpy_from(path):
        list = []
        for r, d, f in os.walk(path):
            for file in f:
                data_file = np.load(os.path.join(path,file))
                list.append(data_file)
        return list

    #all_numpy_ZS = concat_all_numpy_from(os.path.join(path_files_treated,time))
    #np.save(os.path.join(path_files_treated,time,"X_ZS.npy"),all_numpy_ZS)












