"""
Lecture et traitement des données du corpus Haskins (voir https://yale.app.box.com/s/cfn8hj2puveo65fq54rp1ml2mk7moj3h/folder/30415804819)
Il y a un fichier .mat par phrase. Et les données doivent être au préalable téléchargéees dans ..\Donnees_pretraitees.
Les données articulatoires sont consitituées des 12 basiques + de 4 supplémentaires que nous ignorons ici, car c'est le
seul corpus qui fournit ces données articulatoires.
"""
import os
import time
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
import scipy.io as sio



""" after this script the order of the articulators is the following : """

order_arti_haskins = ['td_x','td_y','tb_x','tb_y','tt_x','tt_y','ul_x','ul_y',"ll_x","ll_y",
                      "ml_x","ml_y","li_x","li_y","jl_x","jl_y"]

order_arti =    [ 'tt_x', 'tt_y', 'td_x', 'td_y', 'tb_x', 'tb_y', 'li_x', 'li_y',
        'ul_x', 'ul_y', 'll_x', 'll_y']

def traitement_general_haskins(speaker):

    def detect_silence(data):
        try :#tous les fichiers ne sont pas organisés dans le même ordre dans le dictionnaire, il semble y avoir deux cas

            debut = data[0][5][0][0][1][0][1]
            fin = data[0][5][0][-1][1][0][0]
        except :
            debut = data[0][6][0][0][1][0][1]
            fin   =  data[0][6][0][-1][1][0][0]
        return [debut,fin]


    print("GOING TO DO HASKINS PRE PROCESS FOR SPEAKER {}".format(speaker))
    root_path = dirname(dirname(os.path.realpath(__file__)))

    sampling_rate_ema = 100 # voir si toujours le même, récupérable dans le fichier .mat
    sampling_rate_mfcc = 44100
    path_files_brutes = os.path.join(root_path, "Donnees_brutes","Haskins_IEEE_Rate_Comparison_DB",speaker,"data")
    path_files_treated = os.path.join(root_path, "Donnees_pretraitees","Haskins_"+speaker)
    path_files_annotation = os.path.join(root_path, "Donnees_brutes","Haskins_"+speaker,"trans")
    EMA_files = sorted(  [name[:-4] for name in os.listdir(path_files_brutes) if "palate" not in name])
   # EMA_files = [x for x in EMA_files if x not in files_pbm]
    N = len(EMA_files)

    xtrm = 3
    frame_time = 25
    hop_time = 10  # en ms
    hop_length = int((hop_time * sampling_rate_mfcc) / 1000)
    frame_length = int((frame_time * sampling_rate_mfcc) / 1000)
    n_coeff = 13
    window = 5
    ALL_EMA = []
    ALL_MFCC = []
    cutoff = 30

    def create_directories():

        if not os.path.exists(os.path.join(path_files_treated, "ema")):
            os.makedirs(os.path.join(path_files_treated, "ema"))


        if not os.path.exists(os.path.join(path_files_treated, "ema_filtered")):
            os.makedirs(os.path.join(path_files_treated, "ema_filtered"))

        if not os.path.exists(os.path.join(path_files_treated, "mfcc")):
            os.makedirs(os.path.join(path_files_treated, "mfcc"))


        if not os.path.exists(os.path.join(root_path, "Donnees_brutes","Haskins_IEEE_Rate_Comparison_DB",speaker,"wav")):
            os.makedirs(os.path.join(root_path, "Donnees_brutes","Haskins_IEEE_Rate_Comparison_DB",speaker,"wav"))

    create_directories()

    for i in range(N) :
        if i%200==0:
            print("{} out of {}".format(i,N))
        data = sio.loadmat(os.path.join(path_files_brutes,EMA_files[i]+".mat"))[EMA_files[i]][0]
       # sampling_rate_mfcc = data[0][1][0][0] # 44100 voir si toujours le même
        wav = data[0][2]
        np.save(os.path.join(root_path, "Donnees_brutes","Haskins_IEEE_Rate_Comparison_DB",speaker,"wav",EMA_files[i]),wav)

        ema = np.zeros((len(data[1][2]), len(order_arti_haskins)))
        for arti in range(1, len(data)): # lecture des trajectoires articulatoires dans le dicionnaire
            ema[:, (arti - 1) * 2] = data[arti][2][:, 0]
            ema[:, arti * 2 - 1] =  data[arti][2][:, 2]

        [debut,fin] = detect_silence(data)
        xtrm_ema = [int(np.floor(debut * sampling_rate_ema)), int(np.floor(fin * sampling_rate_ema) + 1)]
        xtrm_wav = [int(np.floor(debut * sampling_rate_mfcc)), int(np.floor(fin* sampling_rate_mfcc) + 1)]
        ema = ema[ xtrm_ema[0]:xtrm_ema[1],:]
        wav = np.reshape(wav[xtrm_wav[0]:xtrm_wav[1]],-1)
        new_order_arti = [order_arti_haskins.index(col) for col in order_arti]
        ema = ema[:,new_order_arti]

        mfcc = librosa.feature.mfcc(y=wav, sr=sampling_rate_mfcc, n_mfcc=n_coeff,
                                    n_fft=frame_length, hop_length=hop_length).T
        dyna_features = get_delta_features(mfcc)
        dyna_features_2 = get_delta_features(dyna_features)
        mfcc = np.concatenate((mfcc, dyna_features, dyna_features_2), axis=1)
        padding = np.zeros((window, mfcc.shape[1]))
        frames = np.concatenate([padding, mfcc, padding])
        full_window = 1 + 2 * window
        mfcc = np.concatenate([frames[i:i + len(mfcc)] for i in range(full_window)], axis=1)
        if np.isnan(ema).sum() != 0:
            print("nan")

        n_frames_wanted = mfcc.shape[0]
        ema = scipy.signal.resample(ema, num=n_frames_wanted)

        np.save(os.path.join(path_files_treated, "ema", EMA_files[i]), ema)
        np.save(os.path.join(path_files_treated, "mfcc", EMA_files[i]), mfcc)

        ALL_EMA.append(ema)
        ALL_MFCC.append(mfcc)

    weights = low_pass_filter_weight(cut_off=10, sampling_rate=sampling_rate_ema)

    n_pad= 30
    all_mean_ema = np.reshape(np.array([np.mean(ALL_EMA[i], axis=0) for i in range(len(ALL_EMA))]),(len(ALL_EMA),12))

    weights_moving_average = low_pass_filter_weight(cut_off=10, sampling_rate=sampling_rate_ema)
    moving_average = np.concatenate([np.expand_dims(np.pad(all_mean_ema[:, k], (n_pad, n_pad), "symmetric"), 1)
                                     for k in range(all_mean_ema.shape[1])], axis=1)
    smoothed_moving_average = np.concatenate(
        [np.expand_dims(np.convolve(channel, weights_moving_average, mode='same'), 1)
         for channel in moving_average.T], axis=1)

    smoothed_moving_average = smoothed_moving_average[n_pad:-n_pad, :]

    std_ema = np.mean(np.array([np.std(x, axis=0) for x in ALL_EMA]), axis=0)
    mean_ema = np.mean(np.array([np.mean(x, axis=0) for x in ALL_EMA]), axis=0)
    std_mfcc = np.mean(np.array([np.std(x, axis=0) for x in ALL_MFCC]), axis=0)
    mean_mfcc = np.mean(np.array([np.mean(x, axis=0) for x in ALL_MFCC]), axis=0)

    np.save(os.path.join("norm_values", "moving_average_ema_" + speaker), smoothed_moving_average)
    np.save(os.path.join("norm_values", "std_ema_" + speaker), std_ema)
    np.save(os.path.join("norm_values", "mean_ema_" + speaker), mean_ema)

    for i in range(N):
        ema = np.load(os.path.join(path_files_treated, "ema", EMA_files[i]+".npy"))
        mfcc = np.load(os.path.join(path_files_treated, "mfcc", EMA_files[i] + ".npy"))
        ema = (ema - smoothed_moving_average[i, :]) / max(std_ema)
        mfcc = (mfcc - mean_mfcc) / std_mfcc

        ema_filtered = np.concatenate([np.expand_dims(np.pad(ema[:, k], (xtrm, xtrm), "symmetric"), 1)
                                       for k in range(ema.shape[1])], axis=1)

        ema_filtered = np.concatenate([np.expand_dims(np.convolve(channel, weights, mode='same'), 1)
                                       for channel in ema_filtered.T], axis=1)

        ema_filtered = ema_filtered[xtrm:-xtrm, :]


        np.save(os.path.join(path_files_treated, "ema", EMA_files[i]), ema)

        np.save(os.path.join(path_files_treated, "ema_filtered", EMA_files[i]), ema_filtered)
        np.save(os.path.join(path_files_treated, "mfcc", EMA_files[i]), mfcc)



speakers = ["F01","F02","F03","F04","M01","M02","M03","M04"]

for sp in speakers:
    traitement_general_haskins(sp)
