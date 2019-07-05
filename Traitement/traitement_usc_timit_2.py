""" Lecture des données EMA pour le corpus MNGU0. On ne conserve que les données concernant les articulateurs indiqués
 dans articulators cest a dire 6 articulateurs en 2Dimensions.
 on ajoute une colonne correspondant à l'ouverture des lèvres, cest donc la 13ème colonne
 on ne normalise pas les données mais on conserve la std et mean des mfcc et ema pour pouvoir normaliser par la suiite)

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

""" after this script the order of the articulators is the following : """
order_arti_usctimit = [
        'tt_x', 'tt_y', 'td_x', 'td_y', 'tb_x', 'tb_y', 'li_x', 'li_y',
        'ul_x', 'ul_y', 'll_x', 'll_y']

def traitement_general_usc_timit(speaker,N):
    print("GOING TO DO USC TIMIT PRE PROCESS FOR SPEAKER {}".format(speaker))
    root_path = dirname(dirname(os.path.realpath(__file__)))

    sampling_rate_ema = 100
    #articulators NOT in the same order that those of MOCHA
    articulators = ["ul_x", "ul_y", "ll_x", "ll_y", "li_x", "li_y", "td_x", "td_y", "tb_x", "tb_y", "tt_x", "tt_y"]
    new_order_arti =  [articulators.index(col) for col in order_arti_usctimit] #change the order from the initial

    n_col_ema = len(articulators) #lip aperture

    path_files_brutes = os.path.join(root_path, "Donnees_brutes","usc_timit",speaker)
    path_files_treated = os.path.join(root_path, "Donnees_pretraitees","usc_timit_"+speaker)
    path_files_annotation = os.path.join(root_path, "Donnees_brutes","usc_timit",speaker,"trans")

    def empty_and_create_dirs():
        if not os.path.exists(os.path.join(path_files_brutes, "mat_cut")) :
           os.makedirs(os.path.join(path_files_brutes, "mat_cut"))

        if not os.path.exists(os.path.join(path_files_brutes, "wav_cut")) :
            os.makedirs(os.path.join(path_files_brutes, "wav_cut"))

        if not os.path.exists(os.path.join(path_files_treated, "mfcc")):
            os.makedirs(os.path.join(path_files_treated, "mfcc"))

        if not os.path.exists(os.path.join(path_files_treated, "ema_filtered")):
            os.makedirs(os.path.join(path_files_treated, "ema_filtered"))

        if not os.path.exists(os.path.join(path_files_treated, "ema")):
            os.makedirs(os.path.join(path_files_treated, "ema"))

        files = glob.glob(os.path.join(path_files_brutes,"wav_cut","*"))
        files += glob.glob(os.path.join(path_files_brutes,"mat_cut","*"))
        files += glob.glob(os.path.join(path_files_treated, "ema_filtered", "*"))
        files += glob.glob(os.path.join(path_files_treated, "mfcc", "*"))

        for f in files:
            os.remove(f)

    empty_and_create_dirs()
    # cols_index = None
    # n_columns = 87
    window=5
    # wav_files = sorted([name[:-4] for name in os.listdir(path_wav_files) if name.endswith('.wav')])
    sampling_rate_mfcc = 20000
    frame_time = 25
    hop_time = 10  # en ms
    hop_length = int((hop_time * sampling_rate_mfcc) / 1000)
    frame_length = int((frame_time * sampling_rate_mfcc) / 1000)
    n_coeff = 13
    n_col_mfcc = n_coeff*(2*window+1)*3
    EMA_files = sorted([name[:-4] for name in os.listdir(os.path.join(path_files_brutes, "mat")) if name.endswith(".mat")])

    def cut_all_files(i):
        path_wav = os.path.join(path_files_brutes, "wav", EMA_files[i] + '.wav')
        wav, sr = librosa.load(path_wav, sr=sampling_rate_mfcc)  # chargement de données
        ema = sio.loadmat(os.path.join(path_files_brutes, "mat", EMA_files[i] + ".mat"))
        ema = ema[EMA_files[i]][0]  # dictionnaire où les données sont associées à la clé EMA_files[i]pritn()
        ema = np.concatenate([ema[arti][2][:, [0, 1]] for arti in range(1, 7)], axis=1)
        with open(os.path.join(path_files_annotation, EMA_files[i] + ".trans")) as file:
            labels = np.array([row.strip("\n").split(",") for row in file])
            phone_details = labels[:, [0, 1, -1]]
            id_phrase = set(phone_details[:, 2])
            id_phrase.remove("")

            for k in set(id_phrase):
                temp = phone_details[phone_details[:, 2] == k]
                xtrm = [float(temp[:, 0][0]), float(temp[:, 1][-1])]
                xtrm_temp_ema = [int(np.floor(xtrm[0] * sampling_rate_ema)), int(np.floor( xtrm[1] * sampling_rate_ema) + 1)]
                xtrm_temp_wav = [int(np.floor(xtrm[0] * sampling_rate_mfcc)), int(np.floor( xtrm[1] * sampling_rate_mfcc) + 1)]

                ema_temp = ema[xtrm_temp_ema[0]:xtrm_temp_ema[1], :]
                wav_temp = wav[xtrm_temp_wav[0]:xtrm_temp_wav[1]]

                if os.path.exists(os.path.join(path_files_brutes, "mat_cut", EMA_files[i][:-7] + str(k)+".npy")):
                    premiere_partie_ema = np.load(os.path.join(path_files_brutes, "mat_cut", EMA_files[i][:-7] + str(k)+".npy"))
                    ema_temp = np.concatenate((ema_temp,premiere_partie_ema ), axis=0)
                    premiere_partie_wav = np.load(os.path.join(path_files_brutes, "wav_cut", EMA_files[i][:-7] + str(k)+".npy"))
                    wav_temp = np.concatenate((wav_temp,premiere_partie_wav ), axis=0)

                np.save(os.path.join(path_files_brutes, "mat_cut", EMA_files[i][:-7] + str(k)),  ema_temp)
                np.save(os.path.join(path_files_brutes, "wav_cut", EMA_files[i][:-7]  + str(k)), wav_temp)
    print("cut files - 1 step out of 4 ")
  #  N=len(EMA_files)
    if N =='All' :
        N = len(EMA_files)
    for i in range(N):
        if i%50 == 0:
            print(i)
        cut_all_files(i)
    EMA_files_2 = sorted([name[:-4] for name in os.listdir(os.path.join(path_files_brutes, "wav_cut")) if name.endswith(".npy")])

    def treat_ema_files(i):
        ema = np.load(os.path.join(path_files_brutes,"mat_cut",EMA_files_2[i]+".npy"))
        ema = ema / 10  # je pense que les données sont en 10^-4m on les met en mm
        if np.isnan(ema).sum() != 0:
            print("nan")
            spline = scipy.interpolate.splrep(np.argwhere(~np.isnan(ema).ravel()),ema[~np.isnan(ema)], k=3)
            for j in np.argwhere(np.isnan(ema)).ravel():
                ema[j] = scipy.interpolate.splev(j, spline)
        ema = ema[:, new_order_arti]  # change order of arti to have the one wanted
        np.save(os.path.join(path_files_treated,"ema",EMA_files_2[i]),ema)
    N_2 = len(EMA_files_2)
    print("treat ema files - 2 step out of 4 ")

    for i in range(N_2):
        if i % 100 == 0:
            print(i)
        treat_ema_files(i)

    def treat_wav_files(i):
        """
           :param i: index de l'uttérence (ie numero de phrase) dont les données WAV seront extraites
           :return: les MFCC en format npy pour l'utterance i avec les premiers traitements.
           :traitement : lecture du fichier .wav, extraction des mfcc avec librosa, ajout des Delta et DeltaDelta
           (features qui représentent les dérivées premières et secondes des mfcc)
           On conserve les 13 plus grands MFCC pour chaque frame de 25ms.
           En sortie nparray de dimension (K',13*3)=(K',39). Ou K' dépend de la longueur de la phrase
           ( Un frame toutes les 10ms, donc K' ~ duree_en_sec/0.01 )
           """
        path_wav = os.path.join(path_files_brutes, "wav_cut",EMA_files_2[i] + '.npy')
        data = np.load(path_wav)  # chargement de données
        mfcc = librosa.feature.mfcc(y=data, sr=sampling_rate_mfcc, n_mfcc=n_coeff,
                                    n_fft=frame_length, hop_length=hop_length ).T
        dyna_features = get_delta_features(mfcc)
        dyna_features_2 = get_delta_features(dyna_features)
        mfcc = np.concatenate((mfcc, dyna_features, dyna_features_2), axis=1)
        padding = np.zeros((window, mfcc.shape[1]))
        frames = np.concatenate([padding, mfcc, padding])
        full_window = 1 + 2 * window
        mfcc = np.concatenate([frames[i:i + len(mfcc)] for i in range(full_window)], axis=1)
        np.save(os.path.join(path_files_treated,"mfcc",EMA_files_2[i]),mfcc)
    print("treat wav files - 3 step out of 4 ")
    for i in range(N_2) :
        if i % 100 == 0 :
            print("{} out of {}".format(i,N_2))
        treat_wav_files(i)

    def for_normalisation():
        print("calculating norm values - 3 step out of 4 ")

        ALL_EMA = []
        ALL_MFCC = []
        n_pad = 30

        for i in range(len(EMA_files_2)):
            ema_i = np.load(os.path.join(path_files_treated,"ema",EMA_files_2[i]+".npy"))
            mfcc_i= np.load(os.path.join(path_files_treated,"mfcc",EMA_files_2[i]+".npy"))
            n_frames_wanted = len(mfcc_i)
            ema_i = scipy.signal.resample(ema_i, num=n_frames_wanted)
            np.save(os.path.join(path_files_treated, "ema", EMA_files_2[i]), ema_i)

            if len(ema_i)!= len(mfcc_i):
                print("pbmmm ",EMA_files_2[i],len(ema_i),len(mfcc_i))
            ALL_EMA.append(ema_i)
            ALL_MFCC.append(mfcc_i)
        all_mean_ema = np.array([np.mean(ALL_EMA[i], axis=0) for i in range(len(ALL_EMA))])
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
        np.save("norm_values","moving_average_ema_" + speaker, smoothed_moving_average)
        np.save("norm_values","std_ema_" + speaker, std_ema)
        np.save("norm_values","mean_ema_" + speaker, mean_ema)
        return std_ema, mean_ema, smoothed_moving_average, mean_mfcc,std_mfcc
    cutoff=10
    weights = low_pass_filter_weight(cut_off=cutoff, sampling_rate=sampling_rate_ema)

    std_ema, mean_ema, smoothed_moving_average, mean_mfcc,std_mfcc = for_normalisation()
    xtrm = 30
    def normalization_and_lowpass(std_ema, mean_ema, smoothed_moving_average, mean_mfcc,std_mfcc):
        print("cutting files between each sentence - 4 step out of 4 ")

        for i in range(len(EMA_files_2)):
            if i % 100 == 0:
                print("{} out of {}".format(i, len(EMA_files_2)))
            ema = np.load(os.path.join(path_files_treated,"ema",EMA_files_2[i]+".npy"))
            mfcc = np.load(os.path.join(path_files_treated,"mfcc",EMA_files_2[i]+".npy"))

            if len(ema)!= len(mfcc):
                print("pbmmm 2",EMA_files_2[i],len(ema),len(mfcc))
            ema = (ema - smoothed_moving_average[i,:] ) / max(std_ema)
         #   ema = (ema - mean_ema)/max(std_ema)
            mfcc = (mfcc-mean_mfcc)/std_mfcc

            ema_filtered = np.concatenate([np.expand_dims(np.pad(ema[:, k], (xtrm, xtrm), "symmetric"), 1)
                                           for k in range(ema.shape[1])], axis=1)

            ema_filtered = np.concatenate([np.expand_dims(np.convolve(channel, weights, mode='same'), 1)
                                           for channel in ema_filtered.T], axis=1)
            ema_filtered = ema_filtered[xtrm:-xtrm, :]
            np.save(os.path.join(path_files_treated, "ema", EMA_files_2[i]), ema)

            np.save(os.path.join(path_files_treated, "ema_filtered", EMA_files_2[i]), ema_filtered)
            np.save(os.path.join(path_files_treated, "mfcc", EMA_files_2[i]), mfcc)

    normalization_and_lowpass( std_ema, mean_ema, smoothed_moving_average, mean_mfcc,std_mfcc)

N = "All"

speakers = ["F1","F5","M1","M3"]

for sp in speakers :
#traitement_general_usc_timit(speakers[0],N)
    traitement_general_usc_timit(sp,N)

def rename(): #the trans folder of usc timit for the speaker m3 have the wrong name folders (mri instead of ema) the script
    #rename all the trans files for this speaker
    folder = r"C:\Users\Maud Parrot\Documents\stages\STAGE LSCP\Maud_travaux\inversion_articulatoire\Donnees_brutes\usc_timit\M3\trans"
    trans_files = sorted([name for name in os.listdir(folder)])
    for name in trans_files :
        os.rename(os.path.join(folder,name),os.path.join(folder,name.replace("mri","ema")))

#rename()
