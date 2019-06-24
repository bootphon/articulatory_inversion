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

def traitement_general_usc_timit(N):
    speaker = "F1"
    root_path = dirname(dirname(os.path.realpath(__file__)))
    path_files_annotation = os.path.join(root_path, "Donnees_brutes","usc_timit",speaker,"trans")

    sampling_rate_ema = 100
    #articulators in the same order that those of MOCHA
    articulators = ["UL_x","UL_y", "LL_x","LL_y","JAW_x" ,"JAW_y", "TD_x","TD_y", "TB_x","TB_y"]

    n_col_ema = len(articulators)+1 #lip aperture
    path_ema_files = os.path.join(root_path, "Donnees_brutes","usc_timit",speaker,"mat")
    EMA_files = sorted([name[:-4] for name in os.listdir(path_ema_files) if name.endswith(".mat")])

    cols_index = None
    n_columns = 87
    window=5
    path_wav_files = os.path.join(root_path, "Donnees_brutes","usc_timit",speaker,"wav")
   # wav_files = sorted([name[:-4] for name in os.listdir(path_wav_files) if name.endswith('.wav')])
    sampling_rate_mfcc = 20000
    frame_time = 25
    hop_time = 10  # en ms
    hop_length = int((hop_time * sampling_rate_mfcc) / 1000)
    frame_length = int((frame_time * sampling_rate_mfcc) / 1000)
    n_coeff = 13
    n_col_mfcc = n_coeff*(2*window+1)*3

    def first_step_ema_data(i):
        """
        :param i: index de l'uttérence (ie numero de phrase) dont les données EMA seront extraites
        :return: les données EMA en format npy pour l'utterance i avec les premiers traitements.
        :traitement : lecture du fichier .ema et recup des données, filtre sur les articulateurs qui nous intéressent ,
        ajout du lip aperture, interpolation pour données manquantes
        En sortie nparray de dimension (K,13), où K dépend de la longueur de la phrase
         (fréquence d'échantillonnage de 200Hz donc K = 200*durée_en_sec)
        """

        ema_data = sio.loadmat(os.path.join(path_ema_files,  EMA_files[i] + ".mat"))
        ema_data = ema_data[ EMA_files[i]][0]  # dictionnaire où les données sont associées à la clé EMA_files[i]pritn()
        arti = 2
        ema_data = np.concatenate([ema_data[arti][2][:, [0, 1]] for arti in range(1, 7)], axis=1)
        ind_1, ind_2 = [articulators.index("UL_x"), articulators.index("UL_y")]
        lip_aperture = (ema_data[:, ind_1] - ema_data[:, ind_2]).reshape(len(ema_data), 1)
        ema_data = np.concatenate((ema_data, lip_aperture), axis=1)

        #retourner une liste ou chaque élement est un np array (K,12) correspondant à une phrase
        return ema_data

    def first_step_wav_data(i): #reste à enlever les blancs et normaliser et ajouter trames passées et futures
        """
           :param i: index de l'uttérence (ie numero de phrase) dont les données WAV seront extraites
           :return: les MFCC en format npy pour l'utterance i avec les premiers traitements.
           :traitement : lecture du fichier .wav, extraction des mfcc avec librosa, ajout des Delta et DeltaDelta
           (features qui représentent les dérivées premières et secondes des mfcc)
           On conserve les 13 plus grands MFCC pour chaque frame de 25ms.
           En sortie nparray de dimension (K',13*3)=(K',39). Ou K' dépend de la longueur de la phrase
           ( Un frame toutes les 10ms, donc K' ~ duree_en_sec/0.01 )
           """

        path_wav = os.path.join(path_wav_files, EMA_files[i] + '.wav')
        data, sr = librosa.load(path_wav, sr=sampling_rate_mfcc)  # chargement de données
        mfcc = librosa.feature.mfcc(y=data, sr=sampling_rate_mfcc, n_mfcc=n_coeff,
                                    n_fft=frame_length, hop_length=hop_length
                                    ).T
        dyna_features = get_delta_features(mfcc)
        dyna_features_2 = get_delta_features(dyna_features)
        mfcc = np.concatenate((mfcc, dyna_features, dyna_features_2), axis=1)
        return mfcc

    def second_step_data(i,ema,mfcc):
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

        all_ema_i = []
        all_mfcc_i = []
        with open(os.path.join(root_path,"Donnees_brutes","usc_timit",speaker, "trans", EMA_files[i] + ".trans")) as file:
            labels = np.array([row.strip("\n").split(",") for row in file])
            phone_details = labels[:, [0, 1, -1]]
            id_phrase = set(phone_details[:, 2])
            id_phrase.remove("")
            for i in set(id_phrase):

                xtrm = np.array(phone_details[phone_details[:, 2] == i][:, [0]][[0, -1]]).astype(float)
                xtrm  = [xtrm[0][0],xtrm[1][0]]
                xtrm_temp_ema = np.floor(xtrm * sampling_rate_ema).astype(int)
                xtrm_temp_mfcc = np.floor(xtrm * sampling_rate_mfcc).astype(int)# on a 100 point par secondes, en xtrm secondes on a xtrm*100 points

                ema_temp =ema[xtrm_temp_ema[0]:xtrm_temp_ema[0] + 1, :]
                mfcc_temp = mfcc[xtrm_temp_mfcc[0]:xtrm_temp_mfcc[1] + 1, :]

                ema_temp = scipy.signal.resample(ema_temp, num=len(mfcc_temp))       #sous echantillonnage de EMA pour synchro avec WAV
                #  padding de sorte que l'on intègre les dépendences temporelles : on apprend la trame du milieu
                # mais on ajoute des trames précédent et suivant pour ajouter de l'informatio temporelle
                padding = np.zeros((window, mfcc_temp.shape[1]))
                frames = np.concatenate([padding, mfcc_temp, padding])
                full_window = 1 + 2 * window
                mfcc_temp = np.concatenate([frames[i:i + len(mfcc_temp)] for i in range(full_window)], axis=1)
                all_ema_i.append(ema_temp)
                all_mfcc_i.append(mfcc_temp)

        return all_ema_i,all_mfcc_i

    ALL_EMA = np.zeros((1,n_col_ema))
    ALL_MFCC = np.zeros((1,n_col_mfcc))
    cutoff = 25
    weights = low_pass_filter_weight(cut_off=cutoff, sampling_rate=sampling_rate_ema)

    #traitement uttérance par uttérance des phrases
    if N == "All":
        N = len(EMA_files)

    for i in range(N):
        if i%1 ==0:
            print(i," out of ",N)
        ema = first_step_ema_data(i)
        mfcc = first_step_wav_data(i)
        print("first step",ema.shape,mfcc.shape)
        ema, mfcc = second_step_data(i, ema, mfcc)  #liste des données pour chaque phrase

        print("second step ",[[ema[i].shape,mfcc[i].shape] for i in range(len(ema))])
        for k in range(len(ema)):
            if ema[k].shape[0] != mfcc[k].shape[0]:
                print(ema[k].shape,mfcc[k].shape)
                print("probleme de shape")
            if not os.path.exists( os.path.join(root_path, "Donnees_pretraitees","usc_timit",speaker,"ema")):
                os.makedirs( os.path.join(root_path, "Donnees_pretraitees","usc_timit",speaker,"ema"))
            if not os.path.exists(os.path.join(root_path, "Donnees_pretraitees", "usc_timit",speaker, "mfcc")):
                os.makedirs(os.path.join(root_path, "Donnees_pretraitees", "usc_timit",speaker, "mfcc"))

            np.save(os.path.join(root_path, "Donnees_pretraitees","usc_timit",speaker,"ema", EMA_files[i]+"_"+str(k)),ema[k]) #sauvegarde temporaire pour la récup après
            np.save(os.path.join(root_path,  "Donnees_pretraitees","usc_timit",speaker,"mfcc", EMA_files[i]+"_"+str(k)),mfcc[k]) #sauvegarde temporaire pour la récup après
            print("shape",np.array(ema).shape)
            ALL_EMA = np.concatenate((ALL_EMA,np.array(ema)),axis=0)
        ALL_MFCC = np.concatenate((ALL_MFCC,np.array(mfcc)),axis=0)

    ALL_EMA  =ALL_EMA[1:]  #concaténation de toutes les données EMA
    ALL_MFCC = ALL_MFCC[1:] #concaténation de toutes les frames mfcc

    # de taille 429 : moyenne et std de chaque coefficient
    mean_mfcc = np.mean(ALL_MFCC,axis=0)
    std_mfcc = np.std(ALL_MFCC,axis=0)

    # de taille 13 : moyenne et std de chaque articulateur
    mean_ema = np.mean(ALL_EMA, axis=0)
    std_ema = np.std(ALL_EMA, axis=0)
    np.save("std_ema_usc_timit",std_ema)
    np.save("mean_ema_usc_timit",mean_ema)
    np.save("std_mfcc_usc_timit", std_mfcc)
    np.save("mean_mfcc_usc_timit", mean_mfcc)


    print("std ema",std_ema)

    # construction du filtre passe bas que lon va appliquer à chaque frame mfcc et trajectoire d'articulateur
    # la fréquence de coupure réduite de 0.1 a été choisi manuellement pour le moment, et il se trouve qu'on
    # n'a pas besoin d'un filtre différent pour mfcc et ema
   # order = 5
   # filt_b, filt_a = scipy.signal.butter(order, 0.1, btype='lowpass', analog=False) #fs=sampling_rate_ema)

    EMA_files_2 = sorted([name[:-4] for name in os.listdir(os.path.join(root_path, "Donnees_pretraitees","usc_timit","ema")) if name.endswith(".npy")])
    N_2 = len(EMA_files_2)
    print("number of files :",N_2)

    for i in range(N_2):

        ema = np.load(os.path.join(root_path, "Donnees_pretraitees","usc_timit","ema", EMA_files_2[i]+".npy"))
        ema = (ema - mean_ema) /std_ema

        mfcc = np.load(os.path.join(root_path, "Donnees_pretraitees","usc_timit","mfcc", EMA_files_2[i] + ".npy"))
        mfcc = (mfcc - mean_mfcc) / std_mfcc
        np.save(os.path.join(root_path, "Donnees_pretraitees","usc_timit","ema", EMA_files_2[i]), ema)
        np.save(os.path.join(root_path, "Donnees_pretraitees","usc_timit","mfcc", EMA_files_2[i]),mfcc)

        ema_filtered = np.concatenate([np.expand_dims(np.convolve(channel, weights, mode='same'), 1)
                              for channel in ema.T], axis=1)

        difference = len(ema_filtered) - len(ema)
        halfdif = int(difference / 2)
        if difference < 0:  # sequence filtree moins longue que l'originale
            ema_filtered = np.pad(ema_filtered, (halfdif, difference - halfdif), "edge")
        elif difference > 0:
            ema_filtered = ema_filtered[halfdif:-(difference - halfdif)]
        if len(ema_filtered) != len(ema):  # sequence filtree plus longue que loriginale
            print("pbm shape", len(ema_filtered), len(y))
        np.save(os.path.join(root_path,"Donnees_pretraitees","usc_timit","ema_filtered", EMA_files_2[i]),ema_filtered)

        #if mfcc.shape[0]!=ema.shape[0]:
         #   print("PBM DE SHAPE",i,EMA_files[i])
           # print(len(mfcc),len(ema))
          #  diff = len(ema) - len(mfcc)
           # ema = ema[:-diff, :]
        #np.save(os.path.join(root_path, "Donnees_pretraitees\MNGU0\ema_", EMA_files[i]),ema)
      #  np.save(os.path.join(root_path, "Donnees_pretraitees\MNGU0\mfcc", EMA_files[i]),mfcc)
    #path_mfcc_mngu0 = os.path.join(path_files_treated,"MNGU0_mfcc.npy")
    #np.save(path_mfcc_mngu0, ALL_MFCC)

    #path_ema_mngu0 = os.path.join(path_files_treated,"MNGU0_ema.npy")
    #np.save(path_ema_mngu0, ALL_EMA)

N=3
traitement_general_usc_timit(N)



