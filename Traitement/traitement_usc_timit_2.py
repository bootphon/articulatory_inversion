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

""" after this script the order of the articulators is the following : """
order_arti_usctimit = [
        'tt_x', 'tt_y', 'td_x', 'td_y', 'tb_x', 'tb_y', 'li_x', 'li_y',
        'ul_x', 'ul_y', 'll_x', 'll_y','la']


def traitement_general_usc_timit(N,speaker):
    root_path = dirname(dirname(os.path.realpath(__file__)))
    path_files_annotation = os.path.join(root_path, "Donnees_brutes","usc_timit",speaker,"trans")

    sampling_rate_ema = 100
    #articulators NOT in the same order that those of MOCHA
  #  articulators = ["UL_x","UL_y", "LL_x","LL_y","JAW_x" ,"JAW_y", "TD_x","TD_y", "TB_x","TB_y","TT_x","TT_y"]
    articulators = ["ul_x", "ul_y", "ll_x", "ll_y", "li_x", "li_y", "td_x", "td_y", "tb_x", "tb_y", "tt_x", "tt_y","la"]
    new_order_arti =  [articulators.index(col) for col in order_arti_usctimit] #change the order from the initial

    n_col_ema = len(articulators) #lip aperture
    path_files_brutes = os.path.join(root_path, "Donnees_brutes","usc_timit",speaker)


    cols_index = None
    n_columns = 87
    window=5
   # wav_files = sorted([name[:-4] for name in os.listdir(path_wav_files) if name.endswith('.wav')])
    sampling_rate_mfcc = 20000
    frame_time = 25
    hop_time = 10  # en ms
    hop_length = int((hop_time * sampling_rate_mfcc) / 1000)
    frame_length = int((frame_time * sampling_rate_mfcc) / 1000)
    n_coeff = 13
    n_col_mfcc = n_coeff*(2*window+1)*3

    def first_step(i):
        path_wav = os.path.join(path_files_brutes,"wav", EMA_files[i] + '.wav')
        wav, sr = librosa.load(path_wav, sr=sampling_rate_mfcc)  # chargement de données

        ema_data = sio.loadmat(os.path.join(path_files_brutes,"mat", EMA_files[i] + ".mat"))
        ema_data = ema_data[EMA_files[i]][0]  # dictionnaire où les données sont associées à la clé EMA_files[i]pritn()
        ema_data = np.concatenate([ema_data[arti][2][:, [0, 1]] for arti in range(1, 7)], axis=1)
        ind_1, ind_2 = [articulators.index("ul_y"), articulators.index("ll_y")]
        lip_aperture = (ema_data[:, ind_1] - ema_data[:, ind_2]).reshape(len(ema_data), 1)
        ema_data = np.concatenate((ema_data, lip_aperture), axis=1)
        ema_data = ema_data / 10  # je pense que les données sont en 10^-4m on les met en mm

        if np.isnan(ema_data).sum() != 0:
            spline = scipy.interpolate.splrep(np.argwhere(~np.isnan(ema_data).ravel()),
                                              ema_data[~np.isnan(ema_data)], k=3)
            for j in np.argwhere(np.isnan(ema_data)).ravel():
                ema_data[j] = scipy.interpolate.splev(j, spline)
            ema_data = ema_data[:, new_order_arti]  # change order of arti to have the one wanted
            # retourner une liste ou chaque élement est un np array (K,12) correspondant à une phrase

        with open(os.path.join(root_path, "Donnees_brutes", "usc_timit", speaker, "trans",  EMA_files[i] + ".trans")) as file:
            labels = np.array([row.strip("\n").split(",") for row in file])
            phone_details = labels[:, [0, 1, -1]]
            id_phrase = set(phone_details[:, 2])
            id_phrase.remove("")
            for k in set(id_phrase):
                xtrm = np.array(phone_details[phone_details[:, 2] == k][:, [0]][[0, -1]]).astype(float)
                xtrm_temp_ema = [int(xtrm[0][0] * sampling_rate_ema), int(xtrm[1][0] * sampling_rate_ema)+ 1]
                xtrm_temp_wav= [int(xtrm[0])[0]*sampling_rate_mfcc, int(xtrm[1][0]*sampling_rate_mfcc)+1]
                ema_temp = ema_data[xtrm_temp_ema[0]:xtrm_temp_ema[1], :]
                wav_temp = wav[xtrm_temp_wav[0]:xtrm_temp_wav[1],:]
                np.save(os.path.join(path_files_brutes,"mat_cut", EMA_files[i] + "_" + str(k)),
                        ema_temp)  # sauvegarde temporaire pour la récup après
                np.save(os.path.join(path_files_brutes,"wav_cut", EMA_files[i] + "_" + str(k)),
                        wav_temp)  # sauvegarde temporaire pour la récup après
                print(len(ema_temp),len(wav_temp))

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

        path_wav = os.path.join(path_wav_files, EMA_files_2[i] + '.wav')
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
            for k in set(id_phrase):
                xtrm = np.array(phone_details[phone_details[:, 2] == k][:, [0]][[0, -1]]).astype(float)

                xtrm_temp_ema =  [int(xtrm[0][0]*sampling_rate_ema),int(np.floor(xtrm[1][0]*sampling_rate_ema))+1]
                xtrm_temp_mfcc = [ int(  xtrm[0][0] * 1000 / hop_time) ,int(np.ceil(xtrm[1][0] * 1000 / hop_time))+1]
                ema_temp =ema[xtrm_temp_ema[0]:xtrm_temp_ema[1], :]
                mfcc_temp = mfcc[xtrm_temp_mfcc[0]:xtrm_temp_mfcc[1], :]
                ema_temp = scipy.signal.resample(ema_temp, num=len(mfcc_temp))       #sous echantillonnage de EMA pour synchro avec WAV
                #  padding de sorte que l'on intègre les dépendences temporelles : on apprend la trame du milieu
                # mais on ajoute des trames précédent et suivant pour ajouter de l'informatio temporelle
                padding = np.zeros((window, mfcc_temp.shape[1]))
                frames = np.concatenate([padding, mfcc_temp, padding])
                full_window = 1 + 2 * window
                mfcc_temp = np.concatenate([frames[j:j + len(mfcc_temp)] for j in range(full_window)], axis=1)
                all_ema_i.append(ema_temp)
                all_mfcc_i.append(mfcc_temp)

        return all_ema_i,all_mfcc_i

    ALL_EMA = np.zeros((1,n_col_ema))
    ALL_MFCC = np.zeros((1,n_col_mfcc))
    cutoff = 25
    weights = low_pass_filter_weight(cut_off=cutoff, sampling_rate=sampling_rate_ema)
    speaker_2 = "usc_timit_"+speaker
    #traitement uttérance par uttérance des phrases
    EMA_files = sorted([name[:-4] for name in os.listdir(os.path.join(path_files_brutes, "mat")) if name.endswith(".mat")])

    if N == "All":
        N = len(EMA_files)
    for i in range(N):
        first_step(i) # load and treat ema files and
    EMA_files_2 = sorted([name[:-4] for name in os.listdir(os.path.join(path_files_brutes, "wav_cut")) if name.endswith(".wav")])
    for i in range(N):

        ema, mfcc = second_step_data(i, ema, mfcc)  #liste des données pour chaque phrase

        if not os.path.exists(os.path.join(root_path, "Donnees_pretraitees",  speaker_2, "ema")):
            os.makedirs(os.path.join(root_path, "Donnees_pretraitees",  speaker_2, "ema"))
        if not os.path.exists(os.path.join(root_path, "Donnees_pretraitees",  speaker_2, "ema_filtered")):
            os.makedirs(os.path.join(root_path, "Donnees_pretraitees",  speaker_2, "ema_filtered"))
        if not os.path.exists(os.path.join(root_path, "Donnees_pretraitees", speaker_2, "mfcc")):
            os.makedirs(os.path.join(root_path, "Donnees_pretraitees",  speaker_2, "mfcc"))

        for k in range(len(ema)):
            if ema[k].shape[0] != mfcc[k].shape[0]:
                print(ema[k].shape,mfcc[k].shape)
                print("probleme de shape")

            if np.isnan(ema[k]).sum() != 0:
                print("1 :ema is nan!!!")
            np.save(os.path.join(root_path, "Donnees_pretraitees",speaker_2,"ema", EMA_files[i]+"_"+str(k)),ema[k]) #sauvegarde temporaire pour la récup après
            np.save(os.path.join(root_path,  "Donnees_pretraitees",speaker_2,"mfcc", EMA_files[i]+"_"+str(k)),mfcc[k]) #sauvegarde temporaire pour la récup après
            ALL_EMA = np.concatenate((ALL_EMA,np.array(ema[k])),axis=0)
            ALL_MFCC = np.concatenate((ALL_MFCC,np.array(mfcc[k])),axis=0)

    ALL_EMA  =ALL_EMA[1:]  #concaténation de toutes les données EMA
    ALL_MFCC = ALL_MFCC[1:] #concaténation de toutes les frames mfcc


    # de taille 429 : moyenne et std de chaque coefficient
    mean_mfcc = np.mean(ALL_MFCC,axis=0)
    std_mfcc = np.std(ALL_MFCC,axis=0)

    # de taille 13 : moyenne et std de chaque articulateur
    mean_ema = np.mean(ALL_EMA, axis=0)
    std_ema = np.std(ALL_EMA, axis=0)
    np.save("std_ema_usc_timit_"+speaker,std_ema)
    np.save("mean_ema_usc_timit_"+speaker,mean_ema)
    np.save("std_mfcc_usc_timit_" + speaker, std_mfcc)
    np.save("mean_mfcc_usc_timit_" + speaker, mean_mfcc)
    print("std ema",std_ema)

    # construction du filtre passe bas que lon va appliquer à chaque frame mfcc et trajectoire d'articulateur
    # la fréquence de coupure réduite de 0.1 a été choisi manuellement pour le moment, et il se trouve qu'on
    # n'a pas besoin d'un filtre différent pour mfcc et ema
   # order = 5
   # filt_b, filt_a = scipy.signal.butter(order, 0.1, btype='lowpass', analog=False) #fs=sampling_rate_ema)

    EMA_files_2 = sorted([name[:-4] for name in os.listdir(os.path.join(root_path, "Donnees_pretraitees",speaker_2,"ema")) if name.endswith(".npy")])
    N_2 = len(EMA_files_2)
    print("number of files :",N_2)

    for i in range(N_2):
        ema = np.load(os.path.join(root_path, "Donnees_pretraitees",speaker_2,"ema", EMA_files_2[i]+".npy"))
        if np.isnan(ema).sum() != 0:
            print("2 :ema is nan!!!")
        ema = (ema - mean_ema) /std_ema
        if np.isnan(ema).sum() != 0:
            print("3 :ema is nan!!!")
        mfcc = np.load(os.path.join(root_path, "Donnees_pretraitees",speaker_2,"mfcc", EMA_files_2[i] + ".npy"))
        mfcc = (mfcc - mean_mfcc) / std_mfcc
        np.save(os.path.join(root_path, "Donnees_pretraitees",speaker_2,"ema", EMA_files_2[i]), ema)
        np.save(os.path.join(root_path, "Donnees_pretraitees",speaker_2,"mfcc", EMA_files_2[i]),mfcc)

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
        np.save(os.path.join(root_path,"Donnees_pretraitees",speaker_2,"ema_filtered", EMA_files_2[i]),ema_filtered)

        #if mfcc.shape[0] != ema.shape[0]:
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


N="All"
N=10

speakers = ["F1","F5","M1","M3"]


traitement_general_usc_timit(N,speakers[0])
#traitement_general_usc_timit(N,speakers[1])
#traitement_general_usc_timit(N,speakers[2])

