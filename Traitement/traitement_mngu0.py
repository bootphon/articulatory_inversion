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
from Apprentissage.utils import low_pass_filter_weight

""" after this script the order of the articulators is the following : """
order_arti_MNGU0 = [
        'tt_x','tt_y','td_y','td_y','tb_x','tb_y',
        'li_x','li_y','ul_x','ul_y',
        'll_x','ll_y']

def traitement_general_mngu0(N):
    root_path = dirname(dirname(os.path.realpath(__file__)))
    path_files_annotation = os.path.join(root_path, "Donnees_brutes\MNGU0\phone_labels")
    sampling_rate_ema = 200
    #articulators in the same order that those of MOCHA
    articulators = [
        'T1_py','T1_pz','T3_py','T3_pz','T2_py','T2_pz',
        'jaw_py','jaw_pz','upperlip_py','upperlip_pz',
        'lowerlip_py','lowerlip_pz']
    n_col_ema = len(articulators)+1 #lip aperture
    path_ema_files = os.path.join(root_path, "Donnees_brutes","MNGU0","ema")
    EMA_files = sorted([name[:-4] for name in os.listdir(path_ema_files) if name.endswith('.ema')])
    path_files_treated = os.path.join(root_path, "Donnees_pretraitees\MNGU0")


    cols_index = None
    n_columns = 87
    window=5
    path_wav_files = os.path.join(root_path, "Donnees_brutes","MNGU0","wav")
   # wav_files = sorted([name[:-4] for name in os.listdir(path_wav_files) if name.endswith('.wav')])
    sampling_rate_mfcc = 16000
    frame_time = 25
    hop_time = 10  # en ms
    hop_length = int((hop_time * sampling_rate_mfcc) / 1000)
    frame_length = int((frame_time * sampling_rate_mfcc) / 1000)
    n_coeff = 13
    n_col_mfcc = n_coeff*(2*window+1)*3
    if N == "All":
        N = len(EMA_files)
    def first_step_ema_data(i):
        """
        :param i: index de l'uttérence (ie numero de phrase) dont les données EMA seront extraites
        :return: les données EMA en format npy pour l'utterance i avec les premiers traitements.
        :traitement : lecture du fichier .ema et recup des données, filtre sur les articulateurs qui nous intéressent ,
        ajout du lip aperture, interpolation pour données manquantes
        En sortie nparray de dimension (K,13), où K dépend de la longueur de la phrase
         (fréquence d'échantillonnage de 200Hz donc K = 200*durée_en_sec)
        """
        path_ema_file = os.path.join(path_ema_files, EMA_files[i] + ".ema")
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
           # ind_1, ind_2 = [articulators.index("upperlip_pz"), articulators.index("lowerlip_pz")]
          #  lip_aperture = (ema_data[:, ind_1] - ema_data[:, ind_2]).reshape(len(ema_data),1)
            #ema_data = np.concatenate((ema_data, lip_aperture), axis=1)

           # ind_1, ind_2 = [articulators.index("upperlip_py"), articulators.index("lowerlip_py")]
           # lip_protrusion = ((ema_data[:, ind_1] + ema_data[:, ind_2]) / 2).reshape(len(ema_data),1)
           # ema_data = np.concatenate((ema_data, lip_protrusion), axis=1)

            #dabord enlever les nan avant de lisser et sous echantillonner
            if np.isnan(ema_data).sum() != 0:
                # Build a cubic spline out of non-NaN values.
                spline = scipy.interpolate.splrep( np.argwhere(~np.isnan(ema_data).ravel()), ema_data[~np.isnan(ema_data)], k=3)
                # Interpolate missing values and replace them.
                for j in np.argwhere(np.isnan(ema_data)).ravel():
                    ema_data[j] = scipy.interpolate.splev(j, spline)
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
        path_annotation = os.path.join(path_files_annotation, EMA_files[i] + '.lab')
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
        mfcc = np.array(mfcc[start_frame_mfcc:end_frame_mfcc])
        start_frame_ema = int(np.floor(start_time * sampling_rate_ema))
        end_frame_ema = int(np.ceil(end_time * sampling_rate_ema))
        ema = ema[start_frame_ema:end_frame_ema]
        #sous echantillonnage de EMA pour synchro avec WAV
        n_frames_wanted = mfcc.shape[0]
        ema = scipy.signal.resample(ema, num=n_frames_wanted)
        #  padding de sorte que l'on intègre les dépendences temporelles : on apprend la trame du milieu
        # mais on ajoute des trames précédent et suivant pour ajouter de l'informatio temporelle
        padding = np.zeros((window, mfcc.shape[1]))
        frames = np.concatenate([padding, mfcc, padding])
        full_window = 1 + 2 * window
        mfcc=  np.concatenate([frames[i:i + len(mfcc)] for i in range(full_window)], axis=1)
        return ema,mfcc

    ALL_EMA = []
    ALL_MFCC =[]
    cutoff = 25
    weights = low_pass_filter_weight(cut_off=cutoff, sampling_rate=sampling_rate_ema)
    if N == "All":
        N=len(EMA_files)
    #traitement uttérance par uttérance des phrases
    for i in range(N):
        if i%100 ==0:
            print(i," out of ",N)
        ema = first_step_ema_data(i)
        mfcc = first_step_wav_data(i)
        ema, mfcc = second_step_data(i, ema, mfcc)
       #  print("second step",ema.shape,mfcc.shape)

        if ema.shape[0] != mfcc.shape[0]:
            print("probleme de shape")
        k = 1
      #  while len(mfcc) > 500:
       #     cut = int(len(mfcc)/2)
        #    ema_1 = ema[:cut]
         #   mfcc_1 = mfcc[:cut]
          #  np.save(os.path.join(path_files_treated, "ema", EMA_files[i]+"_"+str(k)), ema_1)
           # np.save(os.path.join(path_files_treated, "mfcc", EMA_files[i]+"_"+str(k)), mfcc_1)
            #ema = ema[cut:]
            #mfcc = mfcc[cut:]
            #ALL_EMA.append(ema_1)
            #ALL_MFCC.append(mfcc_1)
            #k += 1

        np.save(os.path.join(path_files_treated,"ema", EMA_files[i]),ema) #sauvegarde temporaire pour la récup après
        np.save(os.path.join(path_files_treated,"mfcc", EMA_files[i]),mfcc) #sauvegarde temporaire pour la récup après
        ALL_EMA.append(ema)
        ALL_MFCC.append(mfcc)

    all_mean_ema = np.array([np.mean(ALL_EMA[i], axis=0) for i in range(len(ALL_EMA))])
    EMA_files_2 =  sorted([name[:-4] for name in os.listdir(os.path.join(path_files_treated,"ema")) if name.endswith('.npy')])
    N_2 = len(EMA_files_2)
    xtrm = 30

    weights_moving_average = low_pass_filter_weight(cut_off=10, sampling_rate=sampling_rate_ema)

    moving_average = np.concatenate([np.expand_dims(np.pad(all_mean_ema[:, k], (xtrm, xtrm), "symmetric"), 1)
                                     for k in range(all_mean_ema.shape[1])], axis=1)
    smoothed_moving_average = np.concatenate(
        [np.expand_dims(np.convolve(channel, weights_moving_average, mode='same'), 1)
         for channel in moving_average.T], axis=1)
    smoothed_moving_average = smoothed_moving_average[xtrm:-xtrm, :]

    std_ema = np.mean(np.array([np.std(x, axis=0) for x in ALL_EMA]), axis=0)
    mean_ema = np.mean(np.array([np.mean(x, axis=0) for x in ALL_EMA]), axis=0)  # apres que chaque phrase soit centrée

    std_mfcc = np.mean(np.array([np.std(x, axis=0) for x in ALL_MFCC]), axis=0)
    mean_mfcc = np.mean(np.array([np.mean(x, axis=0) for x in ALL_MFCC]), axis=0)
    np.save("norm_values","moving_average_ema_MNGU0", smoothed_moving_average)
    np.save("norm_values","std_ema_MNGU0", std_ema)
    np.save("norm_values","mean_ema_MNGU0", mean_ema)
    np.save("norm_values","std_mfcc_MNGU0", std_mfcc)
    np.save("norm_values","mean_mfcc_MNGU0", mean_mfcc)
    print(std_ema, "std ema ")

    # construction du filtre passe bas que lon va appliquer à chaque frame mfcc et trajectoire d'articulateur
    # la fréquence de coupure réduite de 0.1 a été choisi manuellement pour le moment, et il se trouve qu'on
    # n'a pas besoin d'un filtre différent pour mfcc et ema
   # order = 5
   # filt_b, filt_a = scipy.signal.butter(order, 0.1, btype='lowpass', analog=False) #fs=sampling_rate_ema)
    print(len(smoothed_moving_average))
    print(N_2)
    for i in range(N_2):
        ema = np.load(os.path.join(path_files_treated,"ema", EMA_files_2[i]+".npy"))
        ema = ((ema - smoothed_moving_average[i, :])) / max(std_ema)

        mfcc = np.load(os.path.join(path_files_treated,"mfcc", EMA_files_2[i] + ".npy"))
        mfcc = (mfcc - mean_mfcc) / std_mfcc
        np.save(os.path.join(path_files_treated, "ema",EMA_files_2[i]), ema)
        np.save(os.path.join(path_files_treated,"mfcc", EMA_files_2[i]),mfcc)

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
        np.save(os.path.join(path_files_treated,"ema_filtered", EMA_files_2[i]),ema_filtered)

traitement_general_mngu0(N="All")
