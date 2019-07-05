

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

""" after this script the order of the articulators is the following : """


def traitement_general_mocha(N=all):
    root_path = dirname(dirname(os.path.realpath(__file__)))
    path_files_treated = os.path.join(root_path, "Donnees_pretraitees")
    order = 5
    sampling_rate_ema = 500
    articulators = [
       'tt_x', 'tt_y', 'td_x', 'td_y', 'tb_x', 'tb_y', 'li_x', 'li_y',
    'ul_x', 'ul_y', 'll_x', 'll_y', 'v_x ', 'v_y ']

    n_col_ema = len(articulators)+1 #plus lip aperture
    def create_missing_dir(speaker):
        if not os.path.exists(os.path.join(root_path, "Donnees_pretraitees", "mocha_" + speaker, "ema")):
            os.makedirs(os.path.join(root_path, "Donnees_pretraitees", "mocha_" + speaker, "ema"))
        if not os.path.exists(os.path.join(root_path, "Donnees_pretraitees", "mocha_" + speaker, "mfcc")):
            os.makedirs(os.path.join(root_path, "Donnees_pretraitees", "mocha_" + speaker, "mfcc"))
        if not os.path.exists(os.path.join(root_path, "Donnees_pretraitees", "mocha_" + speaker, "ema_filtered")):
            os.makedirs(os.path.join(root_path, "Donnees_pretraitees", "mocha_" + speaker, "ema_filtered"))

    def first_step_ema_data(i,speaker):
        """

        :param i: index de l'uttérence (ie numero de phrase) dont les données EMA seront extraites
                speaker : fsew0 ou msak0
        :return: les données EMA en format npy pour l'utterance i avec les premiers traitements.
        :traitement : lecture du fichier .ema et recup des données, filtre sur les articulateurs qui nous intéressent ,
        ajout du lip aperture, interpolation pour données manquantes
        En sortie nparray de dimension (K,13), où K dépend de la longueur de la phrase
         (fréquence d'échantillonnage de 200Hz donc K = 200*durée_en_sec)
        """

        path_files = os.path.join(root_path, "Donnees_brutes\mocha\\" + speaker)
        EMA_files = sorted([name for name in os.listdir(path_files) if "palate" not in name])
        EMA_files = sorted([name[:-4] for name in EMA_files if name.endswith('.ema')])

        path_ema_file = os.path.join(path_files, EMA_files[i] + ".ema")
        N= len(wav_files)

        with open(path_ema_file,'rb') as ema_annotation:
            column_names=[0]*n_columns
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
            try :
                cols_index = [column_names.index(col) for col in articulators]
                ema_data = ema_data[:, cols_index]
                ema_data = ema_data/100 #met en mm, initallement en 10^-1m
                if np.isnan(ema_data).sum() != 0:
                    pritn("nombre de nan ", np.isnan(ema_data).sum())
                    # Build a cubic spline out of non-NaN values.
                    spline = scipy.interpolate.splrep(np.argwhere(~np.isnan(ema_data).ravel()),
                                                      ema_data[~np.isnan(ema_data)], k=3)
                    # Interpolate missing values and replace them.
                    for j in np.argwhere(np.isnan(ema_data)).ravel():
                        ema_data[j] = scipy.interpolate.splev(j, spline)
                return ema_data
            except :
                print("apparemment pas de velum pour le fichier",EMA_files[i])
                return None
          #  lip_aperture=upperlip_y - lowerlip_y
          #  ema_data = np.insert(ema_data,-2,1, axis=1)
           # ema_data[:,-2] = lip_aperture

           # ind_1, ind_2 = [articulators.index("ul_x"), articulators.index("ll_x")]
           # lip_protrusion = (ema_data[:, ind_1] + ema_data[:, ind_2]) / 2
           # ema_data = np.insert(ema_data, -2, 1, axis=1)
           # ema_data[:, -2] = lip_protrusion

            #dabord enlever les nan avant de lisser etsous echantillonner
            # donnees en milimètres

    def first_step_wav_data(i,speaker):
        """
         :param i: index de l'uttérence (ie numero de phrase) dont les données WAV seront extraites
                speaker : msak0 ou fsew0
         :return: les MFCC en format npy pour l'utterance i avec les premiers traitements.
         :traitement : lecture du fichier .wav, extraction des mfcc avec librosa, ajout des Delta et DeltaDelta
         (features qui représentent les dérivées premières et secondes des mfcc)
         On conserve les 13 plus grands MFCC pour chaque frame de 25ms.
         En sortie nparray de dimension (K',13*3)=(K',39). Ou K' dépend de la longueur de la phrase
         ( Un frame toutes les 10ms, donc K' ~ duree_en_sec/0.01 )
         """
        path_files = os.path.join(root_path, "Donnees_brutes\mocha\\" + speaker)
        wav_files = sorted([name[:-4] for name in os.listdir(path_files) if name.endswith('.wav')])
        path_wav = os.path.join(path_files, wav_files[i] + '.wav')
        data, sr = librosa.load(path_wav, sr=sampling_rate_mfcc)  # chargement de données

        mfcc = librosa.feature.mfcc(y=data, sr=sampling_rate_mfcc, n_mfcc=n_coeff,
                                    n_fft=frame_length, hop_length=hop_length
                                    ).T

        dyna_features = get_delta_features(mfcc)
        dyna_features_2 = get_delta_features(dyna_features)

        mfcc = np.concatenate((mfcc, dyna_features, dyna_features_2), axis=1)
        return mfcc

    def second_step_data(i,ema,mfcc,speaker):
        """
              :param i:  index de l'uttérence (ie numero de phrase) pour laquelle on va traiter le fichier EMA et MFCC
                     speaker : msak0 ou fsew0
              :param ema: Données EMA en format .npy en sortie de la fonction first_step_ema_data(i)
              :param mfcc: Données MFCC en format .npy en sortie de la fonction first_step_wav_data(i)
              :return: les npy EMA et MFCC de taille (K,13) et (K,429) avec le même nombre de lignes
              :traitement lecture du fichier d'annotation .lab , on enlève les frames MFCC et EMA qui correspondent à du silence
              On sous échantillone le nparray EMA pour avoir 1 donnée par frame MFCC.
              On ajoute le 'contexte' aux données MFCC ie les 5 frames précédent et les 5 frames suivant chaque frame,
              d'où la taille de mfcc 429 = 5*39 + 5*39 + 39
              """
        path_files = os.path.join(root_path, "Donnees_brutes\mocha\\" + speaker)
        wav_files = sorted([name[:-4] for name in os.listdir(path_files) if name.endswith('.wav')])

        if speaker in ["fsew0","msak0"]:
            path_annotation = os.path.join(path_files, wav_files[i] + '.lab')
            with open(path_annotation) as file:
                labels = [
                    row.strip('\n').strip('\t').replace(' 26 ', '').split(' ')
                    for row in file
                ]
            start_time = float(labels[0][1])  # if labels[0][1] == '#' else 0
            end_time = float(labels[-1][0])  # if labels[-1][1] == '#' else labels[-1][0]
            start_frame_mfcc = int(np.floor(start_time * 1000 / hop_time))  # nombre de frame mfcc avant lesquelles il ny a que du silence
            end_frame_mfcc = int(np.ceil(end_time * 1000 / hop_time))  # nombre de frame mfcc apres lesquelles il ny a que du silence
            mfcc = np.array(mfcc[start_frame_mfcc:end_frame_mfcc])
            start_frame_ema = int(np.floor(start_time * sampling_rate_ema))
            end_frame_ema = int(np.ceil(end_time * sampling_rate_ema))
            ema = ema[start_frame_ema:end_frame_ema]
        #sous echantillonnage de EMA pour synchro avec WAV
        n_frames_wanted = mfcc.shape[0]
        ema = scipy.signal.resample(ema, num=n_frames_wanted)

        ## zero padding de sorte que l'on intègre les dépendences temporelles : on apprend la trame du milieu
        # mais on ajoute des trames précédent et suivant pour ajouter de l'informatio temporelle

        padding = np.zeros((window, mfcc.shape[1]))
        frames = np.concatenate([padding, mfcc, padding])
        full_window = 1 + 2 * window
        mfcc=  np.concatenate( [frames[i:i + len(mfcc)] for i in range(full_window)], axis=1)
        if len(ema)!=len(mfcc):
            print("pbm size",wav_files[i])
        return ema,mfcc

    window = 5
    n_coeff = 13
    n_col_mfcc = n_coeff*(2*window+1)*3
    speakers = ["fsew0","msak0","faet0","falh0","ffes0","mjjn0","maps0"]

    sampling_rate_mfcc = 16000
    frame_time = 25
    hop_time = 10  # en ms
    hop_length = int((hop_time * sampling_rate_mfcc) / 1000)
    frame_length = int((frame_time * sampling_rate_mfcc) / 1000)

    #nyq = 0.5 * sampling_rate_mfcc #250
    #fc = cutoff / nyq #20/250=0.08
    #2*cutoff/sampling_rate_ema = 2*20/500 = 40/500 = 4/50 = 4*2/100 = 0.08
    #filt_b, filt_a = scipy.signal.butter(order, fc, btype='lowpass', analog=False) #fs=sampling_rate_ema)
    #nyq = 0.5 * sampling_rate_ema  # 8000
    #fc = cutoff / nyq  # 20/8000=0.0025
    #filt_b_ema, filt_a_ema = scipy.signal.butter(order, fc, btype='lowpass', analog=False) #fs=sampling_rate_ema)

    cutoff = 30
    weights = low_pass_filter_weight(cut_off=cutoff, sampling_rate=sampling_rate_ema)

    for k in range(len(speakers)) :
        speaker = speakers[k]
        if speaker in ["maps0","mjjn0"]:
            articulators = [
                'tt_x', 'tt_y', 'td_x', 'td_y', 'tb_x', 'tb_y', 'li_x', 'li_y',
                'ul_x', 'ul_y', 'll_x', 'll_y']

        path_files = os.path.join(root_path, "Donnees_brutes\mocha\\" + speaker)
        EMA_files = sorted([name for name in os.listdir(path_files) if "palate" not in name])
        EMA_files = sorted([name[:-4] for name in EMA_files if name.endswith('.ema')])

        cols_index = None
        n_columns = 20
        wav_files = sorted([name[:-4] for name in os.listdir(path_files) if name.endswith('.wav')])

        create_missing_dir(speaker)
        if N == "All":
            N = len(EMA_files)
        ALL_EMA= []
        ALL_MFCC =[]

        for i in range(N):
            if i%50 ==0:
                print(i," out of ",N)

            ema = first_step_ema_data(i,speaker)   # recup ema de occurence i, conserve colonnes utiles, interpole données manquantes, filtre passe bas pour lisser

            mfcc = first_step_wav_data(i,speaker) #recup MFCC de occurence i,  calcule 13 plus grands mfcc sur chaque trame, calcule les delta et deltadelta
            ema, mfcc = second_step_data(i, ema, mfcc,speaker) # enleve les silences en début et fin, ajoute trames alentours pour mfcc, normalise (ema par arti, mfcc en tout)
            if ema.shape[0] != mfcc.shape[0]:
                print("probleme de shape")

            np.save(os.path.join(root_path, "Donnees_pretraitees","mocha_"+speaker,"ema", EMA_files[i]),ema)
            np.save(os.path.join(root_path, "Donnees_pretraitees","mocha_"+speaker,"mfcc", wav_files[i]),mfcc)

            ALL_EMA.append(ema)
            ALL_MFCC.append(mfcc)

        all_mean_ema = np.array([np.mean(ALL_EMA[i], axis=0) for i in range(len(ALL_EMA))])
        xtrm = 30
        weights_moving_average = low_pass_filter_weight(cut_off=10, sampling_rate=sampling_rate_ema)
        moving_average = np.concatenate([np.expand_dims(np.pad(all_mean_ema[:, k], (xtrm, xtrm), "symmetric"), 1)
                                         for k in range(all_mean_ema.shape[1])], axis=1)
        smoothed_moving_average = np.concatenate([np.expand_dims(np.convolve(channel, weights_moving_average, mode='same'), 1)
                                                  for channel in moving_average.T], axis=1)
        smoothed_moving_average = smoothed_moving_average[xtrm:-xtrm, :]


        std_ema = np.mean(  np.array([ np.std(x,axis=0) for x in ALL_EMA])  ,axis=0)
        mean_ema = np.mean( np.array([ np.mean(x,axis=0) for x in ALL_EMA])  ,axis=0) #apres que chaque phrase soit centrée

        std_mfcc = np.mean(np.array([np.std(x, axis=0) for x in ALL_MFCC]), axis=0)
        mean_mfcc = np.mean(np.array([np.mean(x, axis=0) for x in ALL_MFCC]), axis=0)
        np.save(os.path.join("norm_values","moving_average_ema_" + speaker), smoothed_moving_average)
        np.save(os.path.join("norm_values","std_ema_"+speaker), std_ema)
        np.save(os.path.join("norm_values","mean_ema_"+speaker), mean_ema)
        np.save(os.path.join("norm_values","std_mfcc_"+speaker), std_mfcc)
        np.save(os.path.join("norm_values","mean_mfcc_"+speaker), mean_mfcc)
        print(std_ema,"std ema")

        for i in range(N):

            ema = np.load(os.path.join(root_path, "Donnees_pretraitees","mocha_"+speaker,"ema", EMA_files[i]+".npy"))
            ema = ((ema - smoothed_moving_average[i,:]))/max(std_ema)

            mfcc = np.load(os.path.join(root_path, "Donnees_pretraitees","mocha_"+speaker,"mfcc", EMA_files[i]+".npy"))
            mfcc = (mfcc - mean_mfcc) / std_mfcc

            np.save(os.path.join(root_path, "Donnees_pretraitees", "mocha_" + speaker, "ema", EMA_files[i]), ema)
            np.save(os.path.join(root_path, "Donnees_pretraitees", "mocha_" + speaker, "mfcc", wav_files[i]), mfcc)

            #plt.plot(ema[:,0])

            ema_filtered = np.concatenate([np.expand_dims(np.pad(ema[:, k], (xtrm, xtrm), "symmetric"), 1)
                                             for k in range(ema.shape[1])], axis=1)

            ema_filtered = np.concatenate(  [np.expand_dims(np.convolve(channel, weights, mode='same'), 1)
                 for channel in ema_filtered.T], axis=1 )
            ema_filtered = ema_filtered[xtrm:-xtrm, :]

            #difference = len(ema_filtered) - len(ema)
            #print("diff :",difference)
           # halfdif = int(difference / 2)
           # if difference < 0:  # sequence filtree moins longue que l'originale
           #     ema_filtered = np.pad(ema_filtered, (halfdif, difference - halfdif), "edge")
            #elif difference > 0:
            #    ema_filtered = ema_filtered[halfdif:-(difference - halfdif)]

            if len(ema_filtered) != len(ema):  # sequence filtree plus longue que loriginale
                print("pbm shape", len(ema_filtered), len(ema))
            np.save(os.path.join(root_path, "Donnees_pretraitees", "mocha_" + speaker, "ema_filtered", EMA_files[i]), ema_filtered)

N="All"

traitement_general_mocha(N=N)