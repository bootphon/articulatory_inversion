# TODO: ok description à traduire en anglais, et mettre sous la forme décrite en class_corpus.py
""" Lecture des données EMA pour le corpus MNGU0. On ne conserve que les données concernant les articulateurs indiqués
 dans articulators cest a dire 6 articulateurs en 2Dimensions.
 on ajoute une colonne correspondant à l'ouverture des lèvres, cest donc la 13ème colonne
 on ne normalise pas les données mais on conserve la std et mean des mfcc et ema pour pouvoir normaliser par la suiite)
"""

import os
import time
import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)
from os.path import dirname
import numpy as np
import scipy.signal
import matplotlib.pyplot as plt
import scipy.interpolate
import librosa
import scipy.io as sio
import glob
from Traitement.fonctions_utiles import get_fileset_names, get_delta_features, split_sentences

""" after this script the order of the articulators is the following : """


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

import scipy.interpolate
import librosa
from Traitement.fonctions_utiles import get_speakers_per_corpus
from Traitement.class_corpus import Speaker,Corpus
import glob

root_path = dirname(dirname(os.path.realpath(__file__)))


def detect_silence(ma_data):
    # TODO
    try:  # tous les fichiers ne sont pas organisés dans le même ordre dans le dictionnaire, il semble y avoir deux cas
        mon_debut = ma_data[0][5][0][0][1][0][1]
        ma_fin = ma_data[0][5][0][-1][1][0][0]
    except:
        mon_debut = ma_data[0][6][0][0][1][0][1]
        ma_fin = ma_data[0][6][0][-1][1][0][0]
    return [mon_debut, ma_fin]


class Speaker_Haskins(Speaker):
    def __init__(self, sp, N_max = 0 ):
        super().__init__(sp)
        self.path_files_treated = os.path.join(root_path, "Donnees_pretraitees", self.speaker)
        self.path_files_brutes = os.path.join(root_path, "Donnees_brutes", self.corpus, self.speaker, "data")
        self.EMA_files = sorted([name[:-4] for name in os.listdir(self.path_files_brutes) if "palate" not in name])
        self.N_max = N_max

    def create_missing_dir(self):
        if not os.path.exists(os.path.join(self.path_files_treated, "ema")):
            os.makedirs(os.path.join(self.path_files_treated, "ema"))
        if not os.path.exists(os.path.join(self.path_files_treated, "mfcc")):
            os.makedirs(os.path.join(self.path_files_treated, "mfcc"))
        if not os.path.exists(os.path.join(self.path_files_treated, "ema_final")):
            os.makedirs(os.path.join(self.path_files_treated, "ema_final"))

        if not os.path.exists(os.path.join(root_path, "Donnees_brutes", self.corpus, self.speaker, "wav")):
            os.makedirs(os.path.join(root_path, "Donnees_brutes", self.corpus, self.speaker, "wav"))

        files = glob.glob(os.path.join(self.path_files_treated, "ema", "*"))
        files += glob.glob(os.path.join(self.path_files_treated, "mfcc", "*"))
        files += glob.glob(os.path.join(self.path_files_treated, "ema_final", "*"))
        files += glob.glob(os.path.join(root_path, "Donnees_brutes", self.corpus, self.speaker, "wav", "*"))
        for f in files:
            os.remove(f)

    def read_ema_and_wav(self,k):
        # TODO
        order_arti_haskins = ['td_x', 'td_y', 'tb_x', 'tb_y', 'tt_x', 'tt_y', 'ul_x', 'ul_y', "ll_x", "ll_y",
                              "ml_x", "ml_y", "li_x", "li_y", "jl_x", "jl_y"]

        order_arti = ['tt_x', 'tt_y', 'td_x', 'td_y', 'tb_x', 'tb_y', 'li_x', 'li_y',
                      'ul_x', 'ul_y', 'll_x', 'll_y']

        data = sio.loadmat(os.path.join(self.path_files_brutes, self.EMA_files[k] + ".mat"))[self.EMA_files[k]][0]
        my_ema = np.zeros((len(data[1][2]), len(order_arti_haskins)))

        for arti in range(1, len(data)):  # lecture des trajectoires articulatoires dans le dicionnaire
            my_ema[:, (arti - 1) * 2] = data[arti][2][:, 0]
            my_ema[:, arti * 2 - 1] = data[arti][2][:, 2]
        new_order_arti = [order_arti_haskins.index(col) for col in order_arti]
        my_ema = my_ema[:, new_order_arti]

        wav_data = data[0][2][:, 0]
        librosa.output.write_wav(os.path.join(root_path, "Donnees_brutes", self.corpus, self.speaker,
                                              "wav", self.EMA_files[k] + ".wav"), wav_data, self.sampling_rate_wav)
        wav, sr = librosa.load(os.path.join(root_path, "Donnees_brutes", self.corpus, self.speaker, "wav",
                                            self.EMA_files[k] + ".wav") , sr=self.sampling_rate_wav_wanted)
        # np.save(os.path.join(root_path, "Donnees_brutes", corpus, speaker, "wav",
        #                      EMA_files[k]), wav)
        wav = 0.5 * wav / np.max(wav)
        my_mfcc = librosa.feature.mfcc(y=wav, sr=self.sampling_rate_wav_wanted, n_mfcc=self.n_coeff,
                                       n_fft=self.frame_length, hop_length=self.hop_length).T
        dyna_features = get_delta_features(my_mfcc)
        dyna_features_2 = get_delta_features(dyna_features)
        my_mfcc = np.concatenate((my_mfcc, dyna_features, dyna_features_2), axis=1)
        padding = np.zeros((self.window, my_mfcc.shape[1]))
        frames = np.concatenate([padding, my_mfcc, padding])
        full_window = 1 + 2 * self.window
        my_mfcc = np.concatenate([frames[i:i + len(my_mfcc)] for i in range(full_window)], axis=1)

        marge = 0
        xtrm = detect_silence(data)
        xtrm = [max(xtrm[0] - marge, 0), xtrm[1] + marge]

        xtrm_temp_ema = [int(np.floor(xtrm[0] * self.sampling_rate_ema)),
                         int(min(np.floor(xtrm[1] * self.sampling_rate_ema) + 1, len(my_ema)))]
        xtrm_temp_mfcc = [int(np.floor(xtrm[0] / self.hop_time)),
                          int(np.ceil(xtrm[1] / self.hop_time))]
        my_ema = my_ema[xtrm_temp_ema[0]:xtrm_temp_ema[1], :]
        my_mfcc = my_mfcc[xtrm_temp_mfcc[0]:xtrm_temp_mfcc[1]]

        if np.isnan(my_ema).sum() != 0:
            print("number of nan :", np.isnan(my_ema.sum()))
        n_frames_wanted = my_mfcc.shape[0]
        my_ema = scipy.signal.resample(my_ema, num=n_frames_wanted)
        return my_ema, my_mfcc

    def traitement_speaker_haskins(self):
        N = len(self.EMA_files)
        if self.N_max != 0:
            N = int(self.N_max)  # on coupe N fichiers

        for i in range(N):
            ema, mfcc = self.read_ema_and_wav(i)
            ema_VT = self.add_vocal_tract(ema)
            ema_VT_smooth = self.smooth_data(ema_VT)  # filtrage pour meilleur calcul des norm_values
            np.save(os.path.join(root_path, "Donnees_pretraitees", self.speaker, "ema", self.EMA_files[i]), ema_VT)
            np.save(os.path.join(root_path, "Donnees_pretraitees", self.speaker, "mfcc", self.EMA_files[i]), mfcc)
            np.save(os.path.join(root_path, "Donnees_pretraitees", self.speaker, "ema_final",
                                 self.EMA_files[i]), ema_VT_smooth)
            self.list_EMA_traj.append(ema_VT_smooth)
            self.list_MFCC_frames.append(mfcc)
        self.calculate_norm_values()

        for i in range(N):
            ema_VT_smooth = np.load(os.path.join(
                root_path, "Donnees_pretraitees", self.speaker, "ema_final", self.EMA_files[i] + ".npy"))
            mfcc = np.load(os.path.join(
                root_path, "Donnees_pretraitees", self.speaker, "mfcc", self.EMA_files[i] + ".npy"))
            ema_VT_smooth_norma, mfcc = self.normalize_phrase(i, ema_VT_smooth, mfcc)
            new_sr = 1 / self.hop_time
            ema_VT_smooth_norma = self.smooth_data(ema_VT_smooth_norma, new_sr)

            np.save(os.path.join(root_path, "Donnees_pretraitees", self.speaker, "mfcc", self.EMA_files[i]), mfcc)
            np.save(os.path.join(root_path, "Donnees_pretraitees", self.speaker, "ema_final", self.EMA_files[i]),
                    ema_VT_smooth_norma)
        #  split_sentences(speaker)
        get_fileset_names(self.speaker)


def traitement_general_haskins(N_max):
    # TODO
    corpus = 'Haskins'
    speakers_Has = get_speakers_per_corpus(corpus)
    for sp in speakers_Has :
        print("En cours Haskins ",sp)
        speaker = Speaker_Haskins(sp,N_max)
        speaker.traitement_speaker_haskins()
        print("Done Haskins ",sp)


#Test :
#traitement_general_haskins(N_max=50)