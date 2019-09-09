#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    Created august 2019
    by Maud Parrot
    script to read data from the Haskins database
    It's free and available here "https://yale.app.box.com/s/cfn8hj2puveo65fq54rp1ml2mk7moj3h/folder/30415804819"
    There are 8 speakers, the rawfiles for the speaker X need to be in "Donnees_brutes/Haskins/X".
    For one sentence the acoustic & arti data are in a matlab file "data".
    The extraction&preprocessing are done for one speaker after the other.
    For one sentence of speaker X the script saves 3 files in "Donnees_pretraitees/X"  : mfcc (K,429), ema (K,18),
    ema_final (K,18) [same as ema but normalized]; where K depends on the duration of the recording
    The script write a .wav file in "Donnees_brutes/Haskins/X/wav".
    The script also saves for each the "norm values" [see class_corpus, calculate_norm_values()]
"""
import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)
import scipy.signal
import scipy.interpolate
import scipy.io as sio
from Preprocessing.fonctions_utiles import get_fileset_names, get_delta_features, split_sentences

from os.path import dirname
import numpy as np
import scipy.signal

import scipy.interpolate
import librosa
from Preprocessing.fonctions_utiles import get_speakers_per_corpus
from Preprocessing.class_corpus import Speaker
import glob

root_path = dirname(dirname(os.path.realpath(__file__)))


def detect_silence(ma_data):
    """
    :param ma_data: one "data" file containing the beginning and end of one sentence
    :return: the beginning and end (in seconds) of the entence
    We test 2 cases since the "ma_data" are not all organized in the same order.
    """
    for k in [5, 6]:
        try:
            mon_debut = ma_data[0][k][0][0][1][0][1]
            ma_fin = ma_data[0][k][0][-1][1][0][0]
        except:
            pass
    return [mon_debut, ma_fin]


class Speaker_Haskins(Speaker):
    """
    class for 1 speaker of Haskins, child of the Speaker class (in class_corpus.py),
    then inherits of some preprocessing scripts and attributes
    """
    def __init__(self, sp, N_max=0 ):
        """
        :param sp:  name of the speaker
        :param N_max:  # max of files we want to preprocess (0 is for All files), variable useful for test
        """
        super().__init__(sp)  # gets the attributes of the Speaker class
        self.path_files_treated = os.path.join(root_path, "Donnees_pretraitees", self.speaker)
        self.path_files_brutes = os.path.join(root_path, "Donnees_brutes", self.corpus, self.speaker, "data")
        self.EMA_files = sorted([name[:-4] for name in os.listdir(self.path_files_brutes) if "palate" not in name])
        self.N_max = N_max

    def create_missing_dir(self):
        """
        delete all previous preprocessing, create needed directories
        """
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

    def read_ema_and_wav(self, k):
        """
        :param k: index wrt EMA_files list of the file to read
        :return: ema positions for 12 arti (K',12) , acoustic features (K,429); where K in the # of frames.
        read and reorganize the ema traj,
        calculations of the mfcc with librosa , + Delta and DeltaDelta, + 10 context frames
        # of acoustic features per frame: 13 ==> 13*3 = 39 ==> 39*11 = 429.
        parameters for mfcc calculation are defined in class_corpus
        """
        order_arti_haskins = ['td_x', 'td_y', 'tb_x', 'tb_y', 'tt_x', 'tt_y', 'ul_x', 'ul_y', "ll_x", "ll_y",
                              "ml_x", "ml_y", "li_x", "li_y", "jl_x", "jl_y"]

        order_arti = ['tt_x', 'tt_y', 'td_x', 'td_y', 'tb_x', 'tb_y', 'li_x', 'li_y',
                      'ul_x', 'ul_y', 'll_x', 'll_y']

        data = sio.loadmat(os.path.join(self.path_files_brutes, self.EMA_files[k] + ".mat"))[self.EMA_files[k]][0]
        ema = np.zeros((len(data[1][2]), len(order_arti_haskins)))

        for arti in range(1, len(data)):  # lecture des trajectoires articulatoires dans le dicionnaire
            ema[:, (arti - 1) * 2] = data[arti][2][:, 0]
            ema[:, arti * 2 - 1] = data[arti][2][:, 2]
        new_order_arti = [order_arti_haskins.index(col) for col in order_arti]
        ema = ema[:, new_order_arti]

        wav_data = data[0][2][:, 0]
        librosa.output.write_wav(os.path.join(root_path, "Donnees_brutes", self.corpus, self.speaker,
                                              "wav", self.EMA_files[k] + ".wav"), wav_data, self.sampling_rate_wav)
        wav, sr = librosa.load(os.path.join(root_path, "Donnees_brutes", self.corpus, self.speaker, "wav",
                                            self.EMA_files[k] + ".wav"), sr=self.sampling_rate_wav_wanted)
        # np.save(os.path.join(root_path, "Donnees_brutes", corpus, speaker, "wav",
        #                      EMA_files[k]), wav)
        wav = 0.5 * wav / np.max(wav)
        mfcc = librosa.feature.mfcc(y=wav, sr=self.sampling_rate_wav_wanted, n_mfcc=self.n_coeff,
                                       n_fft=self.frame_length, hop_length=self.hop_length).T
        dyna_features = get_delta_features(mfcc)
        dyna_features_2 = get_delta_features(dyna_features)
        mfcc = np.concatenate((mfcc, dyna_features, dyna_features_2), axis=1)
        padding = np.zeros((self.window, mfcc.shape[1]))
        frames = np.concatenate([padding, mfcc, padding])
        full_window = 1 + 2 * self.window
        mfcc = np.concatenate([frames[i:i + len(mfcc)] for i in range(full_window)], axis=1)

        marge = 0
        xtrm = detect_silence(data)
        xtrm = [max(xtrm[0] - marge, 0), xtrm[1] + marge]

        xtrm_temp_ema = [int(np.floor(xtrm[0] * self.sampling_rate_ema)),
                         int(min(np.floor(xtrm[1] * self.sampling_rate_ema) + 1, len(ema)))]
        xtrm_temp_mfcc = [int(np.floor(xtrm[0] / self.hop_time)),
                          int(np.ceil(xtrm[1] / self.hop_time))]
        ema = ema[xtrm_temp_ema[0]:xtrm_temp_ema[1], :]
        mfcc = mfcc[xtrm_temp_mfcc[0]:xtrm_temp_mfcc[1]]

        n_frames_wanted = mfcc.shape[0]
        ema = scipy.signal.resample(ema, num=n_frames_wanted)
        return ema, mfcc

    def Preprocessing_general_speaker(self):
        """
        Go through each sentence doing the preprocessing + adding the trajectoires and mfcc to a list, in order to
        calculate the norm values over all sentences of the speaker
        :return:
        """
        self.create_missing_dir()
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


def Preprocessing_general_haskins(N_max):
    """
    :param N_max: #max of files to treat (0 to treat all files), useful for test
    go through all the speakers of Haskins
    """
    corpus = 'Haskins'
    speakers_Has = get_speakers_per_corpus(corpus)
    for sp in speakers_Has :
        print("En cours Haskins ",sp)
        speaker = Speaker_Haskins(sp,N_max)
        speaker.Preprocessing_general_speaker()
        print("Done Haskins ",sp)


#Test :
#Preprocessing_general_haskins(N_max=50)