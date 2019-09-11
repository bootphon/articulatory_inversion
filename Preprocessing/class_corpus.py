#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    Created august 2019
    by Maud Parrot
    A class for speaker instance, useful because speakers in one corpus share some attributes and the preprocessing
    functions. Also, all speakers share some attributes.
    This class is used in each preprocessing script.
"""
import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)

import os
import numpy as np
import scipy.signal
import scipy.interpolate
from Preprocessing.tools_preprocessing import low_pass_filter_weight
import csv

root_folder = os.path.dirname(os.getcwd())

class Speaker():
    """
    The speakers share some preprocessing function.
    They have some specific attributes that are defined by the parent class (Corpus)
    This class is used in each preprocessing script
    """
    def __init__(self,speaker):
        """
        :param name:  name of the speaker
        """
        self.speaker = speaker
        self.speakers = None
        self.corpus = None
        self.get_corpus_name()
        self.sampling_rate_wav_wanted = 16000
        self.frame_time = 25 / 1000
        self.hop_time = 10 / 1000
        self.hop_length = int(self.hop_time * self.sampling_rate_wav_wanted)
        self.frame_length = int(self.frame_time * self.sampling_rate_wav_wanted)
        self.window = 5
        self.n_coeff = 13
        self.sampling_rate_ema = None
        self.sampling_rate_wav = None
        self.speakers = None
        self.articulators = ['tt_x', 'tt_y', 'td_x', 'td_y', 'tb_x', 'tb_y', 'li_x', 'li_y'
            , 'ul_x', 'ul_y', 'll_x', 'll_y']
        self.speakers_with_velum = ["fsew0", "msak0", "faet0", "ffes0", "falh0"]
        self.init_corpus_param()
        self.EMA_files = None
        if self.speaker in self.speakers_with_velum:
            self.articulators = ['tt_x', 'tt_y', 'td_x', 'td_y', 'tb_x', 'tb_y', 'li_x', 'li_y',
                                 'ul_x', 'ul_y', 'll_x', 'll_y', 'v_x', 'v_y']
        self.list_EMA_traj = []
        self.list_MFCC_frames = []

        self.std_ema = None
        self.moving_average_ema = None
        self.mean_ema = None
        self.std_mfcc = None
        self.mean_mfcc = None

    def get_corpus_name(self):
        """
        define the corpus the speaker comes from
        """
        if self.speaker == "MNGU0":
            corpus = "MNGU0"
        elif self.speaker in ["F1", "F5", "M1", "M3"]:
            corpus ="usc"
        elif self.speaker in   ["F01", "F02", "F03", "F04", "M01", "M02", "M03", "M04"] :
            corpus = "Haskins"
        elif self.speaker in  ["fsew0", "msak0", "faet0", "ffes0", "maps0", "mjjn0", "falh0"]:
            corpus = "mocha"
        else:
            raise NameError("vous navez pas choisi un des speasker")
        self.corpus = corpus

    def init_corpus_param(self):
        """
        Initialize some parameters depending on the corpus
        """
        if self.corpus == "mocha":
            self.sampling_rate_wav = 16000
            self.sampling_rate_ema = 500
            self.cutoff = 10

        elif self.corpus == "MNGU0":
            self.sampling_rate_wav = 16000
            self.sampling_rate_ema = 200
            self.cutoff = 10

        elif self.corpus == "usc":
            self.sampling_rate_wav = 20000
            self.sampling_rate_ema = 100
            self.cutoff = 10

        elif self.corpus == "Haskins":
            self.sampling_rate_wav = 44100
            self.sampling_rate_ema = 100
            self.cutoff = 20

    def smooth_data(self, ema, sr=0):
        """
        :param ema: one ema trajectory
        :param sr: sampling rate of the ema trajectory
        :return:  the smoothed ema trajectory
        """
        pad = 30
        if sr == 0:
            sr = self.sampling_rate_ema
        cutoff = self.cutoff
        weights = low_pass_filter_weight(cut_off=cutoff,sampling_rate= sr)

        my_ema_filtered = np.concatenate([np.expand_dims(np.pad(ema[:, k], (pad, pad), "symmetric"), 1)
                                          for k in range(ema.shape[1])], axis=1)

        my_ema_filtered = np.concatenate([np.expand_dims(np.convolve(channel, weights, mode='same'), 1)
                                          for channel in my_ema_filtered.T], axis=1)
        my_ema_filtered = my_ema_filtered[pad:-pad, :]
        return my_ema_filtered

    def calculate_norm_values(self):
        """
        based on all the EMA trajectories and frames MFCC calculate the norm values :
        - mean of ema and mfcc
        - std of ema and mfcc
        - moving average for ema on 60 sentences
        then save those norm values
        """
        list_EMA_traj = self.list_EMA_traj
        list_MFCC_frames = self.list_MFCC_frames

        pad = 30
        all_mean_ema = np.array([np.mean(traj, axis=0) for traj in list_EMA_traj])
        np.save(os.path.join("norm_values", "all_mean_ema_" + self.speaker), all_mean_ema)
        #    weights_moving_average = low_pass_filter_weight(cut_off=10, sampling_rate=self.sampling_rate_ema)
        all_mean_ema = np.concatenate([np.expand_dims(np.pad(all_mean_ema[:, k], (pad, pad), "symmetric"), 1)
                                       for k in range(all_mean_ema.shape[1])], axis=1)  # rajoute pad avant et apres

        moving_average = np.array(
            [np.mean(all_mean_ema[k - pad:k + pad], axis=0) for k in range(pad, len(all_mean_ema) - pad)])

        all_EMA_concat = np.concatenate([traj for traj in list_EMA_traj], axis=0)
        std_ema = np.std(all_EMA_concat, axis=0)
        std_ema[std_ema < 1e-3] = 1

        mean_ema = np.mean(np.array([np.mean(traj, axis=0) for traj in list_EMA_traj]),
                           axis=0)
        std_mfcc = np.mean(np.array([np.std(frame, axis=0) for frame in list_MFCC_frames]), axis=0)
        mean_mfcc = np.mean(np.array([np.mean(frame, axis=0) for frame in list_MFCC_frames]), axis=0)

        np.save(os.path.join("norm_values", "moving_average_ema_" + self.speaker), moving_average)
        np.save(os.path.join("norm_values", "std_ema_" + self.speaker), std_ema)
        np.save(os.path.join("norm_values", "mean_ema_" + self.speaker), mean_ema)
        np.save(os.path.join("norm_values", "std_mfcc_" + self.speaker), std_mfcc)
        np.save(os.path.join("norm_values", "mean_mfcc_" + self.speaker), mean_mfcc)

        self.std_ema = std_ema
        self.moving_average_ema = moving_average
        self.mean_ema = mean_ema
        self.mean_mfcc = mean_mfcc
        self.std_mfcc = std_mfcc

    def add_vocal_tract(self , my_ema):
        """
        calculate 4 'vocal tract' and reorganize the data into a 18 trajectories in a precised order
        :param my_ema: EMA trajectory with K points
        :return: a np array (18,K) where the trajectories are sorted, and unavailable trajectories are at 0
        """
        def add_lip_aperture(ema):
            """
            :param ema: 1 ema trajectory
            :return: return lip aperture trajectory upperlip_y - lowerlip_y
            """
            ind_1, ind_2 = [self.articulators.index("ul_y"), self.articulators.index("ll_y")]
            lip_aperture = ema[:, ind_1] - ema[:, ind_2]  # upperlip_y - lowerlip_y
            return lip_aperture

        def add_lip_protrusion(ema):
            """
            :param ema: 1 ema trajectory
            :return: return lip protrusion trajectory (upperlip_x + lowerlip_x)/2
            """
            ind_1, ind_2 = [self.articulators.index("ul_x"), self.articulators.index("ll_x")]
            lip_protrusion = (ema[:, ind_1] + ema[:, ind_2]) / 2
            return lip_protrusion

        def add_TTCL(ema):  # tongue tip constriction location in degree
            """
           :param ema: 1 ema trajectory
           :return: return tongue tip constriction location in degree trajectory .
           Formula to check again , corresponds to the angle between the horizontal and the tongue tip location
           """
            ind_1, ind_2 = [self.articulators.index("tt_x"), self.articulators.index("tt_y")]
            TTCL = ema[:, ind_1] / np.sqrt(ema[:, ind_1] ** 2 + ema[:, ind_2] ** 2)
            return TTCL

        def add_TBCL(ema):
            """
            :param ema: 1 ema trajectory
           :return: return tongue body constriction location in degree trajectory .
           Formula to check again , corresponds to the angle between the horizontal and the tongue body location
            """
            ind_1, ind_2 = [self.articulators.index("tb_x"), self.articulators.index("tb_y")]
            TBCL = ema[:, ind_1] / np.sqrt(ema[:, ind_1] ** 2 + ema[:, ind_2] ** 2)  # upperlip_y - lowerlip_y
            return TBCL

        def arti_not_available():
            """
            reads a csv that contains for each speaker a list of 18 0/1 , element i is 1 if the arti i is available.
            :return: index of articulations that are not available for this speaker. Based on the local csv file
            """
            arti_per_speaker = os.path.join(root_folder, "Preprocessing", "articulators_per_speaker.csv")
            csv.register_dialect('myDialect', delimiter=';')
            with open(arti_per_speaker, 'r') as csvFile:
                reader = csv.reader(csvFile, dialect="myDialect")
                next(reader)
                for row in reader:
                    if row[0] == self.speaker: # we look for our speaker
                        arti_to_consider = row[1:19]  # 1 if available
            arti_not_avail = [k for k, n in enumerate(arti_to_consider) if n == "0"] # 1 of NOT available
            return arti_not_avail

        lip_aperture = add_lip_aperture(my_ema)
        lip_protrusion = add_lip_protrusion(my_ema)
        TTCL = add_TTCL(my_ema)
        TBCL = add_TBCL(my_ema)

        if self.speaker in self.speakers_with_velum:  # 14 arti de 0 à 13 (2*6 + 2)
            my_ema = np.concatenate((my_ema, np.zeros((len(my_ema), 4))), axis=1)
            my_ema[:, 16:18] = my_ema[:, 12:14]  # met les velum dans les 2 dernieres arti
            my_ema[:, 12:16] = 0  # les 4 autres colonnes vont etre remplies avec les VT par la suite

        else:
            my_ema = np.concatenate((my_ema, np.zeros((len(my_ema), 6))), axis=1)

        my_ema[:, 12] = lip_aperture
        my_ema[:, 13] = lip_protrusion
        my_ema[:, 14] = TTCL
        my_ema[:, 15] = TBCL
        idx_to_ignore = arti_not_available()
        my_ema[:, idx_to_ignore] = 0
        return my_ema

    def normalize_sentence(self,i,my_ema_filtered,my_mfcc):
        """
        :param i: index of the ema traj (to get the moving average)
        :param my_ema_filtered: the ema smoothed ema traj
        :param my_mfcc: mfcc frames
        :return: the normalized EMA et MFCC data
        """
        my_ema_VT = (my_ema_filtered - self.moving_average_ema[i, :]) / self.std_ema
        my_mfcc = (my_mfcc - self.mean_mfcc) / self.std_mfcc
        return my_ema_VT,my_mfcc

    def synchro_ema_mfcc(self,my_ema, my_mfcc):
        """
        :param my_ema: ema traj
        :param my_mfcc: corresponding mfcc frames
        :return: ema and mfcc synchronized
        the ema traj is downsampled to have 1 position for 1 frame mfcc
        """
        my_ema = scipy.signal.resample(my_ema, num=len(my_mfcc))
        return my_ema, my_mfcc


