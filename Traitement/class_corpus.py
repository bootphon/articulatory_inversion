import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)

import os
import time
from os.path import dirname
import numpy as np
import scipy.signal
from scipy import stats
import matplotlib.pyplot as plt
import scipy.interpolate
import librosa
from scipy.fftpack import fft, ifft
from Traitement.fonctions_utiles import low_pass_filter_weight
import shutil

from Traitement.create_filesets import get_fileset_names
import glob
import csv
import multiprocessing as mp




root_folder = os.path.dirname(os.getcwd())


class Corpus():

    def __init__(self,name):
        super(Corpus, self).__init__()
        self.name = name
        self.sampling_rate_ema = None
        self.sampling_rate_wav = None
        self.speakers = None
        self.get_speakers()
        self.articulators = [ 'tt_x', 'tt_y', 'td_x', 'td_y', 'tb_x', 'tb_y', 'li_x', 'li_y',
        'ul_x', 'ul_y', 'll_x', 'll_y']
        self.speakers_with_velum = ["fsew0", "msak0", "faet0", "ffes0", "falh0"]

        self.init_variables()



    def get_speakers(self):
        if self.name == "MNGU0":
            speakers = ["MNGU0"]
        elif self.name == "usc":
            speakers = ["F1", "F5", "M1", "M3"]
        elif self.name ==  "Haskins":
            speakers = ["F01", "F02", "F03", "F04", "M01", "M02", "M03", "M04"]
        elif self.name == "mocha":
            speakers = ["fsew0", "msak0", "faet0", "ffes0", "maps0", "mjjn0", "falh0"]
        else:
            raise NameError("vous navez pas choisi un des corpus")
        self.speakers =  speakers


    def init_variables(self):  #créer un script qui lit un fichier csv avec ces données
        if self.name == "mocha":
            self.sampling_rate_wav = 16000
            self.sampling_rate_ema = 500
            self.cutoff = 10

        elif self.name == "MNGU0":
            self.sampling_rate_wav = 16000
            self.sampling_rate_ema = 200
            self.cutoff = 10

        elif self.name == "usc":
            self.sampling_rate_wav = 20000
            self.sampling_rate_ema = 100
            self.cutoff = 10

        elif self.name == "Haskins":
            self.sampling_rate_wav = 44100
            self.sampling_rate_ema = 100
            self.cutoff = 20


class Speaker():
    def __init__(self,name):
        self.name = name
        self.speakers = None
        self.corpus_name = None
        self.get_corpus_name()
        self.corpus = Corpus(self.corpus_name)

        self.EMA_files = None
        self.sampling_rate_wav  =self.corpus.sampling_rate_wav
        self.sampling_rate_ema = self.corpus.sampling_rate_ema
        self.cutoff  = self.corpus.cutoff
        self.articulators = self.corpus.articulators
        self.speakers_with_velum = self.corpus.speakers_with_velum

        if self.name in self.speakers_with_velum:
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
        if self.name == "MNGU0":
            corpus = "MNGU0"
        elif self.name in ["F1", "F5", "M1", "M3"]:
            corpus ="usc"
        elif self.name in   ["F01", "F02", "F03", "F04", "M01", "M02", "M03", "M04"] :
            corpus = "Haskins"
        elif self.name in  ["fsew0", "msak0", "faet0", "ffes0", "maps0", "mjjn0", "falh0"]:
            corpus = "mocha"
        else:
            raise NameError("vous navez pas choisi un des speasker")
        self.corpus_name =  corpus

    def smooth_data(self,ema,sr = 0):
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
            list_EMA_traj = self.list_EMA_traj
            list_MFCC_frames = self.list_MFCC_frames

            pad = 30
            all_mean_ema = np.array([np.mean(traj, axis=0) for traj in list_EMA_traj])
            np.save(os.path.join("norm_values", "all_mean_ema_" + self.name), all_mean_ema)
            #    weights_moving_average = low_pass_filter_weight(cut_off=10, sampling_rate=self.sampling_rate_ema)
            all_mean_ema = np.concatenate([np.expand_dims(np.pad(all_mean_ema[:, k], (pad, pad), "symmetric"), 1)
                                           for k in range(all_mean_ema.shape[1])], axis=1)  # rajoute pad avant et apres

            moving_average = np.array(
                [np.mean(all_mean_ema[k - pad:k + pad], axis=0) for k in range(pad, len(all_mean_ema) - pad)])

            all_EMA_concat = np.concatenate([traj for traj in list_EMA_traj], axis=0)
            std_ema = np.std(all_EMA_concat, axis=0)
            std_ema[std_ema < 1e-3]=1

            mean_ema = np.mean(np.array([np.mean(traj, axis=0) for traj in list_EMA_traj]),
                               axis=0)  # apres que chaque phrase soit centrée
            std_mfcc = np.mean(np.array([np.std(frame, axis=0) for frame in list_MFCC_frames]), axis=0)
            mean_mfcc = np.mean(np.array([np.mean(frame, axis=0) for frame in list_MFCC_frames]), axis=0)

            np.save(os.path.join("norm_values", "moving_average_ema_" + self.name), moving_average)
            np.save(os.path.join("norm_values", "std_ema_" + self.name), std_ema)
            np.save(os.path.join("norm_values", "mean_ema_" + self.name), mean_ema)
            np.save(os.path.join("norm_values", "std_mfcc_" + self.name), std_mfcc)
            np.save(os.path.join("norm_values", "mean_mfcc_" + self.name), mean_mfcc)

            self.std_ema = std_ema
            self.moving_average_ema = moving_average
            self.mean_ema = mean_ema
            self.mean_mfcc = mean_mfcc
            self.std_mfcc = std_mfcc

    def add_vocal_tract(self,my_ema):
        """
        :param my_ema: trajectoires EMA (K points) disponibles
        :return: tableau (18,K) en rajoutant les 4 vocal tract , mettant à 0 les trajectoires non disponibles,
        et reorganise dans l'ordre les trajectoires.
        """
        #   print("adding vocal tracts for speaker {}".format(speaker))
        def add_lip_aperture(ema):
            ind_1, ind_2 = [self.articulators.index("ul_y"), self.articulators.index("ll_y")]
            lip_aperture = ema[:, ind_1] - ema[:, ind_2]  # upperlip_y - lowerlip_y
            return lip_aperture

        def add_lip_protrusion(ema):
            ind_1, ind_2 = [self.articulators.index("ul_x"), self.articulators.index("ll_x")]
            lip_protrusion = (ema[:, ind_1] + ema[:, ind_2]) / 2
            return lip_protrusion

        def add_TTCL(ema):  # tongue tip constriction location in degree
            ind_1, ind_2 = [self.articulators.index("tt_x"), self.articulators.index("tt_y")]
            TTCL = ema[:, ind_1] / np.sqrt(ema[:, ind_1] ** 2 + ema[:, ind_2] ** 2)  # upperlip_y - lowerlip_y
            return TTCL

        def add_TBCL(ema):  # tongue body constriction location in degree
            ind_1, ind_2 = [self.articulators.index("tb_x"), self.articulators.index("tb_y")]
            TBCL = ema[:, ind_1] / np.sqrt(ema[:, ind_1] ** 2 + ema[:, ind_2] ** 2)  # upperlip_y - lowerlip_y
            return TBCL

        def get_idx_to_ignore():
            arti_per_speaker = os.path.join(root_folder, "Traitement", "articulators_per_speaker.csv")
            csv.register_dialect('myDialect', delimiter=';')
            with open(arti_per_speaker, 'r') as csvFile:
                reader = csv.reader(csvFile, dialect="myDialect")
                next(reader)
                for row in reader:
                    if row[0] == self.name:
                        arti_to_consider = row[1:19]
            idx_to_ignore = [k for k, n in enumerate(arti_to_consider) if n == "0"]
            return idx_to_ignore

        lip_aperture = add_lip_aperture(my_ema)
        lip_protrusion = add_lip_protrusion(my_ema)
        TTCL = add_TTCL(my_ema)
        TBCL = add_TBCL(my_ema)

        if self.name in self.speakers_with_velum:  # 14 arti de 0 à 13 (2*6 + 2)
            my_ema = np.concatenate((my_ema, np.zeros((len(my_ema), 4))), axis=1)
            my_ema[:, 16:18] = my_ema[:, 12:14]  # met les velum dans les 2 dernieres arti
            my_ema[:, 12:16] = 0  # les 4 autres colonnes vont etre remplies avec les VT par la suite

        else:
            my_ema = np.concatenate((my_ema, np.zeros((len(my_ema), 6))), axis=1)

        my_ema[:, 12] = lip_aperture
        my_ema[:, 13] = lip_protrusion
        my_ema[:, 14] = TTCL
        my_ema[:, 15] = TBCL
        idx_to_ignore = get_idx_to_ignore()
        my_ema[:, idx_to_ignore] = 0
        return my_ema

    def normalize_phrase(self,i,my_ema_filtered,my_mfcc):
        my_ema_VT = (my_ema_filtered - self.moving_average_ema[i, :]) / self.std_ema
        my_mfcc = (my_mfcc - self.mean_mfcc) / self.std_mfcc
        return my_ema_VT,my_mfcc

    def synchro_ema_mfcc(self,my_ema, my_mfcc):
        my_ema = scipy.signal.resample(my_ema, num=len(my_mfcc))
        return my_ema, my_mfcc


#aa = Speaker("fsew0")
#aa.smooth_data("yo")
#print(aa.speakers_with_velum)