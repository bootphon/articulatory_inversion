#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    Created august 2019
    by Maud Parrot
    script to read data from the usc database, 4 speakers
    It's free and available here "https://sail.usc.edu/span/usc-timit/"
    data for speaker X has to be in "Raw_data/X"
    the format is special : 1 file for 18sec of recording, so several sentences per file,
    sometimes 1 sentence over 2 files ==> we use the trans file to have 1 file per sentence
"""
import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)
import scipy.signal
import scipy.interpolate
from Preprocessing.tools_preprocessing import get_fileset_names, get_delta_features, split_sentences
import scipy.io as sio

from os.path import dirname
import numpy as np
import scipy.signal

import scipy.interpolate
import librosa
from Preprocessing.tools_preprocessing import get_speakers_per_corpus
from Preprocessing.class_corpus import Speaker
import glob

root_path = dirname(dirname(os.path.realpath(__file__)))

class Speaker_usc(Speaker):
    """
    class for the speaker of usc, child of the Speaker class (in class_corpus.py),
    then inherits of some preprocessing scripts and attributes
    """
    def __init__(self, sp, N_max=0 ):
        """
        :param sp:  name of the speaker
        :param N_max:  # max of files we want to preprocess (0 is for All files), variable useful for test
        """
        super().__init__(sp)  # gets the attributes of the Speaker class

        self.N_max = N_max
        self.path_files_treated = os.path.join(root_path, "Preprocessed_data", self.speaker)
        self.path_files_brutes = os.path.join(root_path, "Raw_data", self.corpus, self.speaker)
        self.path_files_annotation = os.path.join(root_path, "Raw_data", self.corpus, self.speaker, "trans")
        self.EMA_files = sorted([name[:-4] for name in os.listdir(
            os.path.join(self.path_files_brutes, "mat")) if name.endswith(".mat")])
        self.EMA_files_2 = None

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
        if not os.path.exists(os.path.join(self.path_files_brutes, "mat_cut")):
            os.makedirs(os.path.join(self.path_files_brutes, "mat_cut"))
        if not os.path.exists(os.path.join(self.path_files_brutes, "wav_cut")):
            os.makedirs(os.path.join(self.path_files_brutes, "wav_cut"))

        files = glob.glob(os.path.join(self.path_files_treated, "ema", "*"))
        files += glob.glob(os.path.join(self.path_files_treated, "mfcc", "*"))
        files += glob.glob(os.path.join(self.path_files_treated, "ema_final", "*"))
        files += glob.glob(os.path.join(self.path_files_brutes, "wav_cut","*"))
        files += glob.glob(os.path.join(self.path_files_brutes, "mat_cut","*"))

        for f in files:
            os.remove(f)

    def get_data_per_sentence(self):
        """
        Initially 1 file for several sentences pronounced successively (with silence between them).
        The scripts reads the transcription file and generates new files to have 1 EMA and 1 WAV per sentence.

        For the sentence with id 7 the names of the files will be "usctimit_ema_sp_7". Sometimes one sentence is
        pronounced over 2 files. If we want to save a file with a name already in our directory it means that we are
        in that case. So we just concatenate the 2 ema and the 2 MFCC files.

        After this function in ema_cut and in wav_cut we have 1 file per sentence, just as in other corpus

        """
        N = len(self.EMA_files)
        if self.N_max != 0:
            N = min(int(self.N_max / 3), N)
        marge = 0
        for j in range(N):    # run through the files
            path_wav = os.path.join(self.path_files_brutes, "wav", self.EMA_files[j] + '.wav')
            wav, sr = librosa.load(path_wav, sr=self.sampling_rate_wav_wanted)    # 1 wav containing several sentences
            wav = 0.5 * wav / np.max(wav)

            ema = sio.loadmat(os.path.join(self.path_files_brutes, "mat",
                                           self.EMA_files[j] + ".mat"))  # 1 ema containing several sentnces

            # two lines of code to obtain the ema traj as a np array
            ema = ema[self.EMA_files[j]][0]
            ema = np.concatenate([ema[arti][2][:, [0, 1]] for arti in range(1, 7)], axis=1)

            with open(os.path.join(self.path_files_annotation, self.EMA_files[j] + ".trans")) as file:
                labels = np.array([row.strip("\n").split(",") for row in file])
                phone_details = labels[:, [0, 1, -1]]    # all the info about bebginning/end of sentences if here

                # 3 lines of code to get the id of the sentences of this file
                id_sentence = set(phone_details[:, 2])
                id_sentence.remove("")
                id_sentence = sorted([int(id) for id in id_sentence])

                for k in id_sentence:
                    temp = phone_details[phone_details[:, 2] == str(k)]     # we focus on one sentence
                    xtrm = [max(float(temp[:, 0][0]) - marge, 0),
                            float(temp[:, 1][-1]) + marge]               # beginning and end of the sentences in second

                    xtrm_temp_ema = [int(np.floor(xtrm[0] * self.sampling_rate_ema)), int(
                        min(np.floor(xtrm[1] * self.sampling_rate_ema) + 1, len(ema)))]
                    xtrm_temp_wav = [int(int(np.floor(xtrm[0] * self.sampling_rate_wav_wanted))),
                                     int(min(int(np.floor(xtrm[1] * self.sampling_rate_wav_wanted) + 1), len(wav)))]

                    ema_temp = ema[xtrm_temp_ema[0]:xtrm_temp_ema[1], :]
                    wav_temp = wav[xtrm_temp_wav[0]:xtrm_temp_wav[1]]


                    # if we already have a file for id k it means that the sentence was pronounced over the 2 files
                    # we have to concatenante the previous and current data.
                    if os.path.exists(os.path.join(self.path_files_brutes, "mat_cut",
                                                   self.EMA_files[j][:-7] + str(k) + ".npy")):

                        premiere_partie_ema = np.load(os.path.join(self.path_files_brutes,"mat_cut",
                                                                   self.EMA_files[j][:-7] + str(k) + ".npy"))

                        ema_concatenated = np.concatenate((ema_temp, premiere_partie_ema), axis=0) # Final ema for id k

                        premiere_partie_wav, sr = librosa.load(os.path.join(self.path_files_brutes, "wav_cut",
                                                                            self.EMA_files[j][:-7] + str(k) + ".wav"),
                                                               sr=self.sampling_rate_wav_wanted)

                        wav_concatenated = np.concatenate((wav_temp, premiere_partie_wav), axis=0)  # Final wav for id k

                    np.save(os.path.join(self.path_files_brutes, "mat_cut", self.EMA_files[j][:-7] + str(k)),
                            ema_concatenated)

                    librosa.output.write_wav(
                        os.path.join(self.path_files_brutes, "wav_cut", self.EMA_files[j][:-7] + str(k) + ".wav"),
                        wav_concatenated, self.sampling_rate_wav_wanted)

    def read_ema_file(self,m):
        """
        read the ema file, first preprocessing,
        :param m: utterance index (wrt the list "EMA_files")
        :return: npy array (K,12) , K depends on the duration of the recording, 12 trajectories
        """

        articulators = ["ul_x", "ul_y", "ll_x", "ll_y", "li_x", "li_y", "td_x", "td_y", "tb_x", "tb_y", "tt_x",
                        "tt_y"]

        order_arti_usctimit = [
            'tt_x', 'tt_y', 'td_x', 'td_y', 'tb_x', 'tb_y', 'li_x', 'li_y',
            'ul_x', 'ul_y', 'll_x', 'll_y']

        new_order_arti = [articulators.index(col) for col in order_arti_usctimit]  # change the order from the initial

        ema = np.load(os.path.join(self.path_files_brutes, "mat_cut", self.EMA_files_2[m] + ".npy"))

        if np.isnan(ema).sum() != 0:
            #        print(np.isnan(ema).sum())
            spline = scipy.interpolate.splrep(np.argwhere(~np.isnan(ema).ravel()), ema[~np.isnan(ema)], k=3)
            for j in np.argwhere(np.isnan(ema)).ravel():
                ema[j] = scipy.interpolate.splev(j, spline)
        ema = ema[:, new_order_arti]  # change order of arti to have the one wanted
        return ema

    def from_wav_to_mfcc(self,k):
        """
          :param k: index of the sentence (wrt the list 'EMA_files'
          :return: the acoustic features( K,429); where K in the # of frames.
          calculations of the mfcc with librosa , + Delta and DeltaDelta, + 10 context frames
          # of acoustic features per frame: 13 ==> 13*3 = 39 ==> 39*11 = 429.
          parameters for mfcc calculation are defined in class_corpus
          """
        path_wav = os.path.join(self.path_files_brutes, "wav_cut", self.EMA_files_2[k] + '.wav')
        data, sr = librosa.load(path_wav, sr=self.sampling_rate_wav_wanted)  # chargement de données
        mfcc = librosa.feature.mfcc(y=data, sr=self.sampling_rate_wav_wanted, n_mfcc=self.n_coeff,
                                       n_fft=self.frame_length, hop_length=self.hop_length).T
        dyna_features = get_delta_features(mfcc)
        dyna_features_2 = get_delta_features(dyna_features)
        mfcc = np.concatenate((mfcc, dyna_features, dyna_features_2), axis=1)
        padding = np.zeros((self.window, mfcc.shape[1]))
        frames = np.concatenate([padding, mfcc, padding])
        full_window = 1 + 2 * self.window
        mfcc = np.concatenate([frames[j:j + len(mfcc)] for j in range(full_window)], axis=1)
        return mfcc

    def remove_silences(self,k, ema, mfcc):
        """
       :param k:  utterance index (wrt the list EMA_files)
       :param ema: the ema list of traj
       :param mfcc: the mfcc features
       :return: the data (ema and mfcc) without the silence at the beginning and end of the recording

       """

        marge=0
        n_points_de_silences_ema = int(np.floor(marge * self.sampling_rate_ema))
        xtrm_temp_ema = [n_points_de_silences_ema, len(ema) - n_points_de_silences_ema]

        n_frames_de_silences_mfcc = int(np.floor(marge / self.hop_time))
        xtrm_temp_mfcc = [n_frames_de_silences_mfcc, len(mfcc) - n_frames_de_silences_mfcc]
        #   print("avant ",mfcc.shape)
        mfcc = mfcc[xtrm_temp_mfcc[0]:xtrm_temp_mfcc[1]]
        ema = ema[xtrm_temp_ema[0]:xtrm_temp_ema[1], :]
        # print("apres",mfcc.shape)
        return ema, mfcc

    def Preprocessing_general_speaker(self):
        """
        Go through the sentences one by one.
            - reads ema data and turn it to a (K,18) array where arti are in a precise order, interploate missing values,
        smooth the trajectories, remove silences at the beginning and the end, undersample to have 1 position per
        frame mfcc, add it to the list of EMA traj for this speaker
            - reads the wav file, calculate the associated acoustic features (mfcc+delta+ deltadelta+contextframes) ,
        add it to the list of the MFCC FEATURES for this speaker.
        Then calculate the normvalues based on the list of ema/mfcc data for this speaker
        Finally : normalization and last smoothing of the trajectories.
        Final data are in Preprocessed_data/speaker/ema_final.npy and  mfcc.npy
        """

        self.create_missing_dir()
        EMA_files = sorted(
            [name[:-4] for name in os.listdir(os.path.join(self.path_files_brutes, "mat")) if name.endswith(".mat")])

        N = len(EMA_files)
        if self.N_max != 0:
            N = min(int(self.N_max / 3), N)  # majoration:if we want to preprocess N_max sentences, about N_max/6 files

        self.get_data_per_sentence()   # one file contains several sentences, this create one file per sentence
        self.EMA_files_2 = sorted(
            [name[:-4] for name in os.listdir(os.path.join(self.path_files_brutes, "wav_cut")) if name.endswith(".wav")])
        N_2 = len(self.EMA_files_2)
        if self.N_max != 0:
            N_2 = min(self.N_max, N_2)

        for i in range(N_2):
            ema = self.read_ema_file(i)
            ema_VT = self.add_vocal_tract(ema)
            ema_VT_smooth = self.smooth_data(ema_VT)  # smooth for better calculation of norm values
            mfcc = self.from_wav_to_mfcc(i)
            ema_VT_smooth, mfcc = self.remove_silences(i, ema_VT_smooth, mfcc)
            ema_VT_smooth, mfcc = self.synchro_ema_mfcc(ema_VT_smooth, mfcc)
            np.save(os.path.join(root_path, "Preprocessed_data", self.speaker, "ema", self.EMA_files_2[i]), ema_VT)
            np.save(os.path.join(root_path, "Preprocessed_data", self.speaker, "mfcc", self.EMA_files_2[i]), mfcc)
            np.save(os.path.join(root_path, "Preprocessed_data", self.speaker, "ema_final",
                                 self.EMA_files_2[i]), ema_VT_smooth)
            self.list_EMA_traj.append(ema_VT_smooth)
            self.list_MFCC_frames.append(mfcc)
        self.calculate_norm_values()

        for i in range(N_2):
            ema_VT_smooth = np.load(
                os.path.join(root_path, "Preprocessed_data", self.speaker, "ema_final", self.EMA_files_2[i] + ".npy"))
            mfcc = np.load(os.path.join(root_path, "Preprocessed_data", self.speaker, "mfcc",
                                        self.EMA_files_2[i] + ".npy"))
            ema_VT_smooth_norma, mfcc = self.normalize_sentence(i, ema_VT_smooth, mfcc)
            new_sr = 1 / self.hop_time
            ema_VT_smooth_norma = self.smooth_data(ema_VT_smooth_norma, new_sr)
            np.save(os.path.join(root_path, "Preprocessed_data", self.speaker, "mfcc", self.EMA_files_2[i]), mfcc)
            np.save(os.path.join(root_path, "Preprocessed_data", self.speaker, "ema_final", self.EMA_files_2[i]),
                    ema_VT_smooth_norma)

      #  split_sentences(self.speaker)
        get_fileset_names(self.speaker)


def Preprocessing_general_usc(N_max):
    """
    :param N_max: #max of files to treat (0 to treat all files), useful for test
    go through all the speakers of Haskins
    """
    corpus = 'usc'
    speakers_usc = get_speakers_per_corpus(corpus)
    for sp in speakers_usc :
        print("In progress usc ",sp)
        speaker = Speaker_usc(sp,N_max)
        speaker.Preprocessing_general_speaker()
        print("Done usc ",sp)


#Test :
#Preprocessing_general_usc(N_max=50)