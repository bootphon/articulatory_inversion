#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    Created august 2019
    by Maud Parrot
    script to read data from the MNGU0 database, only 1 speaker
    It's free and available here "http://www.mngu0.org"

"""
import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)
import scipy.signal
import scipy.interpolate
import scipy.io as sio
from Preprocessing.tools_preprocessing import get_fileset_names, get_delta_features, split_sentences

from os.path import dirname
import numpy as np
import scipy.signal
import scipy.interpolate
import librosa
from Preprocessing.class_corpus import Speaker
import glob

root_path = dirname(dirname(os.path.realpath(__file__)))

""" after this script the order of the articulators is the following : """
order_arti_MNGU0 = [
        'tt_x','tt_y','td_y','td_y','tb_x','tb_y',
        'li_x','li_y','ul_x','ul_y',
        'll_x','ll_y']

root_path = dirname(dirname(os.path.realpath(__file__)))


class Speaker_MNGU0(Speaker):
    """
    class for the speaker of MNGU0, child of the Speaker class (in class_corpus.py),
    then inherits of some preprocessing scripts and attributes
    """
    def __init__(self, path_to_raw,  N_max=0 ):
        """
        :param sp:  name of the speaker
        :param N_max:  # max of files we want to preprocess (0 is for All files), variable useful for test
        """
        super().__init__("MNGU0")  # gets the attributes of the Speaker class
        self.root_path = path_to_raw
        self.path_files_annotation = os.path.join(self.root_path, "Raw_data", self.speaker, "phone_labels")
        self.path_ema_files = os.path.join(self.root_path, "Raw_data", self.speaker, "ema")
        self.EMA_files = sorted([name[:-4] for name in os.listdir(self.path_ema_files) if name.endswith('.ema')])
        self.path_files_treated = os.path.join(self.root_path, "Preprocessed_data", self.speaker)
        self.path_files_brutes = os.path.join(self.root_path, "Raw_data", self.speaker)
        self.path_wav_files = os.path.join(self.root_path, "Raw_data", self.speaker, "wav")

        self.N_max = N_max
        self.articulators_init = [
            'T1_py', 'T1_pz', 'T3_py', 'T3_pz', 'T2_py', 'T2_pz',
            'jaw_py', 'jaw_pz', 'upperlip_py', 'upperlip_pz',
            'lowerlip_py', 'lowerlip_pz']
        self.n_columns =  87


    def create_missing_dir(self):
        """
        delete all previous preprocessing, create needed directories
        """
        if not os.path.exists(os.path.join(os.path.join(self.path_files_treated, "ema"))):
            os.makedirs(os.path.join(self.path_files_treated, "ema"))
        if not os.path.exists(os.path.join(os.path.join(self.path_files_treated, "ema_final"))):
            os.makedirs(os.path.join(self.path_files_treated, "ema_final"))
        if not os.path.exists(os.path.join(os.path.join(self.path_files_treated, "mfcc"))):
            os.makedirs(os.path.join(self.path_files_treated, "mfcc"))

        files = glob.glob(os.path.join(self.path_files_treated, "ema", "*"))
        files += glob.glob(os.path.join(self.path_files_treated, "ema_final", "*"))
        files += glob.glob(os.path.join(self.path_files_treated, "mfcc", "*"))

        for f in files:
            os.remove(f)


    def read_ema_file(self,k):
        """
        read the ema file, first preprocessing,
        :param i: utterance index (wrt EMA files)
        :return: npy array (K,12) , K depends on the duration of the recording, 12 trajectories
        """
        path_ema_file = os.path.join(self.path_ema_files, self.EMA_files[k] + ".ema")
        with open(path_ema_file, 'rb') as ema_annotation:
            column_names = [0] * self.n_columns
            for line in ema_annotation:
                line = line.decode('latin-1').strip("\n")
                if line == 'EST_Header_End':
                    break
                elif line.startswith('NumFrames'):
                    n_frames = int(line.rsplit(' ', 1)[-1])
                elif line.startswith('Channel_'):
                    col_id, col_name = line.split(' ', 1)
                    column_names[int(col_id.split('_', 1)[-1])] = col_name

            ema_data = np.fromfile(ema_annotation, "float32").reshape(n_frames, self.n_columns + 2)
            cols_index = [column_names.index(col) for col in self.articulators_init]
            ema_data = ema_data[:, cols_index]
            ema_data = ema_data*100  #initial data in  10^-5m , we turn it to mm
            if np.isnan(ema_data).sum() != 0:
                # Build a cubic spline out of non-NaN values.
                spline = scipy.interpolate.splrep( np.argwhere(~np.isnan(ema_data).ravel()), ema_data[~np.isnan(ema_data)], k=3)
                # Interpolate missing values and replace them.
                for j in np.argwhere(np.isnan(ema_data)).ravel():
                    ema_data[j] = scipy.interpolate.splev(j, spline)
            return ema_data

    def remove_silences(self,k, ema, mfcc):
        """
        :param k:  utterance index (wrt the list EMA_files)
        :param ema: the ema list of traj
        :param mfcc: the mfcc features
        :return: the data (ema and mfcc) without the silence at the beginning and end of the recording
        reads the annotation file to get (in sec) the extremity of the voice,
        calculates the equivalence in # of ema points and # of mfcc frames
        """
        marge = 0
        path_annotation = os.path.join(self.path_files_annotation, self.EMA_files[k] + '.lab')
        with open(path_annotation) as file:
            while next(file) != '#\n':
                pass
            labels = [row.strip('\n').strip('\t').replace(' 26 ', '').split('\t') for row in file]
        labels = [(round(float(label[0]), 2), label[1]) for label in labels]
        start_time = labels[0][0] if labels[0][1] == '#' else 0
        end_time = labels[-2][0] if labels[-1][1] == '#' else labels[-1][0]
        xtrm = [max(start_time-marge,0), end_time+marge]

        xtrm_temp_ema = [int(np.floor(xtrm[0] * self.sampling_rate_ema)),
                         int(min(np.floor(xtrm[1] * self.sampling_rate_ema) + 1, len(ema)))]

        xtrm_temp_mfcc = [int(np.floor(xtrm[0] / self.hop_time)),
                          int(np.ceil(xtrm[1] / self.hop_time))]

        mfcc = mfcc[xtrm_temp_mfcc[0]:xtrm_temp_mfcc[1]]
        ema = ema[xtrm_temp_ema[0]:xtrm_temp_ema[1], :]
        return ema, mfcc

    def from_wav_to_mfcc(self,wav):
        """
        :param wav: list of intensity points of the wav file
        :return: the acoustic features( K,429); where K in the # of frames.
        calculations of the mfcc with librosa , + Delta and DeltaDelta, + 10 context frames
        # of acoustic features per frame: 13 ==> 13*3 = 39 ==> 39*11 = 429.
        parameters for mfcc calculation are defined in class_corpus
        """
        mfcc = librosa.feature.mfcc(y=wav, sr=self.sampling_rate_wav, n_mfcc=self.n_coeff,
                                    n_fft=self.frame_length, hop_length=self.hop_length
                                    ).T
        dyna_features = get_delta_features(mfcc)
        dyna_features_2 = get_delta_features(dyna_features)
        mfcc = np.concatenate((mfcc, dyna_features, dyna_features_2), axis=1)
        padding = np.zeros((self.window, mfcc.shape[1]))
        frames = np.concatenate([padding, mfcc, padding])
        full_window = 1 + 2 * self.window
        mfcc = np.concatenate([frames[i:i + len(mfcc)] for i in range(full_window)], axis=1)
        return mfcc

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
        N = len(self.EMA_files)
        if self.N_max != 0:
            N = self.N_max
        for i in range(N):
            ema = self.read_ema_file(i)
            ema_VT = self.add_vocal_tract(ema)
            ema_VT_smooth = self.smooth_data(ema_VT)
            path_wav = os.path.join(self.path_wav_files, self.EMA_files[i] + '.wav')
            wav, sr = librosa.load(path_wav, sr=self.sampling_rate_wav)
            wav = 0.5 * wav / np.max(wav)
            mfcc = self.from_wav_to_mfcc(wav)
            ema_VT_smooth, mfcc = self.remove_silences(i, ema_VT_smooth, mfcc)
            ema_VT_smooth, mfcc = self.synchro_ema_mfcc(ema_VT_smooth, mfcc)
            np.save(os.path.join(root_path, "Preprocessed_data", self.speaker, "ema", self.EMA_files[i]), ema_VT)
            np.save(os.path.join(root_path, "Preprocessed_data", self.speaker, "mfcc", self.EMA_files[i]), mfcc)
            np.save(os.path.join(root_path, "Preprocessed_data", self.speaker, "ema_final", self.EMA_files[i]), ema_VT_smooth)
            self.list_EMA_traj.append(ema_VT_smooth)
            self.list_MFCC_frames.append(mfcc)
        self.calculate_norm_values()

        for i in range(N):
            ema_VT_smooth = np.load(
                os.path.join(root_path, "Preprocessed_data", self.speaker, "ema_final", self.EMA_files[i] + ".npy"))
            mfcc = np.load(os.path.join(root_path, "Preprocessed_data", self.speaker, "mfcc", self.EMA_files[i] + ".npy"))
            ema_VT_smooth_norma, mfcc = self.normalize_sentence(i, ema_VT_smooth, mfcc)
            np.save(os.path.join(root_path, "Preprocessed_data", self.speaker, "mfcc", self.EMA_files[i]), mfcc)
            np.save(os.path.join(root_path, "Preprocessed_data", self.speaker, "ema_final", self.EMA_files[i]),
                    ema_VT_smooth_norma)
        #  split_sentences(speaker)
        get_fileset_names(self.speaker)


def Preprocessing_general_mngu0(N_max, path_to_raw):
    """
    :param N_max: #max of files to treat (0 to treat all files), useful for tests
    """
    speaker = Speaker_MNGU0(path_to_raw=path_to_raw, N_max=N_max)
    speaker.Preprocessing_general_speaker()
    print("Done MNGU0 ")

#Test :
#Preprocessing_general_MNGU0(N_max=50)