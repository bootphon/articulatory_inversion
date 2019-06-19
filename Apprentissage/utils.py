# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import division
import matplotlib.pyplot as plt
import numpy as np
import os
from os.path import dirname
from numpy.random import choice

def load_filenames(train_on,batch_size):
    """

    :param train_on:  liste des locuteurs sur lesquels on apprend (parmis "fsew0","msak0","MNGU0")
    :param batch_size: nombre de phrase sur lesquelles on veut apprendre au sein d'un batch
    :return: x,y deux listes de longueur batch_size. x contient des numpy array de format (K,429) et y de format (K, nombre_arti)

    choisi #batch_size phrases au hasard parmis tous les locuteurs en respectant la proporition de chaque locuteur dans
    le train set. Plus un locuteur aura de phrases dans le train set, plus il a de chances d'avoir de phrases dans le
    batch.
    """

    path_files = os.path.join(os.path.dirname(os.getcwd()),"Donnees_pretraitees","fileset")
    if len(train_on)==1:
        files = open(os.path.join(path_files,train_on[0]+"_train.txt"), "r").read().split()
        index = np.random.choice(len(files),batch_size)
        train_files = [files[i] for i in index]
    else :
        proba_speakers = {}
        total_number=0
        for speaker in train_on:
            train_speaker = open(os.path.join(path_files,speaker+"_train.txt"), "r").read().split()
            proba_speakers[speaker] = len(train_speaker)
            total_number+=len(train_speaker)
        for speaker in train_on:
            proba_speakers[speaker] = proba_speakers[speaker]/total_number

        speaker_files_chosen = choice(train_on,  batch_size, p=list(proba_speakers.values()))
        train_files=[]
        for speaker in train_on:
            train_speaker = open(os.path.join(path_files, speaker + "_train.txt"), "r").read().split()
            n_train = list(speaker_files_chosen).count(speaker)
            index = np.random.choice(len(train_speaker), n_train)
            train_files.extend([train_speaker[i] for i in index])
    return train_files


def load_data(files_names,filtered=False):
    """

    :param files_names: liste des n
    :param filtered:
    :return:
    """
    folder = os.path.join(os.path.dirname(os.getcwd()), "Donnees_pretraitees")
    x = []
    y = []
    suff = ""
    if filtered :
        suff = "_filtered"
    for file_name in files_names :
        speaker = file_name[0:5]
        speaker_2=speaker
        if speaker in ["fsew0","msak0"]:
            speaker_2 = "mocha_"+speaker
        files_path = os.path.join(folder,speaker_2)
        the_ema_file = np.load(os.path.join(os.path.join(files_path, "ema"+suff), file_name + ".npy"))
        the_mfcc_file = np.load(os.path.join(os.path.join(files_path, "mfcc"+suff), file_name+ ".npy"))

        x.append(the_mfcc_file)
        y.append(the_ema_file)

    return x , y




#filenames = load_filenames(["fsew0","msak0","MNGU0"],10)
#x,y = load_data(filenames)
#print(x[0].shape,y[0].shape)
def chirp(f0, f1, T, fs):
    # f0 is the lower bound of the frequency range, in Hz
    # f1 is the upper bound of the frequency range, in Hz
    # T is the duration of the chirp, in seconds
    # fs is the sampling rate
    slope = (f1-f0)/float(T)

    def chirp_wave(t):
        return np.cos((0.5*slope*t+f0)*2*np.pi*t)
    return [chirp_wave(t) for t in np.linspace(0, T, T*fs).tolist()]


def window(window_type, N):
    def hanning(n):
        return 0.5*(1 - np.cos(2 * np.pi * (n - 1) / (N - 1)))

    def hamming(n):
        return 0.54 - 0.46 * np.cos(2 * np.pi * (n - 1) / (N - 1))

    if window_type == 'hanning':
        return np.asarray([hanning(n) for n in range(N)])
    else:
        return np.asarray([hamming(n) for n in range(N)])


def low_pass_filter_weight_old(cut_off,sampling_rate,len_window,window_type="hanning"):
    N=len_window-1
    if cut_off>sampling_rate/2:
        raise Exception("La frequence de coupure doit etre au moins deux fois la frequence dechantillonnage")
    def hanning_un_point(n):
        value=0.5*(1 - np.cos(2 * np.pi * n/N))
        return value
    hanning_weights = [hanning_un_point(n) for n in range(len_window)]
    cut_off_norm = cut_off/sampling_rate
    def sinc_un_point(n):
        if n!= len_window/2:
            value = np.sin(2*np.pi*cut_off_norm*(n-N/2))/(np.pi*(n-N/2))
        else :
            value=2*cut_off_norm
        return value
    print(cut_off_norm)

    sinc_weights = [sinc_un_point(n) for n in range(len_window)]
    final_weights = [a*b for a,b in zip(hanning_weights,sinc_weights)]
    final_weights = final_weights/np.sum(final_weights)
   # plt.plot(range(len_window),hanning_weights)
   # plt.plot(range(len_window),sinc_weights)
   # plt.plot(range(len_window),final_weights)
   # plt.legend(['hanning','sinc','final'])
   # plt.show()
    return final_weights
#low_pass_filter_weight(cut_off=70,sampling_rate=200,len_window=100)

def low_pass_filter_weight(cut_off,sampling_rate):

    fc = cut_off/sampling_rate# Cutoff frequency as a fraction of the sampling rate (in (0, 0.5)).
    if fc > 0.5:
        raise Exception("La frequence de coupure doit etre au moins deux fois la frequence dechantillonnage")
    b = 0.08  # Transition band, as a fraction of the sampling rate (in (0, 0.5)).

    N = int(np.ceil((4 / b))) #le window
    if not N % 2:
        N += 1  # Make sure that N is odd.
    n = np.arange(N) #int of [0,N]
    h = np.sinc(2 * fc * (n - (N - 1) / 2))  # Compute sinc filter.
    w = 0.5 * (1 - np.cos(2 * np.pi * n / (N-1))) # Compute hanning window.
    h = h * w  # Multiply sinc filter with window.
    h = h / np.sum(h)
    return h


"""
f_1=20
s_r = 500
signal_1 = [np.sin(2*np.pi*f_1*n/s_r) for n in range(300)]
f_2 = f_1*3
signal_2 = [np.sin(2*np.pi*f_2*n/s_r) for n in range(300)]
signal =  [a+b for a,b in zip(signal_1,signal_2)]
#plt.plot(signal_1,alpha=0.5)
#plt.plot(signal_2,alpha=0.5)
#plt.plot(signal)
weights = low_pass_filter_weight(cut_off=30,sampling_rate=s_r)
#print(len(weights),len(signal))
signal_filtre = np.convolve(signal,weights,mode='same')
#print(len(signal_filtre))

€signal_filtre = signal_filtre*np.std(signal_1)/np.std(signal_filtre)
#plt.plot(signal_filtre)
#plt.legend(['s1','s2','somme','filtre'])
#plt.show()


"""