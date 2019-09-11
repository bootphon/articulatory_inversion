#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    Created august 2019
    by Maud Parrot
    Some useful functions for the preprocessing
"""



import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)

import numpy as np
import os
from random import shuffle
import csv
import json
import scipy

root_folder = os.path.dirname(os.getcwd())


def get_delta_features(array, window=5):
    """
    :param array: nparray (K,N) N features per frame, K frames.
    :param window: size of the window to calculate the average speed of the features
    :return: the speed of each feature wrt 5 future and 5 past frames

    """
    all_diff = []
    for lag in range(1, window + 1):
        padding = np.ones((lag, array.shape[1]))
        past = np.concatenate([padding * array[0], array[:-lag]])
        future = np.concatenate([array[lag:], padding * array[-1]])
        all_diff.append(future - past)
    tempo =np.array([all_diff[lag] * lag for lag in range(window)])
    norm = 2 * np.sum(i ** 2 for i in range(1, window + 1))
    delta_features = np.sum(tempo,axis=0)/norm
    return delta_features

def get_speakers_per_corpus(corpus):
    """
    :param corpus: name of the corpus
    :return: list of the speakers in the corpus
    return the list of the speakers names corresponding to the corpus
    """
    if corpus == "MNGU0":
        speakers = ["MNGU0"]
    elif corpus == "usc":
        speakers = ["F1", "F5", "M1", "M3"]
    elif corpus == "Haskins":
        speakers = ["F01", "F02", "F03", "F04", "M01", "M02", "M03", "M04"]
    elif corpus == "mocha":
        speakers = ["fsew0", "msak0", "faet0", "ffes0", "maps0", "mjjn0", "falh0"]
    else:
        raise NameError("vous navez pas choisi un des corpus")
    return speakers


def get_fileset_names(speaker):
    """
    :param speaker: name of a speaker
    once the data are preprocessed, this function split the dataset for a speaker into train/valid/test.
    The repartition is 70% 10% 20%, and the split is random.
    write 3 txt files (sp_train, sp_test, and sp_valid) containing the names of the files concerned.
    These txt files
    """
    donnees_path = os.path.join(root_folder, "Preprocessed_data")
    files_path = os.path.join(donnees_path,speaker)
    EMA_files_names = [name[:-4] for name in os.listdir(os.path.join(files_path,"ema_final")) if name.endswith('.npy') ]
    N = len(EMA_files_names)
    shuffle(EMA_files_names)
    pourcent_train = 0.7
    pourcent_test=0.2
    n_train = int(N*pourcent_train)
    n_test  = int(N*pourcent_test)
    train_files = EMA_files_names[:n_train]
    test_files = EMA_files_names[n_train:n_train+n_test]
    valid_files = EMA_files_names[n_train+n_test:]

    outF = open(os.path.join(root_folder,"Preprocessed_data","fileset",speaker+"_train.txt"), "w")
    outF.write('\n'.join(train_files) + '\n')
    outF.close()

    outF = open(os.path.join(root_folder, "Preprocessed_data", "fileset", speaker + "_test.txt"), "w")
    outF.write('\n'.join(test_files) + '\n')
    outF.close()

    outF = open(os.path.join(root_folder, "Preprocessed_data", "fileset", speaker + "_valid.txt"), "w")
    outF.write('\n'.join(valid_files) + '\n')
    outF.close()


def read_csv_arti_ok_per_speaker():
    """
    create a dictionnary , with different categories as keys (from A to F).
    For a category the value is another dictionnary {"articulators" : list of 18 digit with 1 if arti is
    available for this category,"speakers" : list of speakers in this category}
    The dict is created based on the csv file "articulators_per_speaer"
    """
    arti_per_speaker = os.path.join(root_folder,"Preprocessing", "articulators_per_speaker.csv")
    csv.register_dialect('myDialect', delimiter=';')
    categ_of_speakers = dict()
    with open(arti_per_speaker, 'r') as csvFile:
        reader = csv.reader(csvFile, dialect="myDialect")
        next(reader)
        for categ in ["A", "B", "C", "D", "E", "F"]:
            categ_of_speakers[categ] = dict()
            categ_of_speakers[categ]["sp"] = []
            categ_of_speakers[categ]["arti"] = None
        for row in reader:
            categ_of_speakers[row[19]]["sp"].append(row[0])
            if categ_of_speakers[row[19]]["arti"]  :
                if categ_of_speakers[row[19]]["arti"] != row[1:19]:
                    print("check arti and category for categ {}".format(row[19]))
            else:
                categ_of_speakers[row[19]]["arti"] = row[1:19]

    for cle in categ_of_speakers.keys():
        print("categ ",cle)
        print(categ_of_speakers[cle])

    with open(os.path.join(root_folder,"Training","categ_of_speakers.json"), 'w') as dico:
        json.dump(categ_of_speakers, dico)


def add_voicing(wav, sr):
    """
    estimation of voicing using the short term energy threshold between 0 and 1. This function is not used
    for the moment but voicing could be added to the articulatory representation
    :param wav: wav file
    :param sr:  sampling rate
    :return:  an estimation of the voicing for each point in the wav
    """
    hop_time = 10 / 1000  # en ms
    hop_length = int((hop_time * sr))
    N_frames = int(len(wav) / hop_length)
    window = scipy.signal.get_window("hanning", N_frames)
    ste = scipy.signal.convolve(wav ** 2, window ** 2, mode="same")
    ste = [np.max(min(x, 1), 0) for x in ste]
    return ste



def low_pass_filter_weight(cut_off,sampling_rate):
    """
    :param cut_off:  cutoff of the filter
    :param sampling_rate:  sampling rate of the data 
    :return: the weights of the lowpass filter
    implementation of the weights of a low pass filter windowed with a hanning winow.
    The filter is normalized (gain 1)
    """
    fc = cut_off/sampling_rate # Cutoff frequency as a fraction of the sampling rate (in (0, 0.5)).
    if fc > 0.5:
        raise Exception("La frequence de coupure doit etre au moins deux fois la frequence dechantillonnage")
    b = 0.08  # Transition band, as a fraction of the sampling rate (in (0, 0.5)).
    N = int(np.ceil((4 / b))) # window
    if not N % 2:
        N += 1  # Make sure that N is odd.
    n = np.arange(N)
    h = np.sinc(2 * fc * (n - (N - 1) / 2))  # Compute sinc filter.
    w = 0.5 * (1 - np.cos(2 * np.pi * n / (N-1))) # Compute hanning window.
    h = h * w  # Multiply sinc filter with window.
    h = h / np.sum(h)
    return h


def split_sentences(speaker ,max_length = 300):
    """
    :param speaker:
    :param max_length: max points we want in sentences features (duration = max_lenghts*100sec)
    :return: rien.
    run through all the treated acou and arti features, if lenght > max lenght divide in K slices so that each one has
    less than max_lenght points.
    Warning : when split the original file is removed
              ema files are split only in ema_final (those used for the training)
    """
    Preprocessed_data_path = os.path.join(root_folder, "Preprocessed_data")
    file_names = os.listdir(os.path.join(Preprocessed_data_path, speaker, "ema_final"))
    file_names = [f for f in file_names if 'split' not in f]

    N = len(file_names)
    file_names = file_names[0:N]
    Number_cut = 0
    for f in file_names :
        mfcc = np.load(os.path.join(Preprocessed_data_path,speaker,"mfcc",f))
        ema_VT = np.load(os.path.join(Preprocessed_data_path,speaker,"ema_final",f))
        cut_in_N = int(len(mfcc)/max_length) +1
        if cut_in_N > 1 :
            Number_cut+=1
            temp = 0
            cut_size = int(len(mfcc)/cut_in_N)
            for k in range(cut_in_N-1) :
                mfcc_k = mfcc[temp : temp + cut_size]
                ema_k_vt = ema_VT[temp:temp+cut_size,:]

                temp = temp + cut_size
                np.save(os.path.join(Preprocessed_data_path,speaker,"mfcc",f[:-4]+"_split_"+str(k)),mfcc_k)
                np.save(os.path.join(Preprocessed_data_path,speaker,"ema_final",f[:-4]+"_split_"+str(k)),ema_k_vt)

            mfcc_last = mfcc[temp :]
            ema_last_vt = ema_VT[temp:, :]
            np.save(os.path.join(Preprocessed_data_path, speaker, "mfcc", f[:-4] + "_split_" + str(cut_in_N-1)), mfcc_last)
            np.save(os.path.join(Preprocessed_data_path, speaker, "ema_final", f[:-4] + "_split_" + str(cut_in_N-1)), ema_last_vt)

            os.remove(os.path.join(Preprocessed_data_path,speaker,"mfcc",f))
            os.remove(os.path.join(Preprocessed_data_path,speaker,"ema_final",f))
