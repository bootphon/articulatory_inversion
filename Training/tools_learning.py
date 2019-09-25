#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    Created august 2019
    by Maud Parrot
    Functions that are used in the learning process (train.py)
"""


from __future__ import division
import numpy as np
import os
import gc
import torch
import sys
import psutil
from Preprocessing.tools_preprocessing import get_speakers_per_corpus
import json
import random
import matplotlib.pyplot as plt
from scipy import signal



def load_filenames(speakers, part=["train"]):
    """
    :param speakers: list of speakers we want the filesets
    :param part: list ["train","valid","test"] (or less) of part of fileset we want from the speakers
    :return: a list of the filenames corresponding to the asked part for asked speakers
    based on the fileset files already

    """
    path_files = os.path.join(os.path.dirname(os.getcwd()),"Preprocessed_data","fileset")
    filenames = []
    for speaker in speakers:
        for p in part:
            names = open(os.path.join( path_files , speaker + "_" + p + ".txt"), "r").read().split()
            filenames = filenames + names
    return filenames


def load_np_ema_and_mfcc(filenames):
    """
    :param filenames: list of files we want to load the ema and mfcc data
    :return: x : the list of mfcc features,
            y : the list of ema traj
    Load the numpy arrays correspondign the ema and mfcc of the files in the list filenames
    """
    folder = os.path.join(os.path.dirname(os.getcwd()), "Preprocessed_data")
    x = []
    y = []
    speakers = ["F01", "F02", "F03", "F04", "M01", "M02", "M03", "M04","F1", "F5", "M1", "M3"
        , "maps0", "faet0", 'mjjn0', "ffes0", "MNGU0", "fsew0", "msak0","falh0"]
    for filename in filenames:
        speaker = [s for s in speakers if s.lower() in filename.lower()][0] # we can deduce the speaker from the filename
        files_path = os.path.join(folder,speaker)
        the_ema_file = np.load(os.path.join(files_path, "ema_final", filename + ".npy"))
        the_mfcc_file = np.load(os.path.join(files_path, "mfcc", filename + ".npy"))
        x.append(the_mfcc_file)
        y.append(the_ema_file)
    return x, y

def memReport(all=False):
    """
    :param all: show size of each obj
    use if memory errors
    """
    nb_object = 0
    for obj in gc.get_objects():
        if torch.is_tensor(obj):
            if all:
                print(type(obj), obj.size())
            nb_object += 1
    print('nb objects tensor', nb_object)


def cpuStats():
    """
    use in case of memory errors
    """
    print(sys.version)
    print(psutil.cpu_percent())
    print(psutil.virtual_memory())  # physical memory usage
    pid = os.getpid()
    py = psutil.Process(pid)
    memoryUse = py.memory_info()[0] / 2. ** 30  # memory use in GB...I think
    print('memory GB:', memoryUse)



def criterion_pearson(y, y_pred, cuda_avail , device):
    """
    :param y: nparray (B,K,18) target trajectories of the batch (size B) , padded (K = maxlenght)
    :param y_pred: nparray (B,K,18) predicted trajectories of the batch (size B), padded (K = maxlenght
    :param cuda_avail: bool whether gpu is available
    :param device: the device
    :return: loss function for this prediction for loss = pearson correlation
    for each pair of trajectories (target & predicted) we calculate the pearson correlation between the two
    we sum all the pearson correlation to obtain the loss function
    // Idea : integrate the range of the traj here, making the loss for each sentence as the weighted average of the
    losses with weight proportional to the range of the traj (?)
    """
    y_1 = y.sub(torch.mean(y, dim=1, keepdim=True))
    y_pred_1 = y_pred.sub(torch.mean(y_pred,dim=1, keepdim=True))
    nume = torch.sum(y_1 * y_pred_1, dim=1, keepdim=True)  # (B,1,18)
    deno = torch.sqrt(torch.sum(y_1 ** 2, dim=1, keepdim=True)) * \
        torch.sqrt(torch.sum(y_pred_1 ** 2, dim=1, keepdim=True))  # (B,1,18)

    minim = torch.tensor(0.000001,dtype=torch.float64)  # avoid division by 0
    if cuda_avail:
        minim = minim.to(device=device)
        deno = deno.to(device=device)
        nume = nume.to(device=device)
    nume = nume + minim
    deno = deno + minim
    my_loss = torch.div(nume, deno)  # (B,1,18)
    my_loss = torch.sum(my_loss)
    return -my_loss


def criterion_both(my_y,my_ypred,alpha,cuda_avail,device):
    compl = torch.tensor(1. - float(alpha) / 100., dtype=torch.float64)
    alpha = torch.tensor(float(alpha) / 100., dtype = torch.float64)
    multip = torch.tensor(float(1000), dtype = torch.float64)
    if cuda_avail:
        alpha = alpha.to(device = device)
        multip = multip.to(device = device)
        compl = compl.to(device= device)
    a = alpha * criterion_pearson(my_y, my_ypred, cuda_avail, device)*multip
    b = compl * torch.nn.MSELoss(reduction='sum')(my_y, my_ypred)
    new_loss = a + b
    return new_loss


def plot_filtre(weights):
    """
    :param weights: weights of the low pass filter
    plot the impulse response of the filter, with gain in GB
    """
    print("GAIN", sum(weights))
    freqs, h = signal.freqz(weights)
    freqs = freqs * 100 / (2 * np.pi)  # freq in hz
    plt.plot(freqs, 20 * np.log10(abs(h)), 'r')
    plt.title("Allure filtre passe bas à la fin de l'Training pour filtre en dur")
    plt.ylabel('Amplitude [dB]')
    plt.xlabel("real frequency")
    plt.show()



def which_speakers_to_train_on(corpus_to_train_on, test_on, config):
    """
    :param corpus_to_train_on: list of all the corpus name to train on
    :param test_on: the speaker test name
    :param config:  either specific/dependant/independant
    :return:
            speaker_train_on : list of the speakers to train_on (except the test speaker)
    """
    if config == "spec":  # speaker specific
        speakers_to_train_on = [""]  # only train on the test speaker

    elif config in ["indep", "dep"]:  # train on other corpuses
        speakers_to_train_on = []
        for corpus in corpus_to_train_on:
            print("corpus",corpus)
            sp = get_speakers_per_corpus(corpus)
            speakers_to_train_on = speakers_to_train_on + sp
        if test_on in speakers_to_train_on:
            speakers_to_train_on.remove(test_on)
    return speakers_to_train_on

def give_me_common_articulators(list_speakers):
    """
    Give the indexes of the articulators that are in common for a list of speakers
    :param list_speakers: list of the speakers to consider
    :return: list of indexes that correspond to tha articulators in common
    """
    f_artic = open('articulators_per_speaker.csv', 'r')
    ind = f_artic.readline().replace('\n', '').split(';')
    list_arti_common = range(18)
    for line in f_artic:
        new_line = line.replace('\n', '').split(';')
        if new_line[0] in list_speakers:
            arti_dispo = []
            for i in range(len(new_line[1:-2])):
                if new_line[1 + i] == '1':
                    arti_dispo.append(int(i))
            list_arti_common = list(set(list_arti_common).intersection(arti_dispo))
    return list_arti_common




def give_me_train_valid_test_filenames(train_on, test_on, config, batch_size):
    """
    :param train_on: list of corpus to train on
    :param test_on: the speaker test
    :param config: either spec/dep/indep
    :param batch_size
    :return: files_per_categ :  dictionnary where keys are the categories present in the training set. For each category
    we have a dictionnary with 2 keys (train, valid), and the values is a list of the namefiles for this categ and this
    part (train/valid)
            files_for_test : list of the files of the test set
    3 configurations that impacts the train/valid/test set (if we train a bit on test speaker, we have to be sure that
    the don't test on files that were in the train set)
    - spec : for speaker specific, learning and testing only on the speaker test
    - dep : for speaker dependant, learning on speakers in train_on and a part of the speaker test
    - indep : for speaker independant, learnong on other speakers.
    """
    if config == "spec":
        files_for_train = load_filenames([test_on], part=["train"])
        files_for_valid = load_filenames([test_on], part=["valid"])
        files_for_test = load_filenames([test_on], part=["test"])

    elif config == "dep":
        files_for_train = load_filenames(train_on, part=["train", "test"]) + \
                          load_filenames([test_on], part=["train"])
        files_for_valid = load_filenames(train_on, part=["valid"]) + \
                          load_filenames([test_on], part=["valid"])
        files_for_test = load_filenames([test_on], part=["test"])

    elif config == "indep":
        files_for_train = load_filenames(train_on, part=["train", "test"])
        files_for_valid = load_filenames(train_on, part=["valid"])
        files_for_test = load_filenames([test_on], part=["train", "valid", "test"])

    with open('categ_of_speakers.json', 'r') as fp:
        categ_of_speakers = json.load(fp)  # dictionnary { categ : dict_2} where
                                            # dict_2 :{  speakers : [sp_1,..], arti  : [0,1,1...]  }
    files_per_categ = dict()

    for categ in categ_of_speakers.keys():
        sp_in_categ = categ_of_speakers[categ]["sp"]

        files_train_this_categ = [[f for f in files_for_train if sp.lower() in f.lower()]
                                  for sp in sp_in_categ]  # the speaker name is always in the namefile
        files_train_this_categ = [item for sublist in files_train_this_categ
                                  for item in sublist]  # flatten the list of list

        files_valid_this_categ = [[f for f in files_for_valid if sp.lower() in f.lower()] for sp in sp_in_categ]
        files_valid_this_categ = [item for sublist in files_valid_this_categ for item in sublist]

        if len(files_train_this_categ) > 0:  # meaning we have at least one file in this categ
            files_per_categ[categ] = dict()

            N_iter_categ = int(len(files_train_this_categ)/batch_size)+1
            n_a_ajouter = batch_size*N_iter_categ - len(files_train_this_categ)
            files_train_this_categ = files_train_this_categ +\
                                    files_train_this_categ[:n_a_ajouter]  #so that lenght is a multiple of batchsize
            random.shuffle(files_train_this_categ)
            files_per_categ[categ]["train"] = files_train_this_categ

            N_iter_categ = int(len( files_valid_this_categ) / batch_size) + 1
            n_a_ajouter = batch_size * N_iter_categ - len(files_valid_this_categ)
            files_valid_this_categ = files_valid_this_categ + files_valid_this_categ[:n_a_ajouter]
            random.shuffle(files_valid_this_categ)
            files_per_categ[categ]["valid"] = files_valid_this_categ

    return files_per_categ, files_for_test

def give_me_train_valid_test_filenames_no_cat(train_on, test_on, config):
    """
    :param train_on: list of corpus to train on
    :param test_on: the speaker test
    :param config: either spec/dep/indep
    :param batch_size
    :return: files_per_categ :  dictionnary where keys are the categories present in the training set. For each category
    we have a dictionnary with 2 keys (train, valid), and the values is a list of the namefiles for this categ and this
    part (train/valid)
            files_for_test : list of the files of the test set
    3 configurations that impacts the train/valid/test set (if we train a bit on test speaker, we have to be sure that
    the don't test on files that were in the train set)
    - spec : for speaker specific, learning and testing only on the speaker test
    - dep : for speaker dependant, learning on speakers in train_on and a part of the speaker test
    - indep : for speaker independant, learnong on other speakers.
    """
    if config == "spec":
        files_for_train = load_filenames([test_on], part=["train"])
        files_for_valid = load_filenames([test_on], part=["valid"])
        files_for_test = load_filenames([test_on], part=["test"])

    elif config == "dep":
        files_for_train = load_filenames(train_on, part=["train", "test"]) + \
                          load_filenames([test_on], part=["train"])
        files_for_valid = load_filenames(train_on, part=["valid"]) + \
                          load_filenames([test_on], part=["valid"])
        files_for_test = load_filenames([test_on], part=["test"])

    elif config == "indep":
        files_for_train = load_filenames(train_on, part=["train", "test"])
        files_for_valid = load_filenames(train_on, part=["valid"])
        files_for_test = load_filenames([test_on], part=["train", "valid", "test"])



    return files_for_train, files_for_valid, files_for_test


def get_right_indexes(y, indexes_list):
    #print(y.shape)
    list_array = []
    for i in indexes_list:
        #print(i)
        #print(y[:, :, i:i+1].shape)
        #print(y[:,:,i].shape)
        list_array.append(y[:, :, i:i+1])
    #print(list_array)
    return np.concatenate(tuple(list_array), axis=2)





