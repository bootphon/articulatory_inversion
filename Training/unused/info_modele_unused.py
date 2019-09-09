import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)

import os
import numpy as np


from Apprentissage.class_network import my_bilstm
from Apprentissage.modele import my_ac2art_modele
import sys
import torch
import os
import csv
import sys
from sklearn.model_selection import train_test_split
from Apprentissage.utils import load_filenames, load_data, load_filenames_deter
from Apprentissage.pytorchtools import EarlyStopping
import time
import random
from os.path import dirname
from scipy import signal
import matplotlib.pyplot as plt
from Traitement.fonctions_utiles import get_speakers_per_corpus
import scipy
from os import listdir
import json


root_folder = os.path.dirname(os.getcwd())
fileset_path = os.path.join(root_folder, "Donnees_pretraitees", "fileset")

print(sys.argv)


def see_info(name_file):

    hidden_dim = 300
    input_dim = 429
    batch_size = 10
    #batch_size=1
    output_dim = 18
    cuda_avail = False
    filter_type = 1
    batch_norma=False
    model = my_ac2art_modele(hidden_dim=hidden_dim, input_dim=input_dim, name_file=name_file, output_dim=output_dim,
                      batch_size=batch_size, cuda_avail=cuda_avail,
                      modele_filtered=filter_type,batch_norma=batch_norma)
    model = model.double()
    file_weights = os.path.join("saved_models", name_file +".txt")
   # file_weights = os.path.join("saved_models", name_file +".pt")


    loaded_state = torch.load(file_weights)#, map_location=torch.device('cpu'))

    model.load_state_dict(loaded_state)
    model_dict = model.state_dict()

    model_dict.update(loaded_state)
    model.load_state_dict(model_dict)



    weight_apres = model.lowpass.weight.data[0, 0, :]

    freqs, h = signal.freqz(weight_apres.cpu())
    h = h/np.sum(h)
    freqs = freqs * 100 / (2 * np.pi)  # freq in hz
    plt.plot(freqs, 20 * np.log10(abs(h)), 'r')
    plt.ylabel('Amplitude [dB]')
    plt.xlabel("real frequency")
    plt.title("Poids pour train on Haskins test on F01 à la fin de l'apprentissage")
    plt.show()


if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Train and save a model.')
    parser.add_argument('namefile', type=str,
                        help='0 pas de lissage, 1 lissage en dur, 2 lissage variable crée avec pytorch, 3 lissage variable cree avec en dur')

    args = parser.parse_args()
    name_file =  str(sys.argv[1])
    see_info(name_file)