# TODO

import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)

ncpu="10"
import os
os.environ["OMP_NUM_THREADS"] = ncpu # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = ncpu # export OPENBLAS_NUM_THREADS=4
os.environ["MKL_NUM_THREADS"] = ncpu # export MKL_NUM_THREADS=4
os.environ["VECLIB_MAXIMUM_THREADS"] = ncpu # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = ncpu # export NUMEXPR_NUM_THREADS=4
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
# TODO: c'est toujtours juste ça ? DIre comment utiliser ce script dans le readme, si il y a des choses à changer selon les utilisateurs, il vaut mieux que ce soit de arguments du script
fileset_path = os.path.join(root_folder, "Donnees_pretraitees", "fileset")

print(sys.argv)


def test_model(test_on ,model_name,was_trained_on=False):
    # TODO
    n_epochs = 50
    batch_norma = False
    select_arti = True
    filter_type = 1
    name_corpus_concat = ""
    delta_test=  1
    lr = 0.001
    to_plot = True

    cuda_avail = torch.cuda.is_available()
    if cuda_avail:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    previous_models = os.listdir("saved_models")
    patience = 20
    hidden_dim = 300
    input_dim = 429
    batch_size = 10
    #batch_size=1
    output_dim = 18



    model = my_ac2art_modele(hidden_dim=hidden_dim, input_dim=input_dim, output_dim=output_dim,
                      batch_size=batch_size, cuda_avail=cuda_avail,name_file = model_name,
                      modele_filtered=filter_type,batch_norma=batch_norma)
    model = model.double()

    file_weights = os.path.join("saved_models", model_name +".txt")

    if cuda_avail:
        model = model.to(device=  device)

    loaded_state = torch.load(file_weights, map_location=device )


    model.load_state_dict(loaded_state)
   # model_dict = model.state_dict()
   # loaded_state = {k: v for k, v in loaded_state.items() if
                #    k in model_dict}  # only layers param that are in our current model
   # loaded_state = {k: v for k, v in loaded_state.items() if
                #    loaded_state[k].shape == model_dict[k].shape}  # only if layers have correct shapes
   # model_dict.update(loaded_state)
   # model.load_state_dict(model_dict)

    if was_trained_on :
        files_for_test = load_filenames_deter([test_on], part=["test"])
    else :
        files_for_test = load_filenames_deter([test_on], part=["train","valid","test"])


    plot_filtre_chaque_epochs = False

    random.shuffle(files_for_test)
    x, y = load_data(files_for_test)
    print("evaluation on speaker {}".format(test_on))
    std_speaker = np.load(os.path.join(root_folder,"Traitement","norm_values","std_ema_"+test_on+".npy"))
    arti_per_speaker = os.path.join(root_folder, "Traitement", "articulators_per_speaker.csv")
    csv.register_dialect('myDialect', delimiter=';')
    weight_apres = model.lowpass.weight.data[0, 0, :]
  #  print("GAINAAA",sum(weight_apres.cpu()))
    with open(arti_per_speaker, 'r') as csvFile:
        reader = csv.reader(csvFile, dialect="myDialect")
        next(reader)
        for row in reader:
            if row[0] == test_on:
                arti_to_consider = row[1:19]
                arti_to_consider = [int(x) for x in arti_to_consider]
  #  print("arti to cons",arti_to_consider)
    rmse_per_arti_mean, pearson_per_arti_mean = model.evaluate_on_test(x,y, std_speaker = std_speaker, to_plot=to_plot
                                                                       ,to_consider = arti_to_consider,verbose=False) #,filtered=True)

    if plot_filtre_chaque_epochs:
        weight_apres = model.lowpass.weight.data[0, 0, :]
        print("GAIN",sum(weight_apres.cpu()))

        freqs, h = signal.freqz(weight_apres.cpu())
        freqs = freqs * 100 / (2 * np.pi)  # freq in hz
        plt.plot(freqs, 20 * np.log10(abs(h)), 'r')
        plt.title("Allure filtre passe bas à la fin de l'apprentissage pour filtre en dur")
        plt.ylabel('Amplitude [dB]')
        plt.xlabel("real frequency")
        plt.show()

    return rmse_per_arti_mean, pearson_per_arti_mean

    #x, y = load_data(files_for_test,filtered=False)
   # print("DATA AND MODELE NOT FILTERED")
    #model.evaluate_on_test(x,y, to_plot=to_plot,filtered=False)


if __name__=='__main__':
    # TODO: passe en argparse
    test_on =  sys.argv[1]
    model_name = sys.argv[2]
    was_trained_on = sys.argv[3].lower()=="true"

    rmse,pearson =test_model(test_on = test_on, model_name = model_name
                                                        , was_trained_on = was_trained_on)

    print("results for model ",model_name)
    print("rmse",rmse)
    print("pearson",pearson)
