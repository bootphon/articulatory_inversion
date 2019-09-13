#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    Created august 2019
    by Maud Parrot
    Function to evaluate a model that was already trained : on data the model never saw, calculate the rmse and
    pearson for the prediction made by this model.
"""


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
import argparse
import torch
import os
import csv
import sys
from Training.tools_learning import load_np_ema_and_mfcc, load_filenames
import random
from scipy import signal
import matplotlib.pyplot as plt
from Training.model import my_ac2art_model

root_folder = os.path.dirname(os.getcwd())
fileset_path = os.path.join(root_folder, "Preprocessed_data", "fileset")

print(sys.argv)

def test_model(test_on ,model_name):
    """
    :param test_on:  the speaker test
    :param model_name: the name of the model (of the .txt file, without the ".txt")
    Need to have to weights of the models saved in a txt file located in Training/saved_models/
    for example F01_speaker_indep_Haskins__loss_both_90_filter_fix_0.txt
    The test speaker has to be precised (in fact readable in the begining of the filename ==> future work)
    Depending on the configuration (read in the filename) it tests on parts of the test-speaker the model was not
    trained on.
    It also saves the graphs for one sentence of the predicted and true arti trajectories
    """

    batch_norma = False
    filter_type = "fix"
    to_plot = True

    cuda_avail = torch.cuda.is_available()
    if cuda_avail:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    hidden_dim = 300
    input_dim = 429
    batch_size = 10
    output_dim = 18

    model = my_ac2art_model(hidden_dim=hidden_dim, input_dim=input_dim, output_dim=output_dim,
                             batch_size=batch_size, cuda_avail=cuda_avail, name_file=model_name,
                             filter_type=filter_type, batch_norma=batch_norma)
    model = model.double()

    file_weights = os.path.join("saved_models", model_name + ".txt")

    if cuda_avail:
        model = model.to(device=device)

    loaded_state = torch.load(file_weights, map_location=device)

    model.load_state_dict(loaded_state)

    if "indep" in model_name:  # the model was not trained on the test speaker
        files_for_test = load_filenames([test_on], part=["train", "valid", "test"])

    else:  # specific or dependant learning
        files_for_test = load_filenames([test_on], part=["test"])


    random.shuffle(files_for_test)
    x, y = load_np_ema_and_mfcc(files_for_test)
    print("evaluation on speaker {}".format(test_on))
    std_speaker = np.load(os.path.join(root_folder, "Preprocessing", "norm_values", "std_ema_"+test_on+".npy"))
    arti_per_speaker = os.path.join(root_folder, "Preprocessing", "articulators_per_speaker.csv")
    csv.register_dialect('myDialect', delimiter=';')
    weight_apres = model.lowpass.weight.data[0, 0, :]

    with open(arti_per_speaker, 'r') as csvFile:
        reader = csv.reader(csvFile, dialect="myDialect")
        next(reader)
        for row in reader:
            if row[0] == test_on:
                arti_to_consider = row[1:19]
                arti_to_consider = [int(x) for x in arti_to_consider]
    rmse_per_arti_mean, pearson_per_arti_mean = model.evaluate_on_test(x,y, std_speaker=std_speaker, to_plot=to_plot
                                                                       , to_consider=arti_to_consider, verbose=False)
    show_filter = True  #add it in argument
    if show_filter:
        weight_apres = model.lowpass.weight.data[0, 0, :]
        print("GAIN",sum(weight_apres.cpu()))
        freqs, h = signal.freqz(weight_apres.cpu())
        freqs = freqs * 100 / (2 * np.pi)  # freq in hz
        plt.plot(freqs, 20 * np.log10(abs(h)), 'r')
        plt.title("Allure filtre passe bas à la fin de l'Training pour filtre en dur")
        plt.ylabel('Amplitude [dB]')
        plt.xlabel("real frequency")
        plt.show()

    with open('model_results_test.csv', 'a',newline="") as f:
        writer = csv.writer(f, delimiter=",")
        row_rmse = [model_name,"rmse"] + rmse_per_arti_mean.tolist() + [model.epoch_ref]
        writer.writerow(row_rmse)
        row_pearson = [model_name, "pearson"] + pearson_per_arti_mean.tolist() + [model.epoch_ref]
        print(row_rmse)
        writer.writerow(row_pearson)

    return rmse_per_arti_mean, pearson_per_arti_mean


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Train and save a model.')

    parser.add_argument('test_on', type=str,
                        help='the speaker we want to test on')

    parser.add_argument('model_name', type=str,
                        help='name of the model (without .txt)')
    args = parser.parse_args()

    rmse,pearson = test_model(test_on=args.test_on, model_name=args.model_name)
    print("results for model ",args.model_name)
    print("rmse",rmse)
    print("pearson",pearson)
