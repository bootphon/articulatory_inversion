#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    Created august 2019
    by Maud Parrot
    ---
"""


import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)


from Training.train import train_model
from Preprocessing.tools_preprocessing import get_speakers_per_corpus
import sys
from Training.train import train_model
import numpy as np
import argparse



# CREER UN EXCEL ET Y METTRE LES RESULTATS DES EXPERIENCES, 1 EXCEL (ou csv) PAR EXPERIENCE
# PREMIERE LIGNE EXPLICATION DE LEXPERIENCE ET DES PARAMETRES FIXES
# PUIS PREMIERE COLONNE CE QUI VARIE ET LA SUITE LES RESULTATS


n_epochs = 50
loss_train = "both_90"
patience = 5
select_arti = True
batch_norma = False
filter_type = "fix"
to_plot = False
lr = 0.001
delta_test = 1
corpus = ["Haskins"]
train_a_bit_on_test = False
output_dim = 18 #18 trajectories
speakers = get_speakers_per_corpus("Haskins")
config = "indep"
corpus_to_train_on  = "[Haskins]"

def cross_val_filter():
    """
    performs the cross validation on Haskins corpus for 3 types of filter in order to evaluate the impact of the filter
    the parameters (other than typefilter) are defined above and can be modified
    the results of the experiment are printed (future work : add results in csv expliciting the experiment)
    """
    for filter_type in ["fix","unfix","out"]:
        count = 0
        rmse_all, pearson_all = np.zeros((len(speakers), output_dim)), np.zeros((len(speakers), output_dim))
        for speaker in speakers :
            rmse, pearson = train_model(test_on=speaker, n_epochs=n_epochs, loss_train=loss_train,patience=patience,
                                        select_arti=select_arti, corpus_to_train_on=corpus_to_train_on,
                                        batch_norma=batch_norma, filter_type=filter_type, to_plot=to_plot,
                                        lr=lr, delta_test=delta_test, config=config)
            rmse_all[count, :] = rmse
            pearson_all[count, :] = pearson
            count += 1
        results_rmse = np.mean(rmse_all,axis=0)
        results_pearson = np.mean(pearson_all,axis = 0)
        std_rmse = np.std(rmse_all, axis = 0 )
        std_pearson  = np.std(pearson_all, axis = 0 )
        print("for filter type {} results are".format(filter_type))
        print("RMSE mean ", results_rmse)
        print("RMSE std ", std_rmse)
        print("PEARSON ", results_pearson)
        print(" PEARSON STD",std_pearson)


def cross_val_bath_norma():
    """
    performs the cross validation on Haskins corpus with and without batch normalization
    the parameters (other than bath_norma) are defined above and can be modified
    the results of the experiment are printed
    (future work : maybe batch norma after dense layers and not after lstm )
    """
    corpus_to_train_on = str(corpus)
    for batch_norma in ["True", "False"]:
        count = 0
        rmse_all, pearson_all = np.zeros((len(speakers), output_dim)), np.zeros((len(speakers), output_dim))

        for speaker in speakers :
            rmse, pearson = train_model(test_on=speaker, n_epochs=n_epochs, loss_train=loss_train, patience=patience,
                                        select_arti=select_arti, corpus_to_train_on=corpus_to_train_on,
                                        batch_norma=batch_norma, filter_type=filter_type, to_plot=to_plot,
                                        lr=lr, delta_test=delta_test, config=config)
            rmse_all[count,:] = rmse
            pearson_all[count,:]= pearson
            count += 1
        results_rmse = np.mean(rmse_all,axis=0)
        results_pearson = np.mean(pearson_all,axis = 0)
        std_rmse = np.std(rmse_all, axis=0)
        std_pearson = np.std(pearson_all, axis=0)
        print("for batch_norma {} results are".format(batch_norma))
        print("RMSE mean ", results_rmse)
        print("RMSE std ", std_rmse)
        print("PEARSON ", results_pearson)
        print(" PEARSON STD", std_pearson)


def cross_val_for_alpha():
    """
        performs the cross validation on Haskins corpus for different values of alpha in the combined loss
        experiment to determine the optimal alpha
        the parameters (other than alpha) are defined above and can be modified
        the results of the experiment are printed
        """
    corpus_to_train_on = str(corpus)
    for alpha in [0, 20, 40, 60, 80, 100]:
        count = 0
        loss_train = "both_" + str(alpha)
        rmse_all, pearson_all = np.zeros((len(speakers), output_dim)), np.zeros((len(speakers), output_dim))
        for speaker in speakers:
            rmse, pearson = train_model(test_on=speaker, n_epochs=n_epochs, loss_train=loss_train, patience=patience,
                                        select_arti=select_arti, corpus_to_train_on=corpus_to_train_on,
                                        batch_norma=batch_norma, filter_type=filter_type, to_plot=to_plot,
                                        lr=lr, delta_test=delta_test, config=config)
            rmse_all[count, :] = rmse
            pearson_all[count, :] = pearson
            count += 1
        results_rmse = np.mean(rmse_all, axis=0)
        results_pearson = np.mean(pearson_all, axis=0)
        std_rmse = np.std(rmse_all, axis=0)
        std_pearson = np.std(pearson_all, axis=0)
        print("for alpha {} results are".format(alpha))
        print("RMSE mean ", results_rmse)
        print("RMSE std ", std_rmse)
        print("PEARSON ", results_pearson)
        print(" PEARSON STD", std_pearson)


if __name__=='__main__':

    parser = argparse.ArgumentParser(description='Experiment on the Haskins corpus by cross validation')
    parser.add_argument('experiment', type=str,
                        help='type of experiment (filter alpha or bn)')

    args = parser.parse_args()

    if args.experiment == "filter":
        cross_val_filter()

    elif args.experiment == "alpha":
        cross_val_for_alpha()

    elif args.experiment == "bn":
        cross_val_bath_norma()

