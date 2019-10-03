#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    Created august 2019
    by Maud Parrot
    Script to perform several experiments on the model by cross validation.
    The user can chose which parameter he wants to do an experiment for (filter, alpha, batch normalization).
    The user can also chose the training base for the experiment, it can be one ore several corpus.
    For an experiment, the script evaluate the model for each value of the parameter.
    To evaluate the model, it uses a cross validation scheme :
    it leaves one speaker out of training set and test on this speaker. And this for all speakers of the training set.
    The results are then averaged over all the speakers results.
"""

import random
import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)


from Preprocessing.tools_preprocessing import get_speakers_per_corpus
from Training.tools_learning import give_me_common_articulators
from Training.train import train_model
from Training.train_only_common import train_model_arti_common
import numpy as np
import argparse
import csv
from datetime import date

n_epochs = 500
loss_train = 50
patience = 5
select_arti = True
batch_norma = False
filter_type = "fix"
to_plot = False
lr = 0.001
delta_test = 3
corpus = ["Haskins"]
train_a_bit_on_test = False
output_dim = 18 #18 trajectories
speakers = get_speakers_per_corpus("Haskins")
config = "indep"
delta_valid = 1
def cross_val_config(corpus_to_train_on, only_common = False):
    """
    performs the cross validation on Haskins corpus for 3 types of config in order to evaluate the capacity of
    generalization of the dataset. the parameters (other than config) are defined above and can be modified
    """

    speakers = []
    for co in str(corpus_to_train_on[1:-1]).split(","):
        speakers = speakers + get_speakers_per_corpus(co)
    if only_common:
        output_dim = len(give_me_common_articulators(speakers))
    for config in ["spec", "indep", "dep"]:
        count = 0
        rmse_all, pearson_all = np.zeros((len(speakers), output_dim)), np.zeros((len(speakers), output_dim))
        for speaker in speakers :
            if only_common:
                rmse, pearson = train_model_arti_common(test_on=speaker, n_epochs=n_epochs, loss_train=loss_train,
                                            patience=patience,
                                            corpus_to_train_on=corpus_to_train_on,
                                            batch_norma=batch_norma, filter_type=filter_type, to_plot=to_plot,
                                            lr=lr, delta_test=delta_test, config=config)
            else:
                rmse, pearson = train_model(test_on=speaker, n_epochs=n_epochs, loss_train=loss_train,patience=patience,
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
        print("for config {} results are".format(config))
        print("RMSE mean ", results_rmse)
        print("RMSE std ", std_rmse)
        print("PEARSON ", results_pearson)
        print(" PEARSON STD", std_pearson)
        today = date.today().strftime("%d/%m/%Y")
        with open('experiment_results_config.csv', 'a') as f:
            writer = csv.writer(f)
            row_rmse_mean = [today,corpus_to_train_on, config, "rmse_mean"] + results_rmse.tolist()
            row_rmse_std = [today, corpus_to_train_on,config,"rmse_std"] + std_rmse.tolist()
            row_pearson_mean = [today,corpus_to_train_on, config, "pearson_mean"] + results_pearson.tolist()
            row_pearson_std = [today, corpus_to_train_on,config, "pearson_std"] + std_pearson.tolist()
            for row in [row_rmse_mean, row_rmse_std, row_pearson_mean, row_pearson_std]:
                writer.writerow(row)


def cross_val_filter(corpus_to_train_on, config, only_common = False):
    """
    performs the cross validation on Haskins corpus for 3 types of filter in order to evaluate the impact of the filter
    the parameters (other than typefilter) are defined above and can be modified
    the results of the experiment are printed (future work : add results in csv expliciting the experiment)
    """

    speakers = []
    for co in str(corpus_to_train_on[1:-1]).split(","):
        speakers = speakers + get_speakers_per_corpus(co)
    if only_common:
        output_dim = len(give_me_common_articulators(speakers))
    for filter_type in ["unfix","out","fix"]:
        count = 0
        rmse_all, pearson_all = np.zeros((len(speakers), output_dim)), np.zeros((len(speakers), output_dim))
        for speaker in speakers :
            if only_common:
                rmse, pearson = train_model_arti_common(test_on=speaker, n_epochs=n_epochs, loss_train=loss_train,
                                            patience=patience,
                                            corpus_to_train_on=corpus_to_train_on,
                                            batch_norma=batch_norma, filter_type=filter_type, to_plot=to_plot,
                                            lr=lr, delta_test=delta_test, config=config)
            else:
                rmse, pearson = train_model(test_on=speaker, n_epochs=n_epochs, loss_train=loss_train,patience=patience,
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
        print("for filter type {} results are".format(filter_type))
        print("RMSE mean ", results_rmse)
        print("RMSE std ", std_rmse)
        print("PEARSON ", results_pearson)
        print(" PEARSON STD", std_pearson)
        today = date.today().strftime("%d/%m/%Y")
        with open('experiment_results_filter.csv', 'a') as f:
            writer = csv.writer(f)
            row_rmse_mean = [today, corpus_to_train_on,filter_type, "rmse_mean"] + results_rmse.tolist()
            row_rmse_std = [today,corpus_to_train_on, filter_type,"rmse_std"] + std_rmse.tolist()
            row_pearson_mean = [today, corpus_to_train_on,filter_type, "pearson_mean"] + results_pearson.tolist()
            row_pearson_std = [today, corpus_to_train_on,filter_type, "pearson_std"] + std_pearson.tolist()
            for row in [row_rmse_mean, row_rmse_std, row_pearson_mean, row_pearson_std]:
                writer.writerow(row)

def cross_val_batch_norma(corpus_to_train_on, config):
    """
    performs the cross validation on corpus_to_train_on corpus with and without batch normalization
    the parameters (other than bath_norma) are defined above and can be modified
    the results of the experiment are printed
    (future work : maybe batch norma after dense layers and not after lstm )
    """

    speakers = []

    for co in str(corpus_to_train_on[1:-1]).split(","):
        speakers = speakers + get_speakers_per_corpus(co)

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
        results_rmse = np.mean(rmse_all, axis=0)
        results_pearson = np.mean(pearson_all, axis=0)
        std_rmse = np.std(rmse_all, axis=0)
        std_pearson = np.std(pearson_all, axis=0)
        print("for batch norma {} results are".format(batch_norma))
        print("RMSE mean ", results_rmse)
        print("RMSE std ", std_rmse)
        print("PEARSON ", results_pearson)
        print(" PEARSON STD", std_pearson)
        today = date.today().strftime("%d/%m/%Y")
        with open('experiment_results_batchnorma.csv', 'a') as f:
            writer = csv.writer(f)
            row_rmse_mean = [today, corpus_to_train_on,batch_norma, "rmse_mean"] + results_rmse.tolist()
            row_rmse_std = [today,corpus_to_train_on, batch_norma, "rmse_std"] + std_rmse.tolist()
            row_pearson_mean = [today,corpus_to_train_on, batch_norma, "pearson_mean"] + results_pearson.tolist()
            row_pearson_std = [today, corpus_to_train_on,batch_norma, "pearson_mean"] + std_pearson.tolist()
            for row in [row_rmse_mean, row_rmse_std, row_pearson_mean, row_pearson_std]:
                writer.writerow(row)


def cross_val_for_alpha(corpus_to_train_on,config, only_common = False):
    """
        performs the cross validation on corpus_to_train_on corpus for different values of alpha in the combined loss
        experiment to determine the optimal alpha
        the parameters (other than alpha) are defined above and can be modified
        the results of the experiment are printed
        """
    speakers = []
    for co in str(corpus_to_train_on[1:-1]).split(","):
        speakers = speakers + get_speakers_per_corpus(co)

    # TODO: delete that it is just to make a small test
    #speakers = ["fsew0", "msak0", "MNGU0"]
    speakers = ['F01', 'M01', "fsew0", "msak0", "MNGU0"]
    #speakers = ["msak0", "MNGU0"]
    haskins = ['F01', 'M01']
    mocha_mng = ["fsew0", "msak0", "MNGU0"]
    name = 'experiment_results_alpha_' + '_'.join(speakers) + 'bis.csv'
    print(speakers)
    f = open(name, 'w')
    f.close()
    if only_common:
        output_dim = len(give_me_common_articulators(speakers))

    loss_range = [0, 20, 40, 60, 80, 100]

    for loss_train in loss_range:
        count = 0
        rmse_all, pearson_all = np.zeros((len(speakers), output_dim)), np.zeros((len(speakers), output_dim))

        for speaker in speakers:
            if speaker in haskins:
                speaker_to_valid = str([[sp for sp in mocha_mng if sp != speaker][random.randint(0,2)]])
            if speaker in mocha_mng:
                speaker_to_valid = str([[sp for sp in haskins if sp != speaker][random.randint(0,1)]])

            speaker_to_train = str([sp for sp in speakers if (sp != speaker and sp not in speaker_to_valid)])

            if only_common:
                rmse, pearson = train_model_arti_common(test_on=speaker, n_epochs=n_epochs, loss_train=loss_train, patience=patience,
                                            corpus_to_train_on=corpus_to_train_on,
                                            batch_norma=batch_norma, filter_type=filter_type, to_plot=to_plot,
                                            lr=lr, delta_test=delta_test, config=config, speakers_to_train_on=speaker_to_train,
                                                        speakers_to_valid_on=speaker_to_valid, delta_valid = delta_valid)
            else:
                rmse, pearson = train_model(test_on=speaker, n_epochs=n_epochs, loss_train=loss_train,
                                            patience=patience,
                                            select_arti=select_arti, corpus_to_train_on=corpus_to_train_on,
                                            batch_norma=batch_norma, filter_type=filter_type, to_plot=to_plot,
                                            lr=lr, delta_test=delta_test, config=config,
                                            speakers_to_train_on=speaker_to_train)
            rmse_all[count, :] = rmse
            pearson_all[count, :] = pearson
            count += 1

        results_rmse = np.mean(rmse_all, axis=0)
        results_pearson = np.mean(pearson_all, axis=0)
        std_rmse = np.std(rmse_all, axis=0)
        std_pearson = np.std(pearson_all, axis=0)
        print("for alpha {} results are".format(loss_train))
        print("RMSE mean ", results_rmse)
        print("RMSE std ", std_rmse)
        print("PEARSON ", results_pearson)
        print(" PEARSON STD", std_pearson)
        today = date.today().strftime("%d/%m/%Y")

        with open(name, 'a') as f:
            writer = csv.writer(f)
            row_rmse_mean = [today, corpus_to_train_on, loss_train, "rmse_mean"] + results_rmse.tolist()
            row_rmse_std = [today,corpus_to_train_on, loss_train, "rmse_std"] + std_rmse.tolist()
            row_pearson_mean = [today, corpus_to_train_on,loss_train, "pearson_mean"] + results_pearson.tolist()
            row_pearson_std = [today,corpus_to_train_on, loss_train, "pearson_std"] + std_pearson.tolist()
            for row in [row_rmse_mean, row_rmse_std, row_pearson_mean, row_pearson_std]:
                writer.writerow(row)

def cross_val(corpus_to_train_on, config, only_common = False):
    """
        performs the cross validation on corpus_to_train_on corpus
        the parameters are defined above and can be modified
        the results of the experiment are printed
        """
    speakers = []
    for co in str(corpus_to_train_on[1:-1]).split(","):
        speakers = speakers + get_speakers_per_corpus(co)
    print(speakers)

    if 'mocha' in corpus_to_train_on:
        speakers = ["fsew0", "msak0", "MNGU0"]

    name = 'experiment_results_cross_' + '_'.join(speakers) + '.csv'
    f = open(name, 'w')
    f.close()
    if only_common:
        output_dim = len(give_me_common_articulators(speakers))



    count = 0
    rmse_all, pearson_all = np.zeros((len(speakers), output_dim)), np.zeros((len(speakers), output_dim))

    for speaker in speakers:
        speaker_to_valid = str([[sp for sp in speakers if sp != speaker][0]])
        speaker_to_train = str([sp for sp in speakers if (sp != speaker and sp not in speaker_to_valid)])

        if only_common:
            rmse, pearson = train_model_arti_common(test_on=speaker, n_epochs=n_epochs, loss_train=loss_train, patience=patience,
                                        corpus_to_train_on=corpus_to_train_on,
                                        batch_norma=batch_norma, filter_type=filter_type, to_plot=to_plot,
                                        lr=lr, delta_test=delta_test, config=config, speakers_to_train_on=speaker_to_train,
                                                    speakers_to_valid_on=speaker_to_valid, delta_valid = delta_valid)
        else:
            rmse, pearson = train_model(test_on=speaker, n_epochs=n_epochs, loss_train=loss_train,
                                        patience=patience,
                                        select_arti=select_arti, corpus_to_train_on=corpus_to_train_on,
                                        batch_norma=batch_norma, filter_type=filter_type, to_plot=to_plot,
                                        lr=lr, delta_test=delta_test, config=config,
                                        speakers_to_train_on=speaker_to_train)
        rmse_all[count, :] = rmse
        pearson_all[count, :] = pearson
        count += 1

    results_rmse = np.mean(rmse_all, axis=0)
    results_pearson = np.mean(pearson_all, axis=0)
    std_rmse = np.std(rmse_all, axis=0)
    std_pearson = np.std(pearson_all, axis=0)
    #print("for speaker test {} results are".format(speaker))
    print("RMSE mean ", results_rmse)
    print("RMSE std ", std_rmse)
    print("PEARSON ", results_pearson)
    print(" PEARSON STD", std_pearson)
    today = date.today().strftime("%d/%m/%Y")

    with open(name, 'a') as f:
        writer = csv.writer(f)
        row_rmse_mean = [today, corpus_to_train_on, loss_train, "rmse_mean"] + results_rmse.tolist()
        row_rmse_std = [today,corpus_to_train_on, loss_train, "rmse_std"] + std_rmse.tolist()
        row_pearson_mean = [today, corpus_to_train_on,loss_train, "pearson_mean"] + results_pearson.tolist()
        row_pearson_std = [today,corpus_to_train_on, loss_train, "pearson_std"] + std_pearson.tolist()
        for row in [row_rmse_mean, row_rmse_std, row_pearson_mean, row_pearson_std]:
            writer.writerow(row)

if __name__=='__main__':

    parser = argparse.ArgumentParser(description='Experiment on the asked corpus by cross validation')
    parser.add_argument('corpus_exp', type=str,
                        help='list of corpus on which perform the experiment')

    parser.add_argument('experiment_type', type=str,
                        help='type of experiment (filter alpha cross or bn)')

    parser.add_argument('config', type=str,
                        help='indep, spec train_indep or dep')

    parser.add_argument('--only_common_arti', type=bool, default = False, help='True or False' )

    args = parser.parse_args()

    if args.experiment_type == "config":
        cross_val_config(args.corpus_exp, only_common=args.only_common_arti)

    elif args.experiment_type == "filter":
        if args.config is None :
            print("you have to precise the config for this experiment")
        else:
            cross_val_filter(args.corpus_exp, args.config, only_common=args.only_common_arti)

    elif args.experiment_type == "alpha":
        if args.config is None:
            print("you have to precise the config for this experiment")
        else:
            cross_val_for_alpha(args.corpus_exp,args.config, only_common=args.only_common_arti)

    elif args.experiment_type == "bn":
        if args.config is None:
            print("you have to precise the config for this experiment")
        else :
            cross_val_batch_norma(args.corpus_exp,args.config)
    elif args.experiment_type == "cross":
        if args.config is None:
            print("you have to precise the config for this experiment")
        else:
            cross_val(corpus_to_train_on=args.corpus_exp, config=args.config, only_common=args.only_common_arti)



