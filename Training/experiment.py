#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    Created august 2019
    by Maud Parrot
    Script to perform several experiments on the model by cross validation.

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




def cross_val_indep(corpus_to_train_on, only_common = False):
    """
        performs the cross validation in config setting on corpus_to_train_on corpus
        the parameters are defined above and can be modified
        the results of the experiment are printed
        """
    speakers = []
    for co in str(corpus_to_train_on[1:-1]).split(","):
        speakers = speakers + get_speakers_per_corpus(co)
    print(speakers)

    name = 'experiment_results_cross_' + '_'.join(speakers) + '.csv'
    f = open(name, 'w')
    f.close()
    output_dim = 18
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
                                        lr=lr, delta_test=delta_test, config='train_indep', speakers_to_train_on=speaker_to_train,
                                                    speakers_to_valid_on=speaker_to_valid, delta_valid = delta_valid)
        else:
            rmse, pearson = train_model(test_on=speaker, n_epochs=n_epochs, loss_train=loss_train,
                                        patience=patience,
                                        select_arti=select_arti, corpus_to_train_on=corpus_to_train_on,
                                        batch_norma=batch_norma, filter_type=filter_type, to_plot=to_plot,
                                        lr=lr, delta_test=delta_test, config='train_indep',
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

def cross_val_spec(corpus_to_train_on, only_common = False):
    """
        performs the cross validation in the specific speaker setting on corpus_to_train_on corpus
        the parameters are defined above and can be modified
        the results of the experiment are printed
        """
    art  = ['tt_x', 'tt_y', 'td_x', 'td_y', 'tb_x', 'tb_y', 'li_x', 'li_y',
                    'ul_x', 'ul_y', 'll_x', 'll_y', 'la', 'lp', 'ttcl', 'tbcl', 'v_x', 'v_y']
    speakers = []
    for co in str(corpus_to_train_on[1:-1]).split(","):
        speakers = speakers + get_speakers_per_corpus(co)
    print(speakers)

    name = 'experiment_results_spec_' + '_'.join(speakers) + '.csv'
    f = open(name, 'w')
    f.close()
    count = 0
    output_dim = range(18)
    for speaker in speakers:

        if only_common:
            output_dim = give_me_common_articulators(speaker)

            rmse, pearson = train_model_arti_common(test_on=speaker, n_epochs=n_epochs, loss_train=loss_train, patience=patience,
                                        corpus_to_train_on=corpus_to_train_on,
                                        batch_norma=batch_norma, filter_type=filter_type, to_plot=to_plot,
                                        lr=lr, delta_test=delta_test, config='spec', delta_valid = delta_valid)
        else:
            rmse, pearson = train_model(test_on=speaker, n_epochs=n_epochs, loss_train=loss_train,
                                        patience=patience,
                                        select_arti=select_arti, corpus_to_train_on=corpus_to_train_on,
                                        batch_norma=batch_norma, filter_type=filter_type, to_plot=to_plot,
                                        lr=lr, delta_test=delta_test, config='spec')

        print("for speaker test {} results are".format(speaker))
        print("RMSE mean ", rmse)
        print("PEARSON ", pearson)
        today = date.today().strftime("%d/%m/%Y")

        with open(name, 'a') as f:
            writer = csv.writer(f)
            row_arti = [art[int(i)] for i in output_dim]
            row_rmse_mean = [today, speaker, loss_train, "rmse_mean"] + rmse.tolist()

            row_pearson_mean = [today, speaker, loss_train, "pearson_mean"] + pearson.tolist()

            for row in [row_arti,row_rmse_mean, row_pearson_mean]:
                writer.writerow(row)
        count += 1



if __name__=='__main__':

    parser = argparse.ArgumentParser(description='Experiment on the asked corpus by cross validation')
    parser.add_argument('corpus_exp', type=str,
                        help='list of corpus on which perform the experiment')

    parser.add_argument('experiment_type', type=str,
                        help='type of experiment (cross cross_spec)')


    parser.add_argument('--only_common_arti', type=bool, default = False, help='True or False' )

    args = parser.parse_args()


    if args.experiment_type == "cross":
        cross_val_indep(corpus_to_train_on=args.corpus_exp, only_common=args.only_common_arti)

    elif args.experiment_type == "cross_spec":
        cross_val_spec(corpus_to_train_on=args.corpus_exp, only_common=args.only_common_arti)



