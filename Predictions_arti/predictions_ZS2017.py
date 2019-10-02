#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    Created august 2019
    by Maud Parrot
    Predict the articulatory trajectories corresponding to the wav files of ZeroSpeech2017.
    The user can chose the model for the prediction.
    Computes the MFCC features corresponding to each wav file and save it in "Predictions_arti/mfcc_ZS2017_1s"
    Save the EMA trajectories in "Predictions_arti/name_model/ema_predictions_ZS2017_1s.

    The script also writes fea files to run the testabx easier.
"""

import sys
import torch
import os
import sys
import time
sys.path.append("..")
from Predictions_arti.predictions_arti import preprocess_my_wav_files, predictions_arti
import numpy as np

root_folder = os.path.dirname(os.getcwd())
import argparse


def prediction_arti_ZS(name_model, wav_folder, mfcc_folder, ema_folder, output_dim = 18,Nmax = 0) :
    """
    :param name_model: name of the model we want to predict the trajectories with
    :param Nmax: if we dont want to predict the traj of ALL wav files, precise how many
    arti predictions for the wav files of ZS2017 with the asked model.
    Also writes fea files in order to run the abx test
    """
    preprocess_my_wav_files(wav_folder=wav_folder, mfcc_folder=mfcc_folder,Nmax = Nmax)
    predictions_arti(model_name=name_model,mfcc_folder=mfcc_folder,ema_folder=ema_folder, output_dim=output_dim)
    filenames = os.listdir(os.path.join(root_folder,"Predictions_arti",name_model,ema_folder))
    if not os.path.exists(os.path.join(root_folder,"Predictions_arti", "fea_ZS2017_1s")):
        os.mkdir(os.path.join(root_folder,"Predictions_arti", "fea_ZS2017_1s"))
    if Nmax > 0 :
        filenames = filenames[:Nmax]

    for filename in filenames :
        arti_pred = np.load(os.path.join(root_folder,"Predictions_arti",name_model,"ema_predictions_ZS2017_1s",filename))
        write_fea_file(arti_pred, filename)


def write_fea_file(prediction, filename):
    """
    :param prediction: array with ema prediction for 1 sentence
    :param filename:  name of the file with the sentence (will be the name of the fea)
    save as a fea file the arti representations
    """
    prediction_with_time = np.zeros((prediction.shape[0], prediction.shape[1] + 1))
    prediction_with_time[:, 1:] = prediction
    frame_hop = 0.010
    frame_lenght = 0.025
    all_times = [frame_lenght / 2 + frame_hop * i for i in range(prediction.shape[0])]
    prediction_with_time[:, 0] = all_times
    lines = [' '.join(str(ema) for ema in prediction_with_time[i]) for i in range(len(prediction_with_time))]
    with open(os.path.join(root_folder,"Predictions_arti", "fea_ZS2017_1s", filename + ".fea"), 'w') as f:
        f.writelines("%s\n" % l for l in lines)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Articulatory predictions for some wav files')

    parser.add_argument('name_model', type=str,
                        help='model to use for the inversion')
    parser.add_argument('wav_folder', type=str,
                        help='folder wav')
    parser.add_argument('mfcc_folder', type=str,
                        help='folde to put mfcc')
    parser.add_argument('ema_folder', type=str,
                        help='folder to put ema')
    parser.add_argument('--Nmax', type=int, default=0,
                        help='#max of predictions to do. If 0 do ALL the predictions')
    parser.add_argument('--output_dim', type=int, default=18,
                        help='output dimension of ema')

    args = parser.parse_args()
    prediction_arti_ZS(name_model=args.name_model, Nmax=args.Nmax, wav_folder=args.wav_folder, mfcc_folder=args.mfcc_folder,
                       ema_folder = args.ema_folder, output_dim=args.output_dim)

