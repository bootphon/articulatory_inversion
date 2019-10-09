#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    Created august 2019
    by Juliette MILLET
    script to convert mfccs into the .fea format needed byt the ZS2017 code
"""
import numpy as np
import os
root_folder = os.path.dirname(os.getcwd())
import argparse



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
    with open(os.path.join(root_folder,"Predictions_arti", "fea_ZS2017_1s_mfccs", filename[:-4] + ".fea"), 'w') as f:
        f.writelines("%s\n" % l for l in lines)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Conversion into .fea files')
    parser.add_argument('mfcc_folder', type=str,
                        help='folder where are the mfccs')
    args = parser.parse_args()
    filenames = os.listdir(os.path.join(root_folder, "Predictions_arti", args.mfcc_folder))
    for filename in filenames:
        value = np.load(os.path.join(root_folder, "Predictions_arti", args.mfcc_folder, filename))
        #print(value.shape)
        #break
        write_fea_file(value, filename)

