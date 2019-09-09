
from __future__ import division
import numpy as np
import os
import torch


def load_filenames_deter(train_on,part=["train"]):
    # TODO
    path_files = os.path.join(os.path.dirname(os.getcwd()),"Donnees_pretraitees","fileset")
    filenames = []
    for speaker in train_on:
        for p in part:
            names = open(os.path.join( path_files , speaker + "_" + p + ".txt"), "r").read().split()
            filenames = filenames + names
    return filenames


def load_data(files_names):
    """
    :param files_names: list of files we want to load the ema and mfcc data
    :return: x : the list of mfcc features,
            y : the list of ema traj
    """
    folder = os.path.join(os.path.dirname(os.getcwd()), "Donnees_pretraitees")
    x = []
    y = []
    speakers = ["F01", "F02", "F03", "F04", "M01", "M02", "M03", "M04","F1", "F5", "M1", "M3"
        , "maps0", "faet0", 'mjjn0', "ffes0", "MNGU0", "fsew0", "msak0","falh0"]
    for file_name in files_names :
        speaker = [s for s in speakers if s.lower() in file_name.lower()][0] # we can deduce the speaker from the filename
        files_path = os.path.join(folder,speaker)
        the_ema_file = np.load(os.path.join(files_path, "ema_final", file_name + ".npy"))
        the_mfcc_file = np.load(os.path.join(files_path, "mfcc", file_name+ ".npy"))
        x.append(the_mfcc_file)
        y.append(the_ema_file)
    return x , y

