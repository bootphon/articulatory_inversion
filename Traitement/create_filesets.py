
import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)

import numpy as np
import random
import os
from os.path import dirname
from random import shuffle
from Traitement.fonctions_utiles import get_speakers_per_corpus


root_folder = os.path.dirname(os.getcwd())
donnees_path = os.path.join(root_folder, "Donnees_pretraitees")

def get_fileset_names(speaker):
    """
    :param speaker: pour le moment fsew0,msak0 ou MNGU0
    :return: rien
    lit tous les numpy mfcc et ema correspondant au speaker et les concatène pour avoir une liste de toutes les données.
    On normalise les mfcc et les ema (soustrait moyenne et divise par écart type)
    On divise en deux les phrases plus longues que 800 frames mfcc.

    On a donc une liste X et une liste Y, qu'on va diviser en train et test.
    """

    if speaker in ["msak0","fsew0","maps0","faet0","mjjn0","ffes0","falh0"]:
        speaker_2 = "mocha_"+speaker

    elif speaker == "MNGU0":
        speaker_2 = speaker

    elif speaker in ["F1", "F5", "M1","M3"]:
        speaker_2 = "usc_timit_" + speaker

    elif speaker in ["F01","F02","F03","F04","M01","M02","M03","M04"]:
        speaker_2 = "Haskins_" + speaker

    files_path =  os.path.join(donnees_path,speaker)
    EMA_files_names = [name[:-4] for name in os.listdir(os.path.join(files_path,"ema_filtered")) if name.endswith('.npy') ]
    N = len(EMA_files_names)
    shuffle(EMA_files_names)
    pourcent_train = 0.7
    pourcent_test=0.2
    n_train = int(N*pourcent_train)
    n_test  = int(N*pourcent_test)
    train_files = EMA_files_names[:n_train]
    test_files = EMA_files_names[n_train:n_train+n_test]
    valid_files = EMA_files_names[n_train+n_test:]

    outF = open(os.path.join(root_folder,"Donnees_pretraitees","fileset",speaker+"_train.txt"), "w")
    outF.write('\n'.join(train_files) + '\n')
    outF.close()

    outF = open(os.path.join(root_folder, "Donnees_pretraitees", "fileset", speaker + "_test.txt"), "w")
    outF.write('\n'.join(test_files) + '\n')
    outF.close()

    outF = open(os.path.join(root_folder, "Donnees_pretraitees", "fileset", speaker + "_valid.txt"), "w")
    outF.write('\n'.join(valid_files) + '\n')
    outF.close()

def get_fileset_names_per_corpus(corpus):
    """
    :param speaker: pour le moment fsew0,msak0 ou MNGU0
    :return: rien
    lit tous les numpy mfcc et ema correspondant au speaker et les concatène pour avoir une liste de toutes les données.
    On normalise les mfcc et les ema (soustrait moyenne et divise par écart type)
    On divise en deux les phrases plus longues que 800 frames mfcc.

    On a donc une liste X et une liste Y, qu'on va diviser en train et test.
    """
    speakers = get_speakers_per_corpus(corpus)
    for sp in speakers:
        get_fileset_names(sp)

for corpus in ["mocha","MNGU0","usc"] :
    get_fileset_names_per_corpus(corpus)
