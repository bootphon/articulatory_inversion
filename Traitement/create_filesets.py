
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
import csv
import json



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

    files_path =  os.path.join(donnees_path,speaker)
    EMA_files_names = [name[:-4] for name in os.listdir(os.path.join(files_path,"ema_final")) if name.endswith('.npy') ]
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
    speakers = get_speakers_per_corpus(corpus)
    for sp in speakers :
        try:
            get_fileset_names(sp)
        except :
            print("Pbm pour creer le fileset de sp ,",sp)




def read_csv_arti_ok_per_speaker():
    """
    :return:
    dictionnaire avec en clé les différentes categories de speaker (categ de A à F pour le moment). Au sein
    d'une catégorie les speakers ont les mêmes arti valides. Ces catégories sont tirées du fichier CSV qui est lu
    et peut être modifié par l'utilisateur.
    La valeur associée à une categorie est un autre dictionnaire donnant les speakers concernés par cette catégorie
    et les articulateurs concernés, sous forme d'une liste de 18 0 et 1, avec un 1 pour les arti valides.
    """
    arti_per_speaker = os.path.join(root_folder,"Traitement", "articulators_per_speaker.csv")
    csv.register_dialect('myDialect', delimiter=';')
    categ_of_speakers = dict()
    with open(arti_per_speaker, 'r') as csvFile:
        reader = csv.reader(csvFile, dialect="myDialect")
        next(reader)
        for categ in ["A", "B", "C", "D", "E", "F"]:
            categ_of_speakers[categ] = dict()
            categ_of_speakers[categ]["sp"] = []
            categ_of_speakers[categ]["arti"] = None
        for row in reader:
            categ_of_speakers[row[19]]["sp"].append(row[0])
            if categ_of_speakers[row[19]]["arti"]  :
                if categ_of_speakers[row[19]]["arti"] != row[1:19]:
                    print("check arti and category for categ {}".format(row[19]))
            else:
                categ_of_speakers[row[19]]["arti"] = row[1:19]

    for cle in categ_of_speakers.keys():
        print("categ ",cle)
        print(categ_of_speakers[cle])

    with open(os.path.join(root_folder,"Apprentissage","categ_of_speakers.json"), 'w') as dico:
        json.dump(categ_of_speakers, dico)

#read_csv_arti_ok_per_speaker()