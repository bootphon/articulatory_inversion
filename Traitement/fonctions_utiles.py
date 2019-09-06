import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)



import os
from os.path import dirname
import numpy as np
import numpy as np
import random
import os
from os.path import dirname
from random import shuffle
import csv
import json
import scipy


def get_delta_features(array, window=5):
    #TODO
    all_diff = []
    for lag in range(1, window + 1):
        padding = np.ones((lag, array.shape[1]))
        past = np.concatenate([padding * array[0], array[:-lag]])
        future = np.concatenate([array[lag:], padding * array[-1]])
        all_diff.append(future - past)
    tempo =np.array([ all_diff[lag] * lag for lag in range(window)])
    norm = 2 * np.sum(i ** 2 for i in range(1, window + 1))

    delta_features = np.sum(tempo,axis=0)/norm
    return delta_features


def get_speakers_per_corpus(corpus):
    #TODO
    if corpus == "MNGU0":
        speakers = ["MNGU0"]
    elif corpus == "usc":
        speakers = ["F1", "F5", "M1", "M3"]
    elif corpus == "Haskins":
        speakers = ["F01", "F02", "F03", "F04", "M01", "M02", "M03", "M04"]
    elif corpus == "mocha":
        speakers = ["fsew0", "msak0", "faet0", "ffes0", "maps0", "mjjn0", "falh0"]
    else:
        raise NameError("vous navez pas choisi un des corpus")
    return speakers



root_folder = os.path.dirname(os.getcwd())
donnees_path = os.path.join(root_folder, "Donnees_pretraitees")

def get_fileset_names(speaker):
    """
    #TODO: description de ce que tu fais et en ANGLAIS
    :param speaker: un des speaker
    :return: rien
    Ecrit pour le speaker 3 fichiers txt sp_train, sp_test, sp_valid avec les noms des fichiers du train/test/validation set
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
    """
    # TODO description
    :param corpus: un des corpus "mocha","usc","MNGU0","Haskins"
    :return:  rien, crée les fileset pour tous les speaker du corpus
    """
    speakers = get_speakers_per_corpus(corpus)
    for sp in speakers :
        try:
            get_fileset_names(sp)
        except :
            print("Pbm pour creer le fileset de sp ,",sp)


#get_fileset_names_per_corpus("MNGU0")
def read_csv_arti_ok_per_speaker():
    """
    # TODO description
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


def add_voicing(wav, sr):
    # TODO
    hop_time = 10 / 1000  # en ms
    hop_length = int((hop_time * sr))
    N_frames = int(len(wav) / hop_length)
    window = scipy.signal.get_window("hanning", N_frames)
    ste = scipy.signal.convolve(wav ** 2, window ** 2, mode="same")
  #  ste = scipy.signal.resample(ste, num=len(ema))
    ste = [np.max(min(x, 1), 0) for x in ste]
    return ste

    #read_csv_arti_ok_per_speaker()

def low_pass_filter_weight(cut_off,sampling_rate):
    # TODO
    fc = cut_off/sampling_rate# Cutoff frequency as a fraction of the sampling rate (in (0, 0.5)).
    if fc > 0.5:
        raise Exception("La frequence de coupure doit etre au moins deux fois la frequence dechantillonnage")
    b = 0.08  # Transition band, as a fraction of the sampling rate (in (0, 0.5)).

    N = int(np.ceil((4 / b))) #le window
    if not N % 2:
        N += 1  # Make sure that N is odd.
    n = np.arange(N) #int of [0,N]
    h = np.sinc(2 * fc * (n - (N - 1) / 2))  # Compute sinc filter.
    w = 0.5 * (1 - np.cos(2 * np.pi * n / (N-1))) # Compute hanning window.
    h = h * w  # Multiply sinc filter with window.
    h = h / np.sum(h)
    return h