import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)


from Apprentissage.Entrainement_3 import train_model
from Traitement.fonctions_utiles import get_speakers_per_corpus
import sys
import argparse
corpus = "[mocha,MNGU0,usc,Haskins]"
speakers_cross_val = ["fsew0","msak0","M1","F01","M01","MNGU0"]
def cross_val_for_type_filter():
    patience = 3
    n_epochs = 50
    select_arti = True
    corpus_to_train_on = str(corpus)
    batch_norma = False
    loss_train = "both_90"
    for sp in speakers_cross_val:
        #change filter
        for filter_type in [0,1,2]:
            train_model(test_on=sp, n_epochs=n_epochs, loss_train=loss_train, patience=patience,
                        select_arti=select_arti, corpus_to_train_on=corpus_to_train_on,
                        batch_norma=batch_norma, filter_type=filter_type)


def cross_val_for_alpha():
    patience = 3
    n_epochs = 50
    select_arti = True
    corpus_to_train_on = corpus
    only_one_sp = False
    filter_type = 1
    batch_norma = False
    for sp in speakers_cross_val:
        for alpha in [0,20,40,60,80,100] :
            loss_train = "both_"+alpha
            train_model(test_on=sp, n_epochs=n_epochs, loss_train=loss_train, patience=patience,
                        select_arti=select_arti, corpus_to_train_on=corpus_to_train_on,
                        batch_norma=batch_norma, filter_type=filter_type)


def cross_val_for_bn():
    patience = 3
    n_epochs = 50
    select_arti = True
    corpus_to_train_on = corpus
    filter_type = 1
    loss_train ="both_90"
    for sp in speakers_cross_val:
        train_model(test_on=sp, n_epochs=n_epochs, loss_train=loss_train, patience=patience,
                        select_arti=select_arti, corpus_to_train_on=corpus_to_train_on,
                        batch_norma=False, filter_type=filter_type)

        train_model(test_on=sp, n_epochs=n_epochs, loss_train=loss_train, patience=patience,
                        select_arti=select_arti, corpus_to_train_on=corpus_to_train_on,
                        batch_norma=True, filter_type=filter_type)

if __name__=='__main__':
    experience = sys.argv[1]
    if experience == "filter":
        cross_val_for_type_filter()
    elif experience == "alpha":
        cross_val_for_alpha()
    elif experience == "bn":
        cross_val_for_bn()