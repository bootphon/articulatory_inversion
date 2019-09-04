import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)

from multiprocessing import Process

from Apprentissage.train_all_param import train_model_new
from Traitement.fonctions_utiles import get_speakers_per_corpus
import sys
import argparse


# plus souvent fixe
n_epochs = 50
loss_train = "both_90"
patience = 5
select_arti = True
batch_norma = False # on n'a pas trouvé celà utile, mais à étudier
filter_type = 1 # 0 c'est filtre en dehors, 1 filtre fixe, 2 filtre variable ie les poids se modifient
to_plot = False # plot pour 1 phrase random les trajectoirs articulatoires prédites, en modifiant le code on peut
                # aussi tracer la courbes non lissée.

lr = 0.001
delta_test = 1 # toutes les combien d'époch on veut regarder le validation score
                # autour de 20 épochs pour l'apprentissage  environ

# exemple arbitraire
test_on = "F01"
corpus_to_train_on  = "[Haskins]" #si directement dans la console guillements non nécéssaires
train_a_bit_on_test = False # si False le speaker-test est exclu du training set, si True on rajoute les parties
                            # train et valid du speaker-test au train et valid du modèle, le test se fait alors sur
                            # la partie test du speaker test


# CHOISIR LES PARAMETRES QUE TU VEUX MODIFIER RAPIDEMENT A PARTIR DE LA LIGNE DE COMMANDE ET AJOUTE LES EN DESSOUS

if __name__=='__main__':

    test_on = sys.argv[1] #"F01"
    corpus_to_train_on = sys.argv[2] #"[Haskins]"  # si directement dans la console guillements non nécéssaires
    train_a_bit_on_test = sys.argv[3] # False

    train_model_new(test_on=test_on, n_epochs=n_epochs, loss_train=loss_train, patience=patience,
                    select_arti=select_arti, corpus_to_train_on=corpus_to_train_on, batch_norma=batch_norma,
                    filter_type=filter_type, train_a_bit_on_test=train_a_bit_on_test, to_plot=to_plot
                    , lr=lr, delta_test=delta_test)

