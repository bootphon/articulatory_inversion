import os, sys, inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
from os.path import dirname

from Traitement.traitement_haskins_new import traitement_general_haskins
from Traitement.traitement_mngu0_new import traitement_general_mngu0
from Traitement.traitement_usc_timit_new import traitement_general_usc
from Traitement.traitement_mocha_new import traitement_general_mocha
from Traitement.create_filesets import get_fileset_names_per_corpus
import argparse
from multiprocessing import Process
import time
# from Traitement.create_filesets import get_fileset_names_per_corpus

def main_traitement(corpus_to_treat=["mocha", "usc", "MNGU0", "Haskins"], max="All", split=False):
    root_path = dirname(dirname(os.path.realpath(__file__)))

    if not os.path.exists(os.path.join(os.path.join(root_path, "Traitement", "norm_values"))):
        os.makedirs(os.path.join(os.path.join(root_path, "Traitement", "norm_values")))

    if "mocha" in corpus_to_treat:
        traitement_general_mocha_new(max)

    if "MNGU0" in corpus_to_treat:
        traitement_general_mngu0_new(max)
        # split = True

    if "usc" in corpus_to_treat:
        traitement_general_usc_new(max)

    if "Haskins" in corpus_to_treat:
        traitement_general_haskins_new(max)

    for corpus in corpus_to_treat:
        get_fileset_names_per_corpus(corpus)


def traitement_general_per_corpus(corp,max):
    print("----------------------------",corp,"-------------")
    if corp == "MNGU0":
        traitement_general_mngu0(max)
    elif corp == "usc":
        traitement_general_usc(max)
    elif corp == "Haskins":
        traitement_general_haskins(max)
    elif corp == "mocha":
        traitement_general_mocha(max)

#if __name__ == '__main__':
    #import argparse

   # parser = argparse.ArgumentParser(description='Train and save a model.')
   # parser.add_argument('N_max', metavar='N_max', type=int, help='nombre de fichiers max')

   # args = parser.parse_args()
  #  N_max = int(sys.argv[1])
 #   main_traitement(max=N_max)

if __name__ == '__main__':
    corpus = ["MNGU0","mocha","usc","Haskins"]
    corpus = ["Haskins"]
    procs = []

    parser = argparse.ArgumentParser(description='Train and save a model.')
    parser.add_argument('N_max', metavar='N_max', type=int,
                        help='nombre de fichiers max que lon veut traiter par corpus')

    args = parser.parse_args()
    N_max = int(sys.argv[1])
    for co in corpus:
        proc = Process(target=traitement_general_per_corpus, args=(co,N_max))
        procs.append(proc)
        proc.start()

    for proc in procs:
        proc.join()