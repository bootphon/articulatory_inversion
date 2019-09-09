import os, sys, inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
from os.path import dirname

from Traitement.traitement_haskins import traitement_general_haskins
from Traitement.traitement_mngu0 import traitement_general_mngu0
from Traitement.traitement_usc_timit import traitement_general_usc
from Traitement.traitement_mocha import traitement_general_mocha
import argparse
from multiprocessing import Process


def traitement_general_per_corpus(corp,max):
    """
    :param corp: corpus we want to do the preprocess
    :param max:  max of files to preprocess (useful for test), 0 to treat all files
    perform the preprocess for the asked corpus
     """

    if corp == "MNGU0":
        traitement_general_mngu0(max)
    elif corp == "usc":
        traitement_general_usc(max)
    elif corp == "Haskins":
        traitement_general_haskins(max)
    elif corp == "mocha":
        traitement_general_mocha(max)


if __name__ == '__main__':
    """
    from the cmd to launch preprocess for all the corpuses,
    parallel computing, 1 processor per corpus 
    """
    corpus = ["MNGU0","mocha","usc","Haskins"]
    procs = []

    parser = argparse.ArgumentParser(description='preprocessing of all the corpuses with parallelization')
    parser.add_argument('N_max', metavar='N_max', type=int,
                        help='Nmax we want to preprocess, 0 for all')

    #TO MAKE N_MAX DEFAULT AT 0 ...parser.add_argument('--n_max', metavar='N_max', type=int,help='Nmax we want to preprocess, 0 for all' , default=0)

    args = parser.parse_args()
    N_max = int(sys.argv[1])
    for co in corpus:
        proc = Process(target=traitement_general_per_corpus, args=(co,N_max))
        procs.append(proc)
        proc.start()

    for proc in procs:
        proc.join()



#not used anymore
def main_traitement(corpus_to_treat=["mocha", "usc", "MNGU0", "Haskins"], max=0):
    """
    :param corpus_to_treat: corpus we want to do the preprocess
    :param max:  # max of files to preprocess (useful for test), if 0 all files
    to preprocessing of each corpus one after the other calling the "traitement_general_&corpus" fonction
    """
    root_path = dirname(dirname(os.path.realpath(__file__)))

    if not os.path.exists(os.path.join(os.path.join(root_path, "Traitement", "norm_values"))):
        os.makedirs(os.path.join(os.path.join(root_path, "Traitement", "norm_values")))

    if "mocha" in corpus_to_treat:
        traitement_general_mocha(max)

    if "MNGU0" in corpus_to_treat:
        traitement_general_mngu0(max)

    if "usc" in corpus_to_treat:
        traitement_general_usc(max)

    if "Haskins" in corpus_to_treat:
        traitement_general_haskins(max)
