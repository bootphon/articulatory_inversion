import os, sys, inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
from os.path import dirname

from Preprocessing.preprocessing_haskins import Preprocessing_general_haskins
from Preprocessing.preprocessing_mngu0 import Preprocessing_general_mngu0
from Preprocessing.preprocessing_usc_timit import Preprocessing_general_usc
from Preprocessing.preprocessing_mocha import Preprocessing_general_mocha
import argparse
from multiprocessing import Process


def Preprocessing_general_per_corpus(corp, max, path_to_corpus):
    """
    :param corp: corpus we want to do the preprocess
    :param max:  max of files to preprocess (useful for test), 0 to treat all files
    perform the preprocess for the asked corpus
     """
    if corp == "MNGU0":
        Preprocessing_general_mngu0(max, path_to_raw=path_to_corpus)
    elif corp == "usc":
        Preprocessing_general_usc(max, path_to_raw=path_to_corpus)
    elif corp == "Haskins":
        Preprocessing_general_haskins(max, path_to_raw=path_to_corpus)
    elif corp == "mocha":
        Preprocessing_general_mocha(max, path_to_raw=path_to_corpus)


if __name__ == '__main__':
    """
    from the cmd to launch preprocess for all the corpuses,
    parallel computing, 1 processor per corpus 
    """
    procs = []

    parser = argparse.ArgumentParser(description='preprocessing of all the corpuses with parallelization')
    parser.add_argument('--N_max',  type=int, default=0,
                        help='by default ')
    parser.add_argument('--corpus',  type=str, default=["MNGU0","mocha","usc","Haskins"],
                        help='corpus to preprocess')
    parser.add_argument('--path_to_raw_data', type=str,
                        help='path to the directory where all the folders with the raw data of each corpus are')

    root_folder = os.path.dirname(os.getcwd())

    if not os.path.exists(os.path.join(root_folder,"Preprocessed_data","fileset")):
        os.makedirs(os.path.join(root_folder,"Preprocessed_data","fileset"))

    if not os.path.exists("norm_values"):
        os.makedirs("norm_values")

    args = parser.parse_args()
    if type(args.corpus) is str:
        corpus = args.corpus[1:-1].split(",")
    else:
        corpus = args.corpus
    for co in corpus:
        proc = Process(target=Preprocessing_general_per_corpus, args=(co, args.N_max, args.path_to_raw_data))
        procs.append(proc)
        proc.start()

    for proc in procs:
        proc.join()

