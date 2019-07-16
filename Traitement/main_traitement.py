import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)
from os.path import dirname

from traitement_haskins import traitement_general_haskins
from traitement_mngu0 import traitement_general_mngu0
from traitement_usc_timit_2 import traitement_general_usc_timit
from traitement_mocha import traitement_general_mocha
from add_vocal_tract import add_vocal_tract_per_corpus
from split_sentences import split_sentences
from normalization import normalize_data_per_corpus


def main_traitement(corpus_to_treat = ["mocha","usc","MNGU0","Haskins"],max="All",split=False):
    root_path = dirname(dirname(os.path.realpath(__file__)))

    if not os.path.exists(os.path.join(os.path.join(root_path, "Traitement","norm_values"))):
            os.makedirs(os.path.join(os.path.join(root_path, "Traitement","norm_values")))

    if "mocha" in corpus_to_treat :
        traitement_general_mocha(max=max)

    if "MNGU0" in corpus_to_treat:
        traitement_general_mngu0(max=max)
        #split = True

    if "usc" in corpus_to_treat  :
        traitement_general_usc_timit(max=max)

    if "Haskins" in corpus_to_treat :
        traitement_general_haskins(max=max)

    for corpus in corpus_to_treat :
        normalize_data_per_corpus(corpus)
        add_vocal_tract_per_corpus(corpus)


 #   if split :
  #      split_sentences(corpus=["MNGU0"], max_length=500,max=max)

main_traitement(max=10)