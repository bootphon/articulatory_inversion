import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)


from traitement_haskins import traitement_general_haskins
from traitement_mngu0 import traitement_general_mngu0
from traitement_usc_timit_2 import traitement_general_usc_timit
from traitement_mocha import traitement_general_mocha
from add_vocal_tract import add_vocal_tract_per_corpus
from split_sentences import split_sentences

def main_traitement(corpus_to_treat = ["mocha","usc","MNGU0","Haskins"],N="All"):
    if "mocha" in corpus_to_treat :
        traitement_general_mocha(N=N)

    if "MNGU0" in corpus_to_treat:
        traitement_general_mngu0(N=N)

    if "usc" in corpus_to_treat  :
        traitement_general_usc_timit(N=N)

    if "Haskins" in corpus_to_treat :
        traitement_general_haskins(N=N)

    for corpus in corpus_to_treat :
        add_vocal_tract_per_corpus(corpus,N=N)

    if "MNGU0" in corpus_to_treat :
        split_sentences(corpus=["MNGU0"], max_length=500,N=N)


main_traitement(N="All")


