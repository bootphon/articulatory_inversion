from Traitement.traitement_haskins import traitement_general_haskins
from Traitement.traitement_mngu0 import traitement_general_mngu0
from Traitement.traitement_usc_timit_2 import traitement_general_usc_timit
from Traitement.traitement_mocha import traitement_general_mocha
from Traitement.add_vocal_tract import add_vocal_tract_per_corpus
from Traitement.split_sentences import split_sentences

def main_traitement(corpus_to_treat = ["mocha","usc","MNGU0","Haskins"]):

    if "mocha" in corpus_to_treat :
        traitement_general_mocha(N="All")

    if "MNGU0" in corpus_to_treat:
        traitement_general_mngu0(N="All")

    if "usc" in corpus_to_treat  :
        traitement_general_usc_timit(N="All")

    if "Haskins" in corpus_to_treat :
        traitement_general_haskins(N="All")

    for corpus in corpus_to_treat :
        add_vocal_tract_per_corpus(corpus)

    if "MNGU0" in corpus_to_treat :
        split_sentences(corpus = ["MNGU0"],max_length= 500)




