import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)


from Apprentissage.train import train_model
from Traitement.fonctions_utiles import get_speakers_per_corpus
import sys
from Apprentissage.train_on_speaker import train_model_on_speaker
from multiprocessing import Process
import argparse
corpus = "[Haskins]"
speakers = get_speakers_per_corpus("Haskins")



def cross_val_for_type_filter_has(speaker): #influence du filtre ==> pas bcp
    patience = 3
    n_epochs = 50
    select_arti = True
    corpus_to_train_on = str(corpus)
    batch_norma = False
    loss_train = "both_90"
    for filter_type in [0] :# [0,1,2]:
        train_model(test_on=speaker, n_epochs=n_epochs, loss_train=loss_train, patience=patience,
                    select_arti=select_arti, corpus_to_train_on=corpus_to_train_on,
                    batch_norma=batch_norma, filter_type=filter_type,train_a_bit_on_test=False)


def cross_val_for_alpha_has(speaker): #influence de alpha ==> a voir..
    patience = 3
    n_epochs = 50
    select_arti = True
    corpus_to_train_on = corpus
    filter_type = 1
    batch_norma = False
    for alpha in [0,40,100] :
        loss_train = "both_" + str(alpha)
        train_model(test_on=speaker, n_epochs=n_epochs, loss_train=loss_train, patience=patience,
                    select_arti=select_arti, corpus_to_train_on=corpus_to_train_on,
                    batch_norma=batch_norma, filter_type=filter_type,train_a_bit_on_test=False)


def cross_val_for_bn_has(speaker):  # influence de bn bof

    patience = 3
    n_epochs = 50
    select_arti = True
    corpus_to_train_on = corpus
    filter_type = 1
    loss_train = "both_90"
    train_model(test_on=speaker, n_epochs=n_epochs, loss_train=loss_train, patience=patience,
                select_arti=select_arti, corpus_to_train_on=corpus_to_train_on,
                batch_norma=False, filter_type=filter_type, train_a_bit_on_test=False)

    train_model(test_on=speaker, n_epochs=n_epochs, loss_train=loss_train, patience=patience,
                select_arti=select_arti, corpus_to_train_on=corpus_to_train_on,
                batch_norma=True, filter_type=filter_type, train_a_bit_on_test=False)


def cross_val_for_rmse_has(speaker): #resultats finaux loss rmse generalisation
    patience = 3
    n_epochs = 50
    select_arti = True
    corpus_to_train_on = corpus
    filter_type = 1
    loss_train ="rmse"
    train_model(test_on=speaker, n_epochs=n_epochs, loss_train=loss_train, patience=patience,
                    select_arti=select_arti, corpus_to_train_on=corpus_to_train_on,
                    batch_norma=False, filter_type=filter_type,train_a_bit_on_test=False)


def cross_val_for_rmse_has_and_test_speaker(speaker): #resultats de non généralisation rmse
    patience = 3
    n_epochs = 50
    select_arti = True
    corpus_to_train_on = corpus
    filter_type = 1
    loss_train ="rmse"
    train_model(test_on=speaker, n_epochs=n_epochs, loss_train=loss_train, patience=patience,
                    select_arti=select_arti, corpus_to_train_on=corpus_to_train_on,
                    batch_norma=False, filter_type=filter_type,train_a_bit_on_test=True)

def cross_val_for_both_90_has_and_test_speaker(speaker): #resultats de non généralisation both90
    patience = 3
    n_epochs = 50
    select_arti = True
    corpus_to_train_on = corpus
    filter_type = 1
    loss_train ="both_90"
    train_model(test_on=speaker, n_epochs=n_epochs, loss_train=loss_train, patience=patience,
                    select_arti=select_arti, corpus_to_train_on=corpus_to_train_on,
                    batch_norma=False, filter_type=filter_type,train_a_bit_on_test=True)

def speaker_dependant(speaker):
    train_model_on_speaker(speaker,"rmse","non")
    train_model_on_speaker(speaker,"both_90","non")




if __name__=='__main__':
    experience = sys.argv[1]
    speakers = ["M01", "M02", "M03", "M04"]

    if experience == "filter":
        for sp in speakers:
            for k in range(2):
                cross_val_for_type_filter_has(sp)
     #   for sp in speakers[4:]:
       #     proc = Process(target=cross_val_for_type_filter_has, args=(sp,))
        #    procs.append(proc)
         #   proc.start()

    #    for proc in procs:
     #       proc.join()

    elif experience == "alpha":
        procs = []
        for k in range(3):
            for sp in speakers:
                cross_val_for_bn_has(sp)

    elif experience == "bn":
      for k in range(1):
          for sp in speakers:
              cross_val_for_bn_has(sp)

    elif experience == "rmse":
      for k in range(3):
          for sp in speakers:
              cross_val_for_rmse_has_and_test_speaker(sp)

    elif experience == "also_test_rmse":
      for k in range(3):
          for sp in get_speakers_per_corpus("Haskins"):
              cross_val_for_rmse_has_and_test_speaker(sp)

    elif experience == "also_test_both_90":
      for k in range(3):
          speakers = get_speakers_per_corpus("Haskins")
          speakers = ["M01","M02","M03","M04"]
          for sp in speakers:
              cross_val_for_both_90_has_and_test_speaker(sp)

    elif experience == "speaker_dep":
        speakers = ["M04"]
        for sp in speakers:
            speaker_dependant(sp)


      #  for proc in procs:
     #       proc.join()
    #    procs = []
    #    for sp in speakers[:4]:
     #       proc = Process(target=cross_val_for_bn_has, args=(sp,))
      #      procs.append(proc)
       #     proc.start()

      #  for proc in procs:
       #     proc.join()