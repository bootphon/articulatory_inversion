import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)


from Apprentissage.train import train_model
from Traitement.fonctions_utiles import get_speakers_per_corpus
import sys
from Apprentissage.train_on_speaker import train_model_on_speaker
from Apprentissage.train_all_param import train_model_new
import numpy as np
from multiprocessing import Process
import argparse
corpus = "[Haskins]"
speakers = get_speakers_per_corpus("Haskins")


# CREER UN EXCEL ET Y METTRE LES RESULTATS DES EXPERIENCES, 1 EXCEL (ou csv) PAR EXPERIENCE
# PREMIERE LIGNE EXPLICATION DE LEXPERIENCE ET DES PARAMETRES FIXES
# PUIS PREMIERE COLONNE CE QUI VARIE ET LA SUITE LES RESULTATS

# plus souvent fixe
n_epochs = 50
loss_train = "both_90"
patience = 5
select_arti = True
batch_norma = False
filter_type = 1
to_plot = False
lr = 0.001
delta_test = 1
corpus = ["Haskins"]
train_a_bit_on_test = False
output_dim = 18 #18 trajectories
speakers = get_speakers_per_corpus(corpus)



def cross_val_for_type_filter_has_indep(): #influence du filtre ==> pas bcp
    corpus_to_train_on = str(corpus)
    for filter_type in [0,1,2] :
        count = 0
        rmse_all, pearson_all = np.zeros((len(speakers), output_dim)), np.zeros((len(speakers), output_dim))
        for speaker in speakers :
            rmse,pearson  = train_model_new(test_on=speaker, n_epochs=n_epochs, loss_train=loss_train, patience=patience,
                        select_arti=select_arti, corpus_to_train_on=corpus_to_train_on, batch_norma=batch_norma,
                        filter_type=filter_type, train_a_bit_on_test=train_a_bit_on_test, to_plot=to_plot
                        , lr=lr, delta_test=delta_test)
            rmse_all[count,:] = rmse
            pearson_all[count,:]= pearson
            count += 1
        results_rmse = np.mean(rmse_all,axis=0)
        results_pearson = np.mean(pearson_all,axis = 0)
        std_rmse = np.std(rmse_all, axis = 0 )
        std_pearson  = np.std(pearson_all, axis = 0 )
        print("for filter type {} results are".format(filter_type))
        print("RMSE mean ", results_rmse) #aulieu de plotter mettre dans un csv ?
        print("RMSE std ", std_rmse)
        print("PEARSON ", results_pearson)
        print(" PEARSON STD")


def cross_val_for_bn_has_indep(): #influence du filtre ==> pas bcp
    corpus_to_train_on = str(corpus)
    for batch_norma in ["True","False"] :
        count = 0
        rmse_all, pearson_all = np.zeros((len(speakers), output_dim)), np.zeros((len(speakers), output_dim))

        for speaker in speakers :
            rmse,pearson  = train_model_new(test_on=speaker, n_epochs=n_epochs, loss_train=loss_train, patience=patience,
                        select_arti=select_arti, corpus_to_train_on=corpus_to_train_on, batch_norma=batch_norma,
                        filter_type=filter_type, train_a_bit_on_test=train_a_bit_on_test, to_plot=to_plot
                        , lr=lr, delta_test=delta_test)
            rmse_all[count,:] = rmse
            pearson_all[count,:]= pearson
            count += 1

        results_rmse = np.mean(rmse_all,axis=0)
        results_pearson = np.mean(pearson_all,axis = 0)
        print("for bn {} results are".format(batch_norma))
        print("RMSE ", results_rmse) #aulieu de plotter mettre dans un csv ?
        print("PEARSON ", results_pearson)




def cross_val_for_alpha_has_indep(): #influence du filtre ==> pas bcp
    corpus_to_train_on = str(corpus)
    for alpha in [0,20,40,60,80,100] :
        count = 0
        loss_train = "both_" + str(alpha)
        rmse_all, pearson_all = np.zeros((len(speakers), output_dim)), np.zeros((len(speakers), output_dim))
        for speaker in speakers :
            rmse,pearson  = train_model_new(test_on=speaker, n_epochs=n_epochs, loss_train=loss_train, patience=patience,
                        select_arti=select_arti, corpus_to_train_on=corpus_to_train_on, batch_norma=batch_norma,
                        filter_type=filter_type, train_a_bit_on_test=train_a_bit_on_test, to_plot=to_plot
                        , lr=lr, delta_test=delta_test)
            rmse_all[count,:] = rmse
            pearson_all[count,:]= pearson
            count += 1

        results_rmse = np.mean(rmse_all,axis=0)
        results_pearson = np.mean(pearson_all,axis = 0)
        print("for alpha {} results are".format(alpha))
        print("RMSE ", results_rmse) #aulieu de plotter mettre dans un csv ?
        print("PEARSON ", results_pearson)




def cross_val_indep(): #influence du filtre ==> pas bcp
    corpus_to_train_on = str(corpus)
    for alpha in [0,20,40,60,80,100] :
        count = 0
        loss_train = "both_" + str(alpha)
        rmse_all, pearson_all = np.zeros((len(speakers), output_dim)), np.zeros((len(speakers), output_dim))
        for speaker in speakers :
            rmse,pearson  = train_model_new(test_on=speaker, n_epochs=n_epochs, loss_train=loss_train, patience=patience,
                        select_arti=select_arti, corpus_to_train_on=corpus_to_train_on, batch_norma=batch_norma,
                        filter_type=filter_type, train_a_bit_on_test=train_a_bit_on_test, to_plot=to_plot
                        , lr=lr, delta_test=delta_test)
            rmse_all[count,:] = rmse
            pearson_all[count,:]= pearson
            count += 1

        results_rmse = np.mean(rmse_all,axis=0)
        results_pearson = np.mean(pearson_all,axis = 0)
        print("for alpha {} results are".format(alpha))
        print("RMSE ", results_rmse) #aulieu de plotter mettre dans un csv ?
        print("PEARSON ", results_pearson)





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
          for sp in speakers:
              cross_val_for_both_90_has_and_test_speaker(sp)

    elif experience == "speaker_dep":
        speakers = ["F04","F03","F01"]
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