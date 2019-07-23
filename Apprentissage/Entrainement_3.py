import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)

from Apprentissage.class_network import my_bilstm
from Apprentissage.modele import my_ac2art_modele
import sys
import torch
import os
import csv
import sys
from sklearn.model_selection import train_test_split
from Apprentissage.utils import load_filenames, load_data, load_filenames_deter
from Apprentissage.pytorchtools import EarlyStopping
import time
import random
from os.path import dirname
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from Traitement.fonctions_utiles import get_speakers_per_corpus
import scipy
from os import listdir
#from logger import Logger
import json


root_folder = os.path.dirname(os.getcwd())
fileset_path = os.path.join(root_folder, "Donnees_pretraitees", "fileset")

print(sys.argv)

def train_model(test_on ,n_epochs ,delta_test ,patience ,lr,to_plot,select_arti,corpus_to_train_on):
    data_filtered=True
    modele_filtered=True
    name_corpus_concat = ""
    train_on = []
    corpus_to_train_on = corpus_to_train_on[1:-1].split(",")
    for corpus in corpus_to_train_on :
        print("corpus" , corpus)
        sp = get_speakers_per_corpus(corpus)
        train_on = train_on + sp
        name_corpus_concat = name_corpus_concat+corpus+"_"
    train_on  = ["MNGU0"]
  #  train_on = ["msak0"]
    print(name_corpus_concat)
    if test_on in train_on :
        train_on.remove(test_on)
    print("train_on :",train_on)
    print("test on:",test_on)

    cuda_avail = torch.cuda.is_available()
    print(" cuda ?", cuda_avail)
    output_dim = 18

   # name_file = "test_on_" + test_on+"_idx_"+str(select_arti)+"_onlycorpus_"+str(only_on_corpus)
    name_file = "train_on_"+name_corpus_concat+"test_on_"+test_on+"_idx_"+str(select_arti)
    print("name file : ",name_file)
#   logger = Logger('./log_' + name_file)

    hidden_dim = 300
    input_dim = 429
    batch_size = 10
    print("batch size",batch_size)

    early_stopping = EarlyStopping(name_file,patience=patience, verbose=True)

    # model = my_bilstm(hidden_dim=hidden_dim,input_dim=input_dim,name_file =name_file, output_dim=output_dim,
   #                   batch_size=batch_size,data_filtered=data_filtered,cuda_avail = cuda_avail,modele_filtered=modele_filtered)
    model = my_ac2art_modele(hidden_dim=hidden_dim, input_dim=input_dim, name_file=name_file, output_dim=output_dim,
                      batch_size=batch_size, cuda_avail=cuda_avail,
                      modele_filtered=modele_filtered)

    model = model.double()

   # try :
    file_weights = os.path.join("saved_models", name_file +".txt")
    if not os.path.exists(file_weights):
        print("premiere fois que ce modèle est crée")
        file_weights = os.path.join("saved_models","modele_preentrainement.txt")

    if not cuda_avail:
        loaded_state = torch.load(file_weights, map_location=torch.device('cpu'))

    else :
        cuda2 = torch.device('cuda:1')
        loaded_state = torch.load( file_weights , map_location= cuda2 )
    model_dict = model.state_dict()
    loaded_state = {k: v for k, v in loaded_state.items() if
                    k in model_dict}  # only layers param that are in our current model
    #print("before ", len(loaded_state), loaded_state.keys())
    loaded_state = {k: v for k, v in loaded_state.items() if
                    loaded_state[k].shape == model_dict[k].shape}  # only if layers have correct shapes
    #print("after", len(loaded_state), loaded_state.keys())
    model_dict.update(loaded_state)
    model.load_state_dict(model_dict)

    if cuda_avail:
        model = model.cuda(device=cuda2)

    def criterion_pearson(my_y,my_y_pred): # (L,K,13)
        y_1 = my_y - torch.mean(my_y,dim=1,keepdim=True)
        y_pred_1 = my_y_pred - torch.mean(my_y_pred,dim=1,keepdim=True)
        nume=  torch.sum(y_1* y_pred_1,dim=1,keepdim=True) # y*y_pred multi terme à terme puis on somme pour avoir (L,1,13)
      #pour chaque trajectoire on somme le produit de la vriae et de la predite
        deno =  torch.sqrt(torch.sum(y_1 ** 2,dim=1,keepdim=True)) * torch.sqrt(torch.sum(y_pred_1 ** 2,dim=1,keepdim=True))# use Pearson correlation
        # deno zero veut dire ema constant à 0 on remplace par des 1
        minim = torch.tensor(0.01,dtype=torch.float64)
        if cuda_avail:
            minim = minim.to(device=cuda2)
            deno = deno.to(device=cuda2)
            nume = nume.to(device=cuda2)
        deno = torch.max(deno,minim)
        my_loss = nume/deno
        my_loss = torch.sum(my_loss) #pearson doit etre le plus grand possible
        #loss = torch.div(loss, torch.tensor(y.shape[2],dtype=torch.float64)) # correlation moyenne par arti
       # print("myloss shape",my_loss.shape)
        return -my_loss

    criterion_rmse = torch.nn.MSELoss(reduction='sum')
    with open('categ_of_speakers.json', 'r') as fp:
        categ_of_speakers = json.load(fp) #dictionnaire en clé la categorie en valeur un dictionnaire
                                            # #avec les speakers dans la catégorie et les arti concernées par cette categorie

    optimizer = torch.optim.Adam(model.parameters(), lr=lr ) #, betas = beta_param)

    plt.ioff()
    print("number of epochs : ", n_epochs)

    path_files = os.path.join(os.path.dirname(os.getcwd()),"Donnees_pretraitees","fileset")

    files_for_train = load_filenames_deter(train_on, part=["train", "test"])
    print("len files for train",len(files_for_train))
    files_for_valid = load_filenames_deter(train_on, part=["valid"])
    print("lenfiles for valid",len(files_for_valid))

    files_for_test = load_filenames_deter([test_on], part=["train", "valid", "test"])
    print("len files for test",len(files_for_test))

    files_per_categ = dict()
    for categ in categ_of_speakers.keys():
        sp_in_categ = categ_of_speakers[categ]["sp"]
        sp_in_categ = [sp for sp in sp_in_categ if sp in train_on]
        # fichiers qui appartiennent à la categorie car le nom du speaker apparait touojurs dans le nom du fichier
        files_train_this_categ = [[f for f in files_for_train if sp.lower() in f.lower() ]for sp in sp_in_categ]

        files_train_this_categ = [item for sublist in files_train_this_categ for item in sublist] # flatten la liste de liste
        files_valid_this_categ = [[f for f in files_for_valid if sp.lower() in f.lower()] for sp in sp_in_categ]
        files_valid_this_categ = [item for sublist in files_valid_this_categ for item in sublist]  # flatten la liste de liste

        if len(files_train_this_categ) > 0 : #meaning we have at least one file in this categ
            files_per_categ[categ] = dict()
            N_iter_categ = int(len(files_train_this_categ)/batch_size)+1         # on veut qu'il y a en ait un multiple du batch size , on en double certains
            n_a_ajouter = batch_size*N_iter_categ - len(files_train_this_categ) #si 14 element N_iter_categ vaut 2 et n_a_ajouter vaut 6
            files_train_this_categ = files_train_this_categ + files_train_this_categ[:n_a_ajouter] #nbr de fichier par categorie multiple du batch size
            random.shuffle(files_train_this_categ)
            files_per_categ[categ]["train"] = files_train_this_categ
            N_iter_categ = int(len(  files_valid_this_categ) / batch_size) + 1  # on veut qu'il y a en ait un multiple du batch size , on en double certains
            n_a_ajouter = batch_size * N_iter_categ - len(files_valid_this_categ)  # si 14 element N_iter_categ vaut 2 et n_a_ajouter vaut 6
            files_valid_this_categ = files_valid_this_categ + files_valid_this_categ[:n_a_ajouter] # nbr de fichier par categorie multiple du batch size
            random.shuffle(files_valid_this_categ)
            files_per_categ[categ]["valid"] = files_valid_this_categ


    categs_to_consider = files_per_categ.keys()
    print("categs to consider",categs_to_consider)
    for epoch in range(n_epochs):
        n_this_epoch = 0
        random.shuffle(list(categs_to_consider))
        loss_train_this_epoch = 0
        for categ in categs_to_consider:  # de A à F pour le momen

            files_this_categ_courant = files_per_categ[categ]["train"] #on na pas encore apprit dessus au cours de cette epoch
            random.shuffle(files_this_categ_courant)

            while len(files_this_categ_courant) > 0:
                n_this_epoch+=1
                x, y = load_data(files_this_categ_courant[:batch_size], filtered=data_filtered,VT=True)
                files_this_categ_courant = files_this_categ_courant[batch_size:] #we a re going to train on this 10 files
                x, y = model.prepare_batch(x, y)
                y_pred = model(x).double()
                if cuda_avail:
                    # y_pred = y_pred.cuda()
                    y_pred = y_pred.to(device=cuda2)
                y = y.double()
                optimizer.zero_grad()

                if select_arti:
                    arti_to_consider = categ_of_speakers[categ]["arti"]  # liste de 18 0/1 qui indique les arti à considérer
                    idx_to_ignore = [i for i, n in enumerate(arti_to_consider) if n == "0"]
                    y[:,:,idx_to_ignore]=0
                    y_pred[:,:,idx_to_ignore]=0

                loss = criterion_pearson(y, y_pred)

                loss.backward()
                optimizer.step()
                torch.cuda.empty_cache()

                loss_train_this_epoch += loss.item()
        loss_train_this_epoch = loss_train_this_epoch/n_this_epoch

        if epoch%delta_test ==0:  #toutes les delta_test epochs on évalue le modèle sur validation et on sauvegarde le modele si le score est meilleur
         #   x, y = load_data(files_for_test)
          #  print("evaluation on speaker {}".format(test_on))
           # model.evaluate_on_test(x, y, verbose=True, to_plot=to_plot)

            print("evaluation validation")
            loss_vali = 0
            n_valid = 0
            for categ in categs_to_consider:  # de A à F pour le moment
                files_this_categ_courant = files_per_categ[categ]["valid"]  # on na pas encore apprit dessus au cours de cette epoch
                while len(files_this_categ_courant) >0 :
                    n_valid +=1
                    x, y = load_data(files_this_categ_courant[:batch_size], filtered=data_filtered,VT=True)
                    files_this_categ_courant = files_this_categ_courant[batch_size:]  # on a appris sur ces 10 phrases
                    x, y = model.prepare_batch(x, y)
                    y_pred = model(x).double()
                    torch.cuda.empty_cache()
                    if cuda_avail:
                        y_pred = y_pred.to(device=cuda2)
                    y = y.double()  # (Batchsize, maxL, 18)

                    if select_arti:
                        arti_to_consider = categ_of_speakers[categ]["arti"]  # liste de 18 0/1 qui indique les arti à considérer
                        idx_to_ignore = [i for i, n in enumerate(arti_to_consider) if n == "0"]
                        y[:, :, idx_to_ignore] = 0
                        y_pred[:, :, idx_to_ignore] = 0

                    loss_courant = criterion_pearson(y, y_pred)
                    loss_vali += loss_courant.item()
            loss_vali  = loss_vali/n_valid

            model.all_validation_loss.append(loss_vali)
            model.all_training_loss.append(loss_train_this_epoch)

            if epoch>0:
                if loss_vali > model.all_validation_loss[-1]:
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = param_group['lr'] / 2
                        print(param_group["lr"])
                        patience_temp=0

            #model.all_test_loss += [model.all_test_loss[-1]] * (epoch+previous_epoch - len(model.all_test_loss))
            print("\n ---------- epoch" + str(epoch) + " ---------")
            #early_stopping.epoch = previous_epoch+epoch
            early_stopping(loss_vali, model)
            print("train loss ", loss.item())
            print("valid loss ", loss_vali)

         #   logger.scalar_summary('loss_valid', loss_vali,
          #                        model.epoch_ref)
           # logger.scalar_summary('loss_train', loss.item(), model.epoch_ref)

            torch.cuda.empty_cache()

        if early_stopping.early_stop:
            print("Early stopping")
            break

    if n_epochs>0:
        model.load_state_dict(torch.load(os.path.join("saved_models",name_file+'.pt')))
        torch.save(model.state_dict(), os.path.join( "saved_models",name_file+".txt"))

    random.shuffle(files_for_test)
    x, y = load_data(files_for_test)
    print("evaluation on speaker {}".format(test_on))
    model.evaluate_on_test(x,y, to_plot=to_plot)

if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Train and save a model.')
    parser.add_argument('test_on', type=str,
                        help='the speaker we want to test on')

    parser.add_argument('n_epochs', type=int,
                        help='max number of epochs to train the model')
    parser.add_argument('delta_test', type=int,
                        help='interval between two validation evaluation')
    parser.add_argument('patience',type=int,
                        help='number of iterations in a row with decreasing validation score before stopping the train ')
    parser.add_argument('lr',    help='learning rate of Adam optimizer ')
    parser.add_argument('to_plot', type=bool,
                        help='si true plot les resultats sur le test')

    parser.add_argument('select_arti', type=bool,
                        help='ssi dans la retropro on ne considere que les arti bons')

    parser.add_argument('corpus_to_train_on',  type=str,
                        help='ssi dans la retropro on ne considere que les arti bons')



    args = parser.parse_args()
    test_on =  sys.argv[1]
    n_epochs = int(sys.argv[2])
    delta_test = int(sys.argv[3])
    patience = int(sys.argv[4])
    lr = float(sys.argv[5])
    to_plot = sys.argv[6].lower()=="true"
    select_arti = sys.argv[7].lower()=="true"
    corpus_to_train_on = str(sys.argv[8])

    train_model(test_on = test_on,n_epochs=n_epochs,delta_test=delta_test,patience=patience,
                lr = lr,to_plot=to_plot,select_arti=select_arti,corpus_to_train_on = corpus_to_train_on)