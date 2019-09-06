import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)

ncpu="10"
import os
os.environ["OMP_NUM_THREADS"] = ncpu # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = ncpu # export OPENBLAS_NUM_THREADS=4
os.environ["MKL_NUM_THREADS"] = ncpu # export MKL_NUM_THREADS=4
os.environ["VECLIB_MAXIMUM_THREADS"] = ncpu # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = ncpu # export NUMEXPR_NUM_THREADS=4
import numpy as np


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
from scipy import signal
import matplotlib.pyplot as plt
from Traitement.fonctions_utiles import get_speakers_per_corpus
import scipy
from os import listdir
from logger import Logger
import json


root_folder = os.path.dirname(os.getcwd())
fileset_path = os.path.join(root_folder, "Donnees_pretraitees", "fileset")

print(sys.argv)


def train_model_on_speaker(test_on ,loss_train,pretrain_model):
    """
    Speaker scpecific learning : apprendre et test sur le même speaker.
    Possibilité d'uiliser un pré-entrainement
    """
    n_epochs = 50
    batch_norma = False
    filter_type = 1
    delta_test=  1
    lr = 0.001
    to_plot = True

    cuda_avail = torch.cuda.is_available()
    print(" cuda ?", cuda_avail)

    name_file = "train_on_and_test_on" +test_on +"_loss_"+str(loss_train)+"_pretrained_on_"+pretrain_model
    file_weights = os.path.join("saved_models", pretrain_model +".txt")

    if not(os.path.exists(file_weights)):
        print("SANS PREENTRAINEMENT")
        name_file = "train_on_and_test_on" + test_on + "_loss_" + str(loss_train)

    patience = 5
    hidden_dim = 300
    input_dim = 429
    batch_size = 10
    output_dim = 18

    early_stopping = EarlyStopping(name_file,patience=patience, verbose=True)

    model = my_ac2art_modele(hidden_dim=hidden_dim, input_dim=input_dim, name_file=name_file, output_dim=output_dim,
                      batch_size=batch_size, cuda_avail=cuda_avail,
                      modele_filtered=filter_type,batch_norma=batch_norma)
    model = model.double()


    if cuda_avail:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    load_old_model = True
    if os.path.exists(file_weights):
        print("AVEC PREENTRAINEMENT")
        loaded_state = torch.load(file_weights, map_location=device)
        model.load_state_dict(loaded_state)
        model_dict = model.state_dict()
        loaded_state = {k: v for k, v in loaded_state.items() if
                        k in model_dict}  # only layers param that are in our current model
        loaded_state = {k: v for k, v in loaded_state.items() if
                        loaded_state[k].shape == model_dict[k].shape}  # only if layers have correct shapes
        model_dict.update(loaded_state)
        model.load_state_dict(model_dict)


    if cuda_avail:
        model = model.to(device =  device)

    def criterion_pearson(my_y,my_y_pred): # (L,K,13)
        y_1 = my_y - torch.mean(my_y,dim=1,keepdim=True)
        y_pred_1 = my_y_pred - torch.mean(my_y_pred,dim=1,keepdim=True)
        nume=  torch.sum(y_1* y_pred_1,dim=1,keepdim=True) # y*y_pred multi terme à terme puis on somme pour avoir (L,1,13)
      #pour chaque trajectoire on somme le produit de la vriae et de la predite
        deno =  torch.sqrt(torch.sum(y_1 ** 2,dim=1,keepdim=True)) * torch.sqrt(torch.sum(y_pred_1 ** 2,dim=1,keepdim=True))# use Pearson correlation
        # deno zero veut dire ema constant à 0 on remplace par des 1
        minim = torch.tensor(0.01,dtype=torch.float64)
        if cuda_avail:
            minim = minim.to(device=device)
            deno = deno.to(device=device)
            nume = nume.to(device=device)
        deno = torch.max(deno,minim)
        my_loss = torch.div(nume,deno)
        my_loss = torch.sum(my_loss) #pearson doit etre le plus grand possible
        return -my_loss

    criterion_rmse = torch.nn.MSELoss(reduction='sum')

    def criterion_both(L):
        L = L/100 #% de pearson dans la loss
        def criterion_both_lbd(my_y,my_ypred):
            a = L * criterion_pearson(my_y, my_ypred)
            b = (1 - L) * criterion_rmse(my_y, my_ypred) / 1000
            new_loss = a + b
          #  print(a,b,new_loss)
           # return new_loss
            return new_loss
        return criterion_both_lbd

    if loss_train == "rmse":
        criterion = criterion_rmse
    elif loss_train == "pearson" :
        print("pearson")
        criterion = criterion_pearson
    elif loss_train[:4] == "both":
        lbd = int(loss_train[5:])
        criterion = criterion_both(lbd)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr ) #, betas = beta_param)

    path_files = os.path.join(os.path.dirname(os.getcwd()),"Donnees_pretraitees","fileset")
    files_for_train = load_filenames_deter([test_on], part=["train"])
    files_for_valid = load_filenames_deter([test_on], part=["valid"])
    files_for_test = load_filenames_deter([test_on], part=["test"])

    N_iter_train = int(len(files_for_train) / batch_size) + 1  # on veut qu'il y a en ait un multiple du batch size , on en double certains
    n_a_ajouter = batch_size * N_iter_train - len(files_for_train)  # si 14 element N_iter_categ vaut 2 et n_a_ajouter vaut 6
    files_for_train = files_for_train + files_for_train[:n_a_ajouter]  # nbr de fichier par categorie multiple du batch size

    N_iter_valid = int(len(files_for_valid) / batch_size) + 1  # on veut qu'il y a en ait un multiple du batch size , on en double certains
    n_a_ajouter = batch_size * N_iter_valid - len(files_for_valid)  # si 14 element N_iter_categ vaut 2 et n_a_ajouter vaut 6
    files_for_valid = files_for_valid + files_for_valid[:n_a_ajouter]  # nbr de fichier par categorie multiple du batch size

    plot_filtre_chaque_epochs = False

    for epoch in range(n_epochs):

        n_this_epoch = 0
        loss_train_this_epoch = 0
        files_for_train_courant = files_for_train
        files_for_valid_courant = files_for_valid
        random.shuffle(files_for_train_courant)
        random.shuffle(files_for_valid)

        while len(files_for_train_courant) > 0:
            n_this_epoch+=1
            x, y = load_data(files_for_train_courant[:batch_size])
            files_for_train_courant = files_for_train_courant[batch_size:] #we a re going to train on this 10 files
            x, y = model.prepare_batch(x, y)

            y_pred = model(x).double()
            if cuda_avail:
                y_pred = y_pred.to(device=device)
            y = y.double()
            optimizer.zero_grad()

            loss = criterion(y, y_pred)
            loss.backward()
            #a partir de là y_pred.grad a des elements
     #       print("ypred grad shape", y_pred.grad.shape)
            optimizer.step()
            torch.cuda.empty_cache()
            loss_train_this_epoch += loss.item()
            weight_apres = model.lowpass.weight.data[0, 0, :]
         #    print("GAIN 1", sum(weight_apres.cpu()))

        loss_train_this_epoch = loss_train_this_epoch/n_this_epoch

        if epoch%delta_test ==0:  #toutes les delta_test epochs on évalue le modèle sur validation et on sauvegarde le modele si le score est meilleur
         #   x, y = load_data(files_for_test)
          #  print("evaluation on speaker {}".format(test_on))
           # model.evaluate_on_test(x, y, verbose=True, to_plot=to_plot)

          #  print("evaluation validation")
            loss_vali = 0
            n_valid = 0
            while len(files_for_valid_courant)>0:
                n_valid +=1
                x, y = load_data(files_for_valid_courant[:batch_size])
                files_for_valid_courant = files_for_valid_courant[batch_size:]  # on a appris sur ces 10 phrases
                x, y = model.prepare_batch(x, y)
                y_pred = model(x).double()
                if cuda_avail:
                    y_pred = y_pred.to(device=device)
                y = y.double()  # (Batchsize, maxL, 18)

                loss_courant = criterion(y, y_pred)
                loss_vali += loss_courant.item()
            loss_vali  = loss_vali/n_valid
            model.all_validation_loss.append(loss_vali)
            model.all_training_loss.append(loss_train_this_epoch)
          #  print("all training loss",model.all_training_loss)
            early_stopping(loss_vali, model)

            if epoch>0:
                if loss_vali > model.all_validation_loss[-1]:
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = param_group['lr'] / 2
                        (param_group["lr"])

            torch.cuda.empty_cache()
            if early_stopping.early_stop:
             print("Early stopping, n epochs : ",model.epoch_ref+epoch)
             break

    if n_epochs>0:
        model.epoch_ref = model.epoch_ref + epoch
        total_epoch =model.epoch_ref
        model.load_state_dict(torch.load(os.path.join("saved_models",name_file+'.pt')))
        torch.save(model.state_dict(), os.path.join( "saved_models",name_file+".txt"))

    random.shuffle(files_for_test)
    x, y = load_data(files_for_test)
    print("evaluation on speaker {}".format(test_on))
    #print("DATA AND MODELE FILTERED")
    std_speaker = np.load(os.path.join(root_folder,"Traitement","norm_values","std_ema_"+test_on+".npy"))
    arti_per_speaker = os.path.join(root_folder, "Traitement", "articulators_per_speaker.csv")
    csv.register_dialect('myDialect', delimiter=';')
    weight_apres = model.lowpass.weight.data[0, 0, :]
  #  print("GAINAAA",sum(weight_apres.cpu()))
    with open(arti_per_speaker, 'r') as csvFile:
        reader = csv.reader(csvFile, dialect="myDialect")
        next(reader)
        for row in reader:
            if row[0] == test_on:
                arti_to_consider = row[1:19]
                arti_to_consider = [int(x) for x in arti_to_consider]
  #  print("arti to cons",arti_to_consider)
    rmse_per_arti_mean, pearson_per_arti_mean = model.evaluate_on_test(x,y, std_speaker = std_speaker, to_plot=to_plot
                                                                       ,to_consider = arti_to_consider,
                                                                       ) #,filtered=True)
    print("name file : ",name_file)

    with open('resultats_modeles.csv', 'a') as f:
        writer = csv.writer(f)
        row_rmse = [name_file]+rmse_per_arti_mean.tolist()+[model.epoch_ref]
        row_pearson  =[name_file]+pearson_per_arti_mean.tolist() + [model.epoch_ref]
        writer.writerow(row_rmse)
        writer.writerow(row_pearson)
    plot_filtre_chaque_epochs = False

    if plot_filtre_chaque_epochs:
        weight_apres = model.lowpass.weight.data[0, 0, :]
        print("GAIN",sum(weight_apres.cpu()))

        freqs, h = signal.freqz(weight_apres.cpu())
        freqs = freqs * 100 / (2 * np.pi)  # freq in hz
        plt.plot(freqs, 20 * np.log10(abs(h)), 'r')
        plt.title("Allure filtre passe bas à la fin de l'apprentissage pour filtre en dur")
        plt.ylabel('Amplitude [dB]')
        plt.xlabel("real frequency")
        plt.show()

    return rmse_per_arti_mean, pearson_per_arti_mean

    #x, y = load_data(files_for_test,filtered=False)
   # print("DATA AND MODELE NOT FILTERED")
    #model.evaluate_on_test(x,y, to_plot=to_plot,filtered=False)


if __name__=='__main__':
    n_per_model = 1
    test_on =  sys.argv[1]
    loss_train = sys.argv[2]
    pretrain_model = str(sys.argv[3])
    rmse_all,pearson_all = np.zeros((n_per_model,18)), np.zeros((n_per_model,18))
    for k in range(n_per_model):
        rmse_all[k, :], pearson_all[k, :] = train_model_on_speaker(test_on = test_on, loss_train = loss_train
                                                        , pretrain_model = pretrain_model)

    rmse_moy = np.mean(rmse_all,axis=0)
    rmse_std = np.std(rmse_all,axis=0)

    pearson_moy = np.mean(pearson_all,axis=0)
    pearson_std = np.std(pearson_all,axis=0)

    print("final model result for model pretrained on  ",pretrain_model)
    print("rmse : \n",rmse_moy,"\n",rmse_std)
    print("pearson :\n ",pearson_moy,"\n",pearson_std)