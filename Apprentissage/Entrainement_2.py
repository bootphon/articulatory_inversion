### ETUDIER ARTI 6

from class_network import my_bilstm
import sys
import torch
import os
import sys
from sklearn.model_selection import train_test_split
from utils import load_filenames, load_data
from pytorchtools import EarlyStopping
import time

from os.path import dirname
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import scipy
from os import listdir

root_folder = os.path.dirname(os.getcwd())
fileset_path = os.path.join(root_folder, "Donnees_pretraitees", "fileset")

print(sys.argv)


def train_model(test_on ,n_epochs ,delta_test ,patience ,lr=0.09,to_plot=False):
    data_filtered=True
    modele_filtered=True
    train_on = ["MNGU0", "fsew0", "msak0", "F1", "F5", "M1", "M3", "maps0", "faet0", 'mjjn0', "ffes0"]
    train_on.remove(test_on)
    print("train_on :",train_on)
    print("test on:",test_on)

    cuda_avail = torch.cuda.is_available()
    print(" cuda ?", cuda_avail)
    output_dim = 17

    name_file = "test_on_" + test_on
    print("name file : ",name_file)

    hidden_dim = 300
    input_dim = 429
    batch_size = 10

    print("batch size",batch_size)

    early_stopping = EarlyStopping(name_file,patience=patience, verbose=True)

    model = my_bilstm(hidden_dim=hidden_dim,input_dim=input_dim,name_file =name_file, output_dim=output_dim,
                      batch_size=batch_size,data_filtered=data_filtered,cuda_avail = cuda_avail,modele_filtered=modele_filtered)
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

  #  previous_epoch = 0

  #  print("previous epoch  :", previous_epoch)
    if cuda_avail:
        model = model.cuda(device=cuda2)
        torch.backends.cuda.cufft_plan_cache.max_size


    def criterion_pearson(y,y_pred): # (L,K,13)
        y_1 = y - torch.mean(y,dim=1,keepdim=True)  # (L,K,13) - (L,1,13) ==> utile ? normalement proche de 0
        y_pred_1 = y_pred - torch.mean(y_pred,dim=1,keepdim=True)

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
        loss = nume/deno
        loss = torch.sum(loss)
        return -loss
    criterion = criterion_pearson

    optimizer = torch.optim.Adam(model.parameters(), lr=lr ) #, betas = beta_param)

    plt.ioff()
    print("number of epochs : ", n_epochs)
    valid_files_names = []
    N_train,N_valid,N_test=0,0,0
    path_files = os.path.join(os.path.dirname(os.getcwd()),"Donnees_pretraitees","fileset")

    for speaker in train_on:
        print(speaker)
        N_train = N_train + len(open(os.path.join(path_files,speaker+"_train.txt"), "r").read().split())
        N_valid = N_valid + len(open(os.path.join(path_files,speaker+"_valid.txt"), "r").read().split())
        print(N_train)


    N_test = 466
    print('N_train',N_train)
    n_iteration = int(N_train / batch_size)
    n_iteration_validation = int(N_valid/batch_size)
    n_iteration_test = int(N_test/batch_size)
    patience_temp =0
    test_files_names = []

    for epoch in range(n_epochs):
        for ite in range(n_iteration):
          #  if ite % 10 == 0:
           #     print("{} out of {}".format(ite, n_iteration))
            files_for_train = load_filenames(train_on,batch_size,part="train")
            files_for_train = files_for_train+load_filenames(train_on, batch_size, part="test") #on ne va pas tester sur ces speakers

            x,y = load_data(files_for_train,filtered=data_filtered)

       #     y = [y[i][:,:output_dim] for i in range(len(y))]

        #     x, y = X_train[indices], Y_train[indices]
            x, y = model.prepare_batch(x, y)

            y_pred = model(x).double()

          #  print(y_pred)
            torch.cuda.empty_cache()

            if cuda_avail:
                #y_pred = y_pred.cuda()
                y_pred = y_pred.to(device=cuda2)
            y = y.double()
            optimizer.zero_grad()

        #    print("D,E", torch.isnan(model.first_layer.weight.sum()))
            loss = criterion(y,y_pred)

            loss.backward()
            optimizer.step()
          #  print("ll",x.grad)
           # print("G", torch.isnan(model.first_layer.weight.sum()))
        #  model.all_training_loss.append(loss.item())
            torch.cuda.empty_cache()
       # change_lr_frq = 3
       # if epoch%change_lr_frq== 0 :
        #    print("change learning rate",)

        if epoch%delta_test ==0:  #toutes les delta_test epochs on évalue le modèle sur validation et on sauvegarde le modele si le score est meilleur
            loss_vali = 0
            for ite_valid in range(n_iteration_validation):
                files_for_valid = load_filenames(train_on,batch_size,part="valid")
                x,y = load_data(files_for_valid,filtered=data_filtered)
            #    y = [y[i][:,:output_dim] for i in range(len(y))]
                loss_vali+= model.evaluate(x,y,criterion)
            if epoch>0:
                if loss_vali > model.all_validation_loss[-1]:
                    patience_temp +=1
                    if patience_temp == 3 :
                        print("decrease learning rate")
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = param_group['lr'] / 2
                            print(param_group["lr"])
                            patience_temp=0

            loss_vali = loss_vali / n_iteration_validation
            model.all_validation_loss.append(loss_vali)
            model.all_training_loss.append(loss)
            #model.all_validation_loss += [model.all_validation_loss[-1]] * (epoch+previous_epoch - len(model.all_validation_loss))
            loss_test=0
           # if test_on != [""]:
            #    loss_test = model.evaluate_on_test(criterion,X_test = X_test,Y_test = Y_test,to_plot=False,cuda_avail=cuda_avail)
            model.all_test_loss.append(loss_test)
            model.all_training_loss.append(loss)
            #model.all_test_loss += [model.all_test_loss[-1]] * (epoch+previous_epoch - len(model.all_test_loss))
            print("\n ---------- epoch" + str(epoch) + " ---------")
            #early_stopping.epoch = previous_epoch+epoch
            early_stopping(loss_vali, model)
            print("train loss ", loss.item())
            print("valid loss ", loss_vali)
            print("test loss ", loss_test)
            torch.cuda.empty_cache()

        if early_stopping.early_stop:
            print("Early stopping")
            break

    if n_epochs>0:
        model.load_state_dict(torch.load(os.path.join("saved_models",name_file+'.pt')))
        torch.save(model.state_dict(), os.path.join( "saved_models",name_file+".txt"))

    files_for_test = load_filenames([test_on], N_test, part="train")
    files_for_test = files_for_test + load_filenames([test_on], N_test, part="valid")
    files_for_test = files_for_test + load_filenames([test_on], N_test, part="test")

    x, y = load_data(files_for_test)
    print("evaluation on speaker {}".format(test_on))
    speaker_2 = test_on
    if test_on in ["F1", "M1", "F5","M3"]:
        speaker_2 = "usc_timit_" + test_on

    std_speaker=  np.load(os.path.join(root_folder, "Traitement", "norm_values","std_ema_" + speaker_2 + ".npy"))
    model.evaluate_on_test(criterion=criterion,verbose=True, X_test=x, Y_test=y,
                           to_plot=to_plot, std_ema=max(std_speaker), suffix=test_on)


if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Train and save a model.')
    parser.add_argument('test_on', metavar='test_on', type=str,
                        help='the speaker we want to test on')

    parser.add_argument('n_epochs', metavar='n_epochs', type=int,
                        help='max number of epochs to train the model')
    parser.add_argument('delta_test', metavar='delta_test', type=int,
                        help='interval between two validation evaluation')
    parser.add_argument('patience', metavar='patience', type=int,
                        help='number of iterations in a row with decreasing validation score before stopping the train ')
    parser.add_argument('lr', metavar='lr', type=str,
                        help='learning rate of Adam optimizer ')
    parser.add_argument('to_plot', metavar='to_plot', type=bool,         help='si true plot les resultats sur le test')

    args = parser.parse_args()
    test_on =  sys.argv[1]
    n_epochs = int( sys.argv[2] )
    delta_test = int(sys.argv[3])
    patience = int(sys.argv[4])
    lr = float(sys.argv[5])
    to_plot = sys.argv[6].lower()=="true"

    train_model(test_on = test_on,n_epochs=n_epochs,delta_test=delta_test,patience=patience,
                lr = lr,to_plot=to_plot)