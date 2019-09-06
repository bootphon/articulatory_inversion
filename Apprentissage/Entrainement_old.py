# TODO: je suppose que ça tu jettes ?
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
from logger import Logger

root_folder = os.path.dirname(os.getcwd())
fileset_path = os.path.join(root_folder, "Donnees_pretraitees", "fileset")

print(sys.argv)

def train_model(train_on ,test_on ,n_epochs ,delta_test ,patience ,lr=0.09, output_dim=13,data_filtered=False,
                modele_filtered=False,to_plot=False,loss="rmse"): #,norma=True):

    cuda_avail = torch.cuda.is_available()
    print(" cuda ?", cuda_avail)

    train_on = str(train_on[1:-1])
    test_on = str(test_on[1:-1])
    train_on = train_on.split(",")
    test_on = test_on.split(",")
    suff=""
    if data_filtered:
        print("SMOOTHED DATA")
        suff = suff+"_data_filtered"
    if modele_filtered:
        print("MODELE FILTERED")
        suff = suff + "_modele_filtered"
    suff = suff + "_loss_"+loss
    name_file = "train_" + "_".join(train_on) + "_test_" + "_".join(test_on) +suff
    print("name file : ",name_file)
    logger = Logger('./log_' + name_file)

    hidden_dim = 300
    input_dim = 429
    batch_size = 10
    print("batch size",batch_size)

    early_stopping = EarlyStopping(name_file,patience=patience, verbose=True)
    model = my_bilstm(hidden_dim=hidden_dim,input_dim=input_dim,name_file =name_file, output_dim=output_dim,
                      batch_size=batch_size,data_filtered=data_filtered,cuda_avail = cuda_avail,modele_filtered=modele_filtered)
    model = model.double()

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
    print("before ", len(loaded_state), loaded_state.keys())

    loaded_state = {k: v for k, v in loaded_state.items() if
                    loaded_state[k].shape == model_dict[k].shape}  # only if layers have correct shapes
    print("after", len(loaded_state), loaded_state.keys())
    model_dict.update(loaded_state)
    model.load_state_dict(model_dict)

    if cuda_avail:
        model = model.cuda(device=cuda2)
        torch.backends.cuda.cufft_plan_cache.max_size


    def criterion_pearson(y,y_pred): # (L,K,13)
        y_1 = y - torch.mean(y,dim=1,keepdim=True)  # (L,K,13) - (L,1,13) ==> utile ? normalement proche de 0
        y_pred_1 = y_pred - torch.mean(y_pred,dim=1,keepdim=True)
        nume =  torch.sum(y_1* y_pred_1,dim=1,keepdim=True) # y*y_pred multi terme à terme puis on somme pour avoir (L,1,13)
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

    criterion_rmse = torch.nn.MSELoss(reduction='sum')
    if loss == "pearson":
        criterion = criterion_pearson
    elif loss == "rmse":
        criterion = criterion_rmse
    optimizer = torch.optim.Adam(model.parameters(), lr=lr ) #, betas = beta_param)

    plt.ioff()
    print("number of epochs : ", n_epochs)

    N_train,N_valid,N_test=0,0,0
    path_files = os.path.join(os.path.dirname(os.getcwd()),"Donnees_pretraitees","fileset")

    for speaker in train_on:
        N_train =+len(open(os.path.join(path_files,speaker+"_train.txt"), "r").read().split())
        N_valid =+len(open(os.path.join(path_files,speaker+"_valid.txt"), "r").read().split())

    for speaker in test_on:
        N_test =+len(open(os.path.join(path_files,speaker+"_test.txt"), "r").read().split())
    print('N_train',N_train)
    n_iteration = int(N_train / batch_size)
    n_iteration_validation = int(N_valid/batch_size)
    n_iteration_test = int(N_test/batch_size)
    n_iteration = 1
    patience_temp =0

    for epoch in range(n_epochs):
        model.epoch_ref = model.epoch_ref + 1

        for ite in range(n_iteration):
            files_for_train = load_filenames(train_on,batch_size,part=["train"])
            x,y = load_data(files_for_train,filtered=data_filtered)
            y = [y[i][:,:output_dim] for i in range(len(y))]
            x, y = model.prepare_batch(x, y)
            y_pred = model(x).double()
            torch.cuda.empty_cache()

            if cuda_avail:
                #y_pred = y_pred.cuda()
                y_pred = y_pred.to(device=cuda2)
            y = y.double()
            optimizer.zero_grad()
            loss = criterion(y,y_pred)

            loss.backward()
            optimizer.step()
            torch.cuda.empty_cache()


        if epoch%delta_test ==0:  #toutes les delta_test epochs on évalue le modèle sur validation et on sauvegarde le modele si le score est meilleur


            loss_vali = 0
            for ite_valid in range(n_iteration_validation):
                files_for_valid = load_filenames(train_on,batch_size,part=["valid"])
                x,y = load_data(files_for_valid,filtered=data_filtered)
                y = [y[i][:,:output_dim] for i in range(len(y))]
                loss_vali+= model.evaluate(x,y,criterion)

            loss_vali = loss_vali / n_iteration_validation
            if epoch>0:
                if loss_vali > model.all_validation_loss[-1]:
                    patience_temp +=1
                    if patience_temp == 3 :
                        print("decrease learning rate")
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = param_group['lr'] / 2
                            print(param_group["lr"])
                            patience_temp=0

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
            logger.scalar_summary('loss_valid', loss_vali,
                                  model.epoch_ref)
            logger.scalar_summary('loss_train', loss.item(),  model.epoch_ref)

            torch.cuda.empty_cache()

        if early_stopping.early_stop:
            print("Early stopping")
            break


    if n_epochs>0:
        model.load_state_dict(torch.load(os.path.join("saved_models",name_file+'.pt')))
        torch.save(model.state_dict(), os.path.join( "saved_models",name_file+".txt"))

    print("epoch",model.epoch_ref)
    if test_on != [""]:
        for speaker in test_on:
            loss_test = 0
          #  for ite_valid in range(n_iteration_test):
            files_for_test = load_filenames([speaker], N_test, part=["test"])
            x, y = load_data(files_for_test,filtered=data_filtered)
            y = [y[i][:, :output_dim] for i in range(len(y))]
            print("evaluation on speaker {}".format(speaker))
            speaker_2 = speaker

            if speaker in ["F1", "M1", "F5"]:
                speaker_2 = "usc_timit_" + speaker

            std_speaker=  np.load(os.path.join(root_folder, "Traitement", "norm_values","std_ema_" + speaker_2 + ".npy"))
            std_speaker=std_speaker[:output_dim]

            model.evaluate_on_test(criterion=criterion,verbose=True, X_test=x, Y_test=y,
                                   to_plot=to_plot, std_ema=max(std_speaker), suffix=speaker)

            if data_filtered:
                print("----evaluation with data NON filtetered----")
                x_brut, y_brut = load_data(files_for_test, filtered=False)
                y_brut = [y_brut[i][:, :output_dim] for i in range(len(y_brut))]
                model.evaluate_on_test(criterion=criterion, verbose=True, X_test=x_brut, Y_test=y_brut,
                                   to_plot=False, std_ema=max(std_speaker), suffix=speaker)


if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Train and save a model.')
    parser.add_argument('train_on', metavar='train_on', type=list,
                        help='')
    parser.add_argument('test_on', metavar='test_on', type=list,
                        help='')
    parser.add_argument('n_epochs', metavar='n_epochs', type=int,
                        help='max number of epochs to train the model')
#    parser.add_argument('speaker', metavar='speaker', type=str,
 #                       help='corpus on which train model : fsew or msak or MNGU0 or both')
    parser.add_argument('delta_test', metavar='delta_test', type=int,
                        help='interval between two validation evaluation')
    parser.add_argument('patience', metavar='patience', type=int,
                        help='number of iterations in a row with decreasing validation score before stopping the train ')
    parser.add_argument('lr', metavar='lr', type=str,
                        help='learning rate of Adam optimizer ')

    parser.add_argument('output_dim', metavar='output_dim', type=int,
                        help='simple  : 12, +lipaperture : 13, +velu : 15 -attention il faut avoir appris sur mngu0')

    parser.add_argument('data_filtered', metavar='data_filtered', type=bool,
                        help='si true apprend sur les données ema lissées')

    parser.add_argument('modele_filtered', metavar='modele_filtered', type=bool,
                        help='si true apprend sur les données ema lissées')

    # parser.add_argument('norma', metavar='norma', type=bool,
    #                    help='')

    parser.add_argument('to_plot', metavar='to_plot', type=bool,         help='si true plot les resultats sur le test')
    parser.add_argument('loss', metavar='loss', type=str,
                        help='rmse ou pearson')

    args = parser.parse_args()
    train_on =  sys.argv[1]
    test_on = sys.argv[2]
    n_epochs = int( sys.argv[3] )
    delta_test = int(sys.argv[4])
    patience = int(sys.argv[5])
    lr = float(sys.argv[6])
    output_dim = int(sys.argv[7])
    data_filtered = sys.argv[8].lower() == 'true'
    modele_filtered = sys.argv[9].lower() == 'true'

   # norma = bool(sys.argv[8])
    to_plot = sys.argv[10].lower()=="true"
    loss = sys.argv[11]

    train_model(train_on = train_on,test_on = test_on ,n_epochs=n_epochs,delta_test=delta_test,patience=patience,
                lr = lr,output_dim=output_dim,data_filtered=data_filtered,modele_filtered=   modele_filtered,to_plot=to_plot,loss=loss) #,norma=norma)