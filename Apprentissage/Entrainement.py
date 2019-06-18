### ETUDIER ARTI 6

from class_network import my_bilstm
import sys
import torch
import os
import sys
from sklearn.model_selection import train_test_split

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


def train_model(train_on ,test_on ,n_epochs ,delta_test ,patience ,lr=0.09, output_dim=13): #,norma=True):
    cuda_avail = torch.cuda.is_available()
    print(" cuda ?", cuda_avail)

    train_on = str(train_on[1:-1])
    test_on = str(test_on[1:-1])
    train_on = train_on.split(",")
    test_on = test_on.split(",")
    name_file = "train_" + "_".join(train_on) + "_test_" + "_".join(test_on)
    folder_weights = os.path.join("saved_models", name_file)

    X_train, X_test, Y_train, Y_test = [], [], [], []
    speakers_in_lists = list(set(train_on+test_on) & set(["msak0","fsew0","MNGU0"]))
    print("list ",speakers_in_lists)
    for speaker in speakers_in_lists:
        print("SPEAKER ",speaker)
        X_train_sp = np.load(os.path.join(fileset_path, "X_train_" + speaker + ".npy"),allow_pickle=True)
        Y_train_sp = np.load(os.path.join(fileset_path, "Y_train_" + speaker + ".npy"),allow_pickle=True)
        X_test_sp = np.load(os.path.join(fileset_path, "X_test_" + speaker + ".npy"),allow_pickle=True)
        Y_test_sp = np.load(os.path.join(fileset_path, "Y_test_" + speaker + ".npy"),allow_pickle=True)

        Y_train_sp = np.array([Y_train_sp[i][:, :output_dim] for i in range(len(Y_train_sp))])
        Y_test_sp = np.array([Y_test_sp[i][:, :output_dim] for i in range(len(Y_test_sp))])

        if speaker in train_on:

            X_train.extend(X_train_sp)
            Y_train.extend(Y_train_sp)

            print(False in [(X_train[i].shape[0] == Y_train[i].shape[0]) for i in range(len(Y_train))])
            print(False in [(X_test[i].shape[0] == Y_test[i].shape[0]) for i in range(len(Y_test))])
            if speaker not in test_on:

                X_train.extend(X_test_sp)
                Y_train.extend(Y_test_sp)



        if speaker in test_on:

            X_test.extend(X_test_sp)
            Y_test.extend(Y_test_sp)
            if speaker not in train_on:

                X_test.extend(X_train_sp)
                Y_test.extend(Y_train_sp)


    pourcent_valid = 0.05
    hidden_dim = 300
    input_dim = 429
    beta_param = [0.9 , 0.999]
    batch_size = 10
    X_train, X_valid, Y_train, Y_valid = train_test_split(X_train, Y_train, test_size=pourcent_valid, random_state=1)
    X_train, X_valid, Y_train, Y_valid = np.array(X_train),np.array(X_valid),np.array(Y_train),np.array(Y_valid),

    print("X_valid", len(X_valid))
    print("X_train", len(X_train))
    print("X_test", len(X_test))

    early_stopping = EarlyStopping(name_file,patience=patience, verbose=True)

    model = my_bilstm(hidden_dim=hidden_dim,input_dim=input_dim,name_file =name_file, output_dim=output_dim,batch_size=batch_size)
    model = model.double()
    #print("wweights layer",model.first_layer.weight)
    #folder_weights_init =  os.path.join("saved_models", "train_fsew0_test_msak0","train_fsew0_test_msak0.txt")

   # try :
    if not cuda_avail:
        device = torch.device('cpu')
        loaded_state = torch.load(os.path.join(folder_weights, name_file +".txt"), map_location=device)

    else :
        loaded_state = torch.load(os.path.join(folder_weights, name_file +".txt"))
    model_dict = model.state_dict()

    loaded_state = {k: v for k, v in loaded_state.items() if k in model_dict} #only layers param that are in our current model
    print("before ",len(loaded_state))
    loaded_state= {k:v for k,v in loaded_state.items() if loaded_state[k].shape==model_dict[k].shape } #only if layers have correct shapes
    print("after",len(loaded_state))

    model_dict.update(loaded_state)
    model.load_state_dict(model_dict)
    model.all_training_loss=[]
# except :
    #   print('first time, intialisation with Xavier weight...')
       #torch.nn.init.xavier_uniform(my_bilstm.lstm_layer.weight)

  #  print("wweights layer AFTER", model.first_layer.weight)
    print("train size : ", len(X_train))
    print("test size :", len(X_test))
    previous_epoch = 0
    try :
        previous_losses = np.load(os.path.join(folder_weights, "all_losses.npy"))
        a, b, c = previous_losses[0, :], previous_losses[1, :], previous_losses[2, :]
        if len(a) == len(b) == len(c):
            model.all_training_loss = list(a)
            model.all_validation_loss_loss = list(b)
            model.all_test_loss = list(c)
            previous_epoch = len(a)
    except :
        print("seems first time no previous loss")

    print("previous epoch  :", previous_epoch)
    if cuda_avail:
        model = model.cuda()
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = "1"



    def criterion_old(y,y_pred):
        y_1 = y - torch.mean(y,dim=1,keepdim=True)
        y_pred_1 = y_pred - torch.mean(y_pred,dim=1,keepdim=True)
        nume=  torch.sum(y_1 * y_pred_1,dim=1,keepdim=True)
        deno =  torch.sqrt(torch.sum(y_1 ** 2,dim=1,keepdim=True)) * torch.sqrt(torch.sum(y_pred_1 ** 2,dim=1,keepdim=True))# use Pearson correlation
        loss = nume/deno
        loss[torch.isnan(loss)] = 1
        loss = torch.sum(loss)
        return -loss

    criterion = torch.nn.MSELoss(reduction='sum')



    optimizer = torch.optim.Adam(model.parameters(), lr=lr ) #, betas = beta_param)
    plt.ioff()
    print("number of epochs : ", n_epochs)
    n_iteration = int(len(X_train)/batch_size)

    for epoch in range(n_epochs):
        for ite in range(n_iteration):
            if ite % 10 == 0:
                print("{} out of {}".format(ite, n_iteration))
            indices = np.random.choice(len(X_train), batch_size, replace=False)
            x, y = X_train[indices], Y_train[indices]
            print("0","nan" in x)
            x, y = model.prepare_batch(x, y, cuda_avail=cuda_avail)
            print("1",torch.isnan(x).any())
            y_pred= model(x).double()
            print("2",torch.isnan(y_pred).any())
            y = y.double()
            optimizer.zero_grad()
            loss = criterion(y,y_pred)
            print("loss : ",loss)

            loss.backward()
            optimizer.step()
            model.all_training_loss.append(loss.item())
            torch.cuda.empty_cache()

        if epoch%delta_test ==0:  #toutes les delta_test epochs on évalue le modèle sur validation et on sauvegarde le modele si le score est meilleur
            loss_vali = model.evaluate(X_valid,Y_valid,criterion,cuda_avail=cuda_avail)
            model.all_validation_loss.append(loss_vali)
            model.all_validation_loss += [model.all_validation_loss[-1]] * (epoch+previous_epoch - len(model.all_validation_loss))
            loss_test=0
           # if test_on != [""]:
            #    loss_test = model.evaluate_on_test(criterion,X_test = X_test,Y_test = Y_test,to_plot=False,cuda_avail=cuda_avail)
            model.all_test_loss.append(loss_test)
            model.all_test_loss += [model.all_test_loss[-1]] * (epoch+previous_epoch - len(model.all_test_loss))
            print("\n ---------- epoch" + str(epoch) + " ---------")
            early_stopping.epoch = previous_epoch+epoch
            early_stopping(loss_vali, model)
            print("train loss ", loss.item())
            print("valid loss ", loss_vali)
            print("test loss ", loss_test)


        if early_stopping.early_stop:
            print("Early stopping")
            break

    if n_epochs>0:
        model.load_state_dict(torch.load(os.path.join(folder_weights,name_file+'.pt')))
        torch.save(model.state_dict(), os.path.join( folder_weights,name_file+".txt"))

    if test_on != [""]:
        for speaker in test_on:
            print("evaluation on speaker {}".format(speaker))
            X_test_sp = np.load(os.path.join(fileset_path, "X_test_" + speaker + ".npy"),allow_pickle=True)
            Y_test_sp = np.load(os.path.join(fileset_path, "Y_test_" + speaker + ".npy"),allow_pickle=True)
            multi_loss_test=  np.load(os.path.join(root_folder, "Traitement", "std_ema_" + speaker + ".npy"))
            multi_loss_test=multi_loss_test[:output_dim]
            Y_test_sp = np.array([Y_test_sp[i][:, :output_dim] for i in range(len(Y_test_sp))])

            model.evaluate_on_test(criterion=criterion,verbose=True, X_test=X_test_sp, Y_test=Y_test_sp,
                                   to_plot=False, std_ema=multi_loss_test, suffix=speaker, cuda_avail=cuda_avail)

    length_expected = len(model.all_training_loss)
    print("lenght exp", length_expected)
    try :
        model.all_validation_loss += [model.all_validation_loss[-1]] * (length_expected - len(model.all_validation_loss))
        model.all_training_loss = np.array(model.all_training_loss).reshape(1,length_expected)
        model.all_validation_loss = np.array(model.all_validation_loss).reshape(1,length_expected)
        model.all_test_loss += [model.all_test_loss[-1]] * (length_expected - len(model.all_test_loss))
        model.all_test_loss = np.array(model.all_test_loss).reshape((1, length_expected))
    except :
        print("not any train")
    all_losses = np.concatenate(
         ( np.array(model.all_training_loss),
        np.array(model.all_validation_loss),
      np.array(model.all_test_loss) )
          ,axis=0 )


    np.save(os.path.join(folder_weights,"all_losses.npy"),all_losses)

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

   # parser.add_argument('norma', metavar='norma', type=bool,
    #                    help='')

  #  parser.add_argument('to_plot', metavar='to_plot', type=bool,
   #                     help='')

    args = parser.parse_args()

    train_on =  sys.argv[1]
    test_on = sys.argv[2]
    n_epochs = int( sys.argv[3] )
    delta_test = int(sys.argv[4])
    patience = int(sys.argv[5])
    lr = float(sys.argv[6])
    output_dim = int(sys.argv[7])
   # norma = bool(sys.argv[8])
   # to_plot = bool(sys.argv[9])

    train_model(train_on = train_on,test_on = test_on ,n_epochs=n_epochs,delta_test=delta_test,patience=patience,
                lr = lr,output_dim=output_dim) #,norma=norma)