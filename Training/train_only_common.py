#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    Created august 2019
    by Maud Parrot
    Script to train the model for the articulatory inversion.
    Some parameters concern the model itself, others concern the data used.
    Creates a model with the class myac2artmodel for the asked parameters.
    Learn category by category (in a category  all the speakers have the same arti traj available), the gradients are put
    at 0 for the unavailable arti traj so that it learns only on correct data.
    The model stops training by earlystopping if the validation score is several time consecutively increasing
    The weights of the model are saved in Training/saved_models/name_file.txt, with name file containing the info
    about the training/testing set [and not about the model parameters].

    The results of the model are evaluated on the test set, and are the averaged rmse and pearson per articulator.
    Those results are added as 2 new lines in the file "model_results.csv" , with 1 column being the name of the model
    and the last column the number of epochs [future work : add 1 columns per argument to store ALL the info about
    the model]


"""
import os, sys, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)


import os
"""
ncpu="10" # number of cpu available
os.environ["OMP_NUM_THREADS"] = ncpu  # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = ncpu  # export OPENBLAS_NUM_THREADS=4
os.environ["MKL_NUM_THREADS"] = ncpu  # export MKL_NUM_THREADS=4
os.environ["VECLIB_MAXIMUM_THREADS"] = ncpu  # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = ncpu  # export NUMEXPR_NUM_THREADS=4"""
import numpy as np
import argparse
from Training.model import my_ac2art_model
import torch
import os
import csv
from Training.pytorchtools import EarlyStopping
import random
from Training.tools_learning import which_speakers_to_train_on, give_me_train_valid_test_filenames_no_cat, \
    criterion_both, load_np_ema_and_mfcc, plot_filtre, give_me_common_articulators, get_right_indexes
import json

root_folder = os.path.dirname(os.getcwd())

def train_model_arti_common(test_on, n_epochs, loss_train, patience, corpus_to_train_on, batch_norma, filter_type,
                to_plot, lr, delta_valid, delta_test, config, speakers_to_train_on = "", speakers_to_valid_on = ""):
    """
    :param test_on: (str) one speaker's name we want to test on, the speakers and the corpus the come frome can be seen in
    "fonction_utiles.py", in the function "get_speakers_per_corpus'.

    :param n_epochs: (int)  max number of epochs for the training. We use an early stopping criterion to stop the training,
    so usually we dont go through the n_epochs and the early stopping happends before the 30th epoch (1 epoch is when
    have trained over ALL the data in the training set)

    :param loss_train: (int) alpha in the combined loss . can be anything between 0 and 100.
    the loss is the combinated loss alpha*rmse/1000+(1-alpha)*pearson.

    :param patience: (int) the number successive epochs with a validation loss increasing before stopping the training.
    We usually set it to 5. The more data we have, the smaller it can be (i think)

    :param select_arti: (bool) always true, either to use the trick to only train on available articulatory trajectories,
    fixing the predicted trajectory (to zero) and then the gradient will be 0.

    :param corpus_to_train_on: (list) list of the corpuses to train on. Usually at least the corpus the testspeaker comes from.
    (the testspeaker will be by default removed from the training speakers).

    :param batch_norma: (bool) whether or not add batch norm layer after the lstm layers (maybe better to add them after the
    feedforward layers? )

    :param filter_type: (int) either 0 1 or 2. 0 the filter is outside of the network, 1 it is inside and the weight are fixed
    during the training, 2 the weights get adjusted during the training

    :param to_plot: (bool) if true the trajectories of one random test sentence are saved in "images_predictions"

    :param lr: initial learning rate, usually 0.001

    :param delta_test: frequency of validation evaluation, 1 seems good

    :param config : either "spe" "dep", or "indep" or "train_indep", for specific (train only on test sp), dependant (train on test sp
    and others), or independant, train only on other speakers, and training independant for training on a certain list, validation
    on another an d test on another speaker

    :return: [rmse, pearson] . rmse the is the list of the 18 rmse (1 per articulator), same for pearson.
    """

    corpus_to_train_on = corpus_to_train_on[1:-1].split(",")
    speakers_to_train_on = speakers_to_train_on[1:-1].replace("'", "").replace('"', '').replace(' ', '').split(",")
    speakers_to_valid_on = speakers_to_valid_on[1:-1].replace("'", "").replace('"', '').replace(' ', '').split(",")
    if speakers_to_train_on == [""] or speakers_to_train_on == []:
        train_on = which_speakers_to_train_on(corpus_to_train_on, test_on, config)
    else:
        train_on = speakers_to_train_on

    if speakers_to_valid_on == [""] or speakers_to_valid_on == []:
        valid_on = []
    else:

        valid_on = speakers_to_valid_on
        print("You choose to valid on", speakers_to_valid_on)

    arti_common = give_me_common_articulators([test_on] + train_on + valid_on)
    print(arti_common)
    name_corpus_concat = ""
    if config != "spec" : # if spec DOESNT train on other speakers
        for corpus in corpus_to_train_on:
            name_corpus_concat = name_corpus_concat + corpus + "_"

    name_file = 'only_arti_common_' + test_on+"_"+config+"_train_"+'_'.join(train_on)+'_valid_'+ '_'.join(valid_on) + "_loss_"+str(loss_train)+"_filter_"+\
                str(filter_type)+"_bn_"+str(batch_norma)

    f_loss_train = open('training_loss'+ name_file +'.csv', 'w')
    f_loss_valid = open('valid_loss'+ name_file +'.csv', 'w')
    f_loss_test = open('test_loss' + name_file + '.csv', 'w')
    f_loss_valid.write('epoch,training_loss,pearson,rmse\n')
    f_loss_train.write('epoch,training_loss,pearson,rmse\n')
    f_loss_test.write('epoch,training_loss,pearson,rmse\n')
    if not os.path.exists("saved_models"):
        os.mkdir("saved_models")

    previous_models = os.listdir("saved_models")
    previous_models_2 = [x[:len(name_file)] for x in previous_models if x.endswith(".txt")]
    n_previous_same = previous_models_2.count(name_file)  # how many times our model was trained

    if n_previous_same > 0:
        print("this models has alread be trained {} times".format(n_previous_same))
    else :
        print("first time for this model")
    name_file = name_file + "_" + str(n_previous_same)  # each model trained only once ,
    # this script doesnt continue a previous training if it was ended ie if there is a .txt
    print("going to train the model with name",name_file)

    cuda_avail = torch.cuda.is_available()
    print(" cuda ?", cuda_avail)
    if cuda_avail:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    hidden_dim = 300
    input_dim = 429
    batch_size = 10
    output_dim = len(arti_common)
    early_stopping = EarlyStopping(name_file, patience=patience, verbose=True)
    model = my_ac2art_model(hidden_dim=hidden_dim, input_dim=input_dim, name_file=name_file, output_dim=output_dim,
                            batch_size=batch_size, cuda_avail=cuda_avail,
                            filter_type=filter_type, batch_norma=batch_norma)
    model = model.double()
    file_weights = os.path.join("saved_models", name_file +".pt")
    if cuda_avail:
        model = model.to(device=device)
    load_old_model = True
    if load_old_model:
        if os.path.exists(file_weights):
            print("previous model did not finish learning")
            loaded_state = torch.load(file_weights,map_location=device)
            model.load_state_dict(loaded_state)
            model_dict = model.state_dict()
            loaded_state = {k: v for k, v in loaded_state.items() if
                            k in model_dict}  # only layers param that are in our current model
            loaded_state = {k: v for k, v in loaded_state.items() if
                            loaded_state[k].shape == model_dict[k].shape}  # only if layers have correct shapes
            model_dict.update(loaded_state)
            model.load_state_dict(model_dict)



    files_for_train, files_for_valid, files_for_test = give_me_train_valid_test_filenames_no_cat(train_on,test_on,config, valid_on=valid_on)
    print('train on', len(files_for_train), 'valid on', len(files_for_valid), 'test on', len(files_for_test))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    plot_filtre_chaque_epochs = False

    for epoch in range(n_epochs):
        weights = model.lowpass.weight.data[0, 0, :].cpu()
        if plot_filtre_chaque_epochs :
            plot_filtre(weights)
        n_this_epoch = 0
        random.shuffle(files_for_train)
        loss_train_this_epoch = 0
        loss_pearson = 0
        loss_rmse = 0
        nb_batch = len(files_for_train)/ batch_size
        for i in range(int(nb_batch)):

            n_this_epoch+=1

            x, y = load_np_ema_and_mfcc(files_for_train[i*batch_size:(i+1)*batch_size])
            model.output_dim = 18
            x, y = model.prepare_batch(x, y)

            model.output_dim = len(arti_common)
            y = get_right_indexes(y,arti_common)
            if cuda_avail:
                x, y = x.to(device=model.device), torch.from_numpy(y).double().to(device=model.device)
            y_pred = model(x).double()
            if cuda_avail:
                y_pred = y_pred.to(device=device)
            y = y.double ()
            optimizer.zero_grad()

            loss = criterion_both(y, y_pred,alpha=loss_train, cuda_avail = cuda_avail, device=device)
            loss.backward()
            optimizer.step()

            # computation to have evolution of the losses
            loss_2 = criterion_both(y, y_pred, alpha=100, cuda_avail=cuda_avail, device=device)
            loss_pearson += loss_2.item()
            loss_3 = criterion_both(y, y_pred, alpha=0, cuda_avail=cuda_avail, device=device)
            loss_rmse += loss_3.item()
            torch.cuda.empty_cache()
            loss_train_this_epoch += loss.item()

        torch.cuda.empty_cache()

        loss_train_this_epoch = loss_train_this_epoch/n_this_epoch
        print("Training loss for epoch", epoch, ': ', loss_train_this_epoch)
        f_loss_train.write(str(epoch) + ',' + str(loss_train_this_epoch) + ',' + str(loss_pearson/n_this_epoch/1000./batch_size/len(arti_common)*(-1.)) + ',' + str(loss_rmse/n_this_epoch/batch_size/len(arti_common)) + '\n')
        real_batch_size = batch_size
        batch_size = 1
        if epoch%delta_valid == 0:  #toutes les delta_test epochs on évalue le modèle sur validation et on sauvegarde le modele si le score est meilleur
            loss_vali = 0
            n_valid = 0
            loss_pearson = 0
            loss_rmse = 0
            nb_batch = len(files_for_valid) / batch_size
            for i in range(int(nb_batch)):
                n_valid +=1
                x, y = load_np_ema_and_mfcc(files_for_valid[i * batch_size:(i + 1) * batch_size])
                model.output_dim = 18
                x, y = model.prepare_batch(x, y)
                model.output_dim = len(arti_common)
                y = get_right_indexes(y, arti_common)
                if cuda_avail:
                    x, y = x.to(device=model.device), torch.from_numpy(y).double().to(
                        device=model.device)
                y_pred = model(x).double()
                torch.cuda.empty_cache()
                if cuda_avail:
                    y_pred = y_pred.to(device=device)
                y = y.double()  # (Batchsize, maxL, art_common_nb)
                loss_courant = criterion_both(y, y_pred, loss_train, cuda_avail = cuda_avail, device=device)
                loss_vali += loss_courant.item()

                # to follow both losses
                loss_2 = criterion_both(y, y_pred, alpha=100, cuda_avail=cuda_avail, device=device)
                loss_pearson += loss_2.item()
                loss_3 = criterion_both(y, y_pred, alpha=0, cuda_avail=cuda_avail, device=device)
                loss_rmse += loss_3.item()

            loss_vali  = loss_vali/n_valid
            f_loss_valid.write(str(epoch) + ',' + str(loss_vali) + ',' +  str(loss_pearson/n_valid/1000./batch_size/len(arti_common)*(-1.)) +
                               ',' + str(loss_rmse/n_valid/batch_size/len(arti_common)) + '\n')
        # test on the test files
        if epoch % delta_test == 0:  # toutes les delta_test epochs on évalue le modèle sur validation et on sauvegarde le modele si le score est meilleur
            loss_test = 0
            n_test = 0
            loss_pearson = 0
            loss_rmse = 0
            nb_batch = len(files_for_test) / batch_size
            for i in range(int(nb_batch)):
                n_test += 1
                x, y = load_np_ema_and_mfcc(files_for_test[i * batch_size:(i + 1) * batch_size])
                model.output_dim = 18
                x, y = model.prepare_batch(x, y)
                model.output_dim = len(arti_common)
                y = get_right_indexes(y, arti_common)
                if cuda_avail:
                    x, y = x.to(device=model.device), torch.from_numpy(y).double().to(
                        device=model.device)
                y_pred = model(x).double()
                torch.cuda.empty_cache()
                if cuda_avail:
                    y_pred = y_pred.to(device=device)
                y = y.double()  # (Batchsize, maxL, art_common_nb)
                loss_courant = criterion_both(y, y_pred, loss_train, cuda_avail=cuda_avail, device=device)
                loss_test += loss_courant.item()

                # to follow both losses
                loss_2 = criterion_both(y, y_pred, alpha=100, cuda_avail=cuda_avail, device=device)
                loss_pearson += loss_2.item()
                loss_3 = criterion_both(y, y_pred, alpha=0, cuda_avail=cuda_avail, device=device)
                loss_rmse += loss_3.item()

            loss_test = loss_test / n_test
            f_loss_test.write(str(epoch) + ',' + str(loss_test) + ',' + str(
                loss_pearson / n_test / 1000. / batch_size / len(arti_common) * (-1.)) +
                               ',' + str(loss_rmse / n_test / batch_size / len(arti_common)) + '\n')
        batch_size = real_batch_size
        torch.cuda.empty_cache()
        model.all_validation_loss.append(loss_vali)
        model.all_training_loss.append(loss_train_this_epoch)
        early_stopping(loss_vali, model)
        if early_stopping.early_stop:
            print("Early stopping, n epochs : ", model.epoch_ref + epoch)
            break

        if epoch > 0:  # on divise le learning rate par deux dès qu'on surapprend un peu par rapport au validation set
            if loss_vali > model.all_validation_loss[-1]:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = param_group['lr'] / 2.
                    (param_group["lr"])


    if n_epochs > 0:
        model.epoch_ref = model.epoch_ref + epoch  # voir si ca marche vrmt pour les rares cas ou on continue un training
        model.load_state_dict(torch.load(os.path.join("saved_models",name_file+'.pt')))
        torch.save(model.state_dict(), os.path.join( "saved_models",name_file+".txt")) #lorsque .txt ==> training terminé !
    random.shuffle(files_for_test)
    x, y = load_np_ema_and_mfcc(files_for_test)
    #y = get_right_indexes(y, arti_common)
    print("evaluation on speaker {}".format(test_on))
    std_speaker = np.load(os.path.join(root_folder,"Preprocessing","norm_values","std_ema_"+test_on+".npy"))
    arti_to_consider = [1 for i in range(len(arti_common))]
    rmse_per_arti_mean, pearson_per_arti_mean = model.evaluate_on_test(x, y, std_speaker = std_speaker, to_plot=to_plot
                                                                       , to_consider = arti_to_consider, index_common=arti_common)


    """  RESULTS ON VALIDATION SET """

    pearson_valid = np.zeros((1,output_dim))
    nb_batch = len(files_for_valid) / batch_size
    for i in range(int(nb_batch)):
        x, y = load_np_ema_and_mfcc(files_for_valid[i * batch_size:(i + 1) * batch_size])
        #y = get_right_indexes(y, arti_common)
        rien, pearson_valid_temp = model.evaluate_on_test(x,y,std_speaker=1, to_plot=to_plot,
                                                             to_consider=arti_to_consider,verbose=False, index_common=arti_common , no_std = True)
        pearson_valid_temp = np.reshape(np.array(pearson_valid_temp),(1,output_dim))
        pearson_valid = np.concatenate((pearson_valid,pearson_valid_temp),axis=0)
    pearson_valid = pearson_valid[1:,:]
    pearson_valid[np.isnan(pearson_valid)] = 0
    pearson_valid = np.mean(pearson_valid,axis=0)
    print("on validation set :mean :\n",pearson_valid)
    print("training done for : ",name_file)

    articulators = ['tt_x', 'tt_y', 'td_x', 'td_y', 'tb_x', 'tb_y', 'li_x', 'li_y',
                    'ul_x', 'ul_y', 'll_x', 'll_y', 'la', 'lp', 'ttcl', 'tbcl', 'v_x', 'v_y']
    articulators = [art for art in articulators if art in arti_common]
    if not os.path.exists('model_results.csv'):
        with open('model_results.csv', 'a',newline = "") as f:
            writer = csv.writer(f)
            header = ["name file", "test on", "configuration", "train on (if not spec)", "loss",
                      "n_epochs", "evaluation with...", "average"] + articulators
            writer.writerow(header)

    # write result in csv
    with open('model_results.csv', 'a',newline = "") as f:
        writer = csv.writer(f)
        row_details = ['only_common', name_file,test_on,config,name_corpus_concat,loss_train,model.epoch_ref]
        row_rmse = row_details + ["rmse_on_test", np.mean(rmse_per_arti_mean[rmse_per_arti_mean!=0])] +\
                   rmse_per_arti_mean.tolist()

        row_pearson = row_details + ["pearson_on_test", np.mean(pearson_per_arti_mean[pearson_per_arti_mean!=0])]+\
                      pearson_per_arti_mean.tolist()

        row_pearson_val = row_details + ["pearson_on_valid", np.mean(pearson_valid[pearson_valid !=0])] + \
                      pearson_valid.tolist()

        writer.writerow(row_rmse)
        writer.writerow(row_pearson)
        writer.writerow(row_pearson_val)

    weight_apres = model.lowpass.weight.data[0, 0, :].cpu()
    plot_allure_filtre = True
    if plot_allure_filtre:
        plot_filtre(weight_apres)

    return rmse_per_arti_mean, pearson_per_arti_mean


if __name__=='__main__':

    parser = argparse.ArgumentParser(description='Train and save a model.')

    parser.add_argument('test_on', type=str,
                        help='the speaker we want to test on')

    parser.add_argument('--speakers_to_train', type=str, default=[],
                        help='specific speakers to train on')

    parser.add_argument('--speakers_to_valid', type=str, default=[],
                        help='specific speakers to valid on')

    parser.add_argument('--n_epochs', type=int, default=50,
                        help='max number of epochs to train the model')

    parser.add_argument("--loss_train",type = int, default=90,
                        help = "from 0 to 100, coeff of pearson is the combined loss")

    parser.add_argument("--patience",type=int, default=5,
                        help = "patience before early topping")

    parser.add_argument('corpus_to_train_on', type=str,
                        help='list of the corpus we want to train on ')

    parser.add_argument('--batch_norma', type=bool, default= False,
                        help='whether to add batch norma after lstm layyers')

    parser.add_argument('--filter_type', type=str, default="fix",
                        help='"out" filter outside of nn, "fix" filter with fixed weights, "unfix" filter with adaptable weights')

    parser.add_argument('--to_plot', type=bool, default= False,
                        help='whether to save one graph of prediction & target of the test ')

    parser.add_argument('--lr', type = float, default = 0.001,
                        help='learning rate of Adam optimizer ')

    parser.add_argument('--delta_valid', type=int, default=1,
                        help='how often evaluate the validation set')

    parser.add_argument('--delta_test', type=int, default=5,
                        help='how often evaluate the test set')

    parser.add_argument('config', type=str,
                        help='spec or dep or indep that stands for speaker specific/dependant/independant')

    args = parser.parse_args()

    train_model_arti_common(test_on=args.test_on, n_epochs=args.n_epochs, loss_train=args.loss_train,
                patience=args.patience,  corpus_to_train_on=args.corpus_to_train_on,
                batch_norma=args.batch_norma, filter_type=args.filter_type, to_plot=args.to_plot,
                lr=args.lr, delta_valid=args.delta_valid, delta_test=args.delta_test, config=args.config,
                            speakers_to_train_on=args.speakers_to_train, speakers_to_valid_on=args.speakers_to_valid)