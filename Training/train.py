#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    Created august 2019
    by Maud Parrot

"""
import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

ncpu="10"
import os
os.environ["OMP_NUM_THREADS"] = ncpu # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = ncpu # export OPENBLAS_NUM_THREADS=4
os.environ["MKL_NUM_THREADS"] = ncpu # export MKL_NUM_THREADS=4
os.environ["VECLIB_MAXIMUM_THREADS"] = ncpu # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = ncpu # export NUMEXPR_NUM_THREADS=4
import numpy as np
import argparse
import gc
import psutil
from Training.model import my_ac2art_model
import torch
import os
import csv
import sys
from Training.tools_learning import load_filenames_deter, load_data
from Training.pytorchtools import EarlyStopping
import random
from scipy import signal
import matplotlib.pyplot as plt
from Preprocessing.tools_preprocessing import get_speakers_per_corpus
import json
from Training.logger import Logger

root_folder = os.path.dirname(os.getcwd())


def memReport(all=False):
    """
    :param all: show size of each obj
    use if memory errors
    """
    nb_object = 0
    for obj in gc.get_objects():
        if torch.is_tensor(obj):
            if all:
                print(type(obj), obj.size())
            nb_object += 1
    print('nb objects tensor', nb_object)


def cpuStats():
    """
    use in case of memory errors
    """
    print(sys.version)
    print(psutil.cpu_percent())
    print(psutil.virtual_memory())  # physical memory usage
    pid = os.getpid()
    py = psutil.Process(pid)
    memoryUse = py.memory_info()[0] / 2. ** 30  # memory use in GB...I think
    print('memory GB:', memoryUse)


def criterion_pearson(y, y_pred, cuda_avail, device):
    """
    :param y: nparray (B,K,18) target trajectories of the batch (size B) , padded (K = maxlenght)
    :param y_pred: nparray (B,K,18) predicted trajectories of the batch (size B), padded (K = maxlenght
    :param cuda_avail: bool whether gpu is available
    :param device: the device
    :return: loss function for this prediction for loss = pearson correlation
    for each pair of trajectories (target & predicted) we calculate the pearson correlation between the two
    we sum all the pearson correlation to obtain the loss function
    // Idea : integrate the range of the traj here, making the loss for each sentence as the weighted average of the
    losses with weight proportional to the range of the traj (?)
    """
    y_1 = y - torch.mean(y, dim=1, keepdim=True)
    y_pred_1 = y_pred - torch.mean(y_pred,dim=1, keepdim=True)
    nume = torch.sum(y_1 * y_pred_1, dim=1, keepdim=True)  # (B,1,18)
    deno = torch.sqrt(torch.sum(y_1 ** 2, dim=1, keepdim=True)) * \
        torch.sqrt(torch.sum(y_pred_1 ** 2, dim=1, keepdim=True))  # (B,1,18)

    minim = torch.tensor(0.01,dtype=torch.float64)  # avoid division by 0
    if cuda_avail:
        minim = minim.to(device=device)
        deno = deno.to(device=device)
        nume = nume.to(device=device)
    deno = torch.max(deno, minim)  # replace 0 by minimum
    my_loss = torch.div(nume, deno)  # (B,1,18)
    my_loss = torch.sum(my_loss)
    return -my_loss


def criterion_both(L, cuda_avail, device):
    """
    :param L: parameter in the combined loss of rmse and pearson (loss = (1-L)*rmse + L*pearson ) between 0 and 100
    :param cuda_avail: bool if gpu is available
    :param device: device
    :return: the function that calculates the combined loss of 1 prediction. This function will be used as a criterion
    """
    L = L/100

    def criterion_both_lbd(my_y, my_ypred):
        """
        :param my_y: target
        :param my_ypred: perdiction
        :return: the loss for the combined loss
        """
        a = L * criterion_pearson(my_y, my_ypred, cuda_avail, device)
        b = (1 - L) * torch.nn.MSELoss(reduction='sum')(my_y, my_ypred) / 1000
        new_loss = a + b
        return new_loss
    return criterion_both_lbd


def plot_filtre(weights):
    """
    :param weights: weights of the low pass filter
    plot the impulse response of the filter, with gain in GB
    """
    print("GAIN", sum(weights))
    freqs, h = signal.freqz(weights)
    freqs = freqs * 100 / (2 * np.pi)  # freq in hz
    plt.plot(freqs, 20 * np.log10(abs(h)), 'r')
    plt.title("Allure filtre passe bas à la fin de l'Training pour filtre en dur")
    plt.ylabel('Amplitude [dB]')
    plt.xlabel("real frequency")
    plt.show()


def give_me_train_on(corpus_to_train_on, test_on, config):
    """
    :param corpus_to_train_on: list of all the corpus name to train on
    :param test_on: the speaker test name
    :param config:  either specific/dependant/independant
    :return: name_corpus_concat : concatenation of the corpus name, used only for the name of the modele
            train_on : list of the speakers to train_on (except the test speaker)
    """
    name_corpus_concat = ""

    if config == "spec":  # speaker specific
        train_on = [""]  # only train on the test speaker

    elif config in ["indep", "dep"]:  # train on other corpuses
        train_on = []
        corpus_to_train_on = corpus_to_train_on[1:-1].split(",")
        for corpus in corpus_to_train_on:
            sp = get_speakers_per_corpus(corpus)
            train_on = train_on + sp
            name_corpus_concat = name_corpus_concat + corpus + "_"
        if test_on in train_on:
            train_on.remove(test_on)

    return name_corpus_concat, train_on


def give_me_train_valid_test(train_on, test_on, config, batch_size):
    """
    :param train_on: list of corpus to train on
    :param test_on: the speaker test
    :param config: either spec/dep/indep
    :param batch_size
    :return: files_per_categ :  dictionnary where keys are the categories present in the training set. For each category
    we have a dictionnary with 2 keys (train, valid), and the values is a list of the namefiles for this categ and this
    part (train/valid)
            files_for_test : list of the files of the test set
    3 configurations that impacts the train/valid/test set (if we train a bit on test speaker, we have to be sure that
    the don't test on files that were in the train set)
    - spec : for speaker specific, learning and testing only on the speaker test
    - dep : for speaker dependant, learning on speakers in train_on and a part of the speaker test
    - indep : for speaker independant, learnong on other speakers.
    """
    if config == "spec":
        files_for_train = load_filenames_deter([test_on], part=["train"])
        files_for_valid = load_filenames_deter([test_on], part=["valid"])
        files_for_test = load_filenames_deter([test_on], part=["test"])

    elif config == "dep":
        files_for_train = load_filenames_deter(train_on, part=["train", "test"]) + \
                          load_filenames_deter([test_on], part=["train"])
        files_for_valid = load_filenames_deter(train_on, part=["valid"]) + \
                          load_filenames_deter([test_on], part=["valid"])
        files_for_test = load_filenames_deter([test_on], part=["test"])

    elif config == "indep":
        files_for_train = load_filenames_deter(train_on, part=["train", "test"])
        files_for_valid = load_filenames_deter(train_on, part=["valid"])
        files_for_test = load_filenames_deter([test_on], part=["train", "valid", "test"])

    with open('categ_of_speakers.json', 'r') as fp:
        categ_of_speakers = json.load(fp)  # dictionnary { categ : dict_2} where
                                            # dict_2 :{  speakers : [sp_1,..], arti  : [0,1,1...]  }
    files_per_categ = dict()

    for categ in categ_of_speakers.keys():
        sp_in_categ = categ_of_speakers[categ]["sp"]

        files_train_this_categ = [[f for f in files_for_train if sp.lower() in f.lower()]
                                  for sp in sp_in_categ]  # the speaker name is always in the namefile
        files_train_this_categ = [item for sublist in files_train_this_categ
                                  for item in sublist]  # flatten the list of list

        files_valid_this_categ = [[f for f in files_for_valid if sp.lower() in f.lower()] for sp in sp_in_categ]
        files_valid_this_categ = [item for sublist in files_valid_this_categ for item in sublist]

        if len(files_train_this_categ) > 0:  # meaning we have at least one file in this categ
            files_per_categ[categ] = dict()

            N_iter_categ = int(len(files_train_this_categ)/batch_size)+1
            n_a_ajouter = batch_size*N_iter_categ - len(files_train_this_categ)
            files_train_this_categ = files_train_this_categ +\
                                    files_train_this_categ[:n_a_ajouter]  #so that lenght is a multiple of batchsize
            random.shuffle(files_train_this_categ)
            files_per_categ[categ]["train"] = files_train_this_categ

            N_iter_categ = int(len( files_valid_this_categ) / batch_size) + 1
            n_a_ajouter = batch_size * N_iter_categ - len(files_valid_this_categ)
            files_valid_this_categ = files_valid_this_categ + files_valid_this_categ[:n_a_ajouter]
            random.shuffle(files_valid_this_categ)
            files_per_categ[categ]["valid"] = files_valid_this_categ

    return files_per_categ, files_for_test


def train_model(test_on, n_epochs, loss_train, patience, select_arti, corpus_to_train_on, batch_norma, filter_type,
                to_plot, lr, delta_test, config):
    """
    :param test_on: (str) one speaker's name we want to test on, the speakers and the corpus the come frome can be seen in
    "fonction_utiles.py", in the function "get_speakers_per_corpus'.

    :param n_epochs: (int)  max number of epochs for the training. We use an early stopping criterion to stop the training,
    so usually we dont go through the n_epochs and the early stopping happends before the 30th epoch (1 epoch is when
    have trained over ALL the data in the training set)

    :param loss_train: (str) either "rmse", "pearson", or "both_80" where 80 can be anything between 0 and 100. "both_alpha"
    is the combinated loss alpha*rmse/1000+(1-alpha)*pearson.
    Hence "rmse" and "both_0" are equivalent, same for "pearson" and "both_100" # (better to change it in the future)

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

    :param config : either "spe" "dep", or "indep", for specific (train only on test sp), dependant (train on test sp
    and others), or independant, train only on other speakers

    :return: [rmse, pearson] . rmse the is the list of the 18 rmse (1 per articulator), same for pearson.
    """

    name_corpus_concat, train_on = give_me_train_on(corpus_to_train_on, test_on,config)

    name_file = test_on+"_speaker_"+config+name_corpus_concat+"_loss_"+str(loss_train)+"_filter_"+str(filter_type)
    "_bn_"+str(batch_norma)

    previous_models = os.listdir("saved_models")
    previous_models_2 = [x[:len(name_file)] for x in previous_models if x.endswith(".txt")]
    n_previous_same = previous_models_2.count(name_file)  #how many times our model was trained

    if n_previous_same > 0:
        print("this models has alread be trained {} times".format(n_previous_same))
    else :
        print("first time for this model")
    name_file = name_file + "_" + str(n_previous_same)  # each model trained only once ,
    # this script doesnt continue a previous training if it was ended ie if there is a .txt
    print("going to train the model with name",name_file)
    logger = Logger('./logs')

    cuda_avail = torch.cuda.is_available()
    print(" cuda ?", cuda_avail)
    if cuda_avail:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    hidden_dim = 300
    input_dim = 429
    batch_size = 10
    output_dim = 18
    early_stopping = EarlyStopping(name_file, patience=patience, verbose=True)
    model = my_ac2art_modele(hidden_dim=hidden_dim, input_dim=input_dim, name_file=name_file, output_dim=output_dim,
                             batch_size=batch_size, cuda_avail=cuda_avail,
                             filter_type=filter_type, batch_norma=batch_norma)
    model = model.double()

    file_weights = os.path.join("saved_models", name_file +".pt")
    if cuda_avail:
        model = model.to(device = device)
    load_old_model = True
    if load_old_model:
        if os.path.exists(file_weights): # veut dire qu'on sest entraîné avant d'avoir le txt final
            print("modèle précédent pas fini")
            loaded_state = torch.load(file_weights,map_location = device)
            model.load_state_dict(loaded_state)
            model_dict = model.state_dict()
            loaded_state = {k: v for k, v in loaded_state.items() if
                            k in model_dict}  # only layers param that are in our current model
            loaded_state = {k: v for k, v in loaded_state.items() if
                            loaded_state[k].shape == model_dict[k].shape}  # only if layers have correct shapes
            model_dict.update(loaded_state)
            model.load_state_dict(model_dict)

    if loss_train == "rmse":
        lbd = 0
    elif loss_train == "pearson" :
        lbd = 100
    elif loss_train[:4] == "both":
        lbd = int(loss_train[5:])
    criterion = criterion_both(lbd, cuda_avail, device)

    files_per_categ, files_for_test = give_me_train_valid_test(train_on,test_on,config, batch_size)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr )

    categs_to_consider = files_per_categ.keys()
    with open('categ_of_speakers.json', 'r') as fp:
        categ_of_speakers = json.load(fp)  # dictionnaire en clé la categorie en valeur un dictionnaire
    plot_filtre_chaque_epochs = False

    for epoch in range(n_epochs):
        weights = model.lowpass.weight.data[0, 0, :].cpu()
        if plot_filtre_chaque_epochs :
            plot_filtre(weights)
        n_this_epoch = 0
        random.shuffle(list(categs_to_consider))
        loss_train_this_epoch = 0
        for categ in categs_to_consider:  # de A à F pour le momen

            files_this_categ_courant = files_per_categ[categ]["train"]  #on na pas encore apprit dessus au cours de cette epoch
            random.shuffle(files_this_categ_courant)

            while len(files_this_categ_courant) > 0:
                n_this_epoch+=1
                x, y = load_data(files_this_categ_courant[:batch_size])

                files_this_categ_courant = files_this_categ_courant[batch_size:] #we a re going to train on this 10 files
                x, y = model.prepare_batch(x, y)
                y_pred = model(x).double()
                if cuda_avail:
                    y_pred = y_pred.to(device=device)
                y = y.double()
                optimizer.zero_grad()
                if select_arti:
                    arti_to_consider = categ_of_speakers[categ]["arti"]  # liste de 18 0/1 qui indique les arti à considérer
                    idx_to_ignore = [i for i, n in enumerate(arti_to_consider) if n == "0"]
                    y_pred[:, :, idx_to_ignore] = 0 #the grad associated to this value will be zero  : CHECK THAT
                   # y_pred[:,:,idx_to_ignore].detach()
                    #y[:,:,idx_to_ignore].requires_grad = False

                loss = criterion(y, y_pred)
                loss.backward()
                optimizer.step()
                torch.cuda.empty_cache()
                loss_train_this_epoch += loss.item()

        torch.cuda.empty_cache()

        loss_train_this_epoch = loss_train_this_epoch/n_this_epoch

        if epoch%delta_test == 0:  #toutes les delta_test epochs on évalue le modèle sur validation et on sauvegarde le modele si le score est meilleur
            loss_vali = 0
            n_valid = 0
            for categ in categs_to_consider:  # de A à F pour le moment
                files_this_categ_courant = files_per_categ[categ]["valid"]  # on na pas encore apprit dessus au cours de cette epoch
                while len(files_this_categ_courant) >0 :
                    n_valid +=1
                    x, y = load_data(files_this_categ_courant[:batch_size])
                    files_this_categ_courant = files_this_categ_courant[batch_size:]  # on a appris sur ces 10 phrases
                    x, y = model.prepare_batch(x, y)
                    y_pred = model(x).double()
                    torch.cuda.empty_cache()
                    if cuda_avail:
                        y_pred = y_pred.to(device=device)
                    y = y.double()  # (Batchsize, maxL, 18)

                    if select_arti:
                        arti_to_consider = categ_of_speakers[categ]["arti"]  # liste de 18 0/1 qui indique les arti à considérer
                        idx_to_ignore = [i for i, n in enumerate(arti_to_consider) if n == "0"]
                        y_pred[:, :, idx_to_ignore] = 0
                    #    y_pred[:, :, idx_to_ignore].detach()
                   #     y[:, :, idx_to_ignore].requires_grad = False

                    loss_courant = criterion(y, y_pred)
                    loss_vali += loss_courant.item()
            loss_vali  = loss_vali/n_valid
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
                    param_group['lr'] = param_group['lr'] / 2
                    (param_group["lr"])

        logger.scalar_summary('loss_train', loss_train_this_epoch, model.epoch_ref )
        logger.scalar_summary('loss_valid', loss_vali, model.epoch_ref)

    if n_epochs > 0:
        model.epoch_ref = model.epoch_ref + epoch  # voir si ca marche vrmt pour les rares cas ou on continue un training
        model.load_state_dict(torch.load(os.path.join("saved_models",name_file+'.pt')))
        torch.save(model.state_dict(), os.path.join( "saved_models",name_file+".txt")) #lorsque .txt ==> training terminé !
    random.shuffle(files_for_test)
    x, y = load_data(files_for_test)
    print("evaluation on speaker {}".format(test_on))
    std_speaker = np.load(os.path.join(root_folder,"Preprocessing","norm_values","std_ema_"+test_on+".npy"))
    arti_per_speaker = os.path.join(root_folder, "Preprocessing", "articulators_per_speaker.csv")
    csv.register_dialect('myDialect', delimiter=';')
    with open(arti_per_speaker, 'r') as csvFile:
        reader = csv.reader(csvFile, dialect="myDialect")
        next(reader)
        for row in reader:
            if row[0] == test_on:
                arti_to_consider = row[1:19]
                arti_to_consider = [int(x) for x in arti_to_consider]

    rmse_per_arti_mean, pearson_per_arti_mean = model.evaluate_on_test(x, y, std_speaker = std_speaker, to_plot=to_plot
                                                                       , to_consider = arti_to_consider)
    print("training done for : ",name_file)

    # write result in csv
    with open('resultats_modeles.csv', 'a') as f:
        writer = csv.writer(f)
        row_rmse = [name_file]+rmse_per_arti_mean.tolist()+[model.epoch_ref]
        row_pearson = [name_file]+pearson_per_arti_mean.tolist() + [model.epoch_ref]
        writer.writerow(row_rmse)
        writer.writerow(row_pearson)

    weight_apres = model.lowpass.weight.data[0, 0, :].cpu()
    plot_allure_filtre = False
    if plot_allure_filtre :
        plot_filtre(weight_apres)


if __name__=='__main__':

    parser = argparse.ArgumentParser(description='Train and save a model.')

    parser.add_argument('test_on', type=str,
                        help='the speaker we want to test on')

    parser.add_argument('--n_epochs', type=int, default=50,
                        help='max number of epochs to train the model')

    parser.add_argument("--loss_train",type = str, default="both_90",
                        help = " 'both_alpha' with alpha from 0 to 100, coeff of pearson is the combined loss")

    parser.add_argument("--patience",type=int, default=5,
                        help = "patience before early topping")

    parser.add_argument("--select_arti", type = bool,default=True,
                        help = "whether to learn only on available parameters or not")

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

    parser.add_argument('--delta_test', type=int, default=1,
                        help='how often evaluate the validation set')

    parser.add_argument('config', type=str,
                        help='spec or dep or indep that stands for speaker specific/dependant/independant')

    args = parser.parse_args()

    train_model(test_on=args.test_on, n_epochs=args.n_epochs, loss_train=args.loss_train,
                patience=args.patience, select_arti=args.select_arti, corpus_to_train_on=args.corpus_to_train_on,
                batch_norma=args.batch_norma, filter_type=args.filter_type, to_plot=args.to_plot,
                lr=args.lr, delta_test=args.delta_test, config=args.config)