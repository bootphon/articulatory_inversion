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

import gc
import psutil
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
import json
from Apprentissage.logger import Logger

root_folder = os.path.dirname(os.getcwd())
fileset_path = os.path.join(root_folder, "Donnees_pretraitees", "fileset")

print(sys.argv)


def memReport(all = False):
    nb_object = 0
    for obj in gc.get_objects():
        if torch.is_tensor(obj):
            if all:
                print(type(obj), obj.size())
            nb_object += 1
    print('nb objects tensor', nb_object)


def cpuStats():
    print(sys.version)
    print(psutil.cpu_percent())
    print(psutil.virtual_memory())  # physical memory usage
    pid = os.getpid()
    py = psutil.Process(pid)
    memoryUse = py.memory_info()[0] / 2. ** 30  # memory use in GB...I think
    print('memory GB:', memoryUse)


def train_model_complete(test_on ,n_epochs ,loss_train,patience ,select_arti,corpus_to_train_on,batch_norma,filter_type,
                train_a_bit_on_test,to_plot, lr,delta_test,config):
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

    :param train_a_bit_on_test: (bool) wheter or not to add some speaker-test info in the training set.
    If true we add its training part  to the train set and its validation part to the validation set
    (then we test on the test part only)

    :param to_plot: (bool) if true the trajectories of one random test sentence are saved in "images_predictions"

    :param lr: initial learning rate, usually 0.001

    :param delta_test: frequency of validation evaluation, 1 seems good

    :param config : either "spe" "dep", or "indep", for specific (train only on test sp), dependant (train on test sp
    and others), or independant, train only on other speakers

    :return: [rmse, pearson] . rmse the is the list of the 18 rmse (1 per articulator), same for pearson.
    """



    name_corpus_concat = ""
    if config == "spe": #speaker specific
        corpus_to_train_on = ""
        train_on = [""]  # only train on the test speaker

    elif config in ["indep","dep"] : # train on other corpuses
        train_on = []
        corpus_to_train_on = corpus_to_train_on[1:-1].split(",")
        for corpus in corpus_to_train_on:
            sp = get_speakers_per_corpus(corpus)
            train_on = train_on + sp
            name_corpus_concat = name_corpus_concat + corpus + "_"
        if test_on in train_on:
            train_on.remove(test_on)

    name_file = test_on+"_speaker_"+config+corpus_to_train_on+"_loss_"+str(loss_train)+"_filter_"+str(filter_type)+
    "_bn_"+str(batch_norma)
    previous_models = os.listdir("saved_models")
    previous_models_2 = [x[:len(name_file)] for x in previous_models if x.endswith(".txt")]
    n_previous_same = previous_models_2.count(name_file) #how many times our model was trained
    if n_previous_same > 0:
        print("this models has alread be trained {} times".format(n_previous_same))
    else :
        print("first time for this model")
    name_file = name_file + "_" + str(n_previous_same) # each model trained only once ,
    # this script doesnt continue a previous training if it was ended ie if there is a .txt
    print("going to train the model with name",name_file)
    logger = Logger('./logs')


    cuda_avail = torch.cuda.is_available()
    print(" cuda ?", cuda_avail)
    if cuda_avail :
        device = torch.device("cuda")
    else :
        device = torch.device("cpu")


    hidden_dim = 300
    input_dim = 429
    batch_size = 10
    output_dim = 18
    early_stopping = EarlyStopping(name_file,patience=patience, verbose=True)
    model = my_ac2art_modele(hidden_dim=hidden_dim, input_dim=input_dim, name_file=name_file, output_dim=output_dim,
                      batch_size=batch_size, cuda_avail=cuda_avail,
                      modele_filtered=filter_type,batch_norma=batch_norma)
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
        lbd = 0
    elif loss_train == "pearson" :
        lbd = 100
    elif loss_train[:4] == "both":
        lbd = int(loss_train[5:])

    criterion = criterion_both(lbd)

    def plot_filtre(weights) :
        print("GAIN", sum(weights))
        freqs, h = signal.freqz(weights)
        freqs = freqs * 100 / (2 * np.pi)  # freq in hz
        plt.plot(freqs, 20 * np.log10(abs(h)), 'r')
        plt.title("Allure filtre passe bas à la fin de l'apprentissage pour filtre en dur")
        plt.ylabel('Amplitude [dB]')
        plt.xlabel("real frequency")
        plt.show()


    #code qui suit peut être + compact mais ainsi on voit bien sur qui on apprend et test
    if config == "spec":
        files_for_train = load_filenames_deter([test_on],part= ["train"])
        files_for_valid = load_filenames_deter([test_on],part = ["valid"])
        files_for_test = load_filenames_deter([test_on],part=["test"])

    elif config == "dep":
        files_for_train = load_filenames_deter(train_on, part=["train","test"])+\
                          load_filenames_deter([test_on], part=["train"])
        files_for_valid = load_filenames_deter(train_on, part=["valid"]) + \
                          load_filenames_deter([test_on], part=["valid"])
        files_for_test = load_filenames_deter([test_on], part=["test"])

    elif config == "indep":
        files_for_train = load_filenames_deter(train_on, part=["train", "test"])
        files_for_valid = load_filenames_deter(train_on, part=["valid"])
        files_for_test = load_filenames_deter([test_on], part=["train","valid","test"])


    with open('categ_of_speakers.json', 'r') as fp:
        categ_of_speakers = json.load(fp) #dictionnaire en clé la categorie en valeur un dictionnaire
                                            # #avec les speakers dans la catégorie et les arti concernées par cette categorie
    optimizer = torch.optim.Adam(model.parameters(), lr=lr ) #, betas = beta_param)
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

    plot_filtre_chaque_epochs = False

    for epoch in range(n_epochs):

        weights = model.lowpass.weight.data[0, 0, :].cpu()

        if plot_filtre_chaque_epochs :
            plot_filtre(weights)

        n_this_epoch = 0
        random.shuffle(list(categs_to_consider))
        loss_train_this_epoch = 0

        for categ in categs_to_consider:  # de A à F pour le momen
            files_this_categ_courant = files_per_categ[categ]["train"] #on na pas encore apprit dessus au cours de cette epoch
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
                    y_pred[:, :, idx_to_ignore] = 0
                   # y_pred[:,:,idx_to_ignore].detach()
                    #y[:,:,idx_to_ignore].requires_grad = False

                loss = criterion(y, y_pred)
                loss.backward()
                optimizer.step()
                torch.cuda.empty_cache()
                loss_train_this_epoch += loss.item()

        torch.cuda.empty_cache()

        loss_train_this_epoch = loss_train_this_epoch/n_this_epoch

        if epoch%delta_test ==0:  #toutes les delta_test epochs on évalue le modèle sur validation et on sauvegarde le modele si le score est meilleur
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


        model.all_validation_loss.append(loss_vali)
        model.all_training_loss.append(loss_train_this_epoch)
        early_stopping(loss_vali, model)

        # ================================================================== #
        #                        Tensorboard Logging                         #
        # ================================================================== #

        # 1. Log scalar values (scalar summary)
        info = {'loss': loss_train_this_epoch}

        for tag, value in info.items():
           # print("tag valu",tag,value)
            logger.scalar_summary(tag, value, epoch + 1)

        # 2. Log values and gradients of the parameters (histogram summary)
    #    for tag, value in model.named_parameters():
     #       tag = tag.replace('.', '/')
      #      logger.histo_summary(tag, value.data.cpu().numpy(), epoch + 1)
       #     logger.histo_summary(tag + '/grad', value.grad.data.cpu().numpy(), epoch + 1)

        # 3. Log training images (image summary)
       # info = {'images': images.view(-1, 28, 28)[:10].cpu().numpy()}

        #for tag, images in info.items():
         #   logger.image_summary(tag, images, epoch + 1)
#
        if epoch>0 : # on divise le learning rate par deux dès qu'on surapprend un peu par rapport au validation set
            if loss_vali > model.all_validation_loss[-1]:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = param_group['lr'] / 2
                    (param_group["lr"])

            #model.all_test_loss += [model.all_test_loss[-1]] * (epoch+previous_epoch - len(model.all_test_loss))
           # print("\n ---------- epoch" + str(epoch) + " ---------")
            #early_stopping.epoch = previous_epoch+epoch

          #  print("train loss ", loss.item())
          #  print("valid loss ", loss_vali)

         #   logger.scalar_summary('loss_valid', loss_vali,
          #                        model.epoch_ref)
           # logger.scalar_summary('loss_train', loss.item(), model.epoch_ref)

            torch.cuda.empty_cache()

        if early_stopping.early_stop:
            print("Early stopping, n epochs : ",model.epoch_ref+epoch)
            break


    if n_epochs>0:
        model.epoch_ref = model.epoch_ref + epoch
        model.load_state_dict(torch.load(os.path.join("saved_models",name_file+'.pt')))
        torch.save(model.state_dict(), os.path.join( "saved_models",name_file+".txt"))


    random.shuffle(files_for_test)
    x, y = load_data(files_for_test)
    print("evaluation on speaker {}".format(test_on))

    std_speaker = np.load(os.path.join(root_folder,"Traitement","norm_values","std_ema_"+test_on+".npy"))
    arti_per_speaker = os.path.join(root_folder, "Traitement", "articulators_per_speaker.csv")
    csv.register_dialect('myDialect', delimiter=';')


    weight_apres = model.lowpass.weight.data[0, 0, :] #gain du filtre à la fin de l'apprentissage
  #  print("GAIN FILTRE APRES APPRENTISSAGE",sum(weight_apres.cpu()))
    with open(arti_per_speaker, 'r') as csvFile:
        reader = csv.reader(csvFile, dialect="myDialect")
        next(reader)
        for row in reader:
            if row[0] == test_on:
                arti_to_consider = row[1:19]
                arti_to_consider = [int(x) for x in arti_to_consider]

    rmse_per_arti_mean, pearson_per_arti_mean = model.evaluate_on_test(x,y, std_speaker = std_speaker, to_plot=to_plot
                                                                       ,to_consider = arti_to_consider) #,filtered=True)
    print("name file : ",name_file)

    def write_results_in_csv():
        with open('resultats_modeles.csv', 'a') as f:
            writer = csv.writer(f)
            row_rmse = [name_file]+rmse_per_arti_mean.tolist()+[model.epoch_ref]
            row_pearson  =[name_file]+pearson_per_arti_mean.tolist() + [model.epoch_ref]
            writer.writerow(row_rmse)
            writer.writerow(row_pearson)
    add_results_in_csv = True
    if add_results_in_csv :
        write_results_in_csv()

    weight_apres = model.lowpass.weight.data[0, 0, :].cpu()
    plot_allure_filtre = False #if True affiche  la réponse fréquentielle du filtre à la fin de l'apprentissage
    if plot_allure_filtre :
        plot_filtre(weight_apres)

