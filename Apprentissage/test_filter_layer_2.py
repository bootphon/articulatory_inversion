# TODO

"""
ce script est composé de deux fonctions.
La première d'apprendre la correspondance "articulatoires" "velum" sur le corpus MOCHA.
C'est à dire à partir des 12 premières données artiuculatoires prédire les deux dernières (qui sont velum_x et velum_y)
La deuxième fonction permet de remplir un dossier "ema_lar" dans "inversion_articulatoire_2\Donnees_pretraitees\Donnees_breakfast\MNGU0"
Pour chaque phrase on utilise le modèle qui a apprit précédemment pour prédire les trajectoires du velum.

ema_lar contient pour chaque phrase un fichier "mngu0_s1_000i_lar" de forme (Kx14), où K dépend de la longueur de la phrase.
Donc le fichier "mngu0_s1_000i_lar" est le même que "mngu0_s1_000i", en y rajoutant deux colonnes pour velum_x et velum_y.
"""

import torch
import os
import sys
from sklearn.model_selection import train_test_split
from scipy.signal import butter, lfilter, freqz
import time
from os.path import dirname
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import scipy
import utils
from torch.autograd import Variable



def concat_all_numpy_from(path,speaker="",file_type=""):
    " TODO"
    list = []
    for r, d, f in os.walk(path):
        for file in f:
            if speaker in file :
                if file_type in file:
                    data_file = np.load(os.path.join(path,file))
                    list.append(data_file)
    return list #

# TODO: c'est toujours vrai ça ? Si dépend utilisateur: à changer
root_folder = os.path.dirname(os.getcwd())
fileset_path = os.path.join(root_folder,"Donnees_pretraitees","fileset_train_fsew0_test_msak0")
arti_fsew0 = np.load(os.path.join(fileset_path,"Y_train.npy"),allow_pickle=True)
arti_msak0 = np.load(os.path.join(fileset_path,"Y_test.npy"),allow_pickle=True)
arti_mocha = np.concatenate((arti_fsew0,arti_msak0),axis=0)


X = [arti_mocha[i][:,:-2] for i in range(len(arti_mocha))] # input : all the articulatory trajectories except larynx ie last two columns
Y= [arti_mocha[i][:,-2:] for i in range(len(arti_mocha))] #output velum_x and velum_y

class test_filter(torch.nn.Module):
    # TODO
    def __init__(self):
        # TODO
        super(test_filter, self).__init__()
        self.sampling_rate = 500
        self.input_dim = 12 # all arti
        self.output_dim =2 # velum x and y
        self.first_layer = torch.nn.Linear(self.input_dim, 100)
        self.second_layer = torch.nn.Linear(100, 30)
        self.lstm_layer = torch.nn.LSTM(input_size=30,
                                        hidden_size=30, num_layers=1,
                                        bidirectional=True)
        self.readout_layer = torch.nn.Linear(30,self.output_dim) #30 fois 2 si lstm
        self.all_training_loss=[]
        self.all_validation_loss=[]
        self.window=5
        self.cutoff=240
        self.batch_size=10
        self.init_filter_layer()

    def prepare_batch(self, x, y):
        # TODO
        max_lenght = np.max([len(phrase) for phrase in x])
        new_x = torch.zeros((self.batch_size, max_lenght, self.input_dim), dtype=torch.double)
        new_y = torch.zeros((self.batch_size, max_lenght,self.output_dim), dtype=torch.double)
        for j in range(self.batch_size):
            zeropad = torch.nn.ZeroPad2d((0, 0, 0, max_lenght - len(x[j])))
            new_x[j] = zeropad(torch.from_numpy(x[j])).double()
            new_y[j] = zeropad(torch.from_numpy(y[j])).double()

        x = new_x.view((self.batch_size, max_lenght, self.input_dim))
        y = new_y.view((self.batch_size, max_lenght,self.output_dim))
        return x, y

    def get_filter_weights(self,cutoff):
        #TODO
        #print(cutoff)
        c = cutoff.item()
        fc = cutoff.item() / self.sampling_rate  # Cutoff frequency as a fraction of the sampling rate (in (0, 0.5)).

        if fc > 0.5:
            raise Exception("La frequence de coupure doit etre au moins deux fois la frequence dechantillonnage")
        b = 0.08  # Transition band, as a fraction of the sampling rate (in (0, 0.5)).
        N = int(np.ceil((4 / b)))  # le window
        if not N % 2:
            N += 1  # Make sure that N is odd.
        n = torch.from_numpy(np.arange(N) ) # int of [0,N]
        h = np.sinc(2 * fc * (n - (N - 1) / 2))  # Compute sinc filter.
        h = torch.from_numpy(h)
        w = 0.5 * (1 - np.cos(2 * np.pi * n / (N - 1)))  # Compute hanning window.
        h = h * w  # Multiply sinc filter with window.
        h = h / torch.sum(h)
        h = h.view(1,len(h))
        h = torch.cat([h]*self.output_dim,0)
        return h

    def init_filter_layer(self):
        #TODO
        cutoff  = torch.tensor(self.cutoff,dtype=torch.float64).view(1,1)
        self.cutoff = Variable( cutoff,requires_grad=True)
        self.cutoff.require_grads = True
        self.cutoff.retain_grad()
        window_size = 5
        C_in = self.output_dim
        #stride=1
        padding =18 #int(0.5*((C_in-1)*stride-C_in+window_size))+23
        lowpass = torch.nn.Conv1d(C_in,C_in,window_size,  stride=1, padding=padding,
                                  bias=False,groups = self.output_dim)
        lowpass.weight.requires_grad = True
        lowpass_init = self.get_filter_weights(cutoff=self.cutoff)
        lowpass_init = lowpass_init.view((self.output_dim,1,-1)) #2 1 51
        print(lowpass_init.shape)
        lowpass.weight = torch.nn.Parameter(lowpass_init)
        lowpass = lowpass.double()
        self.lowpass = lowpass

    def filter_layer(self,y):
        #TODO
        L= len(y[0])
        y = y.double()
    #    y_smoothed = torch.zeros(self.batch_size,L,self.output_dim)
        y = y.view(self.batch_size,self.output_dim,L)
        y_smoothed= self.lowpass(y)
        Ls = y_smoothed.shape[2] #longueur des phrases lissees
        y_smoothed = y_smoothed.view(self.batch_size,self.output_dim,Ls)
       # for i in range(self.output_dim):
        #    traj_arti = y[:,:,i].view(self.batch_size,1,L)
         #   traj_arti_smoothed = self.lowpass(traj_arti) #prend que une seule dimension
        difference = int((L- Ls) / 2)
       # print(L,Ls,difference)
        #print("bfore",y_smoothed.shape)
        #print("gol",y.shape)
        if difference>0 :
            y_smoothed = torch.nn.ReplicationPad1d(difference)(y_smoothed)
       # print("after", y_smoothed.shape)
        y_smoothed = y_smoothed.view(self.batch_size,L,self.output_dim)

            #traj_arti_smoothed = traj_arti_smoothed.view(self.batch_size,L)
            #y_smoothed[:,:,i] =traj_arti_smoothed
     #   print('1', self.cutoff.grad)
        return y_smoothed

    def forward(self, x):
        #TODO
        dense_out = torch.nn.functional.relu(self.first_layer(x))
        dense_out_2 = torch.nn.functional.relu(self.second_layer(dense_out))
        y_pred = self.readout_layer(dense_out_2)
        y_pred_filter = self.filter_layer(y_pred)

        return y_pred,y_pred_filter



def train_model(lr=0.05,n_epochs=20):
    #TODO
    pourcent_valid = 0.1
    pourcent_test = 0.3
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=pourcent_test, random_state=1)
    X_train, X_valid, Y_train, Y_valid = train_test_split(X_train, Y_train, test_size=pourcent_valid, random_state=1)
    model_velum=test_filter()
    PATH_weights = os.path.join(root_folder, "Apprentissage", "saved_models", "model_velum_without_lstm.txt")
    try :
        model_velum.load_state_dict(torch.load(PATH_weights))
    except :
        print('first time, intialisation...')

    model_velum = model_velum.double()
    criterion = torch.nn.MSELoss(reduction='mean')
    optimizer = torch.optim.Adam(model_velum.parameters(), lr=lr)

    for epoch in range(n_epochs):
        print("--epoch----",epoch)
        print("cutoff :",model_velum.cutoff)
        print(model_velum.cutoff.grad)
        #print(model_velum.first_layer.weight.grad)

        weights_before =model_velum.lowpass.weight.data
        indices = np.random.choice(len(X_train), model_velum.batch_size, replace=False)
        X_train,Y_train = np.array(X_train),np.array(Y_train)
        x,y = X_train[indices], Y_train[indices]
        x,y = model_velum.prepare_batch(x,y)
        y = y.view((y.shape[0],y.shape[1],model_velum.output_dim))
        y_pred,y_pred_smoothed= model_velum(x)
        weights_after_1 =model_velum.lowpass.weight.data

        y_np = y.detach().numpy()
        y_pred_np = y_pred.detach().numpy()
        y_pred_smoothed_np = y_pred_smoothed.detach().numpy()

       #  plt.plot(y_np[3][:,0])
        plt.plot(y_pred_np[3][:,0])
        plt.plot(y_pred_smoothed_np[3][:,0])
      #  plt.legend(['predit','predit + filtered'])
      #  plt.show()

    #  plt.plot(y_np[3][:, 1])
        plt.plot(y_pred_np[3][:, 1])
        plt.plot(y_pred_smoothed_np[3][:, 1])
        plt.legend(['predit', 'predit + filtered','predit_2', 'predit_2 + filtered'])
        plt.show()

        optimizer.zero_grad()
       # print(y_pred.shape)
        print("kkk",y_pred.shape,y.shape)
        loss = criterion(y_pred,y)
        model_velum.cutoff.retain_grad()
        loss.backward()

        optimizer.step()
        weights_after_2 =model_velum.lowpass.weight.data

      #  print("1 et 2 same ? ",weights_before==weights_after_1)
       # print("2 et 3 same ? ",weights_after_2==weights_after_1)

# TODO: c'est normal ça ? SI c'est pour un test il faut le mettre dans un main
train_model(n_epochs=2,lr=0.01)



