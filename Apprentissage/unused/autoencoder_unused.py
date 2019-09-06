
from Apprentissage.class_network import my_bilstm
import os
from sklearn.model_selection import train_test_split

import sys
import torch
import os
import sys
import time
import numpy as np
from os.path import dirname
from numpy.random import choice

class autoencoder(torch.nn.Module):
    def __init__(self ,model_lstm ,hidden_dim_2,batch_size):
        super(autoencoder, self).__init__()
        self.my_bilstm = model_lstm
        self.output_dim = model_lstm.output_dim
        self.hidden_dim = model_lstm.hidden_dim
        self.input_dim = model_lstm.input_dim
        self.hidden_dim_2 = hidden_dim_2
        self.batch_size = batch_size
        self.deco_first_layer = torch.nn.Linear(self.output_dim,self.hidden_dim)
        self.deco_second_layer = torch.nn.Linear(self.hidden_dim,self.input_dim)

    def decodeur(self,y):
        y=y.double()
        y1 = self.deco_first_layer(y)
        y2 = torch.nn.ReLU(True)(y1)
        y3 = self.deco_second_layer(y2)
        x_reco = torch.nn.Tanh()(y3)
        return x_reco


    def forward(self, x):
        arti = self.my_bilstm(x)
        recons = self.decodeur(arti)
        return recons,arti



weight_loss_arti = 0.0001 #hyperparamètre qui dit l'importance de chaque loss
hidden_dim = 300
input_dim = 429
output_dim = 13
batch_size = 10
pourcent_train = 0.9
root_folder = os.path.dirname(os.getcwd())
fileset_path = os.path.join(root_folder, "Donnees_pretraitees","fileset")
X_train, Y_train = dict(),dict()
for speaker in ["fsew0","msak0","MNGU0"]:
    X_train[speaker] =np.load(os.path.join(fileset_path, "X_train_" + speaker + ".npy"))
  #  X_train&speaker = X_train_&speaker.extend(np.load(os.path.join(fileset_path, "X_test_" + speaker + ".npy")))
    Y_train_temp =np.load(os.path.join(fileset_path, "Y_train_" + speaker + ".npy"))
    Y_train[speaker] = np.array([Y_train_temp[i][:, :output_dim] for i in range(len(Y_train_temp))])

#   Y_train_&speaker = Y_train_&speaker.extend(np.load(os.path.join(fileset_path, "Y_test_" + speaker + ".npy")))

X_train["ZS"] =  np.load(os.path.join(root_folder,"Donnees_pretraitees","donnees_challenge_2017","1s","X_ZS.npy"))

#X_train, X_valid, Y_train, Y_valid = train_test_split(X, Y, test_size=1 - pourcent_train, random_state=1)
patience = 5

weight_model = os.path.join(root_folder,"Apprentissage","saved_models","modeles_valides")
model = my_bilstm(hidden_dim=hidden_dim, input_dim=input_dim, name_file="for_autoencoder", output_dim=output_dim,
                  batch_size=batch_size)
model = model.double()
model.load_state_dict(torch.load(os.path.join(weight_model,"train_fsew0_msak0_MNGU0_test_fsew0_msak0_MNGU0.txt")))

my_autoencoder = autoencoder(model,hidden_dim_2=hidden_dim,batch_size=batch_size)
my_autoencoder=my_autoencoder.double()

criterion_arti = torch.nn.MSELoss(reduction='sum')#meme loss ?
criterion_recon = torch.nn.MSELoss(reduction='sum') #meme loss ?
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

n_epochs=500
delta=1
print("number of epochs : ", n_epochs)
for epoch in range(n_epochs):
     # dabord sur breakfast

    proba = [0.2,0.2,0.2,0.4]
    speaker_chosen = choice(["fsew0","msak0","MNGU0","ZS"], 1, p=proba)[0]
    indices = np.random.choice(len(X_train[speaker_chosen]), batch_size, replace=False)
    print("speaker ",speaker_chosen)
    if speaker_chosen != "ZS" :
        x = X_train.get(speaker_chosen)[indices]
        y =  Y_train.get(speaker_chosen)[indices]
        x, y = model.prepare_batch(x, y)
        x_reconstruit,y_encoding = my_autoencoder(x)
        y,y_encoding = y.double(),y_encoding.double()

        optimizer.zero_grad()
        loss_reconstruction = criterion_recon(x, x_reconstruit)
        loss_arti = criterion_arti(y,y_encoding)
        loss = sum([loss_arti*weight_loss_arti,loss_reconstruction])
        loss.backward()
        optimizer.step()
        print("loss  :", loss.item())

    elif speaker_chosen == "ZS":
        x = X_train.get("ZS")[indices]
        inutile = np.array([np.ones((len(x[i]),output_dim)) for i in range(batch_size)])

        x, inutile = model.prepare_batch(x, inutile) #la fonction prepare batch doit prendre en entrée le X et Y
        x_reconstruit,inutile = my_autoencoder(x)
        x,x_reconstruit = x.double(),x_reconstruit.double()
        optimizer.zero_grad()
        loss_reconstruction = criterion_recon(x, x_reconstruit)
        loss = loss_reconstruction
        loss.backward()
        optimizer.step()
        print("loss :", loss.item())




