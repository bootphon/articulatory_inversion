
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

speaker="mocha"
root_folder = os.path.dirname(os.getcwd())
fileset_path = os.path.join(root_folder, "Donnees_pretraitees", "Donnees_breakfast", "fileset")

X_breakfast = np.load(os.path.join(fileset_path, "X_train_" + speaker + ".npy"), allow_pickle=True)
Y_breakfast = np.load(os.path.join(fileset_path, "y_train_" + speaker + ".npy"), allow_pickle=True)

X_ZS  = np.load(os.path.join(root_folder,"Donnees_pretraitees","donnees_challenge_2017","X_ZS.npy"), allow_pickle=True)

weight_loss_arti = 0.0001 #hyperparamètre qui dit l'importance de chaque loss
hidden_dim = 300
input_dim = 429
output_dim = 12
batch_size = 10
pourcent_train = 0.9
#X_train, X_valid, Y_train, Y_valid = train_test_split(X, Y, test_size=1 - pourcent_train, random_state=1)
patience = 5
model = torch.load(os.path.join("saved_models", "model_" + speaker + "_cluster.pth"))
model = model.double()

criterion_arti = torch.nn.MSELoss(reduction='mean')#meme loss ?
criterion_recon = torch.nn.MSELoss(reduction='mean')#meme loss ?

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
my_autoencoder = autoencoder(model,hidden_dim_2=hidden_dim,batch_size=batch_size)
my_autoencoder=my_autoencoder.double()
n_epochs=4
print("number of epochs : ", n_epochs)
for epoch in range(n_epochs):
    print("----------epoch : ", epoch)
    #dabord sur breakfast
    indices = np.random.choice(len(X_breakfast), batch_size, replace=False)
    x, y = X_breakfast[indices], Y_breakfast[indices]
    x, y = model.prepare_batch(x, y)
    x_reconstruit,y_encoding = my_autoencoder(x)
    optimizer.zero_grad()
    loss_reconstruction = criterion_recon(x, x_reconstruit)*weight_loss_arti
    loss_arti = criterion_arti(y,y_encoding)
    loss = sum([loss_arti,loss_reconstruction])
    loss.backward()
    optimizer.step()
    print("loss  :", loss.item())

    #ensuite sur zs
    indices = np.random.choice(len(X_ZS), batch_size, replace=False)
    x = X_ZS[indices]
    inutile = [np.ones((len(x[i]),output_dim)) for i in range(batch_size)]
    x, inutile = model.prepare_batch(x, inutile)
    x_reconstruit,inutile = my_autoencoder(x)
    optimizer.zero_grad()
    loss_reconstruction = criterion_recon(x, x_reconstruit)
    loss = loss_reconstruction
    loss.backward()
    optimizer.step()
    print("loss  :", loss.item())


