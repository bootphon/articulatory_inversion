import torch
import os
import sys
from sklearn.model_selection import train_test_split
from scipy.signal import butter, lfilter, freqz
import time
import math
from scipy.stats import pearsonr
from os.path import dirname
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import scipy
from torch.autograd import Variable

try :
    from Apprentissage import utils
except :  import utils


class my_bilstm(torch.nn.Module):
    def __init__(self, hidden_dim, input_dim, output_dim, batch_size,name_file, sampling_rate=200,
                 window=5, cutoff=20):
        root_folder = os.path.dirname(os.getcwd())
        super(my_bilstm, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.first_layer = torch.nn.Linear(input_dim, hidden_dim)
        self.second_layer = torch.nn.Linear(hidden_dim, hidden_dim)
        self.last_layer = torch.nn.Linear(output_dim,output_dim)
        self.lstm_layer = torch.nn.LSTM(input_size=hidden_dim,
                                        hidden_size=hidden_dim, num_layers=1,
                                        bidirectional=True)
      #  self.lstm_layer_2= torch.nn.LSTM(input_size=hidden_dim*2,
       #                                 hidden_size=hidden_dim, num_layers=1,
        #                                bidirectional=True)
        self.readout_layer = torch.nn.Linear(hidden_dim *2, output_dim)
        self.batch_size = batch_size
        self.sigmoid = torch.nn.Sigmoid()
        self.softmax = torch.nn.Softmax(dim=output_dim)
        self.tanh = torch.nn.Tanh()
        self.sampling_rate = sampling_rate
        self.window = window
        self.cutoff = cutoff
        self.min_valid_error = 100000
        self.all_training_loss = []
        self.all_validation_loss = []
        self.all_test_loss = []
        #self.std = np.load(os.path.join(root_folder,"Traitement","std_ema_"+speaker+".npy"))
        self.name_file = name_file
      #  self.lowpass = None
      #  self.init_filter_layer()

    def prepare_batch(self, x, y, cuda_avail = False):
        max_length = np.max([len(phrase) for phrase in x])
        B = len(x)  # often batch size but not for validation
        new_x = torch.zeros((B, max_length, self.input_dim), dtype=torch.double)
        new_y = torch.zeros((B, max_length, self.output_dim), dtype=torch.double)

        for j in range(B):
            zeropad = torch.nn.ZeroPad2d((0, 0, 0, max_length - len(x[j])))
            new_x[j] = zeropad(torch.from_numpy(x[j])).double()
            new_y[j] = zeropad(torch.from_numpy(y[j])).double()
        x = new_x.view((B, max_length, self.input_dim))
        y = new_y.view((B, max_length, self.output_dim))
        if cuda_avail :
            x,y=x.cuda(),y.cuda()
        return x, y

    def forward(self, x):
        dense_out =  torch.nn.functional.relu(self.first_layer(x))
        dense_out_2 = torch.nn.functional.relu(self.second_layer(dense_out))
        lstm_out, hidden_dim = self.lstm_layer(dense_out_2)
      #  lstm_out, hidden_dim = self.lstm_layer_2(lstm_out)

        lstm_out=torch.nn.functional.relu(lstm_out)
        y_pred = self.readout_layer(lstm_out)
       # y_pred = self.filter_layer(y_pred)
        return y_pred

    def init_filter_layer(self):
        def get_filter_weights():
            # print(cutoff)
            cutoff = torch.tensor(self.cutoff, dtype=torch.float64).view(1, 1)
            fc = torch.div(cutoff,
                  self.sampling_rate)  # Cutoff frequency as a fraction of the sampling rate (in (0, 0.5)).
           # print("0",fc)
            if fc > 0.5:
                raise Exception("La frequence de coupure doit etre au moins deux fois la frequence dechantillonnage")
            b = 0.08  # Transition band, as a fraction of the sampling rate (in (0, 0.5)).
            N = int(np.ceil((4 / b)))  # le window
            if not N % 2:
                N += 1  # Make sure that N is odd.
            n = torch.arange(N).double()

            alpha = torch.mul(fc, 2 * (n - (N - 1) / 2)).double()
           # print("1",alpha)
            h = torch.div(torch.sin(alpha), alpha)
            h[torch.isnan(h)] = 1
            #print("2",h)
            #        h = np.sinc(2 * fc * (n - (N - 1) / 2))  # Compute sinc filter.
            beta = n * 2 * math.pi * (N - 1)

            """ n = torch.from_numpy(np.arange(N) ) # int of [0,N]
        h = np.sinc(2 * fc * (n - (N - 1) / 2))  # Compute sinc filter.
        h = torch.from_numpy(h)
        w = 0.5 * (1 - np.cos(2 * np.pi * n / (N - 1)))  # Compute hanning window.
        h = h * w  # Multiply sinc filter with window.
       """
           # print("2.5",beta)
            w = 0.5 * (1 - torch.cos(beta))  # Compute hanning window.
            #print("3",w)
            h = torch.mul(h, w)  # Multiply sinc filter with window.
            h = torch.div(h, torch.sum(h))
            #print("4",h)
           # h.require_grads = True
          #  self.cutoff = Variable(cutoff, requires_grad=True)
         #   self.cutoff.require_grads = True
         #   self.cutoff.retain_grad()
            #  h = torch.cat([h]*self.output_dim,0)
            return h

        def get_filter_weights_en_dur():
            fc = self.cutoff/self.sampling_rate
            if fc > 0.5:
                raise Exception("La frequence de coupure doit etre au moins deux fois la frequence dechantillonnage")

            b = 0.08  # Transition band, as a fraction of the sampling rate (in (0, 0.5)).
            N = int(np.ceil((4 / b)))  # le window
            if not N % 2:
                N += 1  # Make sure that N is odd.
            n = np.arange(N)
           # print("1",n)
            h = np.sinc(fc*2*(n - (N - 1) / 2))
           # print("2",h)
            w = 0.5 * (1 - np.cos( n * 2 * math.pi / (N - 1)))  # Compute hanning window.
           # print("3",w)

            h = h*w
            #print("4",h)
            h = h/np.sum(h)
#            print("5",h)

            return torch.tensor(h)

        # print("1",self.cutoff)
        # self.cutoff = torch.nn.parameter.Parameter(torch.Tensor(self.cutoff))
        # self.cutoff.requires_grad = True
        window_size = 5
        C_in = 1
        # stride=1
        padding = 5 # int(0.5*((C_in-1)*stride-C_in+window_size))+23
        lowpass = torch.nn.Conv1d(C_in, self.output_dim, window_size, stride=1, padding=padding,              bias=False)
        weight_init = get_filter_weights_en_dur()
        weight_init = weight_init.view((1, 1, -1))
        lowpass.weight = torch.nn.Parameter(weight_init)
        lowpass = lowpass.double()
        self.lowpass = lowpass
        #print("lowpasse ",self.lowpass.weight)

        #self.lowpass.require_grads=True
    def filter_layer(self, y):
        B = len(y) # batch size
        L = len(y[0])
        #     y= y.view(self.batch_size,self.output_dim,L)
        y = y.double()
        y_smoothed = torch.zeros(B, L, self.output_dim)
        for i in range(self.output_dim):
            traj_arti = y[:, :, i].view(B, 1, L)
          #  print("traj arti shape",traj_arti.shape)
            traj_arti_smoothed = self.lowpass(traj_arti)  # prend que une seule dimension
            difference = int((L-traj_arti_smoothed.shape[2])/ 2)
            if difference>0: #si la traj smoothed est plus petite que L on rajoute le meme dernier élément
                traj_arti_smoothed = torch.nn.ReplicationPad1d(difference)(traj_arti_smoothed)
            traj_arti_smoothed = traj_arti_smoothed.view(B, L)
            y_smoothed[:, :, i] = traj_arti_smoothed
        return y_smoothed

    def plot_results(self, y, y_pred,suffix=""):
        plt.figure()
        for j in range(self.output_dim):
            plt.figure()
            #print("10 first :",y_pred[0:10,j])
            plt.plot(y_pred[:, j])
            plt.plot(y[:, j])
            plt.title("prediction_test_{0}_{1}_arti{2}.png".format(self.name_file,suffix ,str(j)))
            plt.legend(["prediction", "vraie"])
            save_pics_path = os.path.join(
                "images_predictions\\{0}_{1}_arti{2}.png".format(self.name_file,suffix,str(j)))
            plt.savefig(save_pics_path)
            plt.close('all')

    def evaluate(self, x_valid, y_valid,criterion,cuda_avail=False):
        x_temp, y_temp = self.prepare_batch(x_valid, y_valid,cuda_avail=cuda_avail) #add zero to have correct size
        y_pred = self(x_temp).double()
        y_temp = y_temp.double()
        loss = criterion(y_pred, y_temp).item()
        return loss

    def evaluate_on_test(self, criterion, verbose=False,X_test=None,Y_test=None,to_plot=False,
                         std_ema = 1 ,suffix= "",cuda_avail=False):

        all_diff = np.zeros((1, self.output_dim))
        all_pearson = np.zeros((1, self.output_dim))

        indices_to_plot=[]
        if to_plot :
            print("you chose to plot")
            indices_to_plot = np.random.choice(len(X_test), 5, replace=False)
        loss_test= 0
        for i in range(len(X_test)):
                L = len(X_test[i])
                x_torch = torch.from_numpy(X_test[i]).view(1,L,self.input_dim)  #x (1,L,429)
                y = Y_test[i].reshape((L, self.output_dim))                     #y (L,13)
                y_torch = torch.from_numpy(y).double().reshape(1,L,self.output_dim) #y (1,L,13)
                if cuda_avail:
                    x_torch = x_torch.cuda()
                y_pred_torch = self(x_torch).double() #sortie y_pred (1,L,13)
                if cuda_avail:
                    y_pred_torch = y_pred_torch.cpu()
                y_pred = y_pred_torch.detach().numpy().reshape((L, self.output_dim))  # y_pred (L,13)
                the_loss = criterion(y_torch, y_pred_torch)  #loss entre données de taillees  (1,L,13)
                loss_test += the_loss.item()
                if i in indices_to_plot:
                    self.plot_results(y, y_pred, suffix=suffix + str(i))

                rmse = np.sqrt(np.mean(np.square(y - y_pred), axis=0))  # calcule du rmse à la main
                rmse = np.reshape(rmse, (1,self.output_dim)) #dénormalisation et taille (1,13)
                all_diff = np.concatenate((all_diff, rmse))

                y_1 = (y_torch - torch.mean(y_torch,dim=[0,1]))*torch.from_numpy(std_ema) #(1,L,13)
                y_pred_1 = (y_pred_torch - torch.mean(y_pred_torch,dim=[0,1]))*torch.from_numpy(std_ema )# (1,L,13)
                pearson_1 = torch.sum(y_1 * y_pred_1,dim=[0,1])  # (13)
                pearson_2 = torch.sqrt(torch.sum(y_1 ** 2,dim=[0,1])) * torch.sqrt(torch.sum(y_pred_1 ** 2,dim=[0,1])) #(13)
                pearson = torch.div(pearson_1,pearson_2).view((1,self.output_dim))
                pearson[torch.isnan(pearson)] = 1
                pearson = pearson.detach().numpy()
                all_pearson = np.concatenate((all_pearson,pearson))



        loss_test = loss_test/len(X_test)
        all_diff = all_diff[1:] #remove first row of zeros #all the errors per arti and per sample
        all_pearson=all_pearson[1:]
        if verbose :
            rmse_per_arti_mean = np.mean(all_diff,axis=0)*std_ema
            rmse_per_arti_std = np.std(all_diff,axis=0)*std_ema
            print("rmse final : ", np.mean(rmse_per_arti_mean))
            print("rmse mean per arti : \n", rmse_per_arti_mean)
        #    print("rmse std per arti : \n", rmse_per_arti_std)

            pearson_per_arti_mean = np.mean(all_pearson, axis=0)
            pearson_per_arti_std = np.std(all_pearson, axis=0)
            print("pearson final : ", np.mean(pearson_per_arti_mean))
            print("pearson mean per arti : \n", pearson_per_arti_mean)
         #   print("pearson std per arti : \n", pearson_per_arti_std)
        return loss_test
