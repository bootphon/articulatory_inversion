import torch
import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)
from sklearn.model_selection import train_test_split
from scipy.signal import butter, lfilter, freqz
import time
import math
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
from os.path import dirname
import numpy as np
from scipy import signal
import scipy
from torch.autograd import Variable
from Apprentissage import utils


class my_ac2art_modele(torch.nn.Module):
    def __init__(self, hidden_dim, input_dim, output_dim, batch_size,name_file="", sampling_rate=100,
                 window=5, cutoff=10,cuda_avail =False,modele_filtered=False,batch_norma=False):
        root_folder = os.path.dirname(os.getcwd())
        super(my_ac2art_modele, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.first_layer = torch.nn.Linear(input_dim, hidden_dim)
        self.second_layer = torch.nn.Linear(hidden_dim, hidden_dim)
        self.last_layer = torch.nn.Linear(output_dim,output_dim)
        self.lstm_layer = torch.nn.LSTM(input_size=hidden_dim,
                                        hidden_size=hidden_dim, num_layers=1,
                                        bidirectional=True)
        self.batch_norm_layer =  torch.nn.BatchNorm1d(hidden_dim*2)

        self.lstm_layer_2= torch.nn.LSTM(input_size=hidden_dim*2,
                                       hidden_size=hidden_dim, num_layers=1,
                                      bidirectional=True)
        self.batch_norm_layer_2 =  torch.nn.BatchNorm1d(hidden_dim*2)

        self.readout_layer = torch.nn.Linear(hidden_dim*2  , output_dim)
        self.batch_size = batch_size
        self.sigmoid = torch.nn.Sigmoid()
    #    if modele_filtered==3:
     #       self.first_layer.weight.data.requires_grad=False
      #      self.second_layer.weight.data.requires_grad=False
       #     for param in self.lstm_layer.parameters():
        #        param.requires_grad = False
         #   for param in self.lstm_layer_2.parameters():
          #      param.requires_grad = False

        self.modele_filtered=modele_filtered
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
        self.lowpass = None
        self.init_filter_layer()
        self.cuda_avail = cuda_avail
      #  if self.cuda_avail:
       #     self.cuda = torch.device('cuda')
        self.epoch_ref = 0
        self.batch_norma = batch_norma# False #tester par la suite si améliore la perf

    def prepare_batch(self, x, y):
        """
        :param x: liste de B données accoustiques de longueurs variables  (B souvent 10 batchsize sauf pour l'éval)
        :param y: liste de B données articulatoires correspondantes de longueurs variables (B souvent 10 batchsize sauf pour l'éval)
        :return: les x et y zeropaddé de telle sorte que chacun ait la même longueur.
        Les tailles des output sont [B, max lenght, 429] et [B, max lenght, 18]
        """
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
        if self.cuda_avail :
            x,y=x.cuda(),y.cuda()
           # x,y = x.to(device=self.cuda),y.to(device=self.cuda)
        return x, y

    def forward(self, x, filter_output =None) :
        if filter_output is None :
            filter_output = (self.modele_filtered != 0)
        dense_out =  torch.nn.functional.relu(self.first_layer(x))
        dense_out_2 = torch.nn.functional.relu(self.second_layer(dense_out))
        lstm_out, hidden_dim = self.lstm_layer(dense_out_2)
        B = lstm_out.shape[0] #presque tjrs batch size
        if self.batch_norma :
            lstm_out_temp = lstm_out.view(B,2*self.hidden_dim,-1)
            lstm_out_temp = torch.nn.functional.relu(self.batch_norm_layer(lstm_out_temp))
            lstm_out= lstm_out_temp.view(B,  -1,2 * self.hidden_dim)
        lstm_out = torch.nn.functional.relu(lstm_out)
        lstm_out, hidden_dim = self.lstm_layer_2(lstm_out)
        if self.batch_norma :
            lstm_out_temp = lstm_out.view(B,2*self.hidden_dim,-1)
            lstm_out_temp = torch.nn.functional.relu(self.batch_norm_layer_2(lstm_out_temp))
            lstm_out= lstm_out_temp.view(B,  -1,2 * self.hidden_dim)
        lstm_out=torch.nn.functional.relu(lstm_out)
        y_pred = self.readout_layer(lstm_out)
        if filter_output:
            y_pred = self.filter_layer(y_pred)
        return y_pred

    def init_filter_layer(self):

        def get_filter_weights():
            cutoff = torch.tensor(self.cutoff, dtype=torch.float64,requires_grad=True).view(1, 1)
            fc = torch.div(cutoff,
                  self.sampling_rate)  # Cutoff frequency as a fraction of the sampling rate (in (0, 0.5)).
            if fc > 0.5:
                raise Exception("La frequence de coupure doit etre au moins deux fois la frequence dechantillonnage")
            b = 0.08  # Transition band, as a fraction of the sampling rate (in (0, 0.5)).
            N = int(np.ceil((4 / b)))  # le window
            if not N % 2:
                N += 1  # Make sure that N is odd.
            n = torch.arange(N).double()
            alpha = torch.mul(fc, 2 * (n - (N - 1) / 2)).double()
            minim = torch.tensor(0.01, dtype=torch.float64) #utile ?
            alpha = torch.max(alpha,minim)#utile ?
            h = torch.div(torch.sin(alpha), alpha)
            beta = n * 2 * math.pi / (N - 1)
            w = 0.5 * (1 - torch.cos(beta))  # Compute hanning window.
            h = torch.mul(h, w)  # Multiply sinc filter with window.
            h = torch.div(h, torch.sum(h))
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
            h = np.sinc(fc*2*(n - (N - 1) / 2))
            w = 0.5 * (1 - np.cos( n * 2 * math.pi / (N - 1)))  # Compute hanning window.
            h = h*w
            h = h/np.sum(h)
            return torch.tensor(h)

        window_size = 5
        C_in = 1
        stride=1
        padding = int(0.5*((C_in-1)*stride-C_in+window_size))+23
        if self.modele_filtered in [0,1,3] : #si 0 le filtre ne sera pas appliqué donc on sen fiche
            weight_init = get_filter_weights_en_dur()
        elif self.modele_filtered == 2:
            weight_init = get_filter_weights()
        weight_init = weight_init.view((1, 1, -1))
        lowpass = torch.nn.Conv1d(C_in,self.output_dim, window_size, stride=1, padding=padding, bias=False)
        if self.modele_filtered == 2:
            lowpass.weight = torch.nn.Parameter(weight_init,requires_grad= True)
        elif self.modele_filtered == 3:
            lowpass.weight = torch.nn.Parameter(weight_init)

        else : #0 balek ou 1 en dur
            lowpass.weight = torch.nn.Parameter(weight_init,requires_grad = False)



        lowpass = lowpass.double()
        self.lowpass = lowpass

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
            #difference = int((L-traj_arti_smoothed.shape[2])/ 2)
            #if difference != 0:
              #  print("PAS MEME SHAPE AVANT ET APRES FILTRAGE !")
             #   print("init L",L)
            #    print("after smoothed ",traj_arti_smoothed.shape[2])
           # if difference>0: #si la traj smoothed est plus petite que L on rajoute le meme dernier élément
           #     traj_arti_smoothed = torch.nn.ReplicationPad1d(difference)(traj_arti_smoothed)
         #   elif difference < 0:  # si la traj smoothed est plus petite que L on rajoute le meme dernier élément
          #      traj_arti_smoothed = traj_arti_smoothed[:,:,0:L]

            traj_arti_smoothed = traj_arti_smoothed.view(B, L)
            y_smoothed[:, :, i] = traj_arti_smoothed
        return y_smoothed

    def plot_results(self, y, y_pred,y_pred_smooth = None,to_cons=[]):
        print("you chose to plot")
        plt.figure()

        idx_to_cons = [k for k in range(len(to_cons)) if to_cons[k]]
        for j in idx_to_cons:
            plt.figure()
            plt.plot(y_pred[:, j],alpha=0.6)
            plt.plot(y[:, j])
            if len(y_pred_smooth)>0:
                plt.plot(y_pred_smooth[:,j])
            plt.title("prediction_test_{0}_arti_{1}.png".format(self.name_file,str(j)))
            plt.legend(["prediction", "vraie","pred smoothed"])
            save_pics_path = os.path.join(
                "images_predictions\\{0}_arti_{1}.png".format(self.name_file,str(j)))
            plt.savefig(save_pics_path)
            plt.close('all')


    def evaluate_on_test(self,X_test,Y_test, std_speaker ,to_plot=False,to_consider=None,verbose=True):
        """
        :param X_test:  list of all the input of the test set
        :param Y_test:  list of all the target of the test set
        :param std_speaker : list of the std of each articulator, useful to calculate the RMSE of the predicction
        :param to_plot: weather or not we want to save some predicted smoothed and not and true trajectory
        :param to_consider: list of 0/1 for the test speaker , 1 if the articulator is ok for the test speaker
        :return: print and return the pearson correlation and RMSE between real and predicted trajectories per articulators.
        """
        idx_to_ignore = [i for i in range(len(to_consider)) if not(to_consider[i])]
        all_diff = np.zeros((1, self.output_dim))
        all_pearson = np.zeros((1, self.output_dim))

        indices_to_plot = np.random.choice(len(X_test), 2, replace=False)
        for i in range(len(X_test)):
                L = len(X_test[i])
                x_torch = torch.from_numpy(X_test[i]).view(1,L,self.input_dim)  #x (1,L,429)
                y = Y_test[i].reshape((L, self.output_dim))                     #y (L,13)
                if self.cuda_avail:
                    x_torch = x_torch.cuda()
                  #  x_torch = x_torch.to(device=self.cuda)
               # with torch.no_grad():
                y_pred_passmooth = self(x_torch,False).double() #sortie y_pred (1,L,13)
                y_pred   = self(x_torch,True).double()
                if self.cuda_avail:
                    y_pred_passmooth = y_pred_passmooth.cpu()
                    y_pred = y_pred.cpu()

                y_pred_passmooth = y_pred_passmooth.detach().numpy().reshape((L, self.output_dim))  # y_pred (L,13)
                y_pred= y_pred.detach().numpy().reshape((L, self.output_dim))  # y_pred (L,13)
                if to_plot:
                   if i in indices_to_plot:
                        self.plot_results(y, y_pred_passmooth, y_pred,to_consider)

                rmse = np.sqrt(np.mean(np.square(y - y_pred), axis=0))  # calcule du rmse à la main
                rmse = np.reshape(rmse, (1, self.output_dim))  # dénormalisation et taille (1,13)
                rmse = rmse*std_speaker #unormalize
                all_diff = np.concatenate((all_diff, rmse))
                pearson = [0]*self.output_dim
                for k in range(self.output_dim):
                    pearson[k]= np.corrcoef(y[:,k].T,y_pred[:,k].T)[0,1]

                pearson = np.array(pearson).reshape((1,self.output_dim))
                all_pearson = np.concatenate((all_pearson,pearson))
        all_pearson=all_pearson[1:]
        all_pearson[:,idx_to_ignore]=0
        all_diff = all_diff[1:]
        all_diff[:,idx_to_ignore] = 0
        pearson_per_arti_mean = np.mean(all_pearson, axis=0)
        rmse_per_arti_mean = np.mean(all_diff, axis=0)
        if verbose:
            #pearson_per_arti_std = np.std(all_pearson, axis=0)
            print("rmse final : ", np.mean(rmse_per_arti_mean[rmse_per_arti_mean!=0]))
            print("rmse mean per arti : \n", rmse_per_arti_mean)
            print("pearson final : ", np.mean(pearson_per_arti_mean[rmse_per_arti_mean!=0]))
            print("pearson mean per arti : \n", pearson_per_arti_mean)
        return rmse_per_arti_mean,pearson_per_arti_mean



