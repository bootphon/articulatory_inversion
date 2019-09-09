#TODO

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
import gc

def memReport(all = False):
    """
    :param all: wheter to detail all size obj
    :return: n objects
    In case of memory troubles call this function
    """
    nb_object = 0
    for obj in gc.get_objects():
        if torch.is_tensor(obj):
            if all:
                print(type(obj), obj.size())
            nb_object += 1
    print('nb objects tensor', nb_object)


class my_ac2art_modele(torch.nn.Module):
    """
    pytorch implementation of neural network
    """
    # TODO: là c'est super important de bien détailler
    def __init__(self, hidden_dim, input_dim, output_dim, batch_size,name_file="", sampling_rate=100,
                 window=5, cutoff=10,cuda_avail =False,modele_filtered=False,batch_norma=False):
        """
        :param hidden_dim: int, hidden dimension of lstm (usually 300)
        :param input_dim: int, input dimension of the acoustic features for 1 frame mfcc (usually 429)
        :param output_dim: int, # of trajectories to predict (usually 18)
        :param batch_size:  int, usually 10
        :param name_file: str, name of the modele
        :param sampling_rate: int, sampling rate of the ema data for the smoothing (usually 100)
        :param window: int, window size for the smoothing usually 5
        :param cutoff: int, intial cutoff frequency for the smoothing, usually 10Hz
        :param cuda_avail: bool, whether gpu is available
        :param modele_filtered: bool, whether to smooth the data during the forward
        :param batch_norma: bool, whether to add batch normalization after the lstm layers
        """
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
        self.name_file = name_file
        self.lowpass = None
        self.init_filter_layer()
        self.cuda_avail = cuda_avail

        self.epoch_ref = 0
        self.batch_norma = batch_norma
        if cuda_avail :
            self.device = torch.device("cuda")

    def prepare_batch(self, x, y):
        """
        :param x: list of B(batchsize) acoustic trajectories of variable lenghts,
        each element of the list is an array (K,18)  (K not always the same)
        :param y: list of B(batchsize) articulatory features,
        each element of the list is an array (K,429) (K not always the same)
        :return: 2 np array of sizes (B, K_max, 18) and (B, K_max, 429
        x,y initially data of the batch with different sizes . the script zeropad the acoustic and
        articulatory sequences so that all element in the batch have the same size
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
            x,y = x.to(device = self.device),y.to(device = self.device)
        return x, y

    def forward(self, x, filter_output =None):
        """
        :param x: (Batchsize,K,429)  acoustic features corresponding to batch size
        :param filter_output: whether or not to pass throught the convolutional layer
        :return: the articulatory prediction (Batchsize, K,18) based on the current weights
        """
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

    def get_filter_weights(self):
            """
            :return: low pass filter weights based on calculus exclusively using tensors so pytorch compatible
            """
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

    def get_filter_weights_en_dur(self):
        """
        :return:  low pass filter weights using classical primitive (not on tensor)
        """
        fc = self.cutoff / self.sampling_rate
        if fc > 0.5:
            raise Exception("La frequence de coupure doit etre au moins deux fois la frequence dechantillonnage")
        b = 0.08  # Transition band, as a fraction of the sampling rate (in (0, 0.5)).
        N = int(np.ceil((4 / b)))  # le window
        if not N % 2:
            N += 1  # Make sure that N is odd.
        n = np.arange(N)
        h = np.sinc(fc * 2 * (n - (N - 1) / 2))
        w = 0.5 * (1 - np.cos(n * 2 * math.pi / (N - 1)))  # Compute hanning window.
        h = h * w
        h = h / np.sum(h)
        return torch.tensor(h)

    def init_filter_layer(self):
        """
        intialize the weights of the convolution in the NN,  so that it acts as a lowpass filter.
        the typefilter determines if the weights will be updated during the optim of the NN
        """


        window_size = 5
        C_in = 1
        stride=1
        padding = int(0.5*((C_in-1)*stride-C_in+window_size))+23

        # maybe the two functions do exactly the same...
        if self.modele_filtered in [0,1,3] :
            weight_init = self.get_filter_weights_en_dur()
        elif self.modele_filtered == 2:
            weight_init = self.get_filter_weights()

        weight_init = weight_init.view((1, 1, -1))
        lowpass = torch.nn.Conv1d(C_in,self.output_dim, window_size, stride=1, padding=padding, bias=False)

        if self.modele_filtered == 2: # 2 we let the weights move
            lowpass.weight = torch.nn.Parameter(weight_init,requires_grad= True)

        else : # 0 we don't care the filter won't be applied, 1 the wieghts are fixed
            lowpass.weight = torch.nn.Parameter(weight_init,requires_grad = False)

        lowpass = lowpass.double()
        self.lowpass = lowpass

    def filter_layer(self, y):
        """
        :param y: (B,L,18) articulatory prediction not smoothed
        :return:  smoothed articulatory prediction
        apply the convolution to each articulation, maybe not the best solution (changing n_channel of the conv layer ?)
        """
        B = len(y)
        L = len(y[0])
        y = y.double()
        y_smoothed = torch.zeros(B, L, self.output_dim)
        for i in range(self.output_dim):
            traj_arti = y[:, :, i].view(B, 1, L)
            traj_arti_smoothed = self.lowpass(traj_arti)
            traj_arti_smoothed = traj_arti_smoothed.view(B, L)
            y_smoothed[:, :, i] = traj_arti_smoothed
        return y_smoothed

    def plot_results(self, y, y_pred,y_pred_smooth = None,to_cons=[]):
        """
        :param y: one TRUE arti trajectory
        :param y_pred: one predicted arti trajectory not smoothed (forwoard with filtered=False)
        :param y_pred_smooth:  one predicted arti trajectory (smoothed)
        :param to_cons:  articulations available to consider (list of 0/1)
        save the graph of each available trajectory predicted and true
        """
        print("you chose to plot")
        plt.figure()
        idx_to_cons = [k for k in range(len(to_cons)) if to_cons[k]]
        for j in idx_to_cons:
            plt.figure()
           # plt.plot(y_pred[:, j],alpha=0.6)
            plt.plot(y_pred_smooth[:,j])
            plt.plot(y[:, j])
            plt.title("prediction_test_{0}_arti_{1}.png".format(self.name_file,str(j)))
            #plt.legend(["prediction", "vraie","pred smoothed"])
            plt.legend(["prediction", "vraie"])
            save_pics_path = os.path.join(
                "images_predictions\\{0}_arti_{1}.png".format(self.name_file,str(j)))
            plt.savefig(save_pics_path)
            plt.close('all')


    def evaluate_on_test(self,X_test,Y_test, std_speaker ,to_plot=False,to_consider=None,verbose=True):
        """
        :param X_test:  list of all the input of the test set
        :param Y_test:  list of all the target of the test set
        :param std_speaker : list of the std of each articulator, useful to calculate the RMSE of the predicction
        :param to_plot: wether or not we want to save some predicted smoothed and not and true trajectory
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
                    x_torch = x_torch.to(device = self.device)

                y_pred_passmooth = self(x_torch,False).double() #output y_pred (1,L,13)
                y_pred = self(x_torch,True).double()
                if self.cuda_avail:
                    y_pred_passmooth = y_pred_passmooth.cpu()
                    y_pred = y_pred.cpu()

                y_pred_passmooth = y_pred_passmooth.detach().numpy().reshape((L, self.output_dim))  # y_pred (L,13)
                y_pred= y_pred.detach().numpy().reshape((L, self.output_dim))  # y_pred (L,13)
                if to_plot:
                   if i in indices_to_plot:
                        self.plot_results(y, y_pred_passmooth, y_pred,to_consider)

                rmse = np.sqrt(np.mean(np.square(y - y_pred), axis=0))  # calculate rmse
                rmse = np.reshape(rmse, (1, self.output_dim))
                rmse = rmse*std_speaker #unormalize
                all_diff = np.concatenate((all_diff, rmse))
                pearson = [0]*self.output_dim
                for k in range(self.output_dim):
                    pearson[k] = np.corrcoef(y[:,k].T,y_pred[:,k].T)[0,1]
                pearson = np.array(pearson).reshape((1,self.output_dim))
                all_pearson = np.concatenate((all_pearson,pearson))
        all_pearson=all_pearson[1:]
        all_pearson[:,idx_to_ignore]=0
        all_diff = all_diff[1:]
        all_diff[:,idx_to_ignore] = 0
        pearson_per_arti_mean = np.mean(all_pearson, axis=0)
        rmse_per_arti_mean = np.mean(all_diff, axis=0)
        if verbose:
            print("rmse final : ", np.mean(rmse_per_arti_mean[rmse_per_arti_mean!=0]))
            print("rmse mean per arti : \n", rmse_per_arti_mean)
            print("pearson final : ", np.mean(pearson_per_arti_mean[rmse_per_arti_mean!=0]))
            print("pearson mean per arti : \n", pearson_per_arti_mean)
        return rmse_per_arti_mean,pearson_per_arti_mean



