import torch
try :
    from Apprentissage.utils import load_filenames, load_data
    from Apprentissage.pytorchtools import EarlyStopping

except :
    from utils import load_filenames, load_data
    from pytorchtools import EarlyStopping


import numpy as np
import os
import math
root_folder = os.path.dirname(os.getcwd())
import sys
import matplotlib.pyplot as plt

class learn_velum(torch.nn.Module):
    def __init__(self, hidden_dim, input_dim, output_dim,name_file):
        super(learn_velum, self).__init__()
        self.first_layer = torch.nn.Linear(input_dim, hidden_dim)
        self.hidden_layer = torch.nn.Linear(hidden_dim, hidden_dim)
        self.readout_layer = torch.nn.Linear(hidden_dim*2, output_dim)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.lstm_layer =   torch.nn.LSTM(input_size=hidden_dim,
                                        hidden_size=hidden_dim, num_layers=1,
                                        bidirectional=True)

        self.name_file = name_file
        self.cutoff=20
        self.sampling_rate=200
        self.init_filter_layer()

    def forward(self,x):
        x_1 = torch.nn.functional.relu(self.first_layer(x))
        x_2 = torch.nn.functional.relu(self.hidden_layer(x_1))
        x_3,sr =self.lstm_layer(x_2)
        x_3 = torch.nn.functional.relu(x_3)
        y = self.readout_layer(x_3)
        y_smoothed = self.filter_layer(y)
        return y_smoothed

    def prepare_batch(self, x, y):
        max_length = np.max([len(phrase) for phrase in x])
        B = len(x)  # often batch size but not for validation

        new_x = torch.zeros((B, max_length, self.input_dim), dtype=torch.double)
        new_y = torch.zeros((B, max_length, self.output_dim), dtype=torch.double)
        for j in range(B):
            if len(x[j]) != len(y[j]):
                print("error size with ",j)
                print("mfcc",len(x[j]))
                print("ema",len(y[j]))

            zeropad = torch.nn.ZeroPad2d((0, 0, 0, max_length - len(x[j])))
            new_x[j] = zeropad(torch.from_numpy(x[j])).double()
            new_y[j] = zeropad(torch.from_numpy(y[j])).double()
        x = new_x.view((B, max_length, self.input_dim))
        y = new_y.view((B, max_length, self.output_dim))

        return x, y

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
            # print("2",h)
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
            # print("3",w)
            h = torch.mul(h, w)  # Multiply sinc filter with window.
            h = torch.div(h, torch.sum(h))
            # print("4",h)
            # h.require_grads = True
            #  self.cutoff = Variable(cutoff, requires_grad=True)
            #   self.cutoff.require_grads = True
            #   self.cutoff.retain_grad()
            #  h = torch.cat([h]*self.output_dim,0)
            return h

        def get_filter_weights_en_dur():
            fc = self.cutoff / self.sampling_rate
            if fc > 0.5:
                raise Exception("La frequence de coupure doit etre au moins deux fois la frequence dechantillonnage")

            b = 0.08  # Transition band, as a fraction of the sampling rate (in (0, 0.5)).
            N = int(np.ceil((4 / b)))  # le window
            if not N % 2:
                N += 1  # Make sure that N is odd.
            n = np.arange(N)
            # print("1",n)
            h = np.sinc(fc * 2 * (n - (N - 1) / 2))
            # print("2",h)
            w = 0.5 * (1 - np.cos(n * 2 * math.pi / (N - 1)))  # Compute hanning window.
            # print("3",w)

            h = h * w
            # print("4",h)
            h = h / np.sum(h)
            #            print("5",h)

            return torch.tensor(h)

        # print("1",self.cutoff)
        # self.cutoff = torch.nn.parameter.Parameter(torch.Tensor(self.cutoff))
        # self.cutoff.requires_grad = True
        window_size = 5
        C_in = 1
        stride = 1
        padding = int(0.5 * ((C_in - 1) * stride - C_in + window_size)) + 23
        lowpass = torch.nn.Conv1d(C_in, self.output_dim, window_size, stride=1, padding=padding, bias=False)
        weight_init = get_filter_weights_en_dur()
        weight_init = weight_init.view((1, 1, -1))
        lowpass.weight = torch.nn.Parameter(weight_init)
        lowpass = lowpass.double()
        self.lowpass = lowpass
        # print("lowpasse ",self.lowpass.weight)

        # self.lowpass.require_grads=True

    def filter_layer(self, y):
        B = len(y)  # batch size
        L = len(y[0])
        #     y= y.view(self.batch_size,self.output_dim,L)
        y = y.double()
        y_smoothed = torch.zeros(B, L, self.output_dim)
        for i in range(self.output_dim):
            traj_arti = y[:, :, i].view(B, 1, L)
            #  print("traj arti shape",traj_arti.shape)
            traj_arti_smoothed = self.lowpass(traj_arti)  # prend que une seule dimension
            difference = int((L - traj_arti_smoothed.shape[2]) / 2)
            if difference != 0:
                print("PAS MEME SHAPE AVANT ET APRES FILTRAGE !")
                print("init L", L)
                print("after smoothed ", traj_arti_smoothed.shape[2])
            if difference > 0:  # si la traj smoothed est plus petite que L on rajoute le meme dernier élément
                traj_arti_smoothed = torch.nn.ReplicationPad1d(difference)(traj_arti_smoothed)
            elif difference < 0:  # si la traj smoothed est plus petite que L on rajoute le meme dernier élément
                traj_arti_smoothed = traj_arti_smoothed[:, :, 0:L]

            traj_arti_smoothed = traj_arti_smoothed.view(B, L)
            y_smoothed[:, :, i] = traj_arti_smoothed
        return y_smoothed

    def evaluate(self, x_valid, y_valid, criterion):
        x_temp, y_temp = self.prepare_batch(x_valid, y_valid)  # add zero to have correct size
        y_pred = self(x_temp).double()
        y_temp = y_temp.double()
        loss = criterion(y_temp, y_pred).item()
        return loss

    def evaluate_on_test(self, criterion, verbose=False,X_test=None,Y_test=None,to_plot=False,
                         std_ema = 1 ,suffix= ""):
        all_diff = np.zeros((1, self.output_dim))
        all_pearson = np.zeros((1, self.output_dim))
        indices_to_plot=[]
        if to_plot :
            print("you chose to plot")
            indices_to_plot = np.random.choice(len(X_test), 2, replace=False)
        loss_test= 0
        for i in range(len(X_test)):
                L = len(X_test[i])
                x_torch = torch.from_numpy(X_test[i]).view(1,L,self.input_dim)  #x (1,L,429)
                y = Y_test[i].reshape((L, self.output_dim))                     #y (L,13)
                y_torch = torch.from_numpy(y).double().reshape(1,L,self.output_dim) #y (1,L,13)
                y_pred_torch = self(x_torch).double() #sortie y_pred (1,L,13)
                y_pred = y_pred_torch.detach().numpy().reshape((L, self.output_dim))  # y_pred (L,13)
                #the_loss = criterion(y_torch, y_pred_torch)  #loss entre données de taillees  (1,L,13)
                #loss_test += the_loss.item()
                if i in indices_to_plot:
                    self.plot_results(y, y_pred, suffix=suffix + str(i))

                #rmse = np.sqrt(np.mean(np.square(y - y_pred), axis=0))  # calcule du rmse à la main
               # rmse = np.reshape(rmse, (1,self.output_dim)) #dénormalisation et taille (1,13)
              #  all_diff = np.concatenate((all_diff, rmse))

                pearson = [0]*self.output_dim
                for i in range(self.output_dim):
                    pearson[i]= np.corrcoef(y[:,i].T,y_pred[:,i].T)[0,1]
                pearson = np.array(pearson).reshape((1,self.output_dim))
                pearson[np.isnan(pearson)] = 1
                all_pearson = np.concatenate((all_pearson,pearson))

     #   all_diff = all_diff[1:] #remove first row of zeros #all the errors per arti and per sample
        all_pearson=all_pearson[1:]
        if verbose :
            rmse_per_arti_mean = np.mean(all_diff,axis=0)*std_ema
         #   print("rmse final : ", np.mean(rmse_per_arti_mean))
          #  print("rmse mean per arti : \n", rmse_per_arti_mean)
            pearson_per_arti_mean = np.mean(all_pearson, axis=0)
            print("pearson final : ", np.mean(pearson_per_arti_mean))
           # print("pearson mean per arti : \n", pearson_per_arti_mean)

    def plot_results(self, y, y_pred, suffix=""):
        plt.figure()
        for j in range(self.output_dim):
            plt.figure()
            plt.plot(y_pred[:, j])
            plt.plot(y[:, j])
            plt.title("prediction_test_{0}_{1}_arti{2}.png".format(self.name_file, suffix, str(j)))
            plt.legend(["prediction", "vraie"])
            save_pics_path = os.path.join(
                "images_predictions\\{0}_{1}_arti{2}.png".format(self.name_file, suffix, str(j)))
            plt.savefig(save_pics_path)
            plt.close('all')


def train_learn_velum(n_epochs=10,patience=5):
    input_dim = 429
    output_dim = 2
    hidden_dim = 200
    lr=0.001
    name_file = "modele_velum"

    model = learn_velum(hidden_dim,input_dim,output_dim,name_file).double()
    model_dict = model.state_dict()
    batch_size=10
    data_filtered=True
    optimizer = torch.optim.Adam(model.parameters(), lr=lr )
   # criterion = torch.nn.MSELoss(reduction='sum')

    def criterion_pearson(y, y_pred):  # (L,K,13)
        y_1 = y - torch.mean(y, dim=1, keepdim=True)  # (L,K,13) - (L,1,13) ==> utile ? normalement proche de 0
        y_pred_1 = y_pred - torch.mean(y_pred, dim=1, keepdim=True)

        nume = torch.sum(y_1 * y_pred_1, dim=1,
                         keepdim=True)  # y*y_pred multi terme à terme puis on somme pour avoir (L,1,13)
        # pour chaque trajectoire on somme le produit de la vriae et de la predite
        deno = torch.sqrt(torch.sum(y_1 ** 2, dim=1, keepdim=True)) * torch.sqrt(
            torch.sum(y_pred_1 ** 2, dim=1, keepdim=True))  # use Pearson correlation
        # deno zero veut dire ema constant à 0 on remplace par des 1
        minim = torch.tensor(0.01, dtype=torch.float64)



        deno = torch.max(deno, minim)
        loss = nume / deno
        loss = torch.sum(loss)

        return -loss
    criterion = criterion_pearson

    speakers= ["fsew0","msak0","faet0","ffes0"]

    early_stopping = EarlyStopping(name_file, patience=patience, verbose=True )
    N= 460 * len(speakers)  # velum for mocha whith 460 sentences
    n_iterations = int(N*0.8/batch_size)
    n_iterations_valid = int(N*0.2/batch_size)
    delta_test=1
    file_weights = os.path.join("saved_models", "modele_velum.txt")
    loaded_state = torch.load(file_weights, map_location=torch.device('cpu'))
    loaded_state = {k: v for k, v in loaded_state.items() if
                    k in model_dict}  # only layers param that are in our current model

    loaded_state = {k: v for k, v in loaded_state.items() if
                    loaded_state[k].shape == model_dict[k].shape}  # only if layers have correct shapes
    model_dict.update(loaded_state)
    model.load_state_dict(model_dict)

    for epoch in range(n_epochs):
        for ite in range(n_iterations) :

            files_for_train = load_filenames(speakers, batch_size, part=["train"])
            x, y = load_data(files_for_train, filtered=data_filtered)
            y = [y[i][:,-2:] for i in range(len(y))]
            x, y = model.prepare_batch(x, y)
            y_pred = model(x).double()
            y = y.double()
            optimizer.zero_grad()
            loss = criterion(y, y_pred)
            loss.backward()
            optimizer.step()

        if epoch%delta_test == 0:
            loss_vali = 0
            files_for_valid = load_filenames(speakers, int(N*0.2), part=["valid"])
            x, y = load_data(files_for_valid, filtered=data_filtered,VT=False)
            y = [y[i][:, -2:] for i in range(len(y))]
            loss_vali = model.evaluate(x, y, criterion)
            early_stopping(loss_vali, model)

            print("epoch : {}".format(epoch))
            print("loss train : {}".format(loss.item()))
            print("loss vali : {}".format(loss_vali))

        if early_stopping.early_stop:
            print("Early stopping")
            break

    model.load_state_dict(torch.load(os.path.join("saved_models", name_file + '.pt')))
    torch.save(model.state_dict(), os.path.join("saved_models", name_file + ".txt"))

    for speaker in speakers:
        files_for_test = load_filenames([speaker], int(0.2*N), part=["test"])
        x, y = load_data(files_for_test, filtered=data_filtered)
        y = [y[i][:, -2:] for i in range(len(y))]
        print("evaluation on speaker {}".format(speaker))
        speaker_2 = speaker
        if speaker in ["F1", "M1", "F5"]:
            speaker_2 = "usc_timit_" + speaker
        std_speaker = np.load(os.path.join(root_folder, "Traitement", "norm_values","std_ema_" + speaker_2 + ".npy"))
        std_speaker = std_speaker[:output_dim]
        model.evaluate_on_test(criterion=criterion, verbose=True, X_test=x, Y_test=y,
                               to_plot=True, std_ema=max(std_speaker), suffix=speaker)



if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Train and save a model.')
    parser.add_argument('n_epochs', metavar='n_epochs', type=int,
                        help='nombre depochs')

    parser.add_argument('patience', metavar='patience', type=int,
                        help='patience')

    args = parser.parse_args()
    n_epochs = int(sys.argv[1])
    patience = int(sys.argv[2])
    train_learn_velum(n_epochs=n_epochs,patience=patience)
