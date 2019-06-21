
import sys
import torch
import os
import sys
import time
sys.path.append("..")
from Apprentissage.class_network import my_bilstm

import numpy as np

def prediction_ZS(name_model,Nmax = 20,start=0):
    def concat_all_numpy_from(path):
        list = []
        for r, d, f in os.walk(path):
            for file in f:
                data_file = np.load(os.path.join(path, file))
                list.append(data_file)
        return list

    cuda_avail = torch.cuda.is_available()
    print("cuda avail?",cuda_avail)
    for time in ['1s'] : #,'120s']:
        print("---time",time)
        #sys.path.insert(0, os.path.dirname(os.getcwd()))
        root_folder = os.path.dirname(os.getcwd())
        model_to_load = os.path.join(root_folder,"Apprentissage","saved_models", "modeles_valides",name_model+".txt")
        model = my_bilstm(hidden_dim=300, input_dim=429, output_dim=13, batch_size=10,
                          name_file=name_model)
        model = model.double()
        if not cuda_avail:
            device = torch.device('cpu')
            loaded_state = torch.load(model_to_load, map_location=device)

        else:
            loaded_state = torch.load(model_to_load)

        model.load_state_dict(loaded_state)


        path_mfcc_treated = os.path.join(root_folder,"Donnees_pretraitees","donnees_challenge_2017",time)
        path_prediction_ema = os.path.join(root_folder,"Predictions_arti",time,name_model)
        mfcc_files = sorted([name[:-4] for name in os.listdir(path_mfcc_treated) if name.endswith('.npy')])
        def write_fea_file(prediction,filename):
            """
            :param prediction: array avec la prédiction des trajectoires pour une phrase,
            :param filename:  nom de fichier en question, ie le nom qu'aura le fichier dans prediction
            :return:  rien - mais sauvegarde un fichier dans  1s/filename.fea avec les prédictions
            dans le bon format pour le text abx puisse tourner dessus
            """
            prediction_with_time = np.zeros((prediction.shape[0],prediction.shape[1]+1))
            prediction_with_time[:,1:] = prediction
            frame_hop = 0.010
            frame_lenght = 0.025
            all_times = [frame_lenght/2+ frame_hop*i for i in range(prediction.shape[0])]
            prediction_with_time[:,0]=all_times
            lines =[ ' '.join(str(ema) for ema in prediction_with_time[i]) for i in range(len(prediction_with_time))]
            with open(os.path.join(path_prediction_ema,"fea",filename+".fea"), 'w') as f:
                f.writelines("%s\n" % l for l in lines)

        if not os.path.exists(path_prediction_ema):
            os.makedirs(path_prediction_ema)
        if not os.path.exists(os.path.join(path_prediction_ema,"fea")):
            os.makedirs(os.path.join(path_prediction_ema,"fea"))
        if not os.path.exists(os.path.join(path_prediction_ema, "npy")):
            os.makedirs(os.path.join(path_prediction_ema, "npy"))
        if Nmax== 'All':
            Nmax = len(mfcc_files)
        print(path_mfcc_treated)
        if time == "1s":
            delta_show = 50
        elif time == "120s":
            delta_show = 5
        for i in range(start,Nmax):
            if not os.path.exists(os.path.join(path_prediction_ema,mfcc_files[i]+".npy")):
                if i%delta_show==0:
                    print("{} out of {}".format(i,Nmax))
                x = np.load(os.path.join(path_mfcc_treated,mfcc_files[i]+".npy"))
                x_2 = torch.from_numpy(x).view((1,len(x),len(x[0])))
                y_pred = model(x_2)
                y_pred = y_pred.detach().numpy().reshape((len(x),13))
                write_fea_file(y_pred,mfcc_files[i])
                np.save(os.path.join(path_prediction_ema,"npy",mfcc_files[i]+".npy"),y_pred)
      #  Y_ZS = concat_all_numpy_from(os.path.join(path_prediction_ema,"npy")) #,extension=".npy")
       # np.save(os.path.join(path_prediction_ema,"Y_ZS"),Y_ZS)
        #X_ZS = concat_all_numpy_from(path_mfcc_treated) #,extension =  ".npy")
        #np.save(os.path.join(path_prediction_ema, "X_ZS"), X_ZS)

models = ["train_fsew0_msak0_test_fsew0_msak0_data_filtered_modele_filtered",
          ]
print(sys.argv)
start = int(sys.argv[1])
Nmax = sys.argv[2]
model = int(sys.argv[3])


print("chosent : ",start,Nmax,models[model])
prediction_ZS(models[model],Nmax=Nmax,start=start)