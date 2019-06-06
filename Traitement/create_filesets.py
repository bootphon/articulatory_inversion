
import numpy as np
import random
import os
from os.path import dirname
from sklearn.model_selection import train_test_split

speaker = 'fsew0'

max_lenght = 500
donnees_path = os.path.join(os.path.dirname(os.getcwd()), "Donnees_pretraitees")
fileset_path = os.path.join(donnees_path, "fileset")

def create_fileset(speaker):
    speaker_2 = speaker
    if speaker in ["msak0","fsew0"]:
        speaker_2 = "mocha_"+speaker
    X = []
    Y = []
    EMA_files_path  = os.path.join(donnees_path,speaker_2,"ema")
    EMA_files_names = sorted([name[:-4] for name in os.listdir(EMA_files_path) if name.endswith('.npy')])
    MFCC_files_path = os.path.join(donnees_path,speaker_2,"mfcc")
    k=0
    N=len(EMA_files_names)

    for i in range(N):
        if i%300==0:
            print("{} files treated out of {}".format(i,len(EMA_files_names)))
        try :
            the_ema_file = np.load(os.path.join(EMA_files_path, EMA_files_names[i]+".npy"))
            the_mfcc_file = np.load(os.path.join(MFCC_files_path,EMA_files_names[i]+".npy"))
            #print(the_ema_file.shape)
        except:
            print("cant find this mfcc file : {)".format(EMA_files_names[i]))

#        print("1",len(the_ema_file))
        while len(the_ema_file)>max_lenght+30:
            k+=1
            to_cut = int(len(the_ema_file)/2)
            the_ema_file_part_1 = the_ema_file[:to_cut]
           # print("ema_file_part_1", the_ema_file_part_1.shape)

            the_ema_file = the_ema_file[to_cut:]

            the_mfcc_file_part_1 = the_mfcc_file[:to_cut]
            the_mfcc_file = the_mfcc_file[to_cut:]

            X.append(the_mfcc_file_part_1)
            Y.append(the_ema_file_part_1)
       # print("3",len(the_ema_file))
        if len(the_ema_file)<10:
            print("too short")
        X.append(the_mfcc_file)
        #print("ema_file", len(the_ema_file))
        Y.append(the_ema_file)

    print("nombre de split :",k)
    print(min([len(X[i]) for i in range(N)]))
    if not os.path.exists(fileset_path):
        os.makedirs(fileset_path)
    np.save(os.path.join(fileset_path,"X_"+speaker),X)
    np.save(os.path.join(fileset_path,"y_"+speaker),Y)
    print("FILESET CREATED FOR SPEAKER {}".format(speaker))

#create_fileset("fsew0")
#create_fileset("msak0")
#create_fileset("MNGU0")

def create_fileset_ZS():
    path_files = os.path.join(os.path.dirname(os.getcwd()), "Donnees_pretraitees\donnees_challenge_2017\\1s")
    wav_files = sorted([name[:-4] for name in os.listdir(path_files) if name.endswith('.npy')])
    def concat_all_numpy_from(path):
        list = []
        for r, d, f in os.walk(path):
            for file in f:
                data_file = np.load(os.path.join(path, file))
                list.append(data_file)
        return list
    all_numpy_ZS = concat_all_numpy_from(path_files)
    np.save(os.path.join(fileset_path, "X_ZS"), all_numpy_ZS)

#create_fileset_ZS()

def create_test_train_OLD(model_name):
    fileset_folder = os.path.join(donnees_path, "fileset_" + model_name)
    if not os.path.exists(fileset_folder):
        os.makedirs(fileset_folder)
    if model_name in ['fsew0', 'msak0', 'MNGU0']:
        print("speaker is unique")
        X = np.load(os.path.join(fileset_path, "X_" + speaker + ".npy"), allow_pickle=True)
        Y = np.load(os.path.join(fileset_path, "Y_" + speaker + ".npy"), allow_pickle=True)

        _OLDpourcent_test = 0.2

        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=pourcent_test, random_state=1)
        print("dim train test :", len(X_train), len(X_train))

    elif model_name == "train_on_two_test_msak0":
        X_train_2 = list(np.load(os.path.join(fileset_path, "X_MNGU0.npy"), allow_pickle=True))
        X_train_1 = list(np.load(os.path.join(fileset_path, "X_fsew0.npy"), allow_pickle=True))
        X_train = np.array(X_train_1 + X_train_2)
        X_test = list(np.load(os.path.join(fileset_path, "X_msak0.npy"), allow_pickle=True, encoding='latin1'))
        Y_train_1 = list(np.load(os.path.join(fileset_path, "y_fsew0.npy"), allow_pickle=True, encoding='latin1'))
        Y_train_2 = list(np.load(os.path.join(fileset_path, "y_MNGU0.npy"), allow_pickle=True, encoding='latin1'))
        Y_train = np.array(Y_train_1 + Y_train_2)
        Y_test = list(np.load(os.path.join(fileset_path, "y_msak0.npy"), allow_pickle=True, encoding='latin1'))

    elif model_name == "train_test_on_three":
        pourcent_test = 0.2
        X_1 = list(np.load(os.path.join(fileset_path, "X_fsew0.npy"), allow_pickle=True))
        Y_1 = list(np.load(os.path.join(fileset_path, "Y_fsew0.npy"), allow_pickle=True))
        X_1_tr,X_1_te,Y_1_tr,Y_1_te = train_test_split(X_1, Y_1, test_size=pourcent_test, random_state=1)

        X_2 = list(np.load(os.path.join(fileset_path, "X_MNGU0.npy"), allow_pickle=True))
        Y_2 = list(np.load(os.path.join(fileset_path, "Y_MNGU0.npy"), allow_pickle=True))
        X_2_tr,X_2_te,Y_2_tr,Y_2_te = train_test_split(X_2, Y_2, test_size=pourcent_test, random_state=1)

        X_3 = list(np.load(os.path.join(fileset_path, "X_msak0.npy"), allow_pickle=True))
        Y_3 = list(np.load(os.path.join(fileset_path, "Y_msak0.npy"), allow_pickle=True))
        X_3_tr,X_3_te,Y_3_tr,Y_3_te = train_test_split(X_3, Y_3, test_size=pourcent_test, random_state=1)

        X_train = np.array(X_1_tr + X_2_tr+ X_3_tr)
        Y_train = np.array(Y_1_tr + Y_2_tr+ Y_3_tr)
        X_test = np.array(X_1_te + X_2_te+ X_3_te)
        Y_test = np.array(Y_1_te + Y_2_te+ Y_3_te)


    elif model_name == "train_fsew0_test_msak0":
        X_train = list(np.load(os.path.join(fileset_path, "X_fsew0.npy"), allow_pickle=True))
        X_test = list(np.load(os.path.join(fileset_path, "X_msak0.npy"), allow_pickle=True, encoding='latin1'))
        Y_train = np.array(list(np.load(os.path.join(fileset_path, "y_fsew0.npy"), allow_pickle=True, encoding='latin1')))
        Y_test = list(np.load(os.path.join(fileset_path, "y_msak0.npy"), allow_pickle=True, encoding='latin1'))
    print("train and test calculés")
    np.save(os.path.join(fileset_folder,"X_train.npy"),X_train)
    np.save(os.path.join(fileset_folder,"X_test.npy"),X_test)
    np.save(os.path.join(fileset_folder,"Y_train.npy"),Y_train)
    np.save(os.path.join(fileset_folder,"Y_test.npy"),Y_test)
    print("train and test sauvegardés pour ",model_name)

#create_test_train("train_fsew0_test_msak0")
#create_test_train("train_test_on_three")
#create_test_train("train_on_two_test_msak0")

def create_test_train():
    fileset_path = os.path.join(donnees_path, "fileset")
    for speaker in ["fsew0","msak0",'MNGU0']:
        X = np.load(os.path.join(fileset_path, "X_" + speaker + ".npy"), allow_pickle=True)
        Y = np.load(os.path.join(fileset_path, "Y_" + speaker + ".npy"), allow_pickle=True)
        pourcent_test = 0.2
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=pourcent_test, random_state=1)
        np.save(os.path.join(fileset_path, "X_train_"+speaker+".npy"), X_train)
        np.save(os.path.join(fileset_path, "Y_train_"+speaker+".npy"), Y_train)
        np.save(os.path.join(fileset_path, "X_test_"+speaker+".npy"), X_test)
        np.save(os.path.join(fileset_path, "Y_test_"+speaker+".npy"), Y_test)

#create_test_train()