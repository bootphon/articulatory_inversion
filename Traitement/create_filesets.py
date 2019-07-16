
import numpy as np
import random
import os
from os.path import dirname
from sklearn.model_selection import train_test_split
from Apprentissage.utils import low_pass_filter_weight
from random import shuffle


max_lenght = 800
root_folder = os.path.dirname(os.getcwd())
donnees_path = os.path.join(root_folder, "Donnees_pretraitees")
fileset_path = os.path.join(donnees_path, "fileset")
fileset_path_non_decoupes = os.path.join(donnees_path, "fileset_non_decoupes")

def create_fileset_OLD(speaker):
    """
    :param speaker: pour le moment fsew0,msak0 ou MNGU0
    :return: rien
    lit tous les numpy mfcc et ema correspondant au speaker et les concatène pour avoir une liste de toutes les données.
    On normalise les mfcc et les ema (soustrait moyenne et divise par écart type)
    On divise en deux les phrases plus longues que 800 frames mfcc.

    On a donc une liste X et une liste Y, qu'on va diviser en train et test.
    """
    speaker_2 = speaker
    if speaker in ["msak0","fsew0"]:
        speaker_2 = "mocha_"+speaker

    if speaker in ["F1", "F5","M1"]:
        speaker_2 = "usc_timit_" + speaker

    X = []
    Y = []

    files_path =  os.path.join(donnees_path,speaker_2)
    #EMA_files_path  = os.path.join(donnees_path,speaker_2,"ema")
    EMA_files_names = sorted([name[:-4] for name in os.listdir(os.path.join(files_path,"ema")) if name.endswith('.npy') ])
    #MFCC_files_path = os.path.join(donnees_path,speaker_2,"mfcc")
    k=0
    N=len(EMA_files_names)
    for i in range(N):
        if i%100==0:
            print("{} files treated out of {}".format(i,len(EMA_files_names)))
        try :
            the_ema_file = np.load(os.path.join(os.path.join(files_path,"ema"), EMA_files_names[i]+".npy"))
            the_mfcc_file = np.load(os.path.join(os.path.join(files_path,"mfcc"),EMA_files_names[i]+".npy"))
         #print(the_ema_file.shape)
        except:
            print("cant find this mfcc file : {)".format(EMA_files_names[i]))

#        print("1",len(the_ema_file))
        while len(the_ema_file)>max_lenght+30:
            k+=1
            #print("before",the_ema_file.shape)
            to_cut = int(len(the_ema_file)/2)
            the_ema_file_part_1 = the_ema_file[:to_cut]
           # print("ema_file_part_1", the_ema_file_part_1.shape)
            the_ema_file = the_ema_file[to_cut:]
          #  print("after",the_ema_file.shape)
            the_mfcc_file_part_1 = the_mfcc_file[:to_cut]
            the_mfcc_file = the_mfcc_file[to_cut:]
            X.append(the_mfcc_file_part_1)
            Y.append(the_ema_file_part_1)
        X.append(the_mfcc_file)
        Y.append(the_ema_file)
    if not os.path.exists(fileset_path):
        os.makedirs(fileset_path)
    pourcent_test = 0.2
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=pourcent_test, random_state=1)
    np.save(os.path.join(fileset_path, "X_train_" + speaker + ".npy"), X_train)
    np.save(os.path.join(fileset_path, "Y_train_" + speaker + ".npy"), Y_train)
    np.save(os.path.join(fileset_path, "X_test_" + speaker + ".npy"), X_test)
    np.save(os.path.join(fileset_path, "Y_test_" + speaker + ".npy"), Y_test)
    print("FILESET CREATED FOR SPEAKER {}".format(speaker))

def get_fileset_names(speaker):
    """
    :param speaker: pour le moment fsew0,msak0 ou MNGU0
    :return: rien
    lit tous les numpy mfcc et ema correspondant au speaker et les concatène pour avoir une liste de toutes les données.
    On normalise les mfcc et les ema (soustrait moyenne et divise par écart type)
    On divise en deux les phrases plus longues que 800 frames mfcc.

    On a donc une liste X et une liste Y, qu'on va diviser en train et test.
    """

    if speaker in ["msak0","fsew0","maps0","faet0","mjjn0","ffes0","falh0"]:
        speaker_2 = "mocha_"+speaker

    elif speaker == "MNGU0":
        speaker_2 = speaker

    elif speaker in ["F1", "F5", "M1","M3"]:
        speaker_2 = "usc_timit_" + speaker

    elif speaker in ["F01","F02","F03","F04","M01","M02","M03","M04"]:
        speaker_2 = "Haskins_" + speaker

    files_path =  os.path.join(donnees_path,speaker_2)
    EMA_files_names = [name[:-4] for name in os.listdir(os.path.join(files_path,"ema_filtered")) if name.endswith('.npy') ]
    N = len(EMA_files_names)
    shuffle(EMA_files_names)
    pourcent_train = 0.7
    pourcent_test=0.2
    n_train = int(N*pourcent_train)
    n_test  = int(N*pourcent_test)
    train_files = EMA_files_names[:n_train]
    test_files = EMA_files_names[n_train:n_train+n_test]
    valid_files = EMA_files_names[n_train+n_test:]

    outF = open(os.path.join(root_folder,"Donnees_pretraitees","fileset",speaker+"_train.txt"), "w")
    outF.write('\n'.join(train_files) + '\n')
    outF.close()

    outF = open(os.path.join(root_folder, "Donnees_pretraitees", "fileset", speaker + "_test.txt"), "w")
    outF.write('\n'.join(test_files) + '\n')
    outF.close()

    outF = open(os.path.join(root_folder, "Donnees_pretraitees", "fileset", speaker + "_valid.txt"), "w")
    outF.write('\n'.join(valid_files) + '\n')
    outF.close()


def get_fileset_names_per_corpus(corpus):
    """
    :param speaker: pour le moment fsew0,msak0 ou MNGU0
    :return: rien
    lit tous les numpy mfcc et ema correspondant au speaker et les concatène pour avoir une liste de toutes les données.
    On normalise les mfcc et les ema (soustrait moyenne et divise par écart type)
    On divise en deux les phrases plus longues que 800 frames mfcc.

    On a donc une liste X et une liste Y, qu'on va diviser en train et test.
    """

    if corpus == "MNGU0":
        speakers = ["MNGU0"]
    elif corpus == "usc":
        speakers = ["F1", "F5", "M1", "M3"]
    elif corpus == "Haskins":
        speakers = ["F01", "F02", "F03", "F04", "M01", "M02", "M03", "M04"]

    elif corpus == "mocha":
        speakers = ["fsew0", "msak0", "faet0", "ffes0", "maps0", "mjjn0", "falh0"]
    else:
        print("vous navez pas choisi un des corpus")

    for sp in speakers:
        get_fileset_names(sp)


speakers =  ["F01","F02","F03","F04","M01","M02","M03","M04","F1","F5","M1","M3"
    ,"maps0","faet0",'mjjn0',"ffes0","msak0","falh0","MNGU0"]
for sp in speakers :
    get_fileset_names(sp)

 #   print("speaker :",sp)

#create_fileset("fsew0")
#create_fileset("msak0")
#create_fileset("MNGU0")

def create_fileset_ZS():
    path_files = os.path.join(os.path.dirname(os.getcwd()), "Donnees_pretraitees\donnees_challenge_2017\\1s")
    wav_files = sorted([name[:-4] for name in os.listdir(path_files) if name.endswith('.npy')])
    def concat_all_numpy_from(path):
        list = []
        for r, d,  f in os.walk(path):
            for file in f:
                data_file = np.load(os.path.join(path, file))
                list.append(data_file)
        return list
    all_numpy_ZS = concat_all_numpy_from(path_files)
    np.save(os.path.join(fileset_path, "X_ZS"), all_numpy_ZS)

#create_fileset_ZS()

def filter_filesets(speaker):

    Y_test = np.load(os.path.join(fileset_path, "Y_test_" + speaker + ".npy"))
    Y_train = np.load(os.path.join(fileset_path, "Y_train_" + speaker + ".npy"))

    if speaker in ["fsew0","msak0"]:
        sampling_rate= 500
        cutoff=30
    elif speaker == "MNGU0":
        sampling_rate=200
        cutoff= 25

    weights = low_pass_filter_weight(cut_off= cutoff, sampling_rate=sampling_rate)

    def filter_seq(y):

        filtered_data_ema = np.concatenate([np.expand_dims(np.convolve(channel, weights,mode="same"), 1)
                                       for channel in y.T], axis=1)
        difference = len(filtered_data_ema) - len(y)
        halfdif = int(difference/2)
        if difference <0: #sequence filtree moins longue que l'originale
            filtered_data_ema = np.pad(filtered_data_ema,(halfdif,difference-halfdif),"edge")
        elif difference > 0:
            filtered_data_ema = filtered_data_ema[halfdif:-(difference-halfdif)]
        if len(filtered_data_ema)!=len(y): #sequence filtree plus longue que loriginale
             print("pbm shape",len(filtered_data_ema),len(y))
        return filtered_data_ema

    Y_test_filtered=[ filter_seq(Y_test[i]) for i in range(len(Y_test))]
    Y_train_filtered = [filter_seq(Y_train[i]) for i in range(len(Y_train))]

    np.save(os.path.join(fileset_path, "Y_test_filtered_" + speaker + ".npy"), Y_test_filtered)
    np.save(os.path.join(fileset_path, "Y_train_filtered_" + speaker + ".npy"), Y_train_filtered)

#filter_filesets("fsew0")
#filter_filesets("msak0")
#filter_filesets("MNGU0")