
import numpy as np
import random
import os
from os.path import dirname
from Preprocessing.tools_preprocessing import get_speakers_per_corpus
root_folder = os.path.dirname(os.getcwd())



# file_names = file_names[0:30]
def split_sentences(speaker="MNGU0" ,max_length = 300,N="All"):
    """

    :param speaker:
    :param max_length: nombre de points max qu'on veut par phrase (durée = max_lenghts*100 secondes)
    :param N: ne s'occupe que des N premières phrases
    :return: rien.
    parcourt les fichiers dans "mfcc" et "ema_final". Si leur longueur est > max_length, alors il est divisé en K
    de telle sorte que chaque bout soit de longueur < max_length.
    - Attention les fichiers non splittées sont alors supprimées, et à la place K fichiers dans mfcc et dans ema_final.
    Si la phrase 80 est trop grande et doit être coupée en deux on n'aura plus les fichiers "mfcc_80" ni "ema_80" mais à
    la place on aura "mfcc_80_split_0", "mfcc_80_split_1" "ema_80_split_0" et "ema_80_split_1".
    - Attention les fichiers ema ne sont splittés que dans ema_final (ce sont ceux qui sont à terme utilsés pour l'Training).
    """
    Preprocessed_data_path = os.path.join(root_folder, "Preprocessed_data")
    file_names = os.listdir(os.path.join(Preprocessed_data_path, speaker, "ema_final"))
    file_names = [f for f in file_names if 'split' not in f]
    if N == "All":
        N = len(file_names)
    file_names = file_names[0:N]

    Number_cut = 0
    for f in file_names :
       # ema = np.load(os.path.join(Preprocessed_data_path,sp,"ema",f))
        mfcc = np.load(os.path.join(Preprocessed_data_path,speaker,"mfcc",f))
        #ema_filtered = np.load(os.path.join(Preprocessed_data_path,sp,"ema_filtered",f))
        ema_VT = np.load(os.path.join(Preprocessed_data_path,speaker,"ema_final",f))
        cut_in_N = int(len(mfcc)/max_length) +1
        if cut_in_N > 1 :
         #   print("cur in ",cut_in_N)
            Number_cut+=1
            temp = 0
            cut_size = int(len(mfcc)/cut_in_N)
            for k in range(cut_in_N-1) : # si k va jusqua 0 ca veut cut_in_N-1 vaut 1 cut_in_N vaut 2
                #print("k ,",k)
                mfcc_k = mfcc[temp : temp + cut_size]
        #        ema_k = ema[temp : temp + cut_size,:]
         #       ema_k_f = ema_filtered[temp : temp + cut_size,:]
                ema_k_vt = ema_VT[temp:temp+cut_size,:]

                temp = temp + cut_size
                np.save(os.path.join(Preprocessed_data_path,speaker,"mfcc",f[:-4]+"_split_"+str(k)),mfcc_k)
             #   np.save(os.path.join(Preprocessed_data_path,sp,"ema",f[:-4]+"_split_"+str(k)),ema_k)
             #   np.save(os.path.join(Preprocessed_data_path,sp,"ema_filtered",f[:-4]+"_split_"+str(k)),ema_k_f)
                np.save(os.path.join(Preprocessed_data_path,speaker,"ema_final",f[:-4]+"_split_"+str(k)),ema_k_vt)

            mfcc_last = mfcc[temp :]
          #  ema_last = ema[temp : ,:]
           # ema_last_f = ema_filtered[temp: , :]
            ema_last_vt = ema_VT[temp:, :]
            np.save(os.path.join(Preprocessed_data_path, speaker, "mfcc", f[:-4] + "_split_" + str(cut_in_N-1)), mfcc_last)
            #np.save(os.path.join(Preprocessed_data_path, sp, "ema", f[:-4] + "_split_" + str(cut_in_N-1)), ema_last)
            #np.save(os.path.join(Preprocessed_data_path, sp, "ema_filtered", f[:-4] + "_split_" + str(k)), ema_last_f)
            np.save(os.path.join(Preprocessed_data_path, speaker, "ema_final", f[:-4] + "_split_" + str(cut_in_N-1)), ema_last_vt)

            os.remove(os.path.join(Preprocessed_data_path,speaker,"mfcc",f))
            #os.remove(os.path.join(Preprocessed_data_path,sp,"ema",f))
            #os.remove(os.path.join(Preprocessed_data_path,sp,"ema_filtered",f))
            os.remove(os.path.join(Preprocessed_data_path,speaker,"ema_final",f))
   # print("number cut for ",speaker," : ",Number_cut)

#split_sentences("MNGU0")

#speakers = get_speakers_per_corpus("usc")
#for sp in speakers:
#    split_sentences(speaker = sp)

