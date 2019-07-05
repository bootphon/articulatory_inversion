
import numpy as np
import random
import os
from os.path import dirname

root_folder = os.getcwd()

donnees_pretraitees_path = os.path.join(os.path.dirname(os.getcwd()),"Donnees_pretraitees")

speakers = os.listdir(donnees_pretraitees_path)
speakers.remove("fileset")
max_length = 500
def split_in_2(ema,mfcc):
    to_cut = int(len(mfcc) / 2)
    mfcc_1 = mfcc[:to_cut]
    ema_1 = ema[:to_cut, :]

for sp in speakers :
    print("speaker ",sp)
    file_names = os.listdir(os.path.join(donnees_pretraitees_path,sp,"ema"))
    for f in file_names :
        ema = np.load(os.path.join(donnees_pretraitees_path,sp,"ema",f))
        mfcc = np.load(os.path.join(donnees_pretraitees_path,sp,"mfcc",f))
        ema_filtered = np.load(os.path.join(donnees_pretraitees_path,sp,"ema_filtered",f))
        ema_VT = np.load(os.path.join(donnees_pretraitees_path,sp,"ema_VT",f))

        too_long = (len(mfcc)>max_length)

        while too_long :
            to_cut = int(len(mfcc)/2)
            mfcc_1 = mfcc[:to_cut]
            ema_1 = ema[:to_cut,:]
            ema_fil_1 = ema_filtered[:to_cut,:]
            ema_vt_1 = ema_VT[:to_cut,:]
            mfcc_2 = mfcc[to_cut:]
            ema_2 = ema[to_cut:,:]






