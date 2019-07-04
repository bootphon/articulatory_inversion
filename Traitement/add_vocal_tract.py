
import os
import librosa
import numpy as np
import scipy
import torch
from Apprentissage.velum_modele import learn_velum

import matplotlib.pyplot as plt
articulators = [
        'tt_x', 'tt_y', 'td_x', 'td_y', 'tb_x', 'tb_y', 'li_x', 'li_y',
        'ul_x', 'ul_y', 'll_x', 'll_y']


articulators_after = [
        'tt_x', 'tt_y', 'td_x', 'td_y', 'tb_x', 'tb_y', 'li_x', 'li_y',
        'ul_x', 'ul_y', 'll_x', 'll_y','la','pro','lar','v_x','v_y'] #détailler

root_folder = os.path.dirname(os.getcwd())


def add_vocal_tract(speaker):
    print("adding vocal tracts for speaker {}".format(speaker))
    def add_lip_aperture(ema):
        ind_1, ind_2 = [articulators.index("ul_y"), articulators.index("ll_y")]
        lip_aperture = ema[:, ind_1] - ema[:, ind_2]  # upperlip_y - lowerlip_y
        ema = np.insert(ema, 12, 1, axis=1)
        ema[:, 12] = lip_aperture
        return ema
    def add_lip_protrusion(ema):
        ind_1, ind_2 = [articulators.index("ul_x"), articulators.index("ll_x")]
        lip_protrusion = (ema[:, ind_1] + ema[:, ind_2]) / 2
        ema = np.insert(ema, 13, 1, axis=1)
        ema[:, 13] = lip_protrusion
        return ema
    def add_voicing(wav,ema,sr):

        hop_time = 10 / 1000  # en ms
        hop_length = int((hop_time * sr))
        N_frames = int(len(wav) / hop_length)
        window = scipy.signal.get_window("hanning", N_frames)
        ste = scipy.signal.convolve(wav ** 2, window ** 2, mode="same")
        ste = scipy.signal.resample(ste, num=len(ema))
        ste = [min(x, 1) for x in ste]
        ema = np.insert(ema, 13, 1, axis=1)
        ema[:, 14] = ste
        return ema

    def add_velum(mfcc,ema):
        if len(mfcc)!= len(ema):
            print("pbm mfcc ema",mfcc.shape,ema.shape)
        model = learn_velum(hidden_dim=200, input_dim=429, output_dim=2, name_file="modele_velum").double()
        model_to_load = os.path.join(root_folder, "Apprentissage", "saved_models", "modeles_valides",
                                     "modele_velum.txt")

        loaded_state = torch.load(model_to_load)
        model.load_state_dict(loaded_state)
        mfcc_2 = torch.from_numpy(mfcc).view((1, len(mfcc), len(mfcc[0])))
        velum_xy = model(mfcc_2).double()
        velum_xy = velum_xy.detach().numpy().reshape((len(mfcc), 2))
        ema = np.insert(ema, (14,14), 1, axis=1)
        ema[:, 13:15] = velum_xy
        return ema

    if speaker in ["msak0", "fsew0","maps0","faet0","mjjn0","ffes0"]:
        speaker_2 = "mocha_" + speaker
        wav_path = os.path.join(root_folder, "Donnees_brutes","mocha", speaker)
        sampling_rate_wav = 16000

    elif speaker in ["F1", "F5", "M1","M3"]:
        speaker_2 = "usc_timit_" + speaker
        wav_path = os.path.join(root_folder, "Donnees_brutes", "usc_timit",speaker,"wav_cut")
        sampling_rate_wav = 20000

    elif speaker == "MNGU0":
        speaker_2 = speaker
        wav_path = os.path.join(root_folder, "Donnees_brutes", speaker,"wav")
        sampling_rate_wav= 16000
    mfcc_path = os.path.join(root_folder, "Donnees_pretraitees", speaker_2,"mfcc")
    files_path = os.path.join(root_folder, "Donnees_pretraitees", speaker_2,"ema_filtered")

    if not os.path.exists(os.path.join(root_folder,"Donnees_pretraitees",speaker_2,"ema_VT")):
        os.makedirs(os.path.join(root_folder,"Donnees_pretraitees",speaker_2,"ema_VT"))

    EMA_files_names = sorted(
        [name[:-4] for name in os.listdir(files_path) if name.endswith('.npy')])
    N = len(EMA_files_names)

    for i in range(N):

        if i%10 ==0:
            print("{} out of {}".format(i, N))
        ema = np.load(os.path.join(files_path,EMA_files_names[i]+".npy"))
        ema = add_lip_aperture(ema)
        ema = add_lip_protrusion(ema)

        if speaker in ["fsew0","msak0","faet0","ffes0"] :
            wav,sr = librosa.load(os.path.join(wav_path, EMA_files_names[i] + ".wav"),sr = sampling_rate_wav)
            ema = add_voicing(wav,ema,sampling_rate_wav)

        elif speaker in ["MNGU0","maps0","mjjn0"]:
            wav, sr = librosa.load(os.path.join(wav_path, EMA_files_names[i] + ".wav"), sr=sampling_rate_wav)
            ema = add_voicing(wav, ema, sampling_rate_wav)
            mfcc = np.load(os.path.join(mfcc_path,EMA_files_names[i]+".npy"))
            ema = add_velum(mfcc,ema)

        elif speaker in ["F1","F5","M1","M3"]:
            wav = np.load(os.path.join(wav_path, EMA_files_names[i] + ".npy"))
            mfcc = np.load(os.path.join(mfcc_path,EMA_files_names[i]+".npy"))
            if len(ema)!= len(mfcc):
                print("pbm shape",len(ema),len(mfcc),EMA_files_names[i])
            ema = add_voicing(wav,ema,sampling_rate_wav)
            ema = add_velum(mfcc,ema)

        np.save(os.path.join(root_folder, "Donnees_pretraitees", speaker_2, "ema_VT",EMA_files_names[i]),ema)

speakers = ["MNGU0","fsew0","msak0","F1","F5","M1","M3","maps0","faet0",'mjjn0',"ffes0"]

for sp in speakers :
    add_vocal_tract(sp)

