
import sys
import torch
import os
import sys
import time
sys.path.append("..")
from Training.model import my_ac2art_model

import numpy as np
import librosa
from Preprocessing.tools_preprocessing import get_delta_features
import argparse

root_folder = os.path.dirname(os.getcwd())


def preprocess_my_wav_files():
    """
    Read all the wav files in "my_wav_files_for_inversion" and preprocess them the extract their acoustic features,
    so that it can be used as input of the my_ac2art model.
    Save the mfcc in "my_mfcc_files_for_inversion" , with the same filename as the corresponding wav.
    Warning : the acoustic features are usually normalized at the speaker level when enough data is available for
    the speaker.
    We let future users modify the code to apply this normalization (coeff = (coeff-meancoeff)/stdcoeff  )
    """
    path_wav = os.path.join(root_folder, "Predictions_arti", "my_wav_files_for_inversion")
    if not os.path.exists(os.path.join(root_folder,"Predictions_arti","my_mfcc_files_for_inversion")):
        os.mkdir(os.path.join(root_folder,"Predictions_arti","my_mfcc_files_for_inversion"))
    frame_time = 25 / 1000
    hop_time = 10 / 1000
    sampling_rate_wav_wanted = 16000
    hop_length = int(hop_time * sampling_rate_wav_wanted)
    frame_length = int(frame_time * sampling_rate_wav_wanted)
    window = 5
    n_coeff = 13
    wav_files = os.listdir(path_wav)
    for filename in wav_files:
        filename = filename[:-4]  #remove extension
        wav, sr = librosa.load(os.path.join(path_wav,filename+".wav"), sr=sampling_rate_wav_wanted)  # chargement de données
        wav = 0.5 * wav / np.max(wav)
        mfcc = librosa.feature.mfcc(y=wav, sr=sr, n_mfcc=n_coeff, n_fft=frame_length, hop_length=hop_length).T
        dyna_features = get_delta_features(mfcc)
        dyna_features_2 = get_delta_features(dyna_features)
        mfcc = np.concatenate((mfcc, dyna_features, dyna_features_2), axis=1)
        padding = np.zeros((window, mfcc.shape[1]))
        frames = np.concatenate([padding, mfcc, padding])
        full_window = 1 + 2 * window
        mfcc = np.concatenate([frames[j:j + len(mfcc)] for j in range(full_window)], axis=1)  # add context
        np.save(os.path.join(root_folder, "Predictions_arti","my_mfcc_files_for_inversion", filename), mfcc)



def predictions_arti(model_name):
    """
    :param model_name: name of model we want to use for the articulatory predictions
    with the weights in model_name, this script perform articulatory predictions corresponding to the wav files
    it takes as input the mfcc features already calculated
    the arti predictions are saved my "my_articulatory_prediction" as np array (K,18)
    """

    if not os.path.exists(os.path.join(root_folder,"Predictions_arti","my_articulatory_prediction")):
        os.mkdir(os.path.join(root_folder,"Predictions_arti","my_articulatory_prediction"))

    hidden_dim = 300
    input_dim = 429
    batch_size = 10
    output_dim = 18

    filter_type = "fix"
    batch_norma = False  # future work : read from model name if true or false
    model = my_ac2art_model(hidden_dim=hidden_dim, input_dim=input_dim, output_dim=output_dim,
                            batch_size=batch_size, cuda_avail=False, name_file=model_name,
                            filter_type=filter_type, batch_norma=batch_norma)
    model = model.double()
    file_weights = os.path.join(root_folder,"Training","saved_models", model_name + ".txt")
    loaded_state = torch.load(file_weights, map_location="cpu")
    model.load_state_dict(loaded_state)

    all_my_mfcc_files = os.listdir(os.path.join(root_folder,"Predictions_arti","my_mfcc_files_for_inversion"))

    for mfcc_file in all_my_mfcc_files :
        mfcc = np.load(os.path.join(root_folder,"Predictions_arti","my_mfcc_files_for_inversion",mfcc_file))
        mfcc_torch = torch.from_numpy(mfcc).view(1, -1, input_dim)
        ema_torch = model(mfcc_torch)
        ema = ema_torch.detach().numpy().reshape((-1,18))
        np.save(os.path.join(root_folder,"Predictions_arti","my_articulatory_prediction",mfcc_file),ema)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Articulatory predictions for some wav files')

    parser.add_argument('--model', type=str, default = "F01_spec_loss_0_filter_fix_bn_False_0",
                        help='name of the model we want to use (without .txt)')

    parser.add_argument('--already_prepro', type=bool, default=False,
                        help='put to True if preprocessin already done for the wav files')

    args = parser.parse_args()
    if not(args.already_prepro):
        print("preprocessing...")
        preprocess_my_wav_files()
    predictions_arti(args.model)







