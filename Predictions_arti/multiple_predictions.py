#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    Created october 2019
    Juliette MILLET
    Predict the articulatory trajectories corresponding to the wav files of ZeroSpeech2017.
    The user can chose the model for the prediction, here you can transform with multile models
    Computes the MFCC features corresponding to each wav file and save it in "Predictions_arti/mfcc_ZS2017_1s"
    Save the EMA trajectories in "Predictions_arti/name_model/ema_predictions_ZS2017_1s.

    The script also writes fea files to run the abx test easier.
"""

from predictions_ZS2017 import prediction_arti_ZS

if __name__ == '__main__':
    dico_output_dim = {'msak0':11, 'fsew0':11, 'M01': 16, 'M02': 16, 'M03':16, 'M04':16, 'F01':16, 'F02': 16, 'F03':16, 'F04': 16, 'F1': 16, 'F5':16, 'M1':16, 'M3':16, 'MNGU0':13}
    """models = ["only_arti_common_M01_train_indep_train_F02_F03_F04_M02_M03_M04_valid_F01_loss_50_filter_fix_bn_False_0",
              "only_arti_common_M02_train_indep_train_F02_F03_F04_M01_M03_M04_valid_F01_loss_50_filter_fix_bn_False_0",
              "only_arti_common_M03_train_indep_train_F02_F03_F04_M01_M02_M04_valid_F01_loss_50_filter_fix_bn_False_0",
              "only_arti_common_M04_train_indep_train_F02_F03_F04_M01_M02_M03_valid_F01_loss_50_filter_fix_bn_False_0",
              "only_arti_common_F01_train_indep_train_F03_F04_M01_M02_M03_M04_valid_F02_loss_50_filter_fix_bn_False_0",
              "only_arti_common_F02_train_indep_train_F03_F04_M01_M02_M03_M04_valid_F01_loss_50_filter_fix_bn_False_0",
              "only_arti_common_F03_train_indep_train_F02_F04_M01_M02_M03_M04_valid_F01_loss_50_filter_fix_bn_False_0",
              "only_arti_common_F04_train_indep_train_F02_F03_M01_M02_M03_M04_valid_F01_loss_50_filter_fix_bn_False_0"]"""
    models = ["only_arti_common_msak0_indep_train_fsew0_valid__loss_50_filter_fix_bn_False_0",
            "only_arti_common_fsew0_indep_train_msak0_valid__loss_50_filter_fix_bn_False_0",
            "only_arti_common_F1_train_indep_train_M1_M3_valid_F5_loss_50_filter_fix_bn_False_0",
            "only_arti_common_F5_train_indep_train_M1_M3_valid_F1_loss_50_filter_fix_bn_False_0",
            "only_arti_common_M1_train_indep_train_F5_M3_valid_F1_loss_50_filter_fix_bn_False_0",
            "only_arti_common_M3_train_indep_train_F5_M1_valid_F1_loss_50_filter_fix_bn_False_0"]
    for model in models:
        print(model)
        test_on = model.split('_')[3] # get the name of indiv it is tested on
        print('This model was tested on', test_on)
        prediction_arti_ZS(name_model = model, wav_folder = 'no_need', mfcc_folder = 'mfcc_ZS2017_1s/',
                           ema_folder = 'ema_ZS2017_1s/', fea_folder = test_on + '_train_indep', output_dim=dico_output_dim[test_on],
                           Nmax=0, prepro_done=True, predic_done=False)