#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    Created october 2019
    by Juliette MILLET
    Function to evaluate a model that was already trained : on data the model never saw, calculate the rmse and
    pearson for the prediction made by this model.
    Here we do it for multiple models
"""

from test_modified import test_model

if __name__ == '__main__':
    """models = ["only_arti_common_M01_train_indep_train_F02_F03_F04_M02_M03_M04_valid_F01_loss_50_filter_fix_bn_False_0",
              "only_arti_common_M02_train_indep_train_F02_F03_F04_M01_M03_M04_valid_F01_loss_50_filter_fix_bn_False_0",
              "only_arti_common_M03_train_indep_train_F02_F03_F04_M01_M02_M04_valid_F01_loss_50_filter_fix_bn_False_0",
              "only_arti_common_M04_train_indep_train_F02_F03_F04_M01_M02_M03_valid_F01_loss_50_filter_fix_bn_False_0",
              "only_arti_common_F01_train_indep_train_F03_F04_M01_M02_M03_M04_valid_F02_loss_50_filter_fix_bn_False_0",
              "only_arti_common_F02_train_indep_train_F03_F04_M01_M02_M03_M04_valid_F01_loss_50_filter_fix_bn_False_0",
              "only_arti_common_F03_train_indep_train_F02_F04_M01_M02_M03_M04_valid_F01_loss_50_filter_fix_bn_False_0",
              "only_arti_common_F04_train_indep_train_F02_F03_M01_M02_M03_M04_valid_F01_loss_50_filter_fix_bn_False_0",
              "only_arti_common_msak0_indep_train_fsew0_valid__loss_50_filter_fix_bn_False_0",
                "only_arti_common_fsew0_indep_train_msak0_valid__loss_50_filter_fix_bn_False_0",
                "only_arti_common_F1_train_indep_train_M1_M3_valid_F5_loss_50_filter_fix_bn_False_0",
                "only_arti_common_F5_train_indep_train_M1_M3_valid_F1_loss_50_filter_fix_bn_False_0",
                "only_arti_common_M1_train_indep_train_F5_M3_valid_F1_loss_50_filter_fix_bn_False_0",
                "only_arti_common_M3_train_indep_train_F5_M1_valid_F1_loss_50_filter_fix_bn_False_0"]"""

    """models = ["only_arti_common_M01_spec_train__valid__loss_50_filter_fix_bn_False_0",
              "only_arti_common_M02_spec_train__valid__loss_50_filter_fix_bn_False_0",
              "only_arti_common_M03_spec_train__valid__loss_50_filter_fix_bn_False_0",
             "only_arti_common_M04_spec_train__valid__loss_50_filter_fix_bn_False_0",
             "only_arti_common_F01_spec_train__valid__loss_50_filter_fix_bn_False_0",
              "only_arti_common_F02_spec_train__valid__loss_50_filter_fix_bn_False_0",
              "only_arti_common_F03_spec_train__valid__loss_50_filter_fix_bn_False_0",
              "only_arti_common_F04_spec_train__valid__loss_50_filter_fix_bn_False_0",
              "only_arti_common_F1_spec_train__valid__loss_50_filter_fix_bn_False_0",
              "only_arti_common_F5_spec_train__valid__loss_50_filter_fix_bn_False_0",
              "only_arti_common_M1_spec_train__valid__loss_50_filter_fix_bn_False_0",
              "only_arti_common_M3_spec_train__valid__loss_50_filter_fix_bn_False_0",
              "only_arti_common_MNGU0_spec_train__valid__loss_50_filter_fix_bn_False_0",
              "only_arti_common_msak0_spec_train__valid__loss_50_filter_fix_bn_False_0",
              "only_arti_common_fsew0_spec_train__valid__loss_50_filter_fix_bn_False_0"]"""
    models = ["only_arti_common_MNGU0_train_indep_train_M02_M03_M04_F01_F02_F03_F04_F1_F5_M3_msak0_valid_M01_fsew0_M1_loss_50_filter_fix_bn_False_0",
              "MNGU0_train_indep_Haskins_usc_mocha_MNGU0_loss_50_filter_fix_bn_False_0",
              "only_arti_common_M01_train_indep_train_F02_F03_F04_M02_M03_M04_valid_F01_loss_50_filter_fix_bn_False_0"]
    for model in models:
        test_model('MNGU0', model, test_on_per_default =False)
        test_model('M01', model)
        test_model('fsew0', model)
        test_model('M1', model)