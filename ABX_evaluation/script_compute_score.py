#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    Code created october 2019
    by Juliette MILLET
    compute new ABX score
"""
import pandas as pd
import numpy as np
from utils import conversion_arpa_ipa

voiced_unvoiced = ['b:p','f:v','ð:θ','d:t','s:z','ʃ:ʒ','tʃ:dʒ','g:k']
oral_nasal = ['m:p','b:m','m:w','n:t','d:n','n:s','n:z','l:n','k:ŋ','g:ŋ','ŋ:w']
h_vowel = ['eɪ:h','h:ɪ','h:i','h:oʊ','h:ɔɪ','h:u']
contrast_to_exclude = voiced_unvoiced + oral_nasal + h_vowel

def result_score(file_model_results, nb_example = 3, bad = False):
    to_exclude = ['ao', 'aw', 'ax', 'ay', 'er']
    df = pd.read_csv(file_model_results, sep='\t')
    # aggregate on speakers
    groups = df.groupby(
        ['phone_1', 'phone_2', 'by'], as_index=False)
    df2 = groups['score'].mean()

    # aggregate on context
    groups2 = df2.groupby(['phone_1', 'phone_2'], as_index=False)
    df3 = groups2['score'].mean()

    groups3 = df.groupby(['phone_1', 'phone_2'], as_index = False)
    df_sum = groups3['n'].sum()

    val = df3.values
    dico_mean = {}
    for k in range(len(val)):
        if val[k][1] + ':' + val[k][0] not in dico_mean.keys(): # we exclude some phones that can be wrong
            dico_mean[val[k][0] + ':' + val[k][1]] = [val[k][2]]
        else:
            dico_mean[val[k][1] + ':' + val[k][0]].append(val[k][2])

    val_new = []
    f = open('file_pair.csv', 'w')
    for cont in dico_mean.keys():
        phone1 = cont.split(':')[0]
        phone2 = cont.split(':')[1]
        trad_1 = conversion_arpa_ipa(phone1)
        trad_2 = conversion_arpa_ipa(phone2)
        if trad_1 == 'g' or trad_2 == 'g':
            print(cont)
        # we compute the contraste to exclude
        if not bad:
            excl = trad_1 + ':' + trad_2 in contrast_to_exclude or trad_2 + ':' + trad_1 in contrast_to_exclude
        else:
            excl = trad_1 + ':' + trad_2 not in contrast_to_exclude and trad_2 + ':' + trad_1 not in contrast_to_exclude
        if len(dico_mean[cont]) > 1 and phone1 not in to_exclude and phone2 not in to_exclude and not excl: # we take only contrast symetric

            f.write(conversion_arpa_ipa(phone1) +':'+ conversion_arpa_ipa(phone2) + '\n')
            val_new.append([cont, np.array(dico_mean[cont]).mean()])


    val_new = sorted(val_new, key=lambda x: x[1])


    count = 0
    contrast_done = []
    mean_error = 0
    for i in range(len(val_new)):
        if val_new[i][0] not in contrast_done:
            contrast_done.append(val_new[i][0])
            phone1 = val_new[i][0].split(':')[0]
            phone2 = val_new[i][0].split(':')[1]
            extract = df_sum.loc[((df_sum['phone_1'] == phone1) & (df_sum['phone_2'] == phone2))].values
            n = extract[0][2]
            extract2 = df_sum.loc[((df_sum['phone_1'] == phone1) & (df_sum['phone_2'] == phone2))].values
            n2 = extract2[0][2]
            if n >= nb_example and n2>=nb_example: # we take only the contrast that have at least nb_example context
                mean_error += 1. - val_new[i][1]
                count += 1

    mean_error = mean_error / float(count)
    return mean_error

if __name__ == '__main__':
    import argparse
    import os
    parser = argparse.ArgumentParser(description='Compute the modified ABX score')
    parser.add_argument('score_folder', type=str,
                        help='original abx score folder given by Zerospeech 2017 score computation')
    parser.add_argument('result_file', type=str,
                        help='csv file to keep the results')
    args = parser.parse_args()

    f = open(args.result_file, 'a')
    f.write(','.join(['model', 'ok within', 'bad within', 'ok across', 'bad across']) + '\n')
    mean_ok_within= result_score(file_model_results=os.path.join(args.score_folder,'tmp/within.csv'), nb_example=3, bad = False)
    mean_bad_within = result_score(file_model_results=os.path.join(args.score_folder,'tmp/within.csv'), nb_example=3, bad = True)
    mean_ok_across = result_score(file_model_results=os.path.join(args.score_folder,'tmp/across.csv'), nb_example=3, bad = False)
    mean_bad_across = result_score(file_model_results=os.path.join(args.score_folder,'tmp/across.csv'), nb_example=3, bad = True)
    f.write(','.join([data, str(mean_ok_within), str(mean_bad_within), str(mean_ok_across), str(mean_bad_across)]) + '\n')
