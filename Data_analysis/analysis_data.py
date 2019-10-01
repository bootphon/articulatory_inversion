#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    Created august 2019
    by Juliette MILLET
    Script_to_analyze the data
"""
import os
import numpy as np
import plotly.graph_objs as go
import plotly.io as pio
mocha = ['faet0', 'falh0', 'ffes0', 'fsew0', 'maps0', 'mjjn0', 'msak0']
haskins = ['F01', 'F02', 'F03', 'F04', 'M01', 'M02', 'M03', 'M04']
arti = ['tt_x', 'tt_y', 'td_x', 'td_y', 'tb_x', 'tb_y', 'li_x', 'li_y',
     'ul_x', 'ul_y', 'll_x', 'll_y', 'la', 'lp', 'ttcl', 'tbcl', 'v_x', 'v_y']
def plot_arti(arti_name, list_speakers, folder_speaker, folder_data, treated_or_not):

    fig = go.Figure()
    for speaker in list_speakers:
        path_to = os.path.join(folder_speaker, speaker)
        path_to = os.path.join(path_to, folder_data)
        directory = os.fsencode(path_to)
        first = True
        count  = 0
        for file in os.listdir(directory):
            print(file)
            filename = os.fsdecode(file)
            if not filename.endswith(".npy") :
                continue
            else:
                ema = np.load(os.path.join(path_to, filename))
                ema_interesting = ema[:, arti.index(arti_name): arti.index(arti_name) + 1] # to keep dimensions
                if first:
                    concat = ema_interesting
                    first = False
                else:
                    concat = np.concatenate((concat, ema_interesting), axis = 0)
                count +=1
            if count > 2:
                break

        lenght = concat.shape[0]
        list_x = np.asarray(range(lenght))
        mark = "triangle"
        if speaker in mocha:
            mark = "circle"
        elif speaker in haskins:
            mark = "square"

        fig.add_scatter(
            x=list_x,
            y=concat[:,0],
            mode='markers+lines',
            name=speaker + ' ' + arti_name + ' ' + treated_or_not,
            showlegend=True,
            marker ={"symbol": mark}
        )
        concat = []
    #fig.show()
    pio.write_image(fig, 'figure_all_haskins_all_mocha_'+ arti_name + '_' +  treated_or_not + '.pdf')



for art in arti:
    plot_arti(art, ['F01', 'F02', 'F03', 'F04', 'M01', 'M02', 'M03', 'M04', 'faet0', 'falh0', 'ffes0', 'fsew0', 'maps0', 'mjjn0', 'msak0'],
              folder_speaker='/home/juliette/Documents/Thèse/Encadrement_stages/Maud_parrot/data_computed/Preprocessed_data/',
              folder_data='ema_final', treated_or_not ='treated')
    plot_arti(art,
              ['F01', 'F02', 'F03', 'F04', 'M01', 'M02', 'M03', 'M04', 'faet0', 'falh0', 'ffes0', 'fsew0', 'maps0',
               'mjjn0', 'msak0'],
              folder_speaker='/home/juliette/Documents/Thèse/Encadrement_stages/Maud_parrot/data_computed/Preprocessed_data/',
              folder_data='ema', treated_or_not = 'not_treated')