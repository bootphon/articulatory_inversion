#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    Code created october 2019
    by Juliette MILLET
    small functions useful
"""


def conversion_arpa_ipa(symbol_arpa):
    Arpabet_dict = dict({"AA": "ɑ", "AE": "æ", "AH": "ʌ", "AO": "ɔ", "AW": "aʊ", "AX": "ə", "AXR": "ɚ", "AY": "aɪ",
                         "EH": "ɛ", "ER": "ɝ", "EY": "eɪ", "IH": "ɪ", "IX": "ɨ", "IY": "i", "OW": "oʊ", "OY": "ɔɪ",
                         "UH": "ʊ", "UW": "u", "UX": "ʉ", "B": "b", "CH": "tʃ",
                         "D": "d", "DH": "ð", "DX": "ɾ", "EL": "l̩", "EM": "m̩", "EN": "n̩", "F": "f", "G": "g",
                         "HH": "h", "JH": "dʒ", "K": "k",
                         "L": "l", "M": "m", "N": "n", "NG": "ŋ", "NX": "ɾ̃", "P": "p", "Q": "ʔ", "R": "ɹ", "S": "s",
                         "SH": "ʃ", "T": "t",
                         "TH": "θ", "V": "v", "W": "w", "WH": "ʍ", "Y": "j", "Z": "z", "ZH": "ʒ"})

    if symbol_arpa.upper() not in Arpabet_dict:
        print("ERROR symbole not recognised", symbol_arpa)
    else:
        return Arpabet_dict[symbol_arpa.upper()]