""" compute short term energy"""




import os
import time
from os.path import dirname
import numpy as np
import scipy.signal
from scipy import stats
import matplotlib.pyplot as plt
import scipy.interpolate
import librosa


import numpy
import scipy.signal

def _sgn(x):
  y = numpy.zeros_like(x)
  y[numpy.where(x >= 0)] = 1.0
  y[numpy.where(x < 0)] = -1.0
  return y

def stzcr(x, win):
  """Compute short-time zero crossing rate."""
  x1 = numpy.roll(x, 1)
  x1[0] = 0.0
  abs_diff = numpy.abs(_sgn(x) - _sgn(x1))
  return scipy.signal.convolve(abs_diff, win, mode="same")

def add_voicing(speaker):
    speaker_2 = speaker
    root_path = dirname(dirname(os.path.realpath(__file__)))
    if speaker in ["fsew0","msak0"]:
        speaker_2 = "mocha_"+speaker
        path_files = os.path.join(root_path, "Donnees_brutes", "mocha", speaker)
    elif speaker in ["F1"]:
        speaker_2 = "usc_timit_"+speaker
        path_files = os.path.join(root_path, "Donnees_brutes", "usc_timit", speaker,"wav")
    elif speaker == "MNGU0":
        path_files = os.path.join(root_path, "Donnees_brutes", speaker,"wav")
    wav_files = sorted([name[:-4] for name in os.listdir(path_files) if name.endswith('.wav')])
    N = len(wav_files)
    N=2
    for k in range(N):
        path_wav = os.path.join(path_files, wav_files[k] + '.wav')
        sampling_rate_mfcc = 16000
        frame_time = 25/1000
        hop_time = 10/1000  # en ms
        hop_length = int((hop_time * sampling_rate_mfcc))
        frame_length = int((frame_time * sampling_rate_mfcc) )
        data, sr = librosa.load(path_wav, sr=sampling_rate_mfcc)  # chargement de données
        N_frames = int(len(data)/hop_length)
    #    energy=[]
     #   for i in range(N_frames):
      #      start = i*hop_length
       #     frame_i = data[start : start + frame_length]
        #    energy_i = sum(frame_i*frame_i)
         #   energy.append(energy_i)
        window = scipy.signal.get_window("hanning", N_frames)
        ste = scipy.signal.convolve(data ** 2, window ** 2, mode="same")
        zero_cross = stzcr(data,window)
        print(len(ste),len(zero_cross))
        mfcc = np.load(os.path.join(root_path,"Donnees_pretraitees",speaker_2,"mfcc",wav_files[k]+".npy"))
       # energy_resampled = scipy.signal.resample(energy, num=len(mfcc))
        ste_resampled = scipy.signal.resample(ste,num=len(mfcc))/np.std(ste)
        zero_cross_resampled = scipy.signal.resample(zero_cross,num=len(mfcc))/np.std(zero_cross)

        voice_ste = ste_resampled > 0.5
        voice_zero_cross = zero_cross_resampled < 0.3
        plt.plot(ste_resampled)
        plt.plot(voice_ste)
       # plt.plot(voice_ste)
     #   plt.plot(zero_cross_resampled)
      #  plt.plot(voice_zero_cross)
        plt.legend(["ste","voicing"])
        plt.xlabel("frame")
        plt.title(speaker_2+" voicing detection                 ")
        plt.show()

### pour F1 encore compliqué car 1 wav contient plusieurs phrases qu'il faudrait diviser en phrases, peut être le faire dès le début et
### mettre le tout dans données prétraitées
#add_voicing("MNGU0")
#add_voicing("fsew0")
#add_voicing("msak0")


def voicing_detection(wav):
    hop_length = 25/1000
    N_frames = int(len(wav)/hop_length)
    window = scipy.signal.get_window("hanning", N_frames)
    ste = scipy.signal.convolve(wav ** 2, window ** 2, mode="same")
    voice_ste = ste > 0.5
    return voice_ste
