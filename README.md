# Inversion-articulatoire

Inversion-articulatoire is a Python library for training/test neural network models of articulatory reconstruction.
the task : based on the acoustic signal guess 18 articulatory trajectories. 
It was created for learning the acoustic to articulatory mapping in a subject independant framework.
For that we use data from 4 public datasets, that contains in all 19 speakers and more than 10 hours of acoustic and articulatory recordings.

It contains 2 main parts :  \\
	- preprocessing that reads/cleans/preprocess/reorganize data \\
	- Feed a neural network with our data. Training  on some speakers and testing on a speaker


# Dependencies
numpy
tensorflow (not used but tensorboard can be used)
pytorch
scipy
librosa

# Requirements
The data from the 4 corpuses have to be in the correct folders.
[ put the URL Links and explain quickly]

# Contents

## Preprocessing : 
main_preprocessing.py : launches the preprocessing for each of the corpuses (scripts preprocessing_namecorpus.py)
class_corpus.py : a speaker class useful that contains common attributes to all/some speakers and preprocessing functions that are shared as well.
preprocessing_namecorpus.py : contains function to preprocess all the speaker's data of the corpus.
tools_preprocessing.py : functions that are used in the preprecessing. S

## Training :
1) modele.py : the pytorch model, neural network bi-LSTM. Define the layers of the network, implementation of the smoothing convolutional layer, can evaluate on test set the model
2) train.py : the training process of our project. We can modulate some parameters of the learning. Especially we can chose on which corpus(es) we train our model, and the dependancy level to the test speaker.
3 main configurations : speaker specific, speaker dependant, speaker independant.  
Several parameters are optional and have default values. The required parameters are test_on, corpus_to_train_on, and config.
3) test.py : test the model (without any training) and saving some graph with the predicted & target articulatory trajectories.
experiments : ...

# Usage
1) download the data and put them in the correct folders.
2) Preprocessing : python main_preprocessing.py 0 (for a test put 50)
3) Training : python train.py "F01" ["Haskins"] "indep"    [ this is to train on all the speakers except F01 of the Haskins corpus with the default parameters ]


