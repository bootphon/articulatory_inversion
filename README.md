# Inversion-articulatoire

Inversion-articulatoire is a Python library for training/test neural network models of articulatory reconstruction.\
the task : based on the acoustic signal guess 18 articulatory trajectories. \
It was created for learning the acoustic to articulatory mapping in a subject independant framework.\
For that we use data from 4 public datasets, that contains in all 19 speakers and more than 10 hours of acoustic and articulatory recordings.

It contains 2 main parts :\ 
	- preprocessing that reads/cleans/preprocess/reorganize data\
	- Feed a neural network with our data. Training  on some speakers and testing on a speaker

The library enables evaluating the generalization capacity of a set of (or one) corpus.
 To do so we train the model in three different configurations. 
 For each configuration we evaluate the model through cross validation, considering successively the speakers as the test speaker, and averaging the results.
 1) "speaker specific", we train and test on the same speaker. This configuration gives a topline of the results, and learn some characteristics of the speaker
 2) "speaker dependent", we train on all speakers (including the test speaker). 
 3) "speaker independant", we train on all speakers EXCEPT the test-speaker. We discover the test-speaker at the evaluation of the model. 
 By analyzing how the scores decrease from configuration 1 to 3 we conclude on the generalization capacity.

# Dependencies
numpy\
tensorflow (not used but tensorboard can be used)\
pytorch\
scipy\
librosa

# Requirements
The data from the 4 corpuses have to be in the correct folders.\
mocha : http://data.cstr.ed.ac.uk/mocha/\
MNGU0 : http://www.mngu0.org/\
usc : https://sail.usc.edu/span/usc-timit/\
Haskins : https://yale.app.box.com/s/cfn8hj2puveo65fq54rp1ml2mk7moj3h/folder/30415804819



# Contents

## Preprocessing :
1) main_preprocessing.py : launches the preprocessing for each of the corpuses (scripts preprocessing_namecorpus.py)
2) class_corpus.py : a speaker class useful that contains common attributes to all/some speakers and preprocessing functions that are shared as well.
3)preprocessing_namecorpus.py : contains function to preprocess all the speaker's data of the corpus.
4)tools_preprocessing.py : functions that are used in the preprocessing. 

## Training 
1) modele.py : the pytorch model, neural network bi-LSTM. Define the layers of the network, implementation of the smoothing convolutional layer, can evaluate on test set the model

2) train.py : the training process of our project. We can modulate some parameters of the learning. Especially we can chose on which corpus(es) we train our model, and the dependancy level to the test speaker.
3 main configurations : speaker specific, speaker dependent, speaker independent.  
Several parameters are optional and have default values. The required parameters are test_on, corpus_to_train_on, and config.

3) test.py : test the model (without any training) and saving some graph with the predicted & target articulatory trajectories.
experiments : ...


# Usage
1) Download the data 
Change some name folders to respect the following schema : in Inversion_articulatoire/Raw_data
mocha :  for each speaker in ["fsew0", "msak0", "faet0", "ffes0", "maps0", "mjjn0", "falh0"] all the files are in Raw_data/mocha/speaker (same filename for same sentence, the extension indicates if its EMA, WAV, TRANS)
MNGU0 : 3 folders in Raw_data/MNGU0 : ema, phone_labels and wav. In each folder all the sentences 
usc : for each speaker in ["F1","F5","M1","M3"] a folder  Raw_data/usc/speaker, then 3 folders : wav, mat, trans  
Haskins : for each speaker in ["F01", "F02", "F03", "F04", "M01", "M02", "M03", "M04"] a folder Raw_data/usc/speaker, then 3 folders : data, TextGrids, wav 

2) Preprocessing 
The argument is for the max number of files you want to preprocess, to preprocess ALL files put 0. This argument is useful for test.
To preprocess 50 files for each speaker type the following in the command line
```bash
python main_preprocessing.py 50 
```

3) Training 
The required parameters of the train function are those concerning the training/test set :\
- the test-speaker : the speaker on which the model will be evaluated),\
-  the corpus we want to train on  : the list the corpus that will be in training set
- the configuration : see above the description of the 3 configuration "indep","dep" or "spec".
The optional parameters are the parameters of the models itself.
To train on all the speakers except F01 of the Haskins corpus with the default parameters  type the following in the commandline
```bash
python train.py "F01" ["Haskins"] "indep"  
```

# Results 
To compare to the state of the art, speaker specific results on fsew0 and msak0 :
[table with results]

First use of the corpus Haskins that provides better results, and very good on generalization 

![Alt text](C:\Users\Maud Parrot\Documents\stages\STAGE LSCP\Images_rapport\spec_F04_lly.png "Real and prediction trajectory of lowerlip y of F04 when trained on F04")



Some plot of trajectories



