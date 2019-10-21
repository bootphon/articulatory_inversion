# Inversion-articulatoire

Inversion-articulatoire is a Python library for training/testing neural network models for the acoustic to articulatory reconstruction.\
The task is the following : based on the acoustic signal of a speech, predict articulatory trajectories of the speaker.

It was created for learning the acoustic to articulatory mapping in a subject independent framework.\
For that we use data from 4 public datasets, that contain in all 19 speakers and more than 10 hours of acoustic and articulatory recordings.

It contains 3 main parts :<br/>
	- preprocessing that reads/cleans/preprocess/reorganize data\
	- Feed a neural network with our data. Training  on some speakers and testing on a speaker\
	- Perform articulatory predictions based on a wav file and a model (already trained)
    
The library enables evaluating the generalization capacity of a set of (or one) corpus.
 To do so we train the model in three different configurations. 
 For each configuration we evaluate the model through cross validation, considering successively the speakers as the test speaker, and averaging the results.
 The two configurations are the following ones:
 1) "speaker specific", we train and test on the same speaker. This configuration gives a topline of the results, and learn some characteristics of the speaker
 3) "speaker train independent", we train on all speakers EXCEPT the test-speaker and EXCEPT the validation speaker(s). 
 By analyzing how the scores decrease from configuration 1 to 2 we conclude on the generalization capacity.

# Dependencies
- python 3.7.3
- numpy 1.16.3
- pytorch 1.1.0
- scipy 1.2.1
- librosa 0.6.3
- matplotlib
- psutil

# Datasets
We used data coming from 4 different dataset, it is not necessary to use all of them. We suggest to use Haskins database, since it gives good results of prediction on a new speaker.
The data from the corpus have to be in the correct folders.
- mocha : http://data.cstr.ed.ac.uk/mocha/ <br/> , download data for the speakers : "fsew0", "msak0", "faet0", "ffes0", "maps0", "mjjn0", "falh0".
 for some speakers it is in the "unchecked folder"
- MNGU0 : http://www.mngu0.org/ <br/>
- usc : https://sail.usc.edu/span/usc-timit/<br/>
- Haskins : https://yale.app.box.com/s/cfn8hj2puveo65fq54rp1ml2mk7moj3h/folder/30415804819<br/>

Once downloaded and unzipped, all the folders should be in "Raw_data", and some folder names should be changed. More details are given in the part "usage".

We analyzed the datasets and found that some articulatory trajectories were wrong, here is a list of the articulatory trajectories we validated for each speaker (1 is for validated, 0 is for deleted):
 
 
 
 |  speaker  |  tt_x  |  tt_y  | td_x | td_y | tb_x | tb_y | li_x | li_y  | ul_x | ul_y | ll_x | ll_y | la | pro | ttcl | tbcl | v_x | v_y  | 
 | ------  |  --  |  -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | --  |
 | fsew0 | 1 | 1 | 1 | 1 | 1 | 0 | 0 | 1 | 0 | 0 | 1 | 1 | 0 | 0 | 1 | 0 | 1 | 1 | 
 | msak0 | 1 | 1 | 1 | 1 | 1 | 0 | 0 | 1 | 0 | 0 | 1 | 1 | 0 | 0 | 1 | 0 | 1 | 1 | 
 | maps0 | 1 | 1 | 1 | 1 | 1 | 0 | 0 | 1 | 0 | 0 | 1 | 1 | 0 | 0 | 1 | 0 | 0 | 0 | 
 | faet0 | 1 | 1 | 1 | 1 | 0 | 0 | 0 | 1 | 0 | 0 | 1 | 1 | 0 | 0 | 1 | 0 | 1 | 1 | 
 | mjjn0 | 1 | 1 | 1 | 1 | 1 | 0 | 0 | 1 | 0 | 0 | 1 | 1 | 0 | 0 | 1 | 0 | 0 | 0 | 
 | ffes0 | 1 | 1 | 1 | 1 | 1 | 0 | 0 | 1 | 0 | 0 | 1 | 1 | 0 | 0 | 1 | 0 | 1 | 1 | 
 | falh0 | 1 | 1 | 0 | 0 | 1 | 0 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 0 | 0 | 0 | 
 | MNGU0 | 1 | 1 | 0 | 1 | 0 | 1 | 1 | 1 | 0 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 0 | 0 | 
 | F1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 0 | 0 | 
 | F5 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 0 | 0 | 
 | M1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 0 | 0 | 
 | M3 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 0 | 0 | 
 | F01 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 0 | 0 | 
 | F02 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 0 | 0 | 
 | F03 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 0 | 0 | 
 | F04 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 0 | 0 | 
 | M01 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 0 | 0 | 
 | M02 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 0 | 0 | 
 | M03 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 0 | 0 | 
 | M04 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 0 | 0 | 


All this can be modified by changing the file articulators_per_speakers.csv that you can find in Preprocessing and Training.
# Contents

## Preprocessing :
- main_preprocessing.py : launches the preprocessing for each of the corpus (scripts preprocessing_namecorpus.py)

- class_corpus.py : a speaker class useful that contains common attributes to all/some speakers and preprocessing functions that are shared as well

- preprocessing_namecorpus.py : contains function to preprocess all the speaker's data of the corpus

- tools_preprocessing.py : functions that are used in the preprocessing. 


## Training 
- modele.py : the pytorch model, neural network bi-LSTM. Define the layers of the network, implementation of the smoothing convolutional layer, can evaluate the model on test set and plot some results.

- train.py : the training process of our project. We can modulate some parameters of the learning. Especially we can chose on which corpus(es). 
we train our model, and the dependancy level to the test speaker (ie one of the configurations : speaker specific, speaker dependent, speaker independent).
The loss is a combination of pearson and rmse : loss = (1-a)*rmse+a*(pearson).
Several parameters are optional and have default values. The required parameters are test_on, corpus_to_train_on, and config.
The optional parameters are : n_epochs, loss_train (a in the loss above, between 0 and 100), patience (before early stopping), 
select_arti (always yes, put gradients to 0 if arti is not available for this speaker) , batch_norma (whether to do a batch normalization),
 filter_type (inside the nn, outside with fix or not fixed weights), to_plot (whether to save some graphs of predicted and target traj), 
 lr (learning rate), delta_test (how often do we evaluate on validation set).

- test.py : tests a model (already trained) and saves some graph with the predicted & target articulatory trajectories. Also saves results in a csv files.
- experiments.py :  can perform different experiments by cross validation. The user chose the training set composed of n_speakers. 
Then for each set of parameters n_speakers models are trained : each time one speaker is left out from the training set to be test on. The results are averaged and saved.

# Usage
1) Data collect\
After the corpus data are downloaded, put them in a folder Raw_data and change some name folders to respect the following schema : 
- mocha :  for each speaker in ["fsew0", "msak0", "faet0", "ffes0", "maps0", "mjjn0", "falh0"] all the files are in Raw_data/mocha/speaker 
(same filename for same sentence, the extension indicates if its EMA, WAV, TRANS)
- MNGU0 : 3 folders in Raw_data/MNGU0 : ema, phone_labels and wav. In each folder all the sentences
- usc : for each speaker in ["F1","F5","M1","M3"] a folder  Raw_data/usc/speaker, then 3 folders : wav, mat, trans
- Haskins : for each speaker in ["F01", "F02", "F03", "F04", "M01", "M02", "M03", "M04"] a folder Raw_data/usc/speaker, then 3 folders : data, TextGrids, wav 

For the speaker falh0 (mocha database), we deleted files from 466 to 470 since there was an issue in the recording.

2) Preprocessing\
The script main_preprocessing takes 1 mandatory argument (the path to the raw data folder) and 2 optional arguments : corpus and N_max. N_max is max number of files to preprocess per speaker, N_max=0 means we want to preprocess all files (it is the default value)
Corpus is the list of corpus for which we want to do the preprocessing, default value is all the corpuses : ["mocha","Haskins","usc","MNGU0"].

To preprocess 50 files for each speaker and for all the corpuses: be in the folder "Preprocessing" and type the following in the command line 
```bash
python main_preprocessing.py  -path_to_raw path/to/parent/directory/of/Raw_data/ -N_max 50 
```
If you want to preprocess only the corpus "mocha" and "Haskins" , and preprocess all their data , then type 
```bash
python main_preprocessing.py --corpus ["mocha","Haskins"] 
```
The preprocessing of all the data takes about 6 hours. with a parallelization on 4 CPU

3) Training\
The script Train.py perform the training. The required parameters of the train function are those concerning the training/test set :
- the test-speaker : the speaker on which the model will be evaluated),
- the corpus we want to train on  : the list the corpus that will be in training set,
- the configuration : see above the description of the 3 configuration "indep","dep" or "spec",
The optional parameters are the parameters of the models itself.
To train on all the speakers except F01 of the Haskins corpus with the default parameters : be in the folder "Training" and type the following in the commandline
```bash
python train.py "F01" ["Haskins"] "indep"  
```

The model name contains information about the training/testing set , and some parameters for which we did experiments\.
The model name is constructed as follow\  
   speaker_test+"_"+config+"_"+name_corpus_concat+"loss_"+str(loss_train)+"_filter_"+str(filter_type)+"_bn_"+str(batch_norma)\
   with config either "indep","dep",or "spec", name corpus concat the list of the corpus to train on.
   loss train the a in the loss function described above, filter type either "out","fix" or "unfix", batch norma is a boolean\
The model weights are saved in "Training/saved_models". The name of the above model would be "F01_spec_loss_90_filter_fix_bn_False_0"\
If we train twice with the same parameters, a new model will be created with an increment in the suffix of the namefile.\
An exception is when the last model didn't end training, in that case the training continue for the model [no increment in the suffix of the namefile].\
At the end of the training (either by early stopping or n_epochs hit), the model is evaluated.
 It calculates the pearson and rmse mean per articulator, and add 2 rows in the csv files "results_models" (one for rmse, one for pearson) with the results.
 It also prints the results.


4) Perform inversion \
Supposed you have acoustic data (.wav) and you want to get the articulatory trajectories. \
The script predictions_arti takes 1 required argument : model_name, and 1 optional argument  : already_prepro.\
model_name is the  name of the model you want to use for the prediction. Be sure that it is in "Training\saved_models" as a .txt file.
The second argument --already_prepro is a boolean that is by default False, set it to true if you don't want to do the preprocess again.\

To perform the inversion : put the wav files in "Predictions_arti/my_wav_files_for_inversion". \
To launch both preprocessing and articulatory predictions with the model "F01_spec_loss_0_filter_fix_bn_False_0.txt",
 be in the folder "Predictions_arti" and type in the command line : 
```bash
python predictions_arti.py  F01_spec_loss_0_filter_fix_bn_False_0
```

If you  already did the preprocessing and want to test with another model :

```bash
python predictions_arti.py  F01_spec_loss_0_filter_fix_bn_False_0 --already_prepro True
```
 
The preprocessing will calculate the mfcc features corresponding to the wav files and save it in "Predictions_arti/my_mfcc_files_for_inversion"\
The predictions of the articulatory trajectories are as nparray in "Predictions_arti/name_model/my_articulatory_prediction" with the same name as the wav (and mfcc) files.\
Warning : usually the mfcc features are normalized at a speaker level when enough data for this speaker is available. The preprocessing normalize the mfcc coeff per file.

5) Modified ABX test


#  Experiments \
The function train.py only trains a model excluding ONE speaker and testing on it.\
For more significant results, one wants to average results obtained by cross validation excluding all the speakers one by one.\
The script Experiments enables to perform this cross_validation and save the result of the cross validation.\

 At the end of the experiment we can see in the csv file the result of each possibility and can compare those easily.


The script experiment.py takes 3 required arguments : corpus and experiment_type.
corpus is the list of the corpus we want to do the experiment on, for instance ["Haskins"] or ["mocha","usc"].
experiment_type is one of "cross" or "cross_spec"

# Test 

To test this models be in the folder "Training" and type this in the command line :
```bash
python test.py "fsew0" "fsew0_spec_loss_0_filter_fix_bn_False_0"
```
"fsew0" indicates the test speaker.  fsew0_spec_loss_0_filter_fix_bn_False_0 is the name of the model that should be in "Training/saved_models" (so change the name of saved_models_examples).\

The script will save in Training/images_prediction some graph. For one random test sentence it will plot and save the target and predicted trajectories for every articulators.\
 
The script will write the rmse, the rmse normlized and pearson result  in a csv file per articulator averaged over the test set. It also adds rows in the csv "results_models_test" with the rmse and pearson per articulator.

You can find the results we obtained here: https://docs.google.com/spreadsheets/d/172osaOYPxoxSziiU6evq4L0OlhEEKZ9bsq0ljmcFTRI/edit?usp=sharing
