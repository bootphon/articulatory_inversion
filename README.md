# Inversion-articulatoire_2

Ce projet contiendra le code implémenté de ac2art en utilisant PyTorch.

Il sera divisé en deux parties :

### Pré traitement

EFfectue le pré taitement nécéssaire aux bases de données MOCHA et MNGU0.
Il contient un script traitement par corpus : traitement_mocha, traitement_MNGU0 et traitement_zerospeech.
Le troisème corpus (zerospeech) est différent des deux premiers, car il ne contient que des fichiers wav (pas de .lab, ni de .ema).

Les étapes de traitement sont les suivantes et dans le même ordre :

Pour chaque locuteur :
	Pour chaque échantillon (= 1 phrase):
*  Récupération de la donnée EMA. 
*  On ne conserve que les données articulatoires qui nous intéressent
*  Interpolation pour remplacer les "NaN"
*  Récupération des données acoustiques .WAV
*  Extraction des 13 plus grand MFCC.
*  Lecture du fichier d'annotation qui donnée le début et la fin de la phrase (hors silence)
*  On enlève les trames MFCC et données EMA correspondant au silence
*  On sous échantillonne les données articulatoires pour avoir autant de données qu'il y a de trames MFCC  
*  Calcul les coefficients dynamiques \Delta et \Delta\Delta associé à chacun des MFCC (2 coefficients en plus par MFCC)
*  On ajoute les 5 trames MFCC précédents et suivant chaque trame.


Une fois qu'on a chargé en mémoire toutes les phrases :
*  Calcul de la moyenne et écart type par articulation/coefficient acoustique (EMA ou Delta Feature),

	Pour chaque échantillon (=1 phrase):
*  Normalisation par rapport à l'ensemble du corpus pour le même speaker ==> enregistrement d'un fichier .npy dans "Donnees_pretraitees/speaker/ema(oumfcc)/nomfichier.npy"
*  flitrage passe bas avec sinc+fenêtre de hanning (à 25Hz pour MNGU0 et 30Hz pour mocha) des données ema ==> enregistrement d'un fichier .npy dans "Donnees_pretraitees/speaker/ema_filtered/nomfichier.npy"


Le script sauvegarde en local dans inversion_articulatoire_2\Donnees_pretraitees\Donnees_breakfast\MNGU0 deux fichiers numpy par phrase.
Un fichier pour les mfcc (dans \mfcc) et un fichier ema (dans \ema)

Le prétraitement zerospeech est plus simple : 
il n'y a que des fichiers wav à charger, calculer les MFCC, les delta et deltadelta, ajouter frames contexte, puis normaliser.

Le script sauvegarde en local dans "inversion_articulatoire_2\Donnees_pretraitees\donnees_challenge_2017\1s" un fichier numpy mfcc par phrase.

Les prérequis de ces scripts de prétraitement est d'avoir bien les données relatives aux corpus dans \inversion_articulatoire_2\Donnees_brutes\Donnees_breakfast
Les fichiers doivent être dans les bons dossiers, ce qui revient mettre les dezipper les dossier téléchargés.
Pour mngu0 : ema dans "ema_basic_data", phonelabels dans "phone_labels", wav dans "wav_16KHz"
pour mocha : les données concernant le locuteur X (fsew0 ou msak0) dans le dossier X. 
Pour une uttérance (ie une phrase) on a des fichier : .EMA, .EPG,.LAB, .LAR, et on ne se sert pas du fichier .EPG.
Pour zerospeech :  dans "inversion_articulatoire_2\Donnees_brutes\donnees_challenge_2017" pour le moment uniquement les fichiers d'une secondes
ils sont dans le dossier "1s" et on a un ensemble de fichiers .wav

Après avoir fait trourner "traitement_mocha","traitement_mngu0", et "traitement_zerospeech", nous avons pour chaque phrase :
 1 fichier .NPY pour les mfcc, et 1 fichier .NPY pour les données EMA (sauf pour ZS2017)


###  Create filesets

Ce script contient une fonction "get_fileset_names(speaker)" qui choisi aléatoirement 70% des phrases du speaker pour en faire le train set, 10% pour le validation set, et 20% le test set.
Le script sauvegarde trois fichier train_speaker.txt , test_speaker.txt, valid_speaker.txt avec la liste des noms de fichiers correspondant. 



### Apprentissage

Les modèles créés se trouvent ici.

Pour le moment le modèle général bilstm se trouve dans "class_network.py".
La fonction d'entrainement est dans "Entrainement.py", l'utilisateur peut choisir : 
 - sur quel(s) speaker(s) il entraîne le  modèle (le validation set sera celui correspondant au(x) speaker(s) sur lesquels on apprend)
 - sur quel(s) speaker(s) il teste le modèle.
- le nombre d'épochs
- la fréquence à laquelle on évalue le score sur le validation set
- la patience (ie combien d'épochs d'overfitting consécutives avant d'arrêter l'apprentissage)
- le learning rate
- si on veut que les données (d'apprentissage et de test) soient filtrées
- si on veut que le modèle lisse les données avec une couche de convolution aux poids ajustés.
- si on veut sauvegarder des  graphes de trajectoires originales et prédites par le modèle sur le test set.


Dans notre code nous avons choisi qu'une epoch correspond à un nombre d'itérations de telle sorte que toutes les phrases aient été prises en compte ( 1 epoch = (n_phrase/batch_size) itération).
A chaque itération on selectionne aléatoirement &batchsize phrases. Si il y a plusieurs speakers sur lesquels apprendre, alors la probabilité de tirer une phrase d'un speaker est proportionnelle au nombre de 
phrases prononcées par ce speaker. Le script qui retourne les phrases sélectionnées se trouve dans Apprentissage\utils.py, et s'appelle load_filenames(train_on,batch_size,part) où part est train valid ou test.
Dans ce même script utils.py la fonction "load_data(filenames, filtered=False)" retourne (x,y) deux listes contenant les mfcc/ema correspondant à chacun des filenames.

Dans ce script est crée la classe "my_bilstm" qui est codée en pytorch.
Elle contient une couche dense à 300 neurones, puis une couche Bi lstm à 300 neurones dans chaque direction.



 