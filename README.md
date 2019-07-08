# Inversion-articulatoire_2

Ce projet contiendra le code implémenté de ac2art en utilisant PyTorch.

Il sera divisé en deux parties :

### Pré traitement

EFfectue le pré taitement nécéssaire aux bases de données dont on dispose, c'est à dire :  MOCHA-TIMIT, MNGU0, USC-TIMIT, et HASKINS. Cette partie pré traite également les données de zerospeech test.
Il contient un script traitement par corpus.

Le dernier corpus (zerospeech) est différent des deux premiers, car il ne contient que des fichiers wav (pas de .lab, ni de .ema).

les articulateurs utils sont dans les deux dimensions : tongue tip, tongue dorsum, tongue blade, upper lip, lower lip , lower incisor.
Les étapes du traitement ne sont pas effectuées toujours suivant la même structure, car celà dépend du format des données en entrées. Les actions effectuées sont les suivantes : 

* Uniquement EMA : récupérer les trajectoires EMA, conserver uniquement les articulateurs utiles (voir plus haut), les mettre dans le format d'un numpy array de taile (Kx12) ou (Kx14) si on dispose des trajectoires du velum. Interpoler les données manquantes avec spline cubique. Lisser les trajectoires en leur appliquant un filtre passe bas à 30Hz.

* Uniquement WAV :  récupérer le waveform en utilisant librosa. Calcule des 13 plus grands MFCC, et des Delta et DeltaDelta pour chaque coefficient (ce qui donne 13*3 = 39 coefficients) . Prise en compte du contexte en ajoutant 5 trames MFCC avant et après la trame courante (ce qui donne 39*11 = 429 coefficients). 

* Pour les données de USC TIMIT nous n'avons pas une phrase par fichier, donc on utilise les fichiers d'annotations pour découper les fichiers wav et ema entre chaque phrase.

* Normalisation : on normalise les coefficients mfcc de façon classique en soustrayant la moyenne et en divisant par l'écart type (moyenne et écart type caculé sur l'ensemble du corpus). Pour les données EMA au lieu de soustraire la moyenne sur l'ensemble du corpus on soustrait la moyenne sur une fenêtre glissante, et on divise par l'écart type de l'articulateur MAXIMUM parmis tous les articulateurs. Les raisons de cette normalisation sont expliquées plus bas.




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

Pour le corpus MNGU0, lorsque la phrase a plus de 500 frames mfcc on divise par deux nos phrases (afin d'avoir des phrases d'à peu près la même longueur que celles du corpus mocha), et on réitère.

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
Elle contient deux couches denses à 300 neurones, puis deux couche Bi lstm à 300 neurones dans chaque direction, ensuite il y a une couche convolutionnelle optionnelle qui agit comme un filtre passe bas à fréquence 30Hz si les données
étaient échantillonnées à une fréquence précise (précisée par l'utilisateur ou par défaut 200Hz). Ceci peut poser problème quand nos données d'apprentissages ne sont pas toutes échantillonées à la même fréquence.
Les poids de la convolution sont pour le moment fixés de la manière suivante : on veut que la sortie soit temporelle convoluée avec une séquence dont la TF est une porte entre 0 et f_cutoff. Il s'agirait d'un sinus cardinal à support infini. 
Comme notre entrée est à support fini on rogne le support du sinus cardinal en le multipliant en temporel avec une fenêtre de Hanning.
Pour le moment les poids de la convolutions sont fixés [à suivre : faire varier fc, ou même un fc par articulateur].



 