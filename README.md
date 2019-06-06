# Inversion-articulatoire_2

Ce projet contiendra le code implémenté de ac2art en utilisant PyTorch.

Il sera divisé en deux parties :

### Pré traitement

EFfectue le pré taitement nécéssaire aux bases de données MOCHA et MNGU0.
Il contient un script traitement par corpus : traitement_mocha, traitement_MNGU0 et traitement_zerospeech.
Le troisème corpus (zerospeech) est différent des deux premiers, car il ne contient que des fichiers wav.
Le traitement qui en résulte est très différent.

Les étapes de traitement sont les suivantes et dans le même ordre :

Pour chaque locuteur :
	Pour chaque échantillon (= 1 phrase):
*  Récupération de la donnée EMA. 
*  On ne conserve que les données articulatoires qui nous intéressent
*  Interpolation pour remplacer les "NaN"
*  Récupération des données acoustiques .WAV
*  Extraction des 13 plus grand MFCC.
*  Lecture du fichier d'annotation qui donnée le début et la fin de la phrase (hors silence)
*  On enlève les trames MFCC correspondant au silence
*  On sous échantillonne les données articulatoires pour avoir autant de données qu'il y a de trames MFCC  
*  Calcul les coefficients dynamiques \Delta et \Delta\Delta associé à chacun des MFCC (2 coefficients en plus par MFCC)

Une fois qu'on a chargé en mémoire toutes les phrases :
*  Calcul de la moyenne et écart type par articulation/coefficient acoustique (EMA ou Delta Feature)

	Pour chaque échantillone (=1 phrase):
*  Normalisation par rapport à l'ensemble du corpus pour le même speaker.
*  On ajoute les 5 trames MFCC précédents et suivant chaque trame.

C'est à dire lit les fichiers audios, en calcule les 15 plus grands coefficients MFCC qu'ils stockent sous format NPY dans un dossier, idem pour EMA (voir).

Le script sauvegarde en local dans inversion_articulatoire_2\Donnees_pretraitees\Donnees_breakfast\MNGU0 deux fichiers numpy par phrase.
Un fichier pour les mfcc (dans \mfcc) et un fichier ema (dans \ema)

Le prétraitement zerospeech est beaucoup plus simple : 
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

Une fois ce prétraitement à l'échelle de la phrase effectué, on construit les np array qui constitueront les 
données du modèle.
Pour nous X (features) sont les MFCC, et Y (target) sont les EMA.
Pour chaque locuteur (MNGU0,fsew0,msak0) on concatène les features et target pour chaque phrase  et créons un fichier  npy X_speaker que nous enregistrons.
Nous faisons bien attention à ce que la ligne i du X corresponde bien à la même phrase que la ligne i du Y.
Pour celà nous parcourons les noms des fichiers .EMA puis cherchons l'équivalent dans les fichiers MFCC. 

Un ajustement a été fait pour que la différence entre les longueurs des phrases ne soit pas trop grande. 
Les phrases MNGU0 peuvent être très longues (jusqu'à 1500 frames mfcc) alors que les autres sont toujours moins que 600 frames.
On découpe en deux les phrases MNGU0 supérieur à 600 frames.

### Apprentissage

Les modèles créés se trouvent ici.

Pour le moment le modèle général bilstm se trouve dans "class_network.py".
La fonction d'entrainement est dans "Entrainement.py", et l'utilisateur peut choisir si il entraine sur un speaker ou tous. 
Si il y a un seul speaker alors le découpage test/train/valid se fait sur le moment, puisque c'est un entraînement temporaire.
Nous voulons infine entraîner un modèle sur l'ensemble des deux speaker fsew0 et MNGU0.
Si l'utilisateur précise que "speaker = both" alors le modèle utilise le découpage train/test qui est toujours le même.


Pour le moment il n'y a qu'un seul script : model.py.
Dans ce script est crée la classe "my_bilstm" qui est codée en pytorch.
Elle contient une couche dense à 300 neurones, puis une couche Bi lstm à 300 neurones dans chaque direction.
Le lissage n'est pas contenu dans le réseau lui même, mais est effectué lorsqu'il s'agit de calculer l'erreur de prédiction sur le validation set ou test set.
Pour l'erreur de prédiction sur le validation set, on la calcul sur les prédictions et vraies trajectoires normalisées.
Ceci est dû au fait que dans la validation set nous ne savons pas quel échantillon provient de quel set. Nous pourrions cependant normaliser nos données sur l'ensemble des corpus.
Mais il faudrait vérifier que les moments sont assez homogène d'un corpus à l'autre.
En revanche pour le test set on utilise 