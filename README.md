# Inversion-articulatoire_2

Ce projet contiendra le code implémenté de ac2art en utilisant PyTorch. Il y a une partie lecture/traitement des données,  et une autre apprentissage. On récupère les données de 4 corpus dans des formats variables, on les lit, les traite, et les homogénéise pour qu'elles aient le même format. 

Il sera divisé en deux parties :

### Pré traitement

EFfectue le pré taitement nécéssaire aux bases de données dont on dispose, c'est à dire :  MOCHA-TIMIT, MNGU0, USC-TIMIT, et HASKINS. Cette partie pré traite également les données de zerospeech test.
Il contient un script traitement par corpus.

Le dernier corpus (zerospeech) est différent des deux premiers, car il ne contient que des fichiers wav (pas de .lab, ni de .ema).

les articulateurs utiles sont dans les deux dimensions : tongue tip, tongue dorsum, tongue blade, upper lip, lower lip , lower incisor.
Les étapes du traitement ne sont pas effectuées toujours suivant la même structure, car celà dépend du format des données en entrées. Les actions effectuées sont les suivantes : 

* Uniquement EMA : récupérer les trajectoires EMA, conserver uniquement les articulateurs utiles (voir plus haut), les mettre dans le format d'un numpy array de taile (Kx12) ou (Kx14) si on dispose des trajectoires du velum. Interpoler les données manquantes avec spline cubique. Lisser les trajectoires en leur appliquant un filtre passe bas à 30Hz. Calculer ce qu'on appelle les 4 "vocal tract" et réarranger les données pour les mettre dans le même format (K*18), avec des 0 pour les trajectoires non disponibles. Les 18 variables en questions sont dans l'ordre suivant ['tt_x', 'tt_y', 'td_x', 'td_y', 'tb_x', 'tb_y', 'li_x', 'li_y', 'ul_x', 'ul_y', 'll_x', 'll_y','la','pro','ttcl','tbcl','v_x','v_y'].



* Uniquement WAV :  récupérer le waveform en utilisant librosa. Calcule des 13 plus grands MFCC, et des Delta et DeltaDelta pour chaque coefficient (ce qui donne 13*3 = 39 coefficients) . Prise en compte du contexte en ajoutant 5 trames MFCC avant et après la trame courante (ce qui donne 39*11 = 429 coefficients). 

* Enlève les silences à partir des fichiers d'annotation (si disponible). On rogne les fichiers EMA , et les frames MFCC (en considérant que la k-ème frame mfcc correspond à la période [k*sr_wav/0.01, (k+1)*sr_wav/0.01] en secondes où sr_wav est le sampling rate du wav, et 0.01 car les frames mfcc sont décalées de 10ms.

* Pour les données de USC TIMIT nous n'avons pas une phrase par fichier, donc on utilise les fichiers d'annotations pour découper les fichiers wav et ema entre chaque phrase.

* Normalisation : on normalise les coefficients mfcc de façon classique en soustrayant la moyenne et en divisant par l'écart type (moyenne et écart type caculé sur l'ensemble du corpus). Pour les données EMA au lieu de soustraire la moyenne sur l'ensemble du corpus on soustrait la moyenne sur une fenêtre glissante, et on divise par l'écart type de l'articulateur sur l'ensemble du corpus. On enlève la moyenne sur une fenêtre glissante car on s'est rendu compte que la bobine bouge ou est replacée au cours de l'enregistrement.


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

Le prétraitement zerospeech est plus simple : 
il n'y a que des fichiers wav à charger, calculer les MFCC, les delta et deltadelta, ajouter frames contexte, puis normaliser.

Le script sauvegarde en local dans "inversion_articulatoire_2\Donnees_pretraitees\donnees_challenge_2017\1s" un fichier numpy mfcc par phrase.

FINALEMENT NON , MAIS PEUT ETRE FAIT SIMPLEMENT EN DECOMMENTANT LA LIGNE SPLIT CORPUS : Pour le corpus MNGU0, lorsque la phrase a plus de 500 frames mfcc on divise par deux nos phrases (afin d'avoir des phrases d'à peu près la même longueur que celles du corpus mocha), et on réitère.

Les prérequis de ces scripts de prétraitement est d'avoir bien les données relatives aux corpus dans \inversion_articulatoire_2\Donnees_brutes
Les fichiers doivent être dans les bons dossiers (leur donner les bons noms), ce qui revient mettre les dezipper les dossier téléchargés.

Pour mngu0 : ema dans "ema", phonelabels dans "phone_labels", wav dans "wav"

Pour mocha : les données concernant le locuteur X (fsew0 , msak0,...) dans le dossier X. 
Pour une uttérance (ie une phrase) on a des fichier : .EMA, .EPG,.LAB, et on ne se sert pas du fichier .EPG ni .LAR

Pour zerospeech :  dans "inversion_articulatoire_2\Donnees_brutes\ZS2017" pour le moment uniquement les fichiers d'une secondes
ils sont dans le dossier "1s" et on a un ensemble de fichiers .wav

Pour usc timit : les données dezippées ie un dossier par speaker et pour chaque speaker un dossier "trans", un "mat", et un "wav"

Pour Haskins : les données dézippées ie un dossier par sepaker et pour chaque speaker un dossier "data".

Les données usctimit et haskins sont fournies en format matlab, et peuvent se lire avec le module sio et scipy. 

Le traitement des données est différent, principalement parce que le format est différent que pour mocha et MNGU0. Ici nous avons un fichier .mat et un ficher .wav pour 18 secondes de parole. Le locuteur récite les phrases une à une, mais les enregistrements sont rognés toutes les 18 secondes. Nous avons autour de 3 phrases par fichiers. Comme déjà précisé précédemment, les silences ne nous intéressent pas, alors il faut redécouper ces signaux pour ne conserver que les parties correspondant à de la parole. Il y a alors deux possibilités : utiliser les fichiers de transcription (.trans) fournis avec le corpus, ou bien utiliser des techniques d'analyse du speech pour détecter les silences. Nous avons trouvé plus simple d'utiliser les fichiers de transcription. Les trajectoires articulatoires nous sont données pour les 6 articulateurs classiques ( tongue tip, tongue dorsum, tongue body, lower lip, upper lip, lower incisor ) en 3 dimensions.

Nous avons créer une classe pour l'objet "corpus" qui contient les paramètres utiles, et également des fonctions communes au traitement des 4 corpus. Ces fonctions communes sont : add_vocal_tract, smooth_data, synchro_ema_mfcc, et calculate_norm_values.

Le script de traitement parcourt 2 fois l'ensemble des enregistrements. La première fois les traitements sont dans l'ordre : lecture ema, lecture wav, wav to mfcc, add_vocal_tract, smooth_ema, synchro_ema_mfcc.
Ensuite les norm values sont calculées et stockées dans les attributs de l'instance corpus crée.
Lors du deuxième parcourt de l'ensemble des enregistrements, il normalise les données EMA et MFCC, puis refiltre les donées EMA. 
(Puis les phrases trop longues sont coupées en 2.)
Enfin les phrases sont découpées en 3 parties train, valid et test). Par ex dans le fichier text  "Donnees_pretraitees/fileset/speaker_train" se trouve la liste des noms des fichiers qui seront dans le training set pour ce speaker.

A l'issu du traitement les sauvegardes locales sont : 
- EMA : les données EMA non normalisées et non lissées avec les vocal tract, et les trajectoires mauvaises mises à 0.
- MFCC : les données MFCC normalisées.
- EMA_final : les données avec les vocal tract lissées et normalisées. Ce sont celles si que nous allons utiliser pour l'apprentissage.
- norm_values : pour chaque speaker std_ema_&speaker, moving_average_&speaker, std_mfcc_&speaker, mean_mfcc_&speaker.




Une fois les données homogénéiisées la fonction get_fileset_name est appelée, et génère 3 listes de noms de fichiers  : train,valid,test. Ce script contient une fonction "get_fileset_names(speaker)" qui choisi aléatoirement 70% des phrases du speaker pour en faire le train set, 10% pour le validation set, et 20% le test set.
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

Le script Entrainement_3 s'ajuste aux articulations "correctes". Chaque batch ne contient que des phrases homogènes en terme d'articulateurs. Nous avons regroupé les speakers en catégorie, de telle sorte qu'au sein d'une catégorie tous les speakers aient les mêmes articulateurs corrects. 



 