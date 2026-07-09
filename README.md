# Detection des debordements des bassins de laverie par IA

![image de la laverie OCP Plant Wash](IMGs\img9.jpeg)

## Sujet du projet

Ce projet a pour objectif de developper un systeme intelligent capable de detecter l'eau dans des images issues de cameras, afin d'aider a identifier les situations de debordement dans les bassins de la laverie.

Le systeme sera base sur des methodes de vision par ordinateur et d'intelligence artificielle. L'idee principale est d'entrainer un modele capable de reconnaitre les zones contenant de l'eau dans une image, puis d'utiliser cette detection pour estimer si un bassin risque de deborder ou si l'eau a deja depasse une zone critique.

Nous disposons de deux sources d'images :

- `water_v2` : dataset issu de Kaggle contenant des images d'eau et des annotations, utilise pour l'entrainement et la validation du modele.
- `IMGs` : images locales provenant de la laverie, utilisees pour tester le modele dans le contexte reel du projet.

## Problematique

La detection de debordement dans un bassin industriel n'est pas simple, car l'apparence de l'eau peut varier fortement selon les conditions :

- changement de luminosite ;
- presence de reflets ;
- presence de mousse, boue ou particules ;
- couleur variable de l'eau ;
- angles de vue differents selon les cameras ;
- formes irregulieres de la surface de l'eau ;
- similarite visuelle entre l'eau, le sol mouille et certaines surfaces brillantes.

Le probleme principal est donc de detecter correctement l'eau dans une image, meme lorsque les conditions visuelles ne sont pas ideales.

Une fois l'eau detectee, le second probleme consiste a determiner si cette eau correspond a une situation normale ou a un debordement. Pour cela, il faudra relier la zone d'eau detectee a une zone critique definie dans l'image, par exemple le bord du bassin ou une zone interdite autour du tank.

## Solutions envisagees

Plusieurs approches peuvent etre etudiees et comparees.

### 1. Methodes classiques de traitement d'image

Ces methodes utilisent des regles basees sur les couleurs, la luminosite, les contours ou la texture.

Exemples :

- seuillage RGB ou HSV ;
- detection de zones brillantes ;
- detection de contours ;
- operations morphologiques ;
- soustraction d'arriere-plan pour camera fixe.

Avantages :

- simples a mettre en place ;
- rapides a executer ;
- faciles a expliquer.

Limites :

- sensibles a la lumiere, aux reflets et aux changements d'environnement ;
- peu robustes pour un contexte industriel complexe ;
- necessitent souvent un reglage manuel pour chaque camera.

Ces methodes pourront servir de baseline pour comparer les resultats avec les approches d'apprentissage automatique.

### 2. Machine Learning classique

Une autre solution consiste a utiliser des caracteristiques extraites des images, puis a entrainer un modele de machine learning classique.

Exemples de modeles :

- SVM ;
- Random Forest ;
- KNN ;
- regression logistique ;
- classification pixel par pixel a partir de descripteurs de couleur et de texture.

Avantages :

- plus flexible que les simples seuils ;
- entrainement moins lourd que les reseaux de neurones profonds ;
- utile pour construire une premiere approche supervisée.

Limites :

- depend beaucoup de la qualite des caracteristiques choisies ;
- moins performant sur des scenes complexes ;
- difficile a generaliser si les images de la laverie sont tres differentes du dataset d'entrainement.

### 3. Deep Learning pour la classification

Une approche possible est d'entrainer un reseau de neurones pour classer une image complete.

Exemple :

- image avec eau ;
- image sans eau ;
- image avec debordement ;
- image sans debordement.

Modeles possibles :

- CNN simple ;
- ResNet ;
- EfficientNet ;
- MobileNet.

Avantages :

- permet une prediction directe ;
- annotation plus simple si l'on travaille seulement avec des labels d'image ;
- peut etre utilise pour une alerte globale.

Limites :

- ne donne pas precisement la position de l'eau ;
- moins adapte si l'on veut mesurer le niveau ou verifier une zone critique ;
- difficile a expliquer si le modele predit seulement une classe.

### 4. Deep Learning pour la segmentation semantique

La segmentation semantique consiste a predire, pour chaque pixel de l'image, s'il appartient a la classe eau ou non-eau.

Cette approche est la plus adaptee a notre objectif, car elle permet d'obtenir un masque precis de la zone d'eau.

Modeles possibles :

- U-Net ;
- DeepLabV3+ ;
- SegFormer ;
- Mask R-CNN ;
- YOLO avec segmentation.

Avantages :

- localisation precise de l'eau ;
- exploitable pour calculer une surface d'eau ;
- permet de verifier si l'eau touche une zone critique ;
- mieux adaptee a la detection de debordement.

Limites :

- necessite des annotations de segmentation ;
- entrainement plus couteux ;
- besoin d'adapter le modele aux images reelles de la laverie.

## Objectif du projet

L'objectif est d'entrainer et de comparer plusieurs approches capables de predire la presence d'eau dans une image, avec une preference pour la segmentation semantique.

Le modele principal devra etre capable de :

- recevoir une image de camera en entree ;
- detecter les pixels correspondant a l'eau ;
- produire un masque de segmentation eau / non-eau ;
- estimer la surface ou la position de l'eau dans l'image ;
- tester si l'eau atteint une zone critique du bassin ;
- fournir un pourcentage de confiance lie au risque de debordement.

Le score final pourra etre exprime sous forme de confiance, par exemple :

- 0 % a 30 % : faible risque de debordement ;
- 30 % a 70 % : risque moyen, situation a surveiller ;
- 70 % a 100 % : fort risque de debordement.

## Demarche proposee

## Organisation modulaire du projet

Le projet sera organise de maniere modulaire afin de comparer facilement plusieurs approches.

Les notebooks seront utilises pour les experimentations, avec un notebook par approche :

- `notebooks/00_exploration_dataset.ipynb` : exploration de `water_v2` et des images locales ;
- `notebooks/01_baseline_seuillage.ipynb` : methode simple par seuillage couleur ;
- `notebooks/02_ml_classique.ipynb` : approche Machine Learning classique ;
- `notebooks/03_deep_learning_unet.ipynb` : approche Deep Learning par segmentation U-Net ;
- `notebooks/04_test_laverie_overflow.ipynb` : test sur les images de la laverie et estimation du risque de debordement.

Les fonctions communes seront placees dans `src/water_detection_methods` :

- `data.py` : chargement des images, des masques et creation des paires image/annotation ;
- `visualization.py` : affichage des images, masques et overlays ;
- `metrics.py` : metriques de segmentation comme IoU, Dice coefficient et accuracy pixel ;
- `overflow.py` : definition des zones critiques et calcul du score de confiance de debordement ;
- `paths.py` : chemins principaux du projet.

Cette organisation evite de dupliquer le meme code dans tous les notebooks et facilite la comparaison des approches.

Pour executer les notebooks, il faut selectionner l'environnement Python du projet (`.venv`). Les helpers ont ete verifies avec cet environnement et detectent actuellement 2400 paires image/masque dans `water_v2` et 10 images locales dans `IMGs`.

### Etape 1 : Exploration des donnees

Analyser la structure du dataset `water_v2` :

- nombre d'images ;
- nombre d'annotations ;
- formats des images ;
- correspondance entre images et masques ;
- qualite des annotations ;
- diversite des scenes contenant de l'eau.

Analyser aussi les images locales du dossier `IMGs` :

- resolution ;
- angle des cameras ;
- visibilite des bassins ;
- presence ou absence d'eau ;
- differences entre les images locales et le dataset Kaggle.

### Etape 2 : Preparation des donnees

Preparer les donnees pour l'entrainement :

- charger les images et les annotations ;
- associer chaque image a son masque ;
- convertir les annotations en masques binaires eau / non-eau si necessaire ;
- redimensionner les images ;
- normaliser les valeurs des pixels ;
- separer les donnees en train, validation et test ;
- appliquer de l'augmentation de donnees si besoin.

Exemples d'augmentation :

- rotation legere ;
- changement de luminosite ;
- contraste ;
- flou ;
- recadrage ;
- bruit.

### Etape 3 : Mise en place des baselines

Avant d'utiliser des modeles avances, mettre en place des methodes simples pour avoir un point de comparaison :

- seuillage couleur en HSV ;
- detection de zones sombres ou brillantes ;
- methode basee sur les textures ;
- eventuellement soustraction d'arriere-plan si les cameras sont fixes.

Ces methodes serviront a evaluer si l'IA apporte une amelioration significative.

### Etape 4 : Entrainement de modeles ML ou Deep Learning

Tester plusieurs modeles :

- un modele de machine learning classique ;
- un CNN de classification si necessaire ;
- un modele de segmentation comme U-Net ;
- eventuellement un modele plus avance comme DeepLabV3+, SegFormer ou YOLO-seg.

Le modele de segmentation sera prioritaire, car il permet de localiser l'eau de maniere precise.

### Etape 5 : Evaluation des modeles

Comparer les resultats avec des metriques adaptees :

- accuracy ;
- precision ;
- recall ;
- F1-score ;
- IoU ;
- Dice coefficient ;
- temps d'inference ;
- robustesse aux reflets et variations de lumiere.

Pour la segmentation, les metriques les plus importantes seront IoU et Dice coefficient.

### Etape 6 : Test sur les images locales de la laverie

Apres l'entrainement sur `water_v2`, tester le modele sur les images du dossier `IMGs`.

Cette etape permettra de verifier si le modele generalise correctement vers le contexte reel de la laverie.

Si les resultats ne sont pas suffisants, il faudra annoter quelques images locales et faire un fine-tuning du modele.

### Etape 7 : Estimation du risque de debordement

Une fois le masque d'eau obtenu, definir une zone critique dans chaque image.

Exemples :

- bord superieur du bassin ;
- zone autour du tank ;
- zone ou l'eau ne doit jamais apparaitre ;
- ligne de niveau maximal autorise.

Le risque de debordement pourra etre estime en fonction de plusieurs criteres :

- proportion de pixels d'eau dans la zone critique ;
- distance entre la zone d'eau et la limite du bassin ;
- surface totale detectee comme eau ;
- confiance moyenne du modele sur les pixels d'eau ;
- evolution temporelle si plusieurs images successives sont disponibles.

Exemple de regle :

```text
Si beaucoup de pixels d'eau sont detectes dans la zone critique,
alors le score de confiance du debordement augmente.
```

### Etape 8 : Resultat attendu

Le systeme final devra produire :

- l'image originale ;
- le masque de segmentation de l'eau ;
- une visualisation de l'eau detectee sur l'image ;
- un score de confiance ;
- une decision finale : normal, surveillance ou debordement probable.

## Architecture generale du systeme

```text
Image camera
    |
    v
Pretraitement
    |
    v
Modele de detection / segmentation de l'eau
    |
    v
Masque eau / non-eau
    |
    v
Analyse de la zone critique
    |
    v
Score de confiance du debordement
    |
    v
Alerte ou decision finale
```

## Comparaison attendue des approches

| Approche | Donnees necessaires | Precision attendue | Robustesse | Complexite | Utilite pour le projet |
|---|---|---:|---:|---:|---|
| Seuillage couleur | Aucune annotation | Faible a moyenne | Faible | Faible | Baseline |
| Traitement d'image classique | Peu de donnees | Moyenne | Faible a moyenne | Faible | Baseline avancee |
| Machine Learning classique | Donnees annotees ou caracteristiques | Moyenne | Moyenne | Moyenne | Comparaison |
| CNN classification | Labels image | Moyenne a bonne | Bonne | Moyenne | Detection globale |
| Segmentation U-Net | Masques de segmentation | Bonne | Bonne | Moyenne | Approche principale |
| DeepLabV3+ / SegFormer | Masques de segmentation | Tres bonne | Tres bonne | Elevee | Approche avancee |
| YOLO-seg | Masques de segmentation | Bonne | Bonne | Moyenne | Temps reel |

## Conclusion

La solution la plus adaptee au projet est d'utiliser une approche de segmentation semantique pour detecter l'eau dans les images. Le dataset `water_v2` servira a entrainer un premier modele, puis les images locales de la laverie dans `IMGs` permettront de tester la generalisation du modele dans l'environnement reel.

Le systeme ne se limitera pas a dire s'il y a de l'eau ou non. Il devra aussi analyser la position de l'eau par rapport aux zones critiques des bassins afin d'estimer un pourcentage de confiance lie au risque de debordement.
