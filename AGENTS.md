# AGENTS.md

## Objet et périmètre

Ce fichier est le guide de travail pour les agents et contributeurs intervenant dans ce dépôt. Il décrit le dépôt tel qu'il existe réellement, pas seulement le système final visé.

Le projet recherche des méthodes de vision par ordinateur pour détecter l'eau dans des images de caméra et, à terme, détecter les débordements autour des bassins de laverie de phosphate. L'objectif technique immédiat est la segmentation sémantique binaire précise (`eau` contre `fond`). La détection de débordement est une étape ultérieure construite à partir d'un masque de segmentation fiable, de la géométrie propre à chaque caméra et d'une confirmation temporelle.

Lisez `PLAN.md` avant tout changement architectural ou expérimental. C'est l'énoncé le plus complet de la stratégie de recherche. `README.md` est la présentation générale du projet. Quand la documentation est en désaccord avec le code exécutable, vérifiez le code et les données actuels avant de modifier le comportement.

## Maturité actuelle

C'est un prototype de recherche, implémenté principalement via des notebooks.

- Le paquet Python réutilisable contient des helpers de chargement de données, de seuillage couleur, de métriques, de visualisation, de détection de débordement, **de définitions de modèles et d'inférence**.
- L'interface graphique OpenCV/Tkinter ne démontre que la baseline par seuillage couleur.
- **L'inférence CLI est fonctionnelle** via `apps/predict.py` — elle charge n'importe quel checkpoint (3 formats supportés) et produit un masque binaire, une superposition bleue, ou une carte de probabilités.
- Le notebook de débordement utilise encore un masque placeholder à zéro ; il n'exécute pas un vrai modèle entraîné.
- `main.py` n'existe pas (supprimé — ce n'est pas le point d'entrée).
- Il n'y a pas de suite de tests automatisés, de CI, de script d'entraînement standardisé, ni de configuration de packaging avec un backend de build.
- Le dépôt n'est pas un système d'alarme de débordement de production. Le score de débordement actuel est une heuristique, pas une probabilité calibrée.

Ne décrivez pas des composants planifiés comme déjà implémentés.

## Arborescence du dépôt

### Sources et documentation suivis

- `src/water_detection_methods/` : helpers Python réutilisables.
  - `__init__.py` : réexporte l'API publique : `UNet`, `BatchNormDoubleConv`, `GroupNormDoubleConv`, `BasicBlock`, `ResNet18Encoder`, `UpBlock`, `PretrainedResNet18UNet`, `build_smp_model`, `load_checkpoint`, `Predictor`, `predict_image`.
  - `paths.py` : constantes de chemins relatives au dépôt.
  - `data.py` : découverte d'images, appariement Water-v2, chargement image/masque, split aléatoire, letterboxing et restauration des masques padding.
  - `baselines.py` : segmentation par dominance bleue et couverture du masque.
  - `metrics.py` : binarisation, IoU, Dice, précision pixel.
  - `visualization.py` : affichage image/masque et superposition bleue.
  - `overflow.py` : zones critiques rectangulaires et scores/étiquettes de débordement.
  - `model.py` : **définitions de modèles et chargement de checkpoint** — `UNet` (4 niveaux BatchNorm), `PretrainedResNet18UNet` (encodeur ResNet18 + décodeur GroupNorm), `build_smp_model()` (smp.Unet / smp.UnetPlusPlus), `load_checkpoint()` (détection automatique des 3 formats).
  - `inference.py` : **pipeline d'inférence** — classe `Predictor` avec letterboxing, normalisation ImageNet, inférence, sigmoid, seuillage, restauration du padding, et superposition. Fonction utilitaire `predict_image()`.
- `apps/threshold_gui.py` : application Tkinter pour le réglage interactif de la baseline par seuillage couleur.
- `apps/predict.py` : **outil CLI d'inférence** — `python apps/predict.py IMG.jpg -c model.pt -o output/ --overlay`. Mode batch, masque/overlay/probabilités, checkpoint et device configurables.
- `notebooks/00_exploration_dataset.ipynb` : découvre et visualise les paires Water-v2 et les images locales.
- `notebooks/01_baseline_seuillage.ipynb` : explique et évalue la baseline par seuillage couleur.
- `notebooks/02_ml_classique_image_level_split.ipynb` : comparaison ML classique pixel-level (RVB+TSV).
- `notebooks/03_deep_learning_unet.ipynb` : expérience U-Net PyTorch léger (U-Net 4 niveaux, BatchNorm). Produit `unet_water_v2_best.pt`.
- `notebooks/03_deep_learning_resnet_encoder.ipynb` : U-Net avec encodeur ResNet18 personnalisé + décodeur GroupNorm. Produit `unet_resnet18_512_best.pt`.
- `notebooks/03_deep_learning_unet++.ipynb` : U-Net++ avec `segmentation-models-pytorch` et encodeur ResNet18. Notebook de comparaison.
- `notebooks/04_test_laverie_overflow.ipynb` : prototype débordement sur images locales avec prédictions placeholder.
- `notebooks/run_inference.ipynb` : notebook vide (à compléter) pour exécuter l'inférence.
- `modal_notebooks/03-deep-learning-unet-fixed.ipynb` : **version corrigée pour entraînement sur Modal T4** avec Volumes persistants. Produit des checkpoints au format SMP.
- `models/unet_water_v2_best.pt` : checkpoint U-Net 4 niveaux (original, notebook 03). Format `model_state_dict` + `config`.
- `models/unet_water_v2_best_modal.pt` : checkpoint U-Net 4 niveaux (entraîné sur Modal).
- `models/unet_resnet18_512_best.pt` : checkpoint ResNet18 U-Net (512×512, entraîné localement).
- `models/unet_resnet18_512_best_trained_on_modal_t4.pt` : checkpoint ResNet18 U-Net (512×512, entraîné sur Modal T4).
- `assets/` : illustrations README/recherche et planche de prédictions.
- `README.md` : présentation générale du projet en français.
- `PLAN.md` : plan de recherche détaillé en français, protocole d'évaluation, adaptation domaine, trajectoire de déploiement et priorités.
- `pyproject.toml` : dépendances directes et configuration Python.
- `uv.lock` : environnement verrouillé ; à préférer pour la reproductibilité.
- `requirements.txt` : généré par `uv pip compile`, mais actuellement incomplet par rapport à `pyproject.toml` (voir Environnement ci-dessous).

### Données locales et matériel généré

- `water_v2/` : dataset Water-v2 local, environ 2,05 Go. Ignoré par Git.
  - `train.txt` : contient `ADE20K` et `river_segs` (~2188 paires).
  - `val.txt` : contient les 22 autres groupes (~212 paires).
- `IMGs/` : images réelles de la laverie. Ignoré par Git via le motif `IMGS/` (insensible à la casse sur Windows).
- `MP4/` : six vidéos réelles de la laverie, ignorées par Git.
- `outputs/` : créé à la demande par l'interface graphique et la CLI. N'est pas ignoré actuellement.
- `logs/` : ignoré.
- `.venv/` : environnement virtuel local, ignoré.
- `.agents/` : actuellement vide ; le `AGENTS.md` racine est le guide de tout le dépôt.
- `__pycache__/` et `*.py[oc]` : générés et ignorés. Ne pas éditer ni commiter.

Ne commitez pas les datasets bruts, les médias industriels locaux, les nouveaux gros checkpoints ou les sorties générées sans approbation explicite. Préservez le checkpoint et les assets suivis actuels sauf si la tâche les remplace explicitement.

## Environnement et dépendances

- Python requis : 3.12 ou plus récent (`.python-version` = `3.12`).
- Gestionnaire d'environnement principal : `uv`.
- Le lock file configure PyTorch depuis l'index CUDA 12.6 explicite `https://download.pytorch.org/whl/cu126`.
- Dépendances directes dans `pyproject.toml` : ipykernel, keras, matplotlib, numpy, opencv-python, pandas, Pillow, scikit-learn, **segmentation-models-pytorch**, tensorflow, torch, torchvision.
- Groupe dev : `modal`.
- Tkinter vient de l'installation Python/OS, pas de PyPI.
- `requirements.txt` a été généré depuis `pyproject.toml`. Ne le traitez pas comme la source de vérité complète tant qu'il n'est pas régénéré et vérifié.

Installation préférée depuis la racine du dépôt :

```powershell
uv sync
```

Utilisation de l'interpréteur du projet :

```powershell
.\.venv\Scripts\python.exe apps\predict.py IMGs\img1.jpg -c models\unet_water_v2_best.pt -o outputs --overlay
.\.venv\Scripts\python.exe apps\threshold_gui.py
```

Les notebooks attendent le kernel `.venv` du projet. Les notebooks et l'interface graphique ajoutent `src/` à `sys.path` ; le projet n'est pas configuré comme paquet installable avec un backend de build actuellement.

Ne changez pas silencieusement l'index CUDA/PyTorch ou la stack de frameworks. Le dépôt contient des dépendances PyTorch et TensorFlow/Keras, mais les expériences deep learning implémentées utilisent PyTorch.

## Conventions de chemins et d'imports

Exécutez les commandes depuis la racine du dépôt sauf si un notebook gère explicitement son propre répertoire de travail.

`src/water_detection_methods/paths.py` est la source des chemins relatifs au dépôt :

- `PROJECT_ROOT`
- `WATER_V2_DIR`
- `LOCAL_IMAGES_DIR`
- `NOTEBOOKS_DIR`
- `OUTPUTS_DIR`
- `MODELS_DIR`

Utilisez `pathlib.Path`, ces constantes, ou des chemins dérivés du fichier courant. N'ajoutez jamais de chemins absolus spécifiques à un utilisateur. Les sorties stockées des notebooks contiennent d'anciens chemins d'autres machines Windows ; ce sont des sorties historiques uniquement et ne doivent pas être copiées dans le code source.

La logique réutilisable appartient à `src/water_detection_methods/`. Gardez les notebooks concentrés sur la configuration, l'exécution, le compte-rendu et la visualisation des expériences. Si du code est nécessaire à l'inférence, à l'évaluation, à une interface graphique et à un notebook, extrayez-le du notebook vers le paquet.

## Contrats de données fondamentaux

Préservez ces contrats sauf si le changement est explicitement une refonte cassante.

### Images

- `load_image()` ouvre avec Pillow, convertit en RGB, redimensionne optionnellement avec interpolation bilinéaire, et retourne un tableau NumPy HWC `float32` normalisé en `[0, 1]`.
- Les `size` de Pillow sont `(largeur, hauteur)`, tandis que les shapes NumPy sont `(hauteur, largeur, canaux)`.
- Les conversions OpenCV doivent utiliser explicitement les constantes RGB comme `cv2.COLOR_RGB2HSV` ; les données chargées par Pillow ne sont pas en BGR.

### Masques

- `load_mask()` convertit en niveaux de gris et retourne un masque 2-D `uint8` contenant `0` et `1`.
- Le redimensionnement de masque doit toujours utiliser l'interpolation au plus proche voisin.
- `load_mask(threshold=...)` compare les valeurs brutes de niveaux de gris, normalement dans `[0, 255]`.
- Les fonctions métriques utilisent des seuils de probabilité normalement dans `[0, 1]`. Ne confondez pas le seuil de chargement avec le seuil métrique/de prédiction.
- Préservez l'alignement en appliquant chaque transformation géométrique de façon identique à l'image et au masque.

### Appariement

`find_water_v2_pairs()` parcourt `water_v2/JPEGImages`, préserve chaque chemin relatif sous `water_v2/Annotations`, et cherche un masque avec `.png`, `.jpg` ou `.jpeg`. Elle retourne uniquement les images qui ont un masque correspondant. L'ordre de découverte trié est intentionnel pour la reproductibilité.

### Redimensionnement et padding

- `load_pair()` redimensionne directement et peut déformer le ratio d'aspect.
- `load_pair_with_padding()` letterboxe une paire image/masque, utilise l'interpolation bilinéaire pour l'image et le plus proche voisin pour le masque, et peut retourner les métadonnées de restauration.
- `restore_mask_from_padding()` retire le padding du letterboxing, binarise le masque recadré, et restaure la résolution originale avec interpolation au plus proche voisin.
- Les nouveaux travaux de segmentation devraient préférer la préservation du ratio d'aspect/padding quand c'est scientifiquement approprié, comme requis par `PLAN.md`.
- Validez les tailles cibles positives et la concordance des tailles image/masque originales. Gardez les clés de métadonnées stables : `original_size`, `resized_size`, `padding`, et `target_size`.

## Faits sur le dataset et politique de split

Le scan local actuel a trouvé :

- 4 413 fichiers image d'entrée sous `water_v2/JPEGImages` répartis dans 22 groupes source.
- 2 400 fichiers d'annotation de type image et 2 400 paires image/masque découvrables.
- `train.txt` contient seulement `ADE20K` et `river_segs`, qui fournissent 2 188 échantillons appariés.
- `val.txt` liste les 22 autres groupes, qui fournissent 212 échantillons appariés.
- Le notebook U-Net original (03) split ces 20 groupes d'évaluation avec `GroupShuffleSplit(test_size=0.5, random_state=42)`, produisant 83 images de validation de 10 groupes et 129 images de test de 10 groupes.
- Le dossier `IMGs/` contient actuellement 10 JPEGs originaux plus `img9_blurred.png`, donc `list_local_images()` retourne actuellement 11 images supportées.
- `MP4/` contient six vidéos.

Beaucoup de dossiers source Water-v2 contiennent plus d'images d'entrée non annotées que de masques. Ne supposez jamais que chaque fichier sous `JPEGImages` est entraînable ; utilisez le helper d'appariement.

L'identité de groupe est le premier composant de répertoire sous `JPEGImages`. Les images d'une même scène/vidéo/groupe source ne doivent pas traverser les partitions d'entraînement, validation ou test. Un split aléatoire au niveau image peut fuiter du contenu de scène quasi-dupliqué et n'est pas acceptable pour des résultats finaux de modèle.

Le split officiel actuel est lui-même une faiblesse connue : seulement deux grands groupes sont utilisés pour l'entraînement tandis que 20 groupes sont réservés pour l'évaluation. `PLAN.md` demande un audit du sens des groupes et la création d'un split équilibré par groupe, environ 14-16 groupes d'entraînement, 3-4 groupes de validation et 3-4 groupes de test final, ou l'utilisation de `GroupKFold`/`StratifiedGroupKFold` quand c'est approprié.

Gardez l'ensemble de test final intact jusqu'à ce que l'architecture du modèle, la loss, le prétraitement et le seuil soient figés. Pour les vraies vidéos de laverie, gardez toutes les images d'une même séquence caméra dans la même partition.

## Helpers implémentés et sémantique

### Baseline (`baselines.py`)

`blue_dominance_threshold()` attend une entrée RGB HWC normalisée. Un pixel est considéré comme eau quand toutes les conditions suivantes sont remplies :

- bleu >= `blue_min` (défaut 0,18)
- bleu >= rouge * `blue_red_ratio` (défaut 1,05)
- bleu >= vert * `blue_green_ratio` (défaut 0,85)
- luminance moyenne RGB >= `brightness_min` (défaut 0,05)

Retourne un masque binaire `uint8`. `mask_coverage()` retourne le pourcentage de pixels positifs. Cette méthode est intentionnellement explicable et fragile ; c'est une baseline, pas la solution cible.

### Métriques (`metrics.py`)

- `binarize()` utilise `>= threshold` et rejette les seuils hors de `[0, 1]`.
- `intersection_over_union()` mesure l'IoU binaire classe eau.
- `dice_coefficient()` mesure le chevauchement binaire de l'eau.
- `pixel_accuracy()` mesure tous les pixels et peut être trompeuse quand le fond domine.
- IoU et Dice utilisent un epsilon au numérateur et dénominateur, donc deux masques vides obtiennent un score d'environ 1,0.

Rapportez l'IoU et le Dice de l'eau comme métriques principales, avec précision, rappel, métriques par image, métriques par groupe, comptes de confusion et vues d'erreur qualitatives. La précision seule n'est jamais suffisante. La feuille de route demande aussi Boundary F1 pour la qualité des contours.

### Visualisation (`visualization.py`)

- Les helpers de visualisation attendent des images RGB et des masques binaires/positifs avec des shapes spatiales correspondantes.
- `overlay_mask()` utilise le bleu `(0,0, 0,45, 1,0)` et alpha `0,45` par défaut, retourne une image flottante clampée, et traite toute valeur de masque positive comme sélectionnée.
- `show_image_mask_overlay()` est la vue standard à trois panneaux : image, masque, superposition.

Pour les comparaisons de modèles, préférez les six vues de diagnostic décrites dans `PLAN.md` : original, vérité terrain, carte de probabilités, prédiction binaire, superposition d'erreur, et rendu distinct faux-positifs/faux-négatifs.

### Règles de débordement (`overflow.py`)

- `rectangular_zone()` convertit des coordonnées relatives en un masque de zone booléen.
- `water_ratio_in_zone()` calcule la fraction des pixels de zone critique marqués comme eau et retourne `0,0` pour une zone vide.
- `overflow_confidence()` mappe linéairement la couverture de zone à `[0, 1]`, saturant à `alert_ratio` (défaut 0,15).
- `overflow_label()` mappe la confiance à `normal` en dessous de 0,3, `surveillance` de 0,3 à moins de 0,7, et `debordement_probable` à partir de 0,7.

La confiance n'est explicitement pas une probabilité calibrée. Les coordonnées, la forme de la zone, `alert_ratio` et les seuils d'étiquette doivent éventuellement être calibrés par caméra fixe avec de vrais événements annotés. Les nouveaux codes doivent valider les plages de coordonnées, les formes de masque compatibles et l'ordre des seuils car les helpers actuels effectuent une validation limitée.

### Modèles (`model.py`)

Le module `model.py` centralise toutes les définitions de modèles et le chargement de checkpoints. Trois architectures sont supportées :

- **`UNet`** (4 niveaux, BatchNorm) — le modèle original des notebooks 03, `base_filters=16`, ~1,94 M paramètres. Charge `models/unet_water_v2_best.pt`.
- **`PretrainedResNet18UNet`** — encodeur ResNet18 personnalisé + décodeur GroupNorm. Charge `models/unet_resnet18_512_best.pt` et `models/unet_resnet18_512_best_trained_on_modal_t4.pt`.
- **SMP** — `build_smp_model()` construit `smp.Unet` ou `smp.UnetPlusPlus` avec n'importe quel encodeur SMP (resnet18, mobilenet_v2, etc.).

`load_checkpoint()` détecte automatiquement le format du checkpoint en inspectant les clés du state dict. Formats supportés :
1. SMP avec tag `"architecture"` et `"state_dict"` (notebook modal/unet++).
2. `"model_state_dict"` avec clés `"encoder."` + `"up"` → `PretrainedResNet18UNet`.
3. `"model_state_dict"` avec clés `"encoder1."` → `UNet` (BatchNorm).

### Inférence (`inference.py`)

La classe `Predictor` encapsule tout le pipeline d'inférence :

1. Chargement du checkpoint via `load_checkpoint()` (détection automatique du format).
2. Letterboxing de l'image d'entrée à la taille d'entraînement (préservation du ratio).
3. Normalisation ImageNet `(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))`.
4. Inférence modèle → sigmoïde → seuil (configurable, défaut 0.5).
5. Restauration du masque à la résolution originale via `restore_mask_from_padding()`.
6. Optionnellement, superposition bleue sur l'image originale.

`predict_image()` est une fonction utilitaire pour l'inférence en une ligne.

## Fichiers de checkpoint et workflow Modal

Le workflow d'entraînement sur le cloud Modal et d'inférence locale est opérationnel :

1. **Entraînement sur Modal T4** : ouvrez `modal_notebooks/03-deep-learning-unet-fixed.ipynb` dans un kernel Modal T4 avec un Volume monté sous `/mnt/water-segmentation`. Les checkpoints sont sauvegardés dans le Volume persistant sous `/artifacts/runs/`.

2. **Récupération locale** : utilisez la CLI Modal pour télécharger le checkpoint :
   ```bash
   modal volume get water-detection-artifacts runs/unetplusplus_resnet18_seed42_<hash>/best.pt ./models/
   ```

3. **Inférence locale** : utilisez la CLI ou le paquet Python :
   ```powershell
   python apps\predict.py IMGs\img1.jpg -c models\mon_checkpoint.pt -o outputs --overlay
   ```

Le format des checkpoints produits par le notebook Modal inclut `"architecture"`, `"state_dict"`, `"encoder_name"`, `"config"` (avec `image_size`, `threshold`, etc.) et `"split_hash"`. Ils sont automatiquement reconnus par `load_checkpoint()`.

## Enregistrement des expériences

Traitez les sorties des notebooks comme des preuves historiques, pas comme des résultats de benchmark automatiquement reproductibles. Enregistrez la révision exacte du code, le split, la seed, le matériel, le prétraitement, le seuil et le checkpoint pour les nouveaux résultats.

### Notebook d'exploration

- La sortie stockée rapporte 2 400 échantillons Water-v2 appariés et 10 images locales au moment de son exécution.
- Les exemples sont directement redimensionnés à `(384, 256)` pour l'affichage.

### Notebook baseline seuillage

- Exécute la règle de dominance bleue par défaut sur seulement le premier échantillon apparié à `(384, 256)`.
- Résultat mono-image stocké : IoU `0,6007723187`, Dice `0,7506030829`.
- Ce n'est pas une baseline au niveau du dataset et ne doit pas être comparé comme tel.

### Notebook ML classique

- Caractéristiques : RGB pixel par pixel plus HSV OpenCV (six colonnes).
- Redimensionnement direct : `(384, 256)`.
- Le notebook actuel divise aléatoirement 2 400 images en 1 920 train et 480 test ; il est disjoint en images mais pas conscient des groupes.
- Il échantillonne jusqu'à 50 000 pixels par image, matérialisant 96 millions de lignes train et 24 millions de lignes test, puis ajuste sur un sous-ensemble train de 250 000 pixels seedé.
- Modèles : Logistic Regression équilibrée, Linear SVM, Decision Tree équilibré, et Random Forest à sous-échantillonnage équilibré.
- Meilleur résultat stocké : Random Forest : accuracy `0,669971`, précision eau `0,400401`, rappel `0,545470`, F1/Dice `0,461811`, IoU `0,300230`.
- Le notebook peut consommer beaucoup de RAM car les matrices pandas train/test complètes sont construites avant le sous-échantillonnage. Reconcevez l'échantillonnage/streaming avant de relancer sur du matériel contraint.
- N'utilisez pas ces résultats comme résultats de généralisation finaux tant que le split n'est pas conscient des groupes.

### Notebook U-Net (4 niveaux BatchNorm)

- Framework : PyTorch avec CUDA obligatoire.
- Environnement d'exécution stocké : PyTorch `2.13.0+cu126`, CUDA `12.6`, GTX 1650 Max-Q 4 Go VRAM.
- Seed : 42.
- Entrée : redimensionnement direct à `256×256`, RVB `[0,1]`, masque binaire.
- Architecture : U-Net 4 niveaux, `BASE_FILTERS=16`, BatchNorm+ReLU, ~1,94 M paramètres.
- Loss : `0.5 * BCEWithLogitsLoss + 0.5 * DiceLoss`.
- Meilleur checkpoint : epoch 21.
- Métriques de validation stockées : loss `0,3311`, Dice `0,7324`, IoU `0,6288`, accuracy `0,8610`.
- Métriques de test stockées : loss `0,3240`, Dice `0,7625`, IoU `0,6517`, accuracy `0,8498`.

### Notebook ResNet18 U-Net

- Encodeur ResNet18 personnalisé (pré-entraîné ImageNet) + décodeur GroupNorm.
- Résolution d'entrée : 512×512 avec letterboxing (préservation du ratio).
- Loss : BCE + Dice avec valid mask pour exclure le padding.
- Produit `unet_resnet18_512_best.pt` (format `model_state_dict`).
- Une version entraînée sur Modal T4 est disponible : `unet_resnet18_512_best_trained_on_modal_t4.pt`.

### Notebook U-Net++

- Utilise `segmentation-models-pytorch` (`smp.UnetPlusPlus`) avec encodeur ResNet18 pré-entraîné.
- Comparaison U-Net vs U-Net++ dans un protocole identique.
- Split 14/4/4 par groupe avec recherche aléatoire équilibrée.
- Prétraitement 512×512 avec padding, normalisation ImageNet, augmentations (flip horizontal, affines légères, couleur).
- Entraînement avec gel initial de l'encodeur (5 epochs), AMP, accumulation de gradient.
- Métriques : IoU, Dice, précision, rappel, Boundary F1 — globales, par image et par groupe.
- Le notebook Modal correspondant est `modal_notebooks/03-deep-learning-unet-fixed.ipynb`.

### Notebook de débordement

- Charge une image locale à `(384, 256)`.
- Utilise un placeholder de prédiction à zéro.
- Définit le quart inférieur comme zone critique rectangulaire.
- Utilise `alert_ratio=0,15` et les seuils d'étiquette à trois niveaux par défaut.
- Il prouve seulement le câblage de l'API géométrie/score ; il ne valide pas la détection d'eau ou de débordement.

## Priorités de recherche depuis `PLAN.md`

Travaillez dans cet ordre sauf si l'utilisateur change explicitement le périmètre :

1. Auditer l'appariement Water-v2, l'alignement, les valeurs de masque, les masques vides, les doublons, les ratios d'eau, les résolutions, les ratios d'aspect, le sens des groupes et la qualité d'annotation.
2. Remplacer le split déséquilibré actuel par un protocole reproductible conscient des groupes et préserver un ensemble de test final intact.
3. Réévaluer l'U-Net actuel avec IoU, Dice, précision, rappel, Boundary F1, métriques par image et par groupe, et erreurs qualitatives.
4. Établir des baselines U-Net avec encodeur pré-entraîné (ResNet18 et/ou MobileNetV3).
5. Comparer les candidats orientés précision dans le même protocole : U-Net++ avec backbone léger, DeepLabV3+ avec MobileNetV3/ResNet18, Attention U-Net, et SegFormer-B0.
6. Optimiser seulement les meilleurs candidats : loss, learning rate, résolution/patches, seuil, terme de contour si justifié, et au moins trois seeds.
7. Produire un script de segmentation mono-image fiable et des artefacts d'expérience complets.
8. Extraire des images diverses des six vidéos réelles, en annoter environ 100-200, mesurer le décalage de domaine, fine-tuner sur les données réelles et valider par vidéo entière.
9. Seulement après des masques fiables en domaine réel, ajouter des polygones/lignes de niveau/zones dangereuses par caméra, hystérésis temporelle, vérifications de santé caméra, alertes et stockage de preuves.
10. Optimiser pour le déploiement plus tard avec ONNX/ONNX Runtime/TensorRT et valider la parité des sorties avant les affirmations de performance.

Le matériel cible a 4 Go de VRAM (GTX 1650 Max-Q et RTX 3050 Laptop). Favorisez les backbones légers, l'AMP, les batch sizes 1-2 si nécessaire, l'accumulation de gradient et les mesures de mémoire reproductibles. La qualité est actuellement plus importante que la vitesse temps réel. La cible future est d'environ 5-10 FPS par caméra avec une latence de décision inférieure à la seconde, sous réserve de validation sur site.

## Conventions de code et d'expériences

- Utilisez la syntaxe compatible Python 3.12 et les bibliothèques déjà déclarées dans `pyproject.toml` quand c'est pratique.
- Suivez le nommage PEP 8 : fonctions/variables en snake_case, classes en PascalCase, constantes de module en majuscules.
- Préférez les annotations de type sur les helpers publics réutilisables, mais n'ajoutez pas d'imports ou de bruit de type inutilisés.
- Écrivez des docstrings concises qui indiquent les shapes de tableaux, dtypes, plages, unités, interpolation et sémantique des seuils.
- La documentation et l'interface utilisateur existantes sont principalement en français ; les docstrings de bas niveau partagées sont principalement en anglais. Suivez le fichier environnant sauf si une tâche demande un nettoyage de langue.
- Évitez de dupliquer des fonctions helpers dans les notebooks. Refactorez le code réutilisable dans `src/` et importez-le.
- Utilisez des seeds déterministes et enregistrez-les. Enregistrez les configs complètes, les noms de split/groupes, la meilleure epoch, le seuil, le temps d'exécution et la VRAM maximale pour les entraînements.
- Comparez les modèles sur le même split, prétraitement, augmentations, règles de sélection de seuil et métriques.
- Sélectionnez les checkpoints en utilisant l'IoU de validation, pas l'accuracy. Ne sélectionnez ou n'ajustez jamais en utilisant les résultats du test final.
- Préservez le ratio d'aspect original quand c'est possible ; si vous utilisez le letterboxing, excluez le padding des métriques ou restaurez la prédiction avant l'évaluation.
- Évitez les rotations larges et irréalistes, les transformations verticales, les changements de couleur extrêmes, ou toute transformation qui change les limites du masque indépendamment de l'image.
- Sauvegardez les artefacts générés dans un répertoire de sortie/log clairement nommé plutôt qu'à côté des sources. Vérifiez `git status` avant de terminer.
- Ne modifiez pas à la main `uv.lock`, les pins de requirements compilés, les pièces jointes binaires des notebooks, les fichiers `.pt` ou les médias.
- Quand les dépendances changent, mettez à jour `pyproject.toml`, rafraîchissez `uv.lock`, et régénérez/vérifiez `requirements.txt` intentionnellement.

## Attentes de validation

Il n'y a pas de suite de tests automatisés aujourd'hui. La validation doit être proportionnelle au changement, et un nouveau comportement réutilisable devrait normalement recevoir des tests ciblés quand un framework de test est introduit.

Vérifications minimales pour les changements de code source Python :

```powershell
.\.venv\Scripts\python.exe -m compileall -q src apps
```

Exécutez aussi des vérifications de fumée ciblées pour les contrats affectés :

- les chargeurs d'images retournent des tableaux RGB `float32` en `[0,1]` ;
- les masques restent des tableaux 2-D `uint8` avec des valeurs `{0,1}` ;
- les dimensions spatiales image/masque correspondent après redimensionnement ou padding ;
- les masques restaurés correspondent à la géométrie originale ;
- les fonctions métriques couvrent le chevauchement parfait, les masques disjoints et les masques vides ;
- les zones de débordement et les masques d'eau ont des formes compatibles ;
- les changements de l'interface graphique sont exercés interactivement sur au moins une image ;
- les refactos de notebook sont vérifiés sur un petit échantillon avant une exécution complète coûteuse ;
- les refactos de checkpoint produisent la même sortie que le notebook pour un échantillon fixe dans une tolérance numérique explicite.

Ne lancez pas le notebook d'entraînement U-Net complet comme étape de validation de routine. Il nécessite CUDA et est coûteux. Ne lancez pas les applications GUI dans des environnements sans tête. Si CUDA n'est pas disponible, dites quelles vérifications ont été sautées plutôt que d'affaiblir la protection du notebook ou d'entraîner silencieusement sur CPU.

Pour les changements d'expérience, rapportez l'IoU/Dice global et par groupe, la précision, le rappel, Boundary F1 quand disponible, les comptes de confusion, la moyenne et la distribution des scores par image, les meilleurs/pires échantillons qualitatifs, le temps d'exécution et la VRAM. Vérifiez que les masques utilisent l'interpolation au plus proche voisin et qu'aucune fuite de groupe ne s'est produite.

## Pièges connus et dette technique

- `apps/predict.py` est le point d'entrée CLI pour l'inférence ; il n'y a pas de `main.py`.
- Le checkpoint U-Net original (`models/unet_water_v2_best.pt`) peut désormais être consommé via l'API du paquet (classes `UNet` dans `model.py`, `Predictor` dans `inference.py`). Cette dette de la version précédente d'AGENTS.md est résorbée.
- Les checkpoints au format `PretrainedResNet18UNet` (`unet_resnet18_512_best*.pt`) et SMP sont aussi supportés.
- L'interface graphique (`threshold_gui.py`) n'utilise pas les modèles deep learning — seulement la baseline couleur.
- Le notebook de débordement local n'effectue pas d'inférence et ne peut pas étayer la performance de débordement.
- `train_val_split()` est déterministe mais aléatoire par élément, pas conscient des groupes ; ne l'utilisez pas pour l'évaluation finale scène/vidéo.
- Le notebook ML classique a aussi un split non conscient des groupes et ses DataFrames complets sont très volumineux.
- Le redimensionnement direct dans les notebooks existants change les ratios d'aspect.
- Le notebook U-Net original (03) utilise une augmentation par flip vertical qui entre en conflit avec la mise en garde du plan contre les transformations irréalistes.
- `requirements.txt` et `pyproject.toml` divergent actuellement sur les dépendances directes exécutables.
- Les sorties des notebooks peuvent être obsolètes par rapport aux fichiers, environnements et comptages de données actuels.
- Plusieurs notebooks contiennent de grandes images/sorties embarquées et des encodages de texte mixtes. Évitez le churn de sortie inutile ou le reformatage généralisé.
- Les métriques moyennées par batch/par image dans les notebooks doivent être clairement distinguées de l'agrégation globale de pixels dans les comparaisons futures.
- Le comportement IoU/Dice des masques vides est défini par le lissage ; énoncez la convention quand vous comparez avec d'autres bibliothèques.
- Les helpers de débordement ne valident pas complètement les limites de coordonnées, les formes de masque, `alert_ratio` ou l'ordre des seuils d'étiquette.
- `outputs/` n'est pas ignoré bien que la CLI et l'interface graphique y créent des fichiers.
- Les images/vidéos industrielles brutes peuvent être des données opérationnelles sensibles. Gardez-les locales et évitez de reproduire du contenu identifiable dans les logs, rapports ou commits sans approbation.
- La performance sur Water-v2 n'établit pas la performance en domaine laverie. Attendez-vous à un décalage de domaine substantiel dû à l'eau boueuse, la mousse, la poussière, les reflets, les angles de caméra fixes, la compression, les personnes et l'équipement.

## Liste de vérification pour achèvement sûr

Avant de finaliser un changement :

1. Confirmez le périmètre de la demande et inspectez `git status` ; préservez les modifications utilisateur non liées.
2. Gardez les données brutes/médias et les artefacts générés volumineux hors du changement sauf demande explicite.
3. Exécutez les vérifications de compilation, de fumée, métriques, GUI ou notebook les plus étroites et significatives.
4. Indiquez si les résultats ont été fraîchement exécutés ou lus à partir de sorties de notebook/checkpoint stockées.
5. Documentez les vérifications sautées, en particulier les vérifications CUDA/GPU, GUI, full-dataset ou full-training.
6. Mettez à jour `README.md` ou `PLAN.md` seulement quand le comportement, les commandes, la structure ou les décisions de recherche changent réellement.
7. Maintenez ce `AGENTS.md` synchronisé lors de l'ajout de points d'entrée, scripts, tests, modules de modèle, dispositions de données ou composants de déploiement.
