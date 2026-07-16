# Plan de segmentation précise de l'eau et trajectoire vers la détection des débordements

## 1. Vision du projet

Le projet est volontairement divisé en deux objectifs successifs.

### Objectif immédiat

Développer un modèle de segmentation sémantique capable de détecter l'eau dans une image et de produire un masque aussi proche que possible du masque réel.

À cette étape, la priorité est la qualité de la segmentation :

- bonne détection de tous les pixels d'eau ;
- peu de faux positifs sur les ombres, reflets et surfaces sombres ;
- contours précis ;
- conservation des petites zones et des détails ;
- résultats stables sur des images jamais vues.

Le temps réel, la détection spécifique des bassins et l'alerte ne sont pas les critères principaux de cette première étape. Ils seront traités après l'obtention d'un modèle de segmentation suffisamment fiable.

### Grand objectif industriel

À terme, le modèle devra être adapté aux images et vidéos réelles d'une laverie de phosphate afin de :

1. segmenter l'eau dans chaque image du flux caméra ;
2. déterminer si l'eau dépasse la limite autorisée d'un bassin ;
3. confirmer le dépassement sur plusieurs images consécutives ;
4. déclencher une alerte ;
5. enregistrer une preuve visuelle de l'événement.

Cette seconde étape nécessitera des données réelles de la laverie, une adaptation au domaine industriel, une logique géométrique propre à chaque caméra et une validation temporelle.

## 2. Données et contraintes connues

### 2.1 Dataset source : Water-v2

Le dataset Water-v2 contient approximativement 2 400 paires image/masque :

- image RGB ;
- masque binaire de la zone d'eau ;
- classe 0 : arrière-plan ;
- classe 1 : eau.

La répartition actuellement observée est :

| Partition | Nombre d'images | Nombre de groupes |
|---|---:|---:|
| Entraînement | 2 188 | 2 |
| Validation | 83 | 10 |
| Test | 129 | 10 |
| **Total** | **2 400** | **22** |

Un premier U-Net a obtenu une accuracy proche de 0,83. Cette valeur ne permet pas encore de conclure sur la qualité du masque.

### 2.2 Domaine cible : laverie de phosphate

Water-v2 ne contient pas les images réelles de la laverie. Les données réelles actuellement disponibles sont très limitées :

- environ 10 images ;
- 6 vidéos.

Ces données réelles ne doivent pas être mélangées sans contrôle avec Water-v2. Elles appartiennent à un autre domaine visuel : bassins industriels, eau chargée en phosphate, mousse, boue, poussière, éclairage industriel, angles fixes et compression vidéo.

Dans l'immédiat, Water-v2 sert à développer et comparer les modèles de segmentation. Les données de la laverie seront utilisées plus tard pour mesurer le décalage de domaine, effectuer le fine-tuning et valider le système final.

### 2.3 Contraintes matérielles

Les entraînements doivent être compatibles avec :

- NVIDIA GTX 1650 Max-Q, 4 Go de VRAM ;
- NVIDIA RTX 3050 Laptop, 4 Go de VRAM.

La RTX 3050 sera la machine d'entraînement principale. La GTX 1650 servira pour des entraînements légers, des vérifications et des benchmarks complémentaires.

## 3. Définition d'un masque « presque identique au réel »

Un masque visuellement satisfaisant ne doit pas être évalué uniquement avec l'accuracy. Si l'arrière-plan représente 83 % des pixels, un modèle prédisant toujours « non-eau » peut atteindre une accuracy de 0,83.

La fidélité du masque sera mesurée avec plusieurs critères complémentaires.

### 3.1 Métriques principales

- **IoU eau** : intersection entre masque prédit et masque réel, divisée par leur union ;
- **Dice/F1 eau** : mesure principale de similarité globale ;
- **précision eau** : proportion des pixels prédits comme eau qui sont corrects ;
- **rappel eau** : proportion des vrais pixels d'eau effectivement détectés ;
- **Boundary F1** : précision des contours ;
- **matrice de confusion des pixels** ;
- **IoU et Dice par image**, et pas seulement sur l'ensemble complet.

### 3.2 Analyse qualitative obligatoire

Pour chaque expérience, produire une planche contenant :

1. image originale ;
2. masque réel ;
3. probabilité prédite ;
4. masque binaire prédit ;
5. superposition des erreurs ;
6. faux positifs et faux négatifs avec des couleurs différentes.

Les erreurs devront être classées selon les catégories suivantes :

- contour imprécis ;
- zone d'eau manquée ;
- ombre confondue avec l'eau ;
- reflet confondu avec l'eau ;
- petite zone supprimée ;
- trou artificiel dans le masque ;
- eau trouble ou peu contrastée non détectée ;
- erreur liée au redimensionnement de l'image.

### 3.3 Objectifs chiffrés

Les seuils définitifs seront fixés après le calcul complet des métriques du U-Net actuel et l'audit de la qualité des annotations.

Les objectifs de progression seront :

- dépasser clairement le U-Net actuel sur IoU eau, Dice et Boundary F1 ;
- obtenir des résultats stables entre les groupes ;
- réduire les faux positifs et faux négatifs sur les cas difficiles ;
- rapprocher la qualité du modèle de la cohérence maximale permise par les annotations.

Une mesure de l'accord entre deux annotateurs sur un petit sous-ensemble est recommandée. Si les humains ne produisent pas exactement les mêmes contours, le modèle ne pourra pas dépasser de manière fiable cette limite d'annotation.

## 4. Audit et préparation de Water-v2

La qualité des données doit être vérifiée avant de rechercher une nouvelle architecture.

### 4.1 Vérifications obligatoires

- vérifier que chaque image possède exactement un masque ;
- contrôler l'alignement image/masque ;
- convertir tous les masques vers une représentation binaire cohérente ;
- vérifier les valeurs des pixels du masque ;
- détecter les masques vides ou presque vides ;
- détecter les doublons et quasi-doublons ;
- vérifier la cohérence des contours ;
- mesurer la proportion de pixels d'eau par image ;
- documenter les résolutions et rapports largeur/hauteur ;
- identifier les images très similaires appartenant à une même scène.

### 4.2 Correction du découpage

Le découpage actuel contient 2 groupes d'entraînement contre 20 groupes de validation et de test. La signification exacte d'un groupe doit être vérifiée.

Si un groupe représente une scène, une séquence ou une origine commune :

- ne jamais répartir les images d'un même groupe entre train, validation et test ;
- augmenter le nombre de groupes présents dans l'entraînement ;
- équilibrer les partitions selon le nombre d'images et la proportion d'eau ;
- conserver un test final composé de groupes jamais utilisés pendant le réglage.

Avec 22 groupes, une répartition indicative est :

- 14 à 16 groupes pour l'entraînement ;
- 3 à 4 groupes pour la validation ;
- 3 à 4 groupes pour le test.

Si les tailles sont très différentes, utiliser `GroupKFold` ou `StratifiedGroupKFold`. Le test final ne doit être consulté qu'après la sélection de l'architecture, de la loss et du seuil.

### 4.3 Prétraitement

- conserver le rapport d'aspect lorsque cela est possible ;
- utiliser du padding plutôt qu'une déformation importante ;
- redimensionner les masques avec une interpolation nearest-neighbor ;
- normaliser les images selon le backbone préentraîné ;
- appliquer exactement les mêmes transformations géométriques à l'image et au masque ;
- vérifier visuellement le résultat après chaque étape de prétraitement.

Pour les images à haute résolution, comparer :

1. redimensionnement global à 512 × 512 ;
2. redimensionnement avec conservation du rapport d'aspect ;
3. découpage en tuiles avec chevauchement si les petits détails disparaissent.

## 5. Modèles à comparer pour la précision

La première campagne doit privilégier les architectures compatibles avec 4 Go de VRAM et adaptées à un dataset de taille modeste.

### 5.1 Classement principal

| Priorité | Modèle | Type | Intérêt principal | Compatibilité 4 Go |
|---:|---|---|---|---|
| 1 | U-Net++ + ResNet18/34 ou EfficientNet-B0 | CNN | Raffinement des détails et des connexions multi-échelles | Bonne |
| 2 | DeepLabV3+ + MobileNetV3 ou ResNet18 | CNN | Contexte multi-échelle et contours | Bonne |
| 3 | Attention U-Net + backbone léger | CNN | Concentration sur les régions pertinentes | Bonne |
| 4 | SegFormer-B0 | Transformer léger | Contexte global et robustesse | Bonne avec réglages |
| 5 | U-Net + backbone préentraîné | CNN | Baseline solide et interprétable | Très bonne |

BiSeNetV2 et Fast-SCNN seront étudiés plus tard lorsque la vitesse de déploiement deviendra prioritaire. YOLO-Seg n'est pas prioritaire pour produire un masque sémantique binaire extrêmement fidèle.

### 5.2 U-Net++

U-Net++ est le premier candidat orienté qualité. Ses connexions imbriquées peuvent aider à mieux reconstruire les détails entre l'encodeur et le décodeur.

Backbones à tester dans cet ordre :

1. ResNet18 préentraîné ;
2. EfficientNet-B0 préentraîné ;
3. ResNet34 préentraîné si la mémoire le permet.

### 5.3 DeepLabV3+

DeepLabV3+ traite le contexte à plusieurs échelles et possède un décodeur permettant de raffiner la segmentation.

Backbones recommandés :

- MobileNetV3 pour la faible consommation ;
- ResNet18 pour une comparaison CNN simple ;
- EfficientNet-B0 si l'implémentation est stable.

Les grands backbones ResNet50/101 ne sont pas prioritaires avec 4 Go de VRAM.

### 5.4 Attention U-Net

Attention U-Net peut aider à réduire les activations provenant de régions non pertinentes. Il est intéressant si les faux positifs sur le fond dominent les erreurs du U-Net actuel.

### 5.5 SegFormer-B0

SegFormer-B0 constitue un benchmark complémentaire non-CNN. Il peut mieux exploiter le contexte global et distinguer l'eau de certaines textures ambiguës. Il doit être comparé aux CNN, mais ne remplace pas les expériences principales U-Net++ et DeepLabV3+.

### 5.6 Baseline U-Net

Le U-Net actuel doit être réévalué. Une seconde baseline utilisera un encodeur préentraîné :

- U-Net + ResNet18 ;
- ou U-Net + MobileNetV3.

Cette baseline permettra de mesurer séparément l'effet du préentraînement et celui de l'architecture.

## 6. Protocole d'entraînement orienté fidélité

Tous les modèles doivent partager le même découpage, les mêmes métriques et les mêmes augmentations pour permettre une comparaison valide.

### 6.1 Configuration de départ

| Paramètre | Valeur initiale |
|---|---|
| Résolution | 512 × 512 |
| Batch RTX 3050 | 2, si possible |
| Batch GTX 1650 | 1 ou 2 |
| Batch effectif | 8 avec accumulation de gradients |
| Mixed precision | Activée |
| Préentraînement | ImageNet ou checkpoint disponible |
| Optimiseur | AdamW |
| Scheduler | Cosine annealing ou ReduceLROnPlateau |
| Epochs maximales | 80 à 150 |
| Early stopping | 15 à 20 epochs |
| Checkpoint principal | Meilleur IoU eau de validation |
| Seeds | Au moins 3 pour le meilleur modèle |

Le batch effectif doit rester aussi constant que possible entre les expériences.

### 6.2 Stratégie de fine-tuning

Pour les encodeurs préentraînés :

1. entraîner temporairement le décodeur avec l'encodeur gelé ;
2. dégeler progressivement l'encodeur ;
3. utiliser un learning rate plus faible pour l'encodeur ;
4. surveiller séparément la loss d'entraînement et les métriques de validation ;
5. sauvegarder le checkpoint ayant le meilleur IoU eau, pas la meilleure accuracy.

### 6.3 Losses à comparer

Expérience de référence :

```text
Loss = 0,5 × BCEWithLogits + 0,5 × Dice
```

Puis, sans changer les autres paramètres :

1. BCE + Dice ;
2. Focal + Dice si l'eau est minoritaire ;
3. Tversky si les faux négatifs dominent ;
4. BCE + Dice + composante de contour si le masque global est correct mais les limites restent imprécises.

Une loss de contour ne doit être ajoutée qu'après l'établissement d'une baseline fiable.

### 6.4 Augmentations

Les augmentations doivent améliorer la généralisation tout en préservant la nature de l'eau :

- petites rotations, translations et changements d'échelle ;
- retournement horizontal si cohérent avec les données ;
- variations modérées de luminosité, contraste et gamma ;
- variations prudentes de saturation et de teinte ;
- flou léger et bruit de capteur ;
- compression JPEG ;
- ombres synthétiques réalistes ;
- baisse modérée de contraste ;
- petites occultations.

Éviter :

- les transformations verticales irréalistes ;
- les rotations importantes ;
- les modifications colorimétriques extrêmes ;
- les déformations qui changent artificiellement la frontière du masque.

### 6.5 Résolution et raffinement des contours

Après la première comparaison à 512 × 512 :

- réentraîner le meilleur modèle à 640 × 640 si la VRAM le permet ;
- sinon utiliser des patches ou tuiles chevauchantes ;
- comparer l'effet de la résolution sur Boundary F1 ;
- ajuster le seuil de binarisation sur la validation ;
- tester un post-traitement morphologique léger ;
- refuser tout post-traitement améliorant seulement quelques images mais dégradant la moyenne globale.

Le seuil `0,5` ne doit pas être supposé optimal. Comparer plusieurs seuils sur la validation, puis figer le seuil avant le test final.

### 6.6 Gestion de la VRAM

En cas de dépassement mémoire :

1. réduire le batch à 1 ;
2. vérifier la mixed precision ;
3. augmenter l'accumulation de gradients ;
4. activer le gradient checkpointing ;
5. utiliser un backbone plus léger ;
6. réduire temporairement la résolution à 448 ou 384 pixels ;
7. revenir à une résolution supérieure uniquement pour le meilleur candidat.

## 7. Plan expérimental immédiat

### Phase 0 — Établir la vraie baseline

- auditer les images et masques ;
- corriger le découpage par groupe ;
- calculer la proportion eau/fond ;
- calculer IoU, Dice, précision, rappel et Boundary F1 du U-Net actuel ;
- produire des visualisations d'erreurs ;
- identifier les cinq principales causes d'échec.

### Phase 1 — Mesurer l'effet du préentraînement

Entraîner avec le même protocole :

1. U-Net actuel ;
2. U-Net + ResNet18 préentraîné ;
3. U-Net + MobileNetV3 préentraîné.

Objectif : déterminer si le gain vient principalement du backbone préentraîné.

### Phase 2 — Comparer les architectures de précision

Entraîner :

1. U-Net++ + ResNet18 ;
2. DeepLabV3+ + MobileNetV3 ou ResNet18 ;
3. Attention U-Net ;
4. SegFormer-B0.

Chaque expérience doit enregistrer :

- configuration complète ;
- seed ;
- durée d'entraînement ;
- consommation maximale de VRAM ;
- meilleur epoch ;
- métriques de validation ;
- métriques par groupe ;
- exemples des meilleures et des pires prédictions.

### Phase 3 — Optimiser le meilleur modèle

Sur les deux meilleurs modèles seulement :

- comparer BCE + Dice, Focal + Dice et Tversky ;
- optimiser le learning rate ;
- tester 512 puis 640 pixels ou des patches ;
- ajuster le seuil de binarisation ;
- ajouter une composante de contour si nécessaire ;
- entraîner avec au moins trois seeds ;
- conserver la moyenne et l'écart-type.

### Phase 4 — Test final

Après avoir figé toutes les décisions :

- charger le meilleur checkpoint ;
- utiliser le seuil figé sur validation ;
- évaluer une seule fois sur le test intact ;
- produire les métriques globales et par groupe ;
- produire une galerie complète des erreurs ;
- comparer avec le U-Net initial ;
- documenter les limites restantes.

## 8. Critères de sélection du modèle de segmentation

Le modèle final de l'objectif immédiat sera sélectionné dans cet ordre :

1. IoU de la classe eau ;
2. Dice de la classe eau ;
3. Boundary F1 ;
4. stabilité entre les groupes ;
5. rappel et précision équilibrés ;
6. absence d'échecs critiques sur les cas difficiles ;
7. coût d'inférence, uniquement en cas de qualité comparable.

Un modèle plus rapide ne sera pas retenu s'il produit des masques nettement moins fidèles. En revanche, si deux modèles sont statistiquement équivalents, le plus léger sera préféré pour faciliter la future étape temps réel.

## 9. Livrables de l'objectif immédiat

- dataset Water-v2 audité et versionné ;
- découpage train/validation/test reproductible ;
- script commun d'entraînement ;
- script commun d'évaluation ;
- implémentations des modèles comparés ;
- checkpoints des meilleurs modèles ;
- tableau des métriques globales et par groupe ;
- courbes d'apprentissage ;
- rapport d'utilisation de la VRAM ;
- planches image/masque réel/masque prédit/erreurs ;
- analyse des faux positifs et faux négatifs ;
- modèle de segmentation sélectionné ;
- script d'inférence sur une image ;
- rapport final comparant le modèle retenu au U-Net initial.

## 10. Passage de Water-v2 aux images réelles de la laverie

Une excellente performance sur Water-v2 ne prouve pas que le modèle fonctionnera dans la laverie. Il faudra traiter explicitement le décalage de domaine.

### 10.1 Exploitation des 6 vidéos réelles

Ne pas extraire toutes les images à 25 ou 30 FPS. Les frames consécutives sont trop similaires.

Procédure recommandée :

1. extraire une frame toutes les 2 à 10 secondes ;
2. supprimer les doublons et quasi-doublons ;
3. conserver davantage de frames lors des changements de niveau, lumière ou activité ;
4. sélectionner les cas de mousse, ombre, boue, reflet et projection ;
5. viser initialement 100 à 200 frames réelles diversifiées ;
6. annoter les images sélectionnées avec vérification humaine.

SAM 2 ou un autre outil interactif peut être utilisé pour accélérer la propagation des masques dans une vidéo. Les masques automatiques doivent être corrigés manuellement.

### 10.2 Séparation par vidéo

Les frames d'une même vidéo ne doivent jamais apparaître dans plusieurs partitions.

Avec seulement 6 vidéos :

- utiliser 4 vidéos pour l'entraînement, 1 pour la validation et 1 pour le test ;
- ou réaliser une validation croisée laissant une vidéo entière de côté ;
- conserver une future nouvelle vidéo comme véritable test final si possible.

Les 10 images réelles doivent également être regroupées par caméra, date et origine.

### 10.3 Fine-tuning sur le domaine réel

La stratégie recommandée est :

1. préentraîner le modèle sur Water-v2 ;
2. évaluer ce modèle sans adaptation sur les images réelles annotées ;
3. fine-tuner le décodeur sur les images réelles ;
4. dégeler progressivement les dernières couches de l'encodeur ;
5. utiliser un learning rate faible ;
6. mélanger des images Water-v2 et des images réelles pendant le fine-tuning ;
7. sélectionner le checkpoint uniquement selon la validation réelle.

Une bonne configuration de départ consiste à suréchantillonner les images réelles afin qu'elles représentent une part importante de chaque batch, sans abandonner complètement Water-v2.

### 10.4 Active learning

Après un premier fine-tuning :

1. exécuter le modèle sur les frames réelles non annotées ;
2. détecter les prédictions incertaines ;
3. sélectionner les faux positifs et faux négatifs visibles ;
4. annoter ces cas difficiles ;
5. réentraîner le modèle ;
6. répéter la boucle.

Cette stratégie doit être privilégiée par rapport à l'annotation aléatoire d'un grand nombre de frames similaires.

## 11. Retour vers le grand objectif : détecter les débordements

Lorsque le modèle produit des masques fiables sur des images réelles de la laverie, le projet peut passer à la détection des débordements.

### Étape 1 — Définir précisément un débordement

Avec les responsables du site, définir :

- la limite normale de chaque bassin ;
- la ligne ou zone correspondant à un niveau dangereux ;
- les situations nécessitant une alerte ;
- le délai maximal acceptable avant l'alerte ;
- la tolérance aux fausses alertes ;
- les actions à réaliser après l'alerte.

Le modèle n'a pas nécessairement besoin d'une classe « débordement ». Il peut segmenter l'eau, puis une règle géométrique détermine si elle se trouve dans une zone interdite.

### Étape 2 — Collecter des données représentatives

Collecter, avec les autorisations et mesures de sécurité nécessaires :

- niveaux d'eau normaux ;
- bassins presque pleins ;
- débordements faibles et importants ;
- eau hors du bassin ;
- différentes heures et conditions d'éclairage ;
- mousse, boue, poussière et projections ;
- présence de personnes ou d'engins ;
- défauts de caméra et occultations.

S'il n'existe aucun exemple réel de débordement, réaliser si possible une simulation contrôlée. Des données synthétiques peuvent compléter l'entraînement, mais ne doivent jamais remplacer une validation réelle.

### Étape 3 — Configurer les zones de chaque caméra

Pour chaque caméra fixe, définir :

- un polygone représentant le bassin ;
- une ligne de niveau maximal ;
- une zone de danger ;
- les zones à ignorer ;
- une référence permettant de détecter un déplacement de caméra.

La décision peut reposer sur :

- le franchissement d'une ligne par le masque d'eau ;
- la proportion d'eau dans la zone de danger ;
- la présence d'eau à l'extérieur du bassin ;
- une combinaison de ces règles.

### Étape 4 — Ajouter la validation temporelle

Une seule frame positive ne doit généralement pas déclencher immédiatement une alerte.

Ajouter :

- un lissage sur plusieurs frames ;
- un nombre minimal de détections positives consécutives ;
- un seuil d'activation ;
- un seuil de désactivation plus faible ;
- un délai anti-rebond ;
- une vérification de disponibilité du flux caméra.

Exemple initial à calibrer :

```text
Alerte si plus de 5 % de la zone dangereuse contient de l'eau
pendant au moins 5 frames consécutives ou pendant 1 seconde.
```

### Étape 5 — Construire le pipeline temps réel

```text
Caméra RTSP/USB
    → capture d'une frame
    → prétraitement
    → segmentation CNN de l'eau
    → masque binaire
    → intersection avec les zones du bassin
    → filtrage temporel
    → décision de débordement
    → alerte et sauvegarde d'une preuve
```

Une cible initiale de 5 à 10 FPS par caméra peut être suffisante si la latence de décision reste inférieure à une seconde. Cette cible doit être validée selon la vitesse réelle des événements et les exigences du site.

### Étape 6 — Optimiser le modèle CNN pour le déploiement

Lorsque la qualité sur données réelles est validée :

1. retenir le meilleur CNN compatible avec le temps réel, par exemple DeepLabV3+, U-Net léger ou BiSeNetV2 ;
2. exporter le modèle en ONNX ;
3. comparer les sorties PyTorch et ONNX ;
4. tester ONNX Runtime CUDA ;
5. tester TensorRT FP16 ;
6. mesurer la latence complète avec `batch=1` ;
7. mesurer l'utilisation de la VRAM ;
8. envisager INT8 uniquement si la perte de qualité est acceptable.

Si le meilleur modèle de précision est trop lent, utiliser la stratégie suivante :

- modèle précis comme enseignant ;
- modèle CNN léger comme élève ;
- distillation ou fine-tuning du modèle léger ;
- comparaison finale sur les vraies vidéos.

### Étape 7 — Développer le système d'alerte

Chaque événement devra enregistrer :

- identifiant de la caméra et du bassin ;
- date et heure ;
- proportion d'eau dans la zone dangereuse ;
- image originale ;
- masque prédit ;
- image annotée ;
- durée du dépassement ;
- état de transmission de l'alerte.

Le système devra également signaler :

- flux caméra interrompu ;
- image noire ou figée ;
- objectif occulté ;
- déplacement important de la caméra ;
- indisponibilité du modèle ou du service d'alerte.

### Étape 8 — Valider le système industriel

Les métriques finales seront :

- taux de débordements détectés ;
- taux de débordements manqués ;
- fausses alertes par heure ;
- délai moyen de détection ;
- FPS et latence ;
- stabilité sur une exécution prolongée ;
- robustesse par caméra, bassin et condition visuelle.

Le passage en production nécessitera une période pilote avec surveillance humaine et journalisation de toutes les alertes.

## 12. Ordre d'exécution recommandé

### Maintenant : qualité du masque sur image

1. auditer Water-v2 et ses annotations ;
2. corriger le découpage par groupe ;
3. réévaluer le U-Net actuel avec les bonnes métriques ;
4. entraîner les baselines U-Net préentraînées ;
5. comparer U-Net++, DeepLabV3+, Attention U-Net et SegFormer-B0 ;
6. optimiser les deux meilleurs modèles ;
7. sélectionner le meilleur modèle selon IoU, Dice et Boundary F1 ;
8. produire un script fiable de segmentation d'une image.

### Ensuite : adaptation à la laverie

1. extraire des frames diversifiées des 6 vidéos ;
2. annoter progressivement 100 à 200 images réelles ;
3. évaluer le modèle Water-v2 sans adaptation ;
4. fine-tuner sur les données réelles ;
5. utiliser l'active learning pour sélectionner les nouvelles annotations ;
6. valider par vidéo entière.

### Enfin : détection et alerte de débordement

1. définir les zones et seuils de chaque bassin ;
2. collecter ou simuler de manière contrôlée des débordements ;
3. ajouter la logique géométrique ;
4. ajouter la confirmation temporelle ;
5. optimiser le CNN en ONNX/TensorRT ;
6. intégrer le flux caméra et le service d'alerte ;
7. mesurer les débordements manqués et les fausses alertes ;
8. réaliser un pilote supervisé avant la production.

La priorité actuelle est donc d'obtenir une segmentation d'eau mesurable, reproductible et visuellement fidèle. Le système de débordement devra être construit sur cette base, après adaptation et validation sur les données réelles de la laverie.
