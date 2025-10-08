# Classification de Bactéries par Coloration Gram

## Description

Pipeline de classification automatique d'images microscopiques de bactéries colorées selon la méthode de Gram. Combine traitement d'images (OpenCV) et modèles de Machine Learning/Deep Learning pour distinguer les bactéries Gram+ (violet) des Gram- (rose).

## Objectif

Classifier automatiquement les images microscopiques en **2 classes** :
- **Gram Positif** : Bactéries violettes (paroi épaisse)
- **Gram Négatif** : Bactéries roses (paroi fine)

## Dataset

**DIBaS (Digital Images of Bacteria Species)**
- 660 images microscopiques (.tif)
- 32 espèces bactériennes différentes
- Source : [Université Jagellon](https://doctoral.matinf.uj.edu.pl/database/dibas/)
- Téléchargement automatique dans le notebook

## Modèles Implémentés

Le projet compare **4 approches** :

1. **SVM** (Support Vector Machine)
   - Kernel RBF
   - Features : Histogrammes HSV + Textures GLCM

2. **Random Forest**
   - 100 arbres
   - Méthode d'ensemble robuste

3. **Arbre de Décision**
   - Modèle simple et interprétable
   - Baseline de référence

4. **CNN** (Réseau Convolutif)
   - 3 couches convolutives
   - Data augmentation
   - Architecture end-to-end

## Technologies

- **Langage** : Python 3.10+
- **Framework DL** : TensorFlow/Keras
- **Traitement d'images** : OpenCV, scikit-image
- **ML Classique** : scikit-learn
- **Visualisation** : matplotlib, pandas
- **Environnement** : Google Colab (recommandé)

## Installation

```bash
pip install opencv-python-headless tensorflow numpy matplotlib scikit-learn scikit-image
```

## Utilisation

### Sur Google Colab (Recommandé)

1. Ouvrez le notebook `1.ipynb` dans Colab
2. Connectez votre Google Drive (pour sauvegarder les modèles)
3. Exécutez les cellules dans l'ordre :
   - Installation des dépendances
   - Téléchargement automatique du dataset DIBaS
   - Prétraitement et augmentation
   - Entraînement des 4 modèles
   - Comparaison des performances

### Localement

```bash
# Cloner le projet
git clone https://github.com/VOTRE_USERNAME/gram-classification.git
cd gram-classification

# Installer les dépendances
pip install -r requirements.txt

# Lancer Jupyter
jupyter notebook 1.ipynb
```

## Pipeline de Traitement

```
Images brutes (.tif)
    ↓
Redimensionnement (128x128)
    ↓
CLAHE (amélioration contraste)
    ↓
Normalisation [0-1]
    ↓
Extraction Features (pour ML) ou Data Augmentation (pour CNN)
    ↓
Entraînement des modèles
    ↓
Évaluation et comparaison
```

## Features Extraites (ML)

- **Couleur** : Histogrammes HSV (96 features)
- **Texture** : Matrice GLCM
  - Contraste
  - Énergie
  - Homogénéité
  - Dissimilarité

## Architecture CNN

```
Conv2D(32, 3x3) → ReLU → MaxPooling(2x2)
Conv2D(64, 3x3) → ReLU → MaxPooling(2x2)
Conv2D(128, 3x3) → ReLU → MaxPooling(2x2)
Flatten
Dense(128) → ReLU → Dropout(0.5)
Dense(1) → Sigmoid
```

**Hyperparamètres** :
- Optimizer : Adam (lr=0.001)
- Loss : Binary Crossentropy
- Batch size : 32
- Epochs : 20
- Augmentation : rotation, shift, zoom, flip

## Résultats Attendus

| Modèle | Précision Test |
|--------|----------------|
| Arbre de Décision | ~70-75% |
| SVM | ~75-80% |
| Random Forest | ~80-85% |
| **CNN** | **~85-90%** |

*Le CNN obtient généralement les meilleures performances grâce à l'apprentissage de features automatique.*

## Visualisations

Le notebook génère :
- Exemples d'images Gram+ et Gram-
- Distribution des classes (Pie chart)
- Tableau comparatif des précisions
- Graphique en barres des performances
- Historique d'entraînement du CNN

## Structure du Projet

```
gram-classification/
│
├── 1.ipynb                    # Notebook principal
├── README.md                  # Ce fichier
├── requirements.txt           # Dépendances
├── .gitignore                # Fichiers à ignorer
│
├── dibas_data/               # Dataset (auto-téléchargé)
│   ├── gram_positive/
│   └── gram_negative/
│
└── models/                   # Modèles sauvegardés
    ├── svm_gram.pkl
    ├── rf_gram.pkl
    ├── dt_gram.pkl
    └── cnn_gram.h5
```

## Sauvegarde des Modèles

Les modèles entraînés sont automatiquement sauvegardés sur Google Drive :
- Modèles ML : format `.pkl` (joblib)
- CNN : format `.h5` (Keras)

## Améliorations Possibles

- [ ] Tester des architectures CNN plus profondes (ResNet, VGG)
- [ ] Transfer Learning avec ImageNet
- [ ] Classification multi-classe (33 espèces)
- [ ] Détection d'anomalies
- [ ] Interface web (Streamlit/Gradio)
- [ ] Déploiement (Docker, FastAPI)

## Contexte Médical

La **coloration de Gram** est une technique fondamentale en microbiologie :
- Permet de distinguer 2 types de bactéries selon leur paroi cellulaire
- Guide le choix des antibiotiques
- Diagnostic rapide en laboratoire

