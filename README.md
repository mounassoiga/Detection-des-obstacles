#  Détection d’Obstacles à partir d’Images – CNN avec TensorFlow/Keras

## Objectif
Ce projet vise à entraîner un **réseau de neurones convolutif (CNN)** pour **classer différents types d’obstacles** (chaise, porte, véhicule, escaliers, etc.) à partir d’un dataset d’images issu de Kaggle.

---

## Structure du projet
- `Detection d'obstacles.ipynb` → Notebook complet d'expérimentations
- `train/`, `test/` → Images utilisées pour l’entraînement et l’évaluation
- `Compte_rendu_CNN_Obstacles.docx` → Rapport expérimental détaillé

---

##  Modèle CNN
Le modèle final comprend :
```python
Conv2D(32, (3,3)) + ReLU + MaxPooling2D
Conv2D(64, (3,3)) + ReLU + MaxPooling2D
Conv2D(128, (3,3)) + ReLU + MaxPooling2D
Flatten + Dense(128, ReLU)
Dropout(0.4)
Dense(10, softmax)
Optimiseur : Adam (learning_rate=1e-4)
Loss : Categorical Crossentropy

---

##Resultats

Accuracy :66%

