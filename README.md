# TP05 — Projet Final Deep Learning

## 2. Groupe

| | Nom |
|---|---|
| Étudiant 1 | JARI Salah Eddine |
| Étudiant 2 | Elyas Abdenbi |

---

## 3. Parcours choisi

**Parcours C — Données tabulaires avec MLP**

---

## 5. Dataset

**Wine dataset** — `sklearn.datasets.load_wine`

| Caractéristique | Valeur |
|---|---|
| Nombre d'échantillons | 178 |
| Nombre de features | 13 (mesures chimiques continues) |
| Nombre de classes | 3 (`class_0`, `class_1`, `class_2`) |
| Variable cible | Catégorie du vin (0, 1 ou 2) |
| Type de tâche | Classification multi-classe supervisée |
| Métrique principale | Accuracy |
| Critère de réussite | Accuracy test > 90 % |
| Source | Intégré dans scikit-learn — aucun téléchargement requis |

### 1. Pourquoi ce dataset a été choisi ?

- Directement accessible dans Google Colab via `sklearn.datasets.load_wine` — aucune API, aucun compte externe, aucun téléchargement.
- Dataset léger (178 échantillons, < 1 KB) : chargement instantané, pas de contrainte mémoire.
- Données suffisamment simples pour permettre un pipeline complet (preprocessing → modèle → évaluation) en une journée.
- Baseline réaliste à améliorer (classificateur majoritaire ~40 %).

### 2. Quel est le problème traité ?

Classifier automatiquement un vin dans l'une des 3 catégories (`class_0`, `class_1`, `class_2`) à partir de 13 mesures chimiques issues d'une analyse de laboratoire (taux d'alcool, acide malique, cendres, magnésium, flavanoïdes, proline, etc.).

### 3. Quelle est la variable cible ?

La colonne `target` — un entier entre 0 et 2 représentant la classe du vin :
- `class_0` : 59 échantillons
- `class_1` : 71 échantillons
- `class_2` : 48 échantillons

### 4. Quelle métrique principale est utilisée ?

**Accuracy** (taux de classification correcte sur le test set).  
Justification : les 3 classes sont relativement équilibrées (59 / 71 / 48), donc l'accuracy n'est pas biaisée par un déséquilibre de classes. Le rapport de classification complet (precision, recall, f1 par classe) est également affiché pour détecter les confusions classe par classe.

### 5. Pourquoi le modèle MLP est-il adapté à ces données ?

- Les 13 features sont **numériques continues** : un MLP avec couches `Dense` est naturellement adapté à ce type d'entrée, contrairement à un CNN (images) ou un RNN (séquences temporelles).
- Les features ont des **échelles très différentes** (`proline` ~700 vs `nonflavanoid_phenols` ~0.3) : après `StandardScaler`, un MLP converge efficacement grâce à la descente de gradient sur des distributions normalisées.
- Les interactions **non-linéaires** entre features chimiques (ex. combinaison alcool + flavanoïdes) justifient l'usage d'activations `relu` plutôt qu'un modèle purement linéaire.
- Architecture légère (3 075 paramètres) suffisante pour 178 échantillons — évite le sur-apprentissage.

### ✅ Vérification faisabilité Google Colab (CPU)

| Critère | Résultat |
|---|---|
| Temps de chargement | < 1 s |
| Temps d'entraînement | **2.95 s** (15 epochs, CPU) |
| Mémoire requise | < 1 MB |
| Dépendances | `sklearn`, `tensorflow`, `numpy` — toutes pré-installées sur Colab |
| Exécution complète notebook | < 30 s |

---

## 8. Modèle final

MLP (Multi-Layer Perceptron) avec Keras / TensorFlow :

```
Input(13)
  → Dense(64, relu)
  → Dropout(0.2)
  → Dense(32, relu)
  → Dense(3, softmax)
```

- **Optimiseur :** Adam (lr=1e-3)
- **Loss :** `sparse_categorical_crossentropy`
- **Régularisation :** Dropout + EarlyStopping (patience=3, restore_best_weights)
- **Epochs :** 15/15 (EarlyStopping non déclenché)
- **Paramètres totaux :** 3 075 (12.01 KB)
- **Temps d'entraînement :** 2.95 s (CPU)

---

## 4. Imports & Dépendances

Toutes les dépendances sont compatibles **Google Colab** — aucune installation supplémentaire requise :

| Package | Usage |
|---|---|
| `tensorflow` / `keras` | Construction et entraînement du modèle |
| `scikit-learn` | Dataset, split, scaler, métriques |
| `numpy` | Manipulation des tableaux |
| `pandas` | Visualisation des données tabulaires |
| `matplotlib` | Courbes d'apprentissage, matrice de confusion |

---

## 13. Amélioration

Comparaison de deux architectures MLP :
- **Baseline** : `64→32`, Dropout 0.2
- **Amélioré** : `128→64→32`, Dropout 0.3

Objectif : vérifier si une plus grande capacité améliore l'accuracy sans aggraver l'overfitting.

---

## 14-16. Sauvegarde, Rechargement & Inférence

- Sauvegarde : `model.save("best_model.keras")`
- Rechargement : `keras.models.load_model("best_model.keras")`
- Inférence sur une nouvelle donnée (vecteur de 13 features standardisées)

---

## 20. Conclusion

| Métrique | Résultat |
|---|---|
| Accuracy test | **94.44 %** ✓ (critère > 90 % atteint) |
| Val accuracy finale | 96.55 % |
| Train accuracy finale | 99.12 % |
| Temps d'entraînement | 2.95 s |
| Erreurs sur 36 tests | 2 (class_1 → class_0) |
| class_2 | Parfaitement classifiée (f1 = 1.00) |

Le MLP améliore largement le classificateur majoritaire (~40 %). Il est légèrement en-dessous de la régression logistique (référence ~97–99 %), ce qui est attendu sur 178 échantillons. Avec plus de données, le MLP deviendrait plus compétitif.

---

## Fichiers

| Fichier | Description |
|---|---|
| `TP05_Projet_Final_Deep_Learning_v2.ipynb` | Notebook principal |
| `best_model.keras` | Modèle sauvegardé (généré après exécution) |
| `README.md` | Ce fichier |

---

## Exécution

Ouvrir dans **Google Colab** et exécuter les cellules de haut en bas (`Runtime → Run all`). Le fichier `best_model.keras` est généré automatiquement à la section 14.
