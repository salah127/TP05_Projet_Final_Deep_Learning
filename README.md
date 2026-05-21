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
| Type de tâche | Classification multi-classe |
| Métrique principale | Accuracy |
| Critère de réussite | Accuracy test > 90 % |
| Source | Intégré dans scikit-learn — aucun téléchargement requis |

### Pourquoi ce dataset ?

- Léger et directement accessible dans Google Colab via `sklearn` (aucune API ni compte externe).
- Problème de classification multi-classe compatible avec un MLP.
- Données suffisamment simples pour une analyse complète en une journée.
- Baseline réaliste à battre (classificateur majoritaire ~40 %).

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
- **Epochs max :** 15

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

Dataset très petit (178 échantillons) — résultats potentiellement sensibles au split aléatoire. Le MLP améliore la baseline majoritaire (~40 %) mais ne dépasse pas significativement la régression logistique sur ce problème quasi-linéairement séparable. Avec plus de temps : validation croisée k-fold + comparaison MLP vs SVM vs RandomForest.

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
