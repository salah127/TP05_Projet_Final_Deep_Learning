# TP05 — Projet Final Deep Learning

## 🔗 Démo en ligne

> **[Cliquer ici pour tester le projet →](https://mlp-t5.vercel.app/)**

---

## Démarrage rapide — Lancer la démo

### Prérequis

- Python 3.10+ installé (testé avec 3.13)
- `h5py` et `numpy` : `pip install h5py numpy`

### Étapes

```bash
# 1. Se placer dans le dossier du projet
cd chemin/vers/Project

# 2. Extraire les poids du modèle Keras → web/weights.json
py -3.13 scripts/extract_weights.py

# 3. Lancer le serveur local
py -3.13 -m http.server 8000

# 4. Ouvrir dans le navigateur
#    http://localhost:8000/web/demo.html   ← modèle réel (poids JSON)
```

> **Vérification** : si `extract_weights.py` affiche `[OK] web/weights.json (61.7 KB) — 3 layers`, tout est prêt.

---

## Navigation

| Section notebook | Titre | Où trouver |
|---|---|---|
| §2 | Informations groupe | Ce README → §2 |
| §3 | Choix du parcours | Ce README → §3 |
| §4 | Imports & configuration | Ce README → §4 |
| §5 | Chargement des données / Dataset | Ce README → §5 |
| §6 | Visualisation rapide | Voir notebook — cellule §6 |
| §7 | Preprocessing | Voir notebook — cellule §7 |
| §8 | Construction du modèle | Ce README → §8 |
| §9 | Entraînement | Voir notebook — cellule §9 |
| §10 | Courbes d'apprentissage | Voir notebook — cellule §10 |
| §11 | Évaluation | Voir notebook — cellule §11 |
| §12 | Analyse d'erreurs | Voir notebook — cellule §12 |
| §13 | Amélioration / comparaison | Ce README → §13 |
| §14 | Sauvegarde du modèle | Ce README → §14-16 |
| §15 | Chargement du modèle | Ce README → §14-16 |
| §16 | Inférence | Ce README → §14-16 |
| §17 | Mini-démo | Voir notebook — cellule §17 |
| §18 | Question WebApp/API | Voir notebook — cellule §18 |
| §19 | Bonus Gradio (facultatif) | Voir notebook — cellule §19 |
| §20 | Conclusion | Ce README → §20 |

---

## 2. Groupe

| | Nom |
|---|---|
| Étudiant 1 | JARI Salah Eddine |
| Étudiant 2 | Elyas Abdenbi |

---

## 3. Parcours choisi

**Parcours C — Données tabulaires avec MLP**

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

## 5. Dataset

**Wine dataset** — `sklearn.datasets.load_wine`

| Caractéristique | Valeur |
|---|---|
| Nombre d'échantillons | 178 (class_0 : 59 / class_1 : 71 / class_2 : 48) |
| Nombre de features | 13 (mesures chimiques continues) |
| Variable cible | Catégorie du vin — entier 0, 1 ou 2 |
| Type de tâche | Classification multi-classe supervisée |
| Métrique principale | Accuracy (classes équilibrées → pas de biais) |
| Critère de réussite | Accuracy test > 90 % |
| Source | Intégré dans scikit-learn — aucun téléchargement requis |

### Pourquoi ce dataset ?

- Directement accessible via `sklearn.datasets.load_wine` — aucune API, aucun compte externe.
- Dataset léger (< 1 KB) : chargement instantané, compatible CPU Google Colab (entraînement < 3 s).
- Données suffisamment simples pour un pipeline complet en une journée.
- Baseline réaliste à améliorer (classificateur majoritaire ~40 %).

### Problème traité

Classifier automatiquement un vin dans l'une des 3 catégories à partir de 13 mesures chimiques (alcool, acide malique, cendres, magnésium, flavanoïdes, proline, etc.).

### Pourquoi le MLP est adapté ?

- Features **numériques continues** → couches `Dense` naturellement adaptées (vs CNN pour images, RNN pour séquences).
- Échelles très différentes entre features → `StandardScaler` nécessaire, MLP converge efficacement après normalisation.
- Interactions **non-linéaires** entre features chimiques → activations `relu` justifiées vs modèle purement linéaire.

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

| Paramètre | Valeur |
|---|---|
| Optimiseur | Adam (lr=1e-3) |
| Loss | `sparse_categorical_crossentropy` |
| Régularisation | Dropout + EarlyStopping (patience=3) |
| Epochs | 15/15 (EarlyStopping non déclenché) |
| Paramètres totaux | 3 075 (12.01 KB) |
| Temps d'entraînement | 2.95 s (CPU) |

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
| `notebook/TP05_Projet_Final_Deep_Learning_v2.ipynb` | Notebook principal |
| `model/best_model.keras` | Modèle sauvegardé |
| `scripts/extract_weights.py` | Exporte les poids Keras → `web/weights.json` |
| `web/weights.json` | Poids pour l'inférence JS (généré) |
| `web/demo.html` | Démo interactive — modèle réel, inférence JS native |
| `web/index.html` | Démo visuelle simulée |
| `README.md` | Ce fichier |

---

## Exécution notebook

Ouvrir dans **Google Colab** et exécuter les cellules de haut en bas (`Runtime → Run all`). Le fichier `best_model.keras` est généré automatiquement à la section 14.
