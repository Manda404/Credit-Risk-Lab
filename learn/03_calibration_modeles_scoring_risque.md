# Module 03 - Calibration Des Modeles De Scoring

Ce module porte sur la fiabilite des probabilites. En scoring credit, un modele
peut tres bien separer les bons et mauvais dossiers tout en produisant des
probabilites mal calibrees.

La calibration est donc essentielle pour utiliser une probabilite comme une
mesure de risque, fixer un seuil operationnel, calculer une perte attendue ou
justifier une decision devant un comite risque.

---

Voici un **cours complet, structuré, pédagogique et professionnel** qui couvre **les deux thématiques précédentes** :

# 1️⃣ **La Calibration des modèles de Machine Learning**

# 2️⃣ **Pourquoi et comment calibrer un modèle (scoring, risque, banque, assurance)**

Ce cours est rédigé comme un chapitre de formation avancée en Data Science appliquée au risque et à la décision.

---

# 📘 **COURS : Calibration des Modèles de Machine Learning & Son Importance en Scoring de Crédit / Risque**

---

# 🎯 **Introduction**

Dans de nombreux domaines — notamment la banque, l’assurance, la finance et la détection de fraude — les modèles de Machine Learning ne sont pas seulement utilisés pour *classer* des individus (approbation / rejet) mais pour *estimer une probabilité de risque*.

**Or, un modèle peut très bien prédire correctement les classes…
tout en produisant des probabilités incorrectes.**

La calibration est donc indispensable pour obtenir des probabilités **fiables**, **interprétables**, **utiles pour la prise de décision** et **conformes aux exigences réglementaires**.

Ce cours explique :

* ce qu’est la calibration,
* pourquoi elle est essentielle,
* comment elle se mesure,
* comment la réaliser,
* et son rôle central dans les systèmes de scoring (crédit / assurance).

---

# 🧩 **1. Qu’est-ce qu’un modèle calibré ?**

## 📌 Définition simple

Un modèle est **calibré** lorsque :

> **La probabilité prédite correspond à la réalité.**

Exemple :

* Si le modèle dit : *“80% de chance d'être approuvé”*
  alors, parmi 100 individus ayant cette prédiction, environ **80 doivent être approuvés dans la réalité**.

C’est la différence entre :

### ✔ Modèle discriminant

→ distingue bien les classes (AUC élevé)

### ✔ Modèle calibré

→ prédit des probabilités correctes (réalisme)

Les deux propriétés sont indépendantes.

---

# 🔍 **2. Pourquoi les modèles sont souvent mal calibrés ?**

Beaucoup de modèles sont naturellement **surconfiants** ou **sous-confiants** :

* Random Forest → surconfiance sur les classes majoritaires
* XGBoost / LightGBM → surconfiance générale
* SVM → pas de probas internes → transformation artificielle
* Réseaux de neurones → probas extrêmes (saturation softmax)

Même CatBoost, pourtant très bon, **n’est pas toujours parfaitement calibré**.

**Un modèle peut donc avoir une AUC = 0.99 mais être très mal calibré.**

---

# 🎯 **3. Pourquoi calibrer un modèle ? (Motivations métier)**

La calibration est essentielle pour les raisons suivantes :

---

## 3.1 **Pour prendre des décisions correctes**

Dans le scoring, on fixe souvent un seuil métier :

* Si *p(default) < 0.20* → on approuve
* Si *p(default) > 0.20* → on refuse

👉 Si les probabilités sont fausses, **les décisions sont fausses** :

* approuver un dossier risqué → pertes financières
* refuser un bon dossier → manque à gagner

---

## 3.2 **Pour calculer le risque et la perte attendue**

La probabilité calibrée permet de calculer :

[
EL = PD \times LGD \times EAD
]

* PD = Probability of Default
* LGD = Loss Given Default
* EAD = Exposure At Default

Si PD est mal calibré → tout s'effondre.

---

## 3.3 **Pour la tarification (pricing)**

Taux d’intérêt = f(risque).
Un modèle non calibré donnera un **pricing incorrect**.

---

## 3.4 **Pour respecter les exigences réglementaires**

Dans la banque et l’assurance :

📘 Bâle II
📘 Bâle III
📘 Solvency II
📘 IFRS 9

Les régulateurs exigent des modèles **stables et calibrés**, car les probabilités influencent :

* capital à immobiliser
* provisions
* validation réglementaire
* audit interne/externe

---

## 3.5 **Pour éviter le surconfiance / sous-confiance**

Exemples de modèles mal calibrés :

* modèle prédit 0.99 → vrai taux 0.70 → surconfiance
* modèle prédit 0.40 → vrai taux 0.70 → sous-confiance

Cela rend les probabilités **inutilisables**.

---

## 3.6 **Pour la stabilité dans le temps**

Un modèle calibré résiste mieux :

* au drift
* aux nouvelles cohortes
* aux profils inhabituels

Un modèle non calibré dérive rapidement.

---

# 🧪 **4. Comment mesurer la calibration ?**

---

## 4.1 **Calibration Plot (Reliability Curve)**

Plus importante que la ROC pour le risque.

* Diagonale = calibration parfaite
* Points au-dessus → sous-confiance
* Points au-dessous → surconfiance

Un modèle bien calibré suit la diagonale.

---

## 4.2 **Brier Score Loss**

[
Brier = \frac{1}{N} \sum (p - y)^2
]

* 0 = parfait
* < 0.1 = excellent
* > 0.2 = mauvais

---

## 4.3 **Expected Calibration Error (ECE)**

Utilisé dans le deep learning.

---

# 🛠 **5. Comment calibrer un modèle ? (Méthodes)**

---

## ⭐ 5.1 Platt Scaling (Logistic Calibration)

Calibre la proba via une régression logistique :

[
\hat{p} = \sigma(a \cdot f(x) + b)
]

Adapté pour :

* SVM
* XGBoost
* Random Forest

---

## ⭐ 5.2 Isotonic Regression

Méthode flexible non paramétrique.

Avantages :

* très bonne calibration
* forme de la fonction libre

Inconvénients :

* overfitting si peu de données

---

## ⭐ 5.3 Temperature Scaling (Deep Learning)

Très utilisé pour calibrer les réseaux neuronaux.

---

## ⭐ 5.4 Ensemble / bagging calibré

---

# 🌡 **6. Le cas CatBoost**

CatBoost est souvent très proche d’un modèle calibré **par conception** :

* encoding interne optimal
* loss logarithmique
* gestion naturelle des catégories
* structure des trees plus lisse

Mais il peut nécessiter :

* isotonic regression
* or Platt scaling

dans des cas techniques.

---

# 🧩 **7. Exemple visuel de calibration parfaite**

Dans ton cas, la courbe était quasiment identique à :

```
Probabilité prédite : 0.80
Probabilité réelle : 0.80
```

=> **Ce résultat est exceptionnel**.

Cela signifie que :

* ton pipeline est propre
* tes features sont bien construites
* CatBoost s'ajuste de manière "naturelle"
* les probabilités sont fiables
* le modèle est prêt pour un cas d’usage métier réel

---

# 📚 **8. Conclusion générale**

La calibration est un **pilier du machine learning appliqué au risque**.

Un modèle **non calibré** :

* peut avoir un AUC de 0.99
* mais être **inutilisable** en production

Un modèle **bien calibré** :

* fournit des probabilités réelles
* est interprétable
* est fiable
* satisfait les exigences réglementaires
* permet une prise de décision correcte
* protège l’entreprise contre les pertes

Dans un contexte bancaire ou assurantiel,
➡️ **on n’utilise jamais un modèle non calibré**
➡️ **on utilise toujours la calibration + brier score + reliability curve**

---

# 🎁 Souhaites-tu maintenant :

### 🔹 un **cours complet sur l’évaluation de modèles déséquilibrés** (MCC, F1, Kappa, KS Statistic…) ?

### 🔹 un **cours sur l’interprétabilité (SHAP, LIME, PDPlot)** ?

### 🔹 un **cours sur l’optimisation des seuils de décision** en scoring ?
