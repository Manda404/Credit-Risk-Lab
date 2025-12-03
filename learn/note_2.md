# 🎓 **COURS 1 — ÉVALUATION DES MODÈLES SUR DONNÉES DÉSÉQUILIBRÉES**

(MCC, F1, Kappa, KS Statistic, ROC, PR Curve…)

---

## 🔎 **1. Pourquoi les jeux de données sont-ils souvent déséquilibrés ?**

Beaucoup de problèmes industriels présentent un fort déséquilibre :

* fraude (1% de fraudeurs)
* churn (5–20% de churners)
* défaut de paiement (3–10%)
* assurance sinistres rares
* anomalies industrielles

Dans ces cas, **accuracy devient inutilisable** :

> Si 95% des cas = classe 0, un modèle stupide qui prédit *toujours 0* a 95% d’accuracy.

➡️ Il faut utiliser des métriques adaptées.

---

## 🎯 **2. Les métriques classiques**

### ✔ *Precision*

Parmi les prédits positifs, combien sont corrects ?

### ✔ *Recall (Sensitivity)*

Parmi les vrais positifs, combien sont trouvés ?

### ✔ *F1-score*

Moyenne harmonique Precision/Recall
→ très utile quand les classes sont déséquilibrées
→ compromis entre détecter et ne pas sur-alerter

**Limite :** F1 ne tient pas compte des TN → pas une mesure globale.

---

## 🎯 **3. MCC — Matthews Correlation Coefficient** (LA métrique reine)

[
MCC = \frac{TP \cdot TN - FP \cdot FN}{\sqrt{(TP+FP)(TP+FN)(TN+FP)(TN+FN)}}
]

* valeur ∈ [-1, 1]
* **0 = prédiction aléatoire**
* **1 = modèle parfait**
* **-1 = modèle inversé**

MCC tient compte de **TP, FP, TN, FN**, donc c’est la métrique *la plus fiable* pour les données déséquilibrées.

---

## 🎯 **4. Cohen’s Kappa**

Mesure l’accord entre prédictions et vérité, corrigé du hasard.

* 1 : accord parfait
* 0 : même performance qu’un modèle aléatoire
* <0 : pire que random

Utilisé en : assurance, segmentation médicale, NLP, scoring.

---

## 🎯 **5. KS Statistic (Kolmogorov–Smirnov)**

Spécifiquement utilisé en **banque / scoring réglementé**.

[
KS = \max |TPR - FPR|
]

Interprétation :

* KS > 0.4 = bon modèle
* KS > 0.6 = excellent modèle
* KS > 0.75 = modèle exceptionnel (ton cas probable)

**Pourquoi c'est important :**
→ mesure la capacité à *séparer* les bons et mauvais payeurs
→ exigence réglementaire dans certains comités risque

---

## 🎯 **6. ROC Curve & AUC**

* Toujours utile
* Montre la capacité à distinguer les classes
* Mais peut être *trompeuse* avec imbalance
  → d’où l’importance des courbes **Precision–Recall**

---

## 🎯 **7. Precision–Recall Curve & Average Precision**

La PR curve est la meilleure courbe pour les datasets déséquilibrés.

* PR haute → modèle bon pour trouver la classe minoritaire
* AP proche de 1 → excellent modèle

---

# 📚 **Conclusion Cours 1**

Pour un vrai système industriel :
👉 **Accuracy NE sert à rien**
👉 **F1, MCC, Kappa, KS et PR Curve sont essentiels**
👉 **ROC ne suffit jamais seule**

---

# 🎓 **COURS 2 — INTERPRÉTABILITÉ : SHAP, LIME, PDP, FEATURES**

---

## 🔎 **1. Pourquoi interpréter un modèle ?**

Dans les systèmes critiques (banque, assurance, santé) :

* On doit expliquer *pourquoi* une décision a été prise.
* On doit identifier les risques de biais.
* Le modèle doit être **audit-able**.
* Le régulateur impose une explication (DDA, IA Act UE, etc.)

---

# 🟦 **2. SHAP (SHapley Additive ExPlanations)**

C’est **la référence absolue** pour interpréter les modèles.

### ✔ SHAP Summary Plot

→ importance globale des features
→ impact directionnel (positif / négatif)

### ✔ SHAP Dependence Plot

→ visualise la relation entre une feature et la probabilité
→ détecte non-linéarités, interactions, effets de seuil

### ✔ SHAP Force Plot

→ explique une prédiction individuelle
→ « Pourquoi le modèle a rejeté ce client ? »

### ✔ SHAP Interaction Values

→ permet de détecter les interactions cachées entre variables

SHAP est :

* local + global
* exact pour les arbres (CatBoost, XGBoost)
* mathématiquement robuste (théorie des jeux)

---

# 🟩 **3. LIME (Local Interpretable Model-agnostic Explanations)**

Méthode locale :

* crée un modèle linéaire « simple » autour d’un point
* très utile pour expliquer une prédiction individuelle

Moins précis que SHAP, mais parfois plus pédagogique.

---

# 🟧 **4. PDP (Partial Dependence Plot)**

Visualise :

[
f(x_i) = E[f(x) | x_i]
]

→ Montre l’effet moyen d’une variable sur la prédiction
→ Utile pour comprendre les relations monotones

---

# 🟨 **5. ICE plots**

Comme le PDP, mais plot individuel par instance.
Permet de voir la variabilité intra-groupe.

---

# 🔥 **6. Quelle méthode utiliser ?**

| Méthode         | Quand l’utiliser ?            |
| --------------- | ----------------------------- |
| SHAP Summary    | importance globale            |
| SHAP Force      | expliquer un dossier          |
| SHAP Dependence | analyser une feature          |
| LIME            | expliquer localement (rapide) |
| PDP             | relations simples             |
| ICE             | variabilité individuelle      |

---

# 🎚 **7. Interprétation dans scoring crédit**

Ces méthodes permettent :

* de justifier une décision auprès d’un client
* d’expliquer un rejet
* de détecter les biais
* de valider la logique métier
* d’assurer la conformité réglementaire (GDPR Art.22, DDA)

---

# 📚 **Conclusion Cours 2**

L’interprétabilité n’est plus un luxe :
➡️ c’est une obligation légale et métier.
Et SHAP est la solution moderne numéro 1.

---

# 🎓 **COURS 3 — OPTIMISATION DU SEUIL DE DÉCISION (THRESHOLD TUNING)**

---

## 🎯 **1. Pourquoi optimiser le seuil ?**

Un modèle binaire utilise par défaut :
**seuil = 0.5**

Mais cela suppose que :

* les classes sont équilibrées (rare)
* toutes les erreurs coûtent pareil
* les probabilités sont calibrées
* l’objectif métier = accuracy

👉 En scoring, c’est faux.

---

## 🧠 **2. Le seuil doit dépendre de la stratégie métier**

Exemples :

* prévention de fraude → Recall très important
* crédit → minimiser fausses approbations
* assurance → risque réglementé
* churn → maximise le F1 ou le recall

---

## 📏 **3. Méthodes pour choisir le seuil**

### ✔ 3.1 Maximiser le F1-score

Pour balance précision/recall.

```python
threshold = thresholds[np.argmax(f1_scores)]
```

---

### ✔ 3.2 Maximiser le MCC

La méthode *académique la plus robuste*.

---

### ✔ 3.3 Maximiser le KS Statistic

Le plus utilisé dans **les banques**.

Seuil optimal = point où
[
|TPR - FPR|
]
est maximal.

Permet de séparer au mieux les bons / mauvais payeurs.

---

### ✔ 3.4 Youden Index (J statistic)

[
J = Sensitivity + Specificity - 1
]

Célèbre en biomédical.

---

### ✔ 3.5 Basé sur le coût (expected monetary cost)

On définit :

* coût d’un faux positif (prêter à mauvais payeur)
* coût d’un faux négatif (refuser un bon dossier)

Puis on calcule :

[
\text{coût total} = FP \cdot C_{fp} + FN \cdot C_{fn}
]

Seuil optimal = coût minimal.

C’est la méthode **professionnelle** en banque/assurance.

---

# 🔧 **4. Comment implémenter l'optimisation ?**

On calcule :

* une liste de seuils
* les métriques pour chaque seuil
* on choisit celui maximisant la métrique désirée
* on recalcul la matrice de confusion finale

---

# 📚 **Conclusion Cours 3**

L’optimisation du seuil :

* **est indispensable** en crédit/fraude/assurance
* permet d’aligner le modèle avec les objectifs métier
* transforme un bon modèle en modèle réellement performant
* optimise les décisions, les coûts, et les risques

---

# 🎉 **Les trois cours sont maintenant complets.**

Si tu veux, je peux aussi :

### 🔹 Intégrer ces trois cours dans un **document PDF**

### 🔹 Ajouter des **schémas, graphiques, exemples**

### 🔹 Créer un **cours vidéo / plan de formation**

### 🔹 Construire un **notebook complet** incluant code + théorie

### 🔹 Créer des **exercices corrigés**

Souhaites-tu une **version PDF**, un **notebook**, ou un **résumé pour examen / oral** ?
