# État de l’art du projet Credit Risk Lab

**Date de l’audit :** 10 juillet 2026  
**Version observée :** 0.1.0  
**Périmètre :** code Python, données, notebooks, configuration, qualité, MLOps, Databricks et gouvernance.

> **Mise à jour après implémentation — 10 juillet 2026 :** les lacunes techniques décrites ci-dessous correspondent à l’état initial audité. Le dépôt contient désormais une validation de données, un nettoyage explicite, un preprocessing sans fuite, des wrappers XGBoost/CatBoost/LightGBM, un split train/validation/test commun, early stopping, courbes de loss, métriques crédit, lift/gains, tests, persistance locale et trois notebooks anglais exécutés. Les limites réglementaires, temporelles, Databricks et de production restent valides.

## 1. Résumé exécutif

Credit Risk Lab est actuellement un **prototype pédagogique d’analyse et de préparation de données de crédit**, et non le framework de scoring « production-ready » annoncé par le README.

Le dépôt possède de bonnes premières briques : organisation inspirée de la Clean Architecture, ports d’accès aux données et de feature engineering, adaptateur CSV, split stratifié, analyse exploratoire, métriques de drift et 17 variables dérivées. Mais la chaîne ML centrale n’existe pas encore : aucun entraînement, pipeline appris, validation, calibration, optimisation de seuil, évaluation, explicabilité, persistance, tracking MLflow, registry, API, CLI, test, CI/CD ou composant Databricks n’est implémenté.

| Axe | Niveau / 5 | Constat |
|---|---:|---|
| Exploration des données | 2 | Présente dans les notebooks, peu industrialisée |
| Architecture logicielle | 2 | Bonne intention, contrats et frontières incohérents |
| Feature engineering | 2 | Riche, mais fuite statistique et règles non validées |
| Modélisation et évaluation | 0 | Absentes |
| MLOps / Databricks | 0 | Dépendances et configuration seulement |
| Tests et qualité | 0 | Aucun test collecté, import bloqué |
| Gouvernance / conformité | 0 | Non traitée |
| Déployabilité | 0 | Aucun artefact ni interface d’inférence |

**Verdict :** une base de laboratoire utile, au stade pré-modélisation. La priorité est de rendre les données, transformations, évaluations et décisions reproductibles, sans fuite, testées et gouvernées.

## 2. Méthode d’audit

L’audit s’appuie sur l’inventaire et la lecture du dépôt, l’inspection des 66 cellules des deux notebooks, l’exécution de contrôles sur le CSV local, la compilation de `src/`, l’exécution de `pytest`, des essais d’import et la comparaison avec les pratiques modernes du scoring crédit.

Les fonctions seulement annoncées dans le README ne sont pas considérées comme implémentées.

## 3. Architecture réellement présente

```text
CSV
 │
 ▼
CSVDatasetRepository ──► SplitDatasetUseCase ──► train.csv / test.csv
 │
 ├──► DataAnalyzer (résumé, types, graphiques)
 ├──► DriftAnalyzer (PSI, KS, Hellinger numériques)
 └──► BuildFeaturesUseCase ──► LoanFeatureEngineer ──► DataFrame enrichi

Après ce point : aucun entraînement, évaluation, registry ou service.
```

### Composants présents

- `config/settings.py` : chemins et paramètres déclaratifs pour modèles, MLflow, drift et API.
- `domain/ports/` : protocoles pour le repository et le feature engineering.
- `application/` : cas d’usage de split et de création de features.
- `infrastructure/data_sources/` : lecture et écriture CSV.
- `infrastructure/feature_engineering/` : transformations Pandas.
- `infrastructure/analytics/` : synthèse, visualisation et drift.
- `shared/logging.py` : logs console et fichier rotatif.
- `notebooks/` et `learn/` : exploration et notes théoriques.

### Capacités annoncées mais absentes

Entités et services métier, DTO, entraînement Logistic Regression/XGBoost/CatBoost/LightGBM, preprocessing sérialisable, champion/challenger, métriques de modèle, MLflow Tracking/Registry, stockage de modèles, CLI, API, monitoring de performance, tests, workflows Databricks, Unity Catalog, serving et CI/CD.

## 4. État mesuré des données

Le fichier local contient **45 000 lignes et 14 colonnes**, sans valeur manquante ni doublon exact. La cible contient 35 000 observations de classe 0 (77,78 %) et 10 000 de classe 1 (22,22 %). Le déséquilibre est modéré : il justifie stratification, PR-AUC et métriques par classe, mais pas automatiquement SMOTE.

### Anomalies et risques

- `person_age` atteint 144 ans et `person_emp_exp` 125 ans : valeurs invraisemblables à contrôler.
- `person_income` atteint 7 200 766 : les extrêmes doivent être analysés de façon robuste.
- Le dataset est synthétique et annoncé comme enrichi par SMOTENC. Il ne permet pas de conclure à une performance, une équité ou une conformité réelles.
- La sémantique de `loan_status` est ambiguë. Le README dit « 1 = approuvé », alors que ce nom désigne souvent un défaut/risque dans ce dataset. La source doit être vérifiée et un dictionnaire de données doit figer la définition.
- Le taux d’intérêt et certains ratios peuvent être déterminés durant l’octroi. Il faut prouver leur disponibilité au moment exact de la décision pour éviter une fuite temporelle ou décisionnelle.
- `person_gender` est sensible. Son dérivé `is_female` devrait par défaut servir à l’audit d’équité, pas à l’entraînement.

## 5. Analyse du code

### Points solides

- Layout `src/` et packaging Poetry.
- Accès CSV injecté derrière un `Protocol`.
- Split stratifié et reproductible.
- Transformations séparées en fonctions testables.
- Copie de l’entrée et validation des colonnes dans l’implémentation concrète.
- Logging centralisé avec rotation.

### Défauts bloquants ou importants

1. **Configuration non importable dans l’environnement audité.** `Settings` lit la variable générique `DEBUG`, qui vaut ici `release`; Pydantic ne peut pas la convertir en booléen. Il faut un préfixe tel que `CRL_`.

2. **API publique du feature engineering cassée.** Le fichier `infrastructure/feature_engineering/__init__.py` contient une seconde classe incomplète appelant des méthodes inexistantes, au lieu d’exporter celle de `pandas_feature_engineer.py`.

3. **Contrat de sauvegarde incohérent.** Le port attend `save(data, path: Path)`, alors que l’adaptateur traite le second argument comme un nom et écrit toujours sous `data/processed/`.

4. **Clean Architecture non stricte.** Les ports du domaine importent Pandas et `Path`; l’application dépend de scikit-learn. C’est acceptable pour un petit laboratoire, mais contredit la promesse « strictement respectée ».

5. **Effets de bord de configuration.** Importer les settings crée des dossiers automatiquement.

6. **Dépendances mal classées.** Le code runtime importe Plotly alors que Plotly est une dépendance de développement.

7. **Code mort ou dupliqué.** `logging_old.py`, fichiers vides, imports inutilisés et deux classes concurrentes augmentent la confusion.

8. **Notebooks non reproductibles.** Un chemin absolu vise un autre checkout ; plusieurs cellules sont vides ou répétées et l’exploration est mélangée à des questions d’entretien.

9. **README non aligné.** Il documente une arborescence, des commandes et des capacités inexistantes.

## 6. Analyse critique du feature engineering

La transformation concrète ajoute 17 colonnes : ratios de solvabilité, revenu logarithmique, charge mensuelle estimée, bandes et normalisation du score, groupes d’âge, ratios d’expérience/historique, variables de prêt, défaut antérieur, encodages métier et interactions.

Principaux risques :

- **Fuite train/test :** la moyenne et l’écart-type de `norm_credit_score` sont recalculés sur chaque DataFrame. L’inférence ne réutilise pas les statistiques du train.
- **Pas de pipeline sérialisable :** aucun `fit`, aucun état appris et aucune version des features ; `is_training` n’est pas utilisé.
- **Mensualité incorrecte :** `loan_amnt * monthly_rate` n’est pas une mensualité amortissable sans durée de prêt.
- **Bandes et mappings arbitraires :** les seuils FICO-like, `loan_intent_risk`, `home_risk` et `edu_level` ne sont ni sourcés ni validés.
- **Signe contre-intuitif :** `risky_default_score` devient négatif au-dessus d’un score de 650.
- **Redondance :** le DTI recouvre `loan_percent_income`; plusieurs produits peuvent accroître colinéarité et instabilité.
- **NaN résiduels :** les infinis deviennent des valeurs manquantes sans stratégie d’imputation.

Cible recommandée : un pipeline unique validation du schéma → règles de qualité → split → `ColumnTransformer` → transformations ajustées sur train uniquement → modèle → calibration. Les statistiques apprises doivent être sérialisées avec le modèle.

## 7. État de l’art d’un scoring crédit moderne

### Données et cible

Définir avant tout modèle : événement cible, horizon d’observation, fenêtre de performance, date de décision, population admissible, exclusions, coûts d’erreur et politique d’acceptation. Avec des données datées, utiliser un test hors temps plutôt qu’un simple split aléatoire.

Contractualiser schéma, types, plages, catégories, fraîcheur, unicité, disponibilité à la décision et lineage. Les données synthétiques conviennent à l’apprentissage technique, pas à la validation économique ou réglementaire.

### Modélisation et évaluation

Commencer par une régression logistique/scorecard interprétable et utiliser le boosting comme challenger. Ne pas sélectionner sur l’accuracy : mesurer ROC-AUC, PR-AUC, KS, Gini, log-loss/Brier, confusion au seuil métier, recall défaut, précision, lift et gains par décile.

Un score doit être calibré : calibration plot, Brier score, Platt ou isotonic sur un jeu distinct. Le seuil doit optimiser une fonction de coût métier — perte attendue, marge, capacité de revue humaine — et non rester arbitrairement à 0,5. Ajouter cross-validation, test hors temps, intervalles de confiance et stress tests.

### Explicabilité et équité

Prévoir explications globales et locales, reason codes de refus, analyse de sensibilité et documentation des limites. Évaluer sélection, TPR/FPR, calibration et performance par groupe protégé et intersections. Une métrique d’équité ne remplace ni la revue juridique ni l’analyse causale.

### MLOps et Databricks

Le standard attendu comprend version des données/code/configuration, runs MLflow, signature et exemple d’entrée, registry avec approbation, tests avant promotion, déploiement immuable, rollback, audit trail et surveillance des données, features, scores, performance différée, calibration et équité.

Sur Databricks, une cible cohérente serait Delta Lake et Unity Catalog pour données/lineage, Workflows et Asset Bundles pour orchestration/déploiement, MLflow pour expériences/modèles, puis Model Serving seulement si le besoin temps réel est établi. Rien de cela n’est actuellement implémenté ici.

### Gouvernance et conformité européenne

L’évaluation de solvabilité de personnes physiques est un usage sensible. Le règlement européen sur l’IA classe certains systèmes de crédit aux personnes physiques parmi les systèmes à haut risque : gestion des risques, gouvernance des données, documentation, traçabilité, supervision humaine, robustesse et suivi doivent être conçus dès le départ. Les lignes directrices EBA encadrent la gouvernance de l’octroi, l’évaluation de solvabilité et les modèles automatisés. Le RGPD encadre les décisions exclusivement automatisées et prévoit notamment intervention humaine et contestation selon les cas.

Références officielles :

- [Règlement (UE) 2024/1689 sur l’intelligence artificielle — EUR-Lex](https://eur-lex.europa.eu/eli/reg/2024/1689/oj)
- [EBA — Guidelines on loan origination and monitoring](https://eba.europa.eu/activities/single-rulebook/regulatory-activities/credit-risk/guidelines-loan-origination-and-monitoring)
- [RGPD, notamment l’article 22 — EUR-Lex](https://eur-lex.europa.eu/legal-content/FR/TXT/?uri=CELEX:32016R0679)

Cette section est une orientation technique, pas un avis juridique.

## 8. Tests et reproductibilité

Résultats locaux :

- `python -m compileall -q src` : succès syntaxique ;
- `pytest -q` : aucun test trouvé, code de sortie 5 ;
- import sans installation/PYTHONPATH : impossible dans l’environnement actuel ;
- import avec `PYTHONPATH=src` : bloqué par `DEBUG=release` ;
- aucun workflow CI détecté ;
- `poetry.lock` est présent.

Tests prioritaires : configuration isolée, conformité ports/adaptateurs, schéma et plages, split sans chevauchement, transformations aux limites, absence de fit sur test, cohérence train/inférence, sérialisation, métriques/seuil, smoke test de bout en bout, équité, drift et non-régression.

## 9. Feuille de route priorisée

### P0 — Rendre le socle fiable

- Préfixer les variables d’environnement et corriger l’import.
- Supprimer la classe feature engineer dupliquée et exporter l’implémentation réelle.
- Aligner le port et l’adaptateur de sauvegarde.
- Clarifier la cible et créer le dictionnaire de données.
- Ajouter validation de schéma et règles pour âge, expérience, revenu et catégories.
- Éliminer les chemins absolus.
- Ajouter pytest et une CI minimale.
- Réaligner le README sur la réalité.

**Sortie attendue :** clone neuf installable, imports fiables, tests verts et exemple minimal reproductible.

### P1 — Construire une baseline ML correcte

- Pipeline `fit/transform` sans fuite.
- Baseline logistique et challenger boosting.
- Train/validation/test, idéalement hors temps.
- ROC-AUC, PR-AUC, KS, Gini, Brier, lift/gains et métriques au seuil.
- Calibration et seuil fondé sur le coût métier.
- Sérialisation du pipeline, schéma et métadonnées.
- MLflow Tracking avec artefacts d’évaluation.

**Sortie attendue :** un run reproductible produit un modèle traçable, rechargeable et évalué.

### P2 — Gouvernance crédit et MLOps

- Model card, data sheet, lineage et registre des risques.
- Attributs sensibles exclus de l’entraînement par défaut ; audit d’équité séparé.
- Reason codes et procédure de revue humaine.
- Registry, validation indépendante, approbation et rollback.
- Jobs Databricks déclaratifs ; serving uniquement si nécessaire.
- Monitoring qualité, drift, calibration, performance différée et équité.

**Sortie attendue :** promotion contrôlée, audit trail, alertes actionnables et procédure d’incident testée.

### P3 — Industrialisation avancée

Backtesting temporel, stress tests, stabilité des features, champion/challenger, canary, tests de charge/sécurité/résilience. Le réentraînement doit dépendre de labels et d’une validation formelle, jamais du seul drift.

## 10. Structure cible suggérée

```text
src/credit_risk_lab/
├── domain/             # politique de décision et objets métier purs
├── application/        # train, evaluate, score, monitor
├── infrastructure/
│   ├── data/           # CSV puis Delta/Unity Catalog
│   ├── features/       # transformeurs fit/transform
│   ├── modeling/       # pipelines et calibration
│   ├── tracking/       # MLflow
│   └── monitoring/     # qualité, drift, performance, équité
├── interfaces/         # CLI; API seulement si nécessaire
└── config/
tests/
├── unit/
├── integration/
└── end_to_end/
configs/
databricks.yml
```

## 11. Conclusion

Le projet dispose d’une intention architecturale claire et d’un jeu de transformations utile pour apprendre. Son principal risque est l’écart entre cette intention et les capacités réellement livrées. La bonne trajectoire est d’assainir le socle, verrouiller les données et la cible, construire une baseline sans fuite et testée, puis seulement ajouter MLflow, Databricks et le déploiement.

À court terme, le terme juste est **laboratoire de préparation et d’analyse du risque crédit**. Le qualificatif **production-ready** ne deviendra défendable qu’après l’implémentation et la preuve des contrôles P0 à P2.

## 12. Suivi — 11 juillet 2026

- **Bug de pipeline corrigé :** le notebook 02 calculait `feature_df` (17 variables métier) mais persistait `clean_df` dans `modeling_dataset.csv`. Le notebook 03 entraînait donc les trois modèles sur les seules colonnes brutes nettoyées, sans aucune des variables de feature engineering. `scripts/build_notebooks.py` sauvegarde désormais `feature_df`. Impact mesuré sur ROC-AUC : marginal (0.9762 → 0.9762 LightGBM, 0.9728 → 0.9731 XGBoost, 0.9714 → 0.9718 CatBoost) car les arbres de boosting redécouvraient déjà une grande partie de l'information des ratios/bandes à partir des variables brutes ; le bug restait néanmoins réel et invalidait la valeur du module de feature engineering.
- **Fuite statistique résiduelle supprimée :** `norm_credit_score` était un z-score recalculé sur le DataFrame reçu par `LoanFeatureEngineer.transform()`, appelé sur l'ensemble du dataset avant le split — donc contaminé par les statistiques test/validation. La variable a été retirée ; le score brut reste standardisé de façon sûre par le `ColumnTransformer` fit sur train uniquement.
- **Attribut sensible exclu de l'entraînement par défaut :** `TrainBoostingModelsUseCase` retire désormais `person_gender` et `is_female` (`Settings.sensitive_columns`) des features avant le preprocessing et le fit, tout en conservant ces colonnes pour le jeu de test dans `TrainingResult.sensitive_test`, en vue d'un audit d'équité séparé.
- **Code mort supprimé :** `application/split_dataset.py` (`SplitDatasetUseCase`, doublon non utilisé de `dataset_splitting.three_way_stratified_split`, defaults non alignés sur `Settings`) et `shared/utils.py` (fichier vide).
- **CI minimale ajoutée :** `.github/workflows/ci.yml` installe les dépendances via Poetry et exécute `pytest` sur chaque push/PR vers `main`.

Ces correctifs restent des corrections de socle (P0/P1). Les limites réglementaires, temporelles et de production listées en section 9 demeurent valides.
