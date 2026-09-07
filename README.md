# Credit Risk Lab

Credit Risk Lab est un projet end-to-end de Machine Learning et MLOps pour le
scoring du risque credit. L'objectif est de construire un pipeline
industrialisable capable d'identifier les demandes de pret a risque eleve, tout
en conservant une separation stricte entre exploration, preprocessing,
entrainement, tuning, evaluation finale et simulation d'inference.

Le projet est structure autour d'une clean architecture : les notebooks servent
uniquement a orchestrer et executer les etapes, tandis que la logique reusable
vit dans `src/credit_risk_lab`.

> Dataset synthetique et usage pedagogique : les resultats ne doivent pas etre
> utilises pour prendre de vraies decisions de credit sans validation metier,
> juridique, model risk management et gouvernance complete.

## Probleme Metier

Dans un contexte credit risk, chaque demande de pret doit etre classee selon son
niveau de risque :

- `loan_status = 0` : profil considere comme faible risque.
- `loan_status = 1` : profil considere comme haut risque.

La difficulte ne se limite pas a obtenir un bon score global. Un modele de
credit doit aussi permettre de comprendre :

- quelles variables influencent la decision ;
- comment eviter le data leakage ;
- comment garder un jeu de test strictement untouched ;
- comment choisir un seuil de decision adapte au risque metier ;
- combien de dossiers haut risque sont captures ou rates ;
- quel est le niveau de calibration des probabilites ;
- si certaines variables sensibles sont exclues de l'entrainement ;
- si le pipeline peut etre rejoue de maniere reproductible.

## Ce Que Le Projet Demontre

Ce repository met en avant une demarche complete de ML Engineering :

- Architecture clean avec couches `domain`, `application`, `infrastructure` et
  `interfaces`.
- Configuration centralisee dans `configs/settings.yaml` et
  `configs/models.yaml`.
- Separation des environnements `development`, `staging` et `production`.
- Split initial avec `train.csv` et `test.csv` dans `data/raw`.
- Jeu de test final reserve uniquement a l'evaluation finale.
- Analyse exploratoire et data quality sans polluer les notebooks avec de la
  logique metier.
- Detection de data leakage avant preprocessing.
- Feature engineering deterministe encapsule dans des classes Python.
- Preprocessing fit uniquement sur le train, puis applique a la validation et au
  test.
- Sauvegarde des datasets transformes dans `data/processed`.
- Selection de baseline models sans tuning agressif.
- Optimisation CatBoost avec Optuna sur `roc_auc`.
- Feature importance du modele CatBoost selectionne et optimise.
- Evaluation finale sur test untouched avec seuil de decision versionne.
- Analyses avancees : ROC-AUC, PR-AUC, KS, Gini, Brier score, ECE, lift/gain,
  deciles, intervalle de confiance bootstrap, calibration, fairness diagnostics.
- Bundle modele + preprocessor pour inference reproductible.
- API FastAPI et simulation de prediction.
- Pipeline end-to-end rejouable en une seule classe Python.
- Tests unitaires et formatage pour securiser les changements.

## Architecture

```text
src/credit_risk_lab/
  domain/
    entities/                 # objets metier et contrats de resultat
    ports/                    # interfaces attendues par le domaine
    services/                 # regles metier pures

  application/
    workflows/                # cas d'usage orchestration end-to-end
    scoring.py                # scoring applicatif
    dataset_splitting.py      # logique de split

  infrastructure/
    analytics/                # inspection, leakage, drift, reports
    data_sources/             # lecture CSV et repositories
    evaluation/               # metriques, seuils, lift, fairness
    feature_engineering/      # transformations metier pandas
    modeling/                 # preprocessing, model wrappers, persistence
    visualization/            # figures Plotly reutilisables

  interfaces/
    api.py                    # API FastAPI
    api_models.py             # schemas Pydantic
    api_service.py            # service d'inference
    api_simulation.py         # simulation d'appels
```

Les notebooks dans `dev/` importent ces classes et fonctions. Ils ne portent pas
la logique principale du projet.

## Pipeline MLOps

### 1. Data Understanding

Notebook : `dev/01_data_understanding.ipynb`

Objectif :

- charger le dataset brut ;
- comprendre les colonnes disponibles ;
- visualiser la distribution de la target ;
- produire un resume clair des colonnes ;
- eviter de dupliquer les analyses plus detaillees du notebook 02.

### 2. Data Quality

Notebook : `dev/02_data_quality.ipynb`

Objectif :

- appliquer les controles qualite ;
- analyser les valeurs manquantes, cardinalites, types et outliers ;
- produire des visualisations numeriques et categorielles ;
- nettoyer les valeurs impossibles selon des regles explicites ;
- preparer un dataset propre avant split/modeling.

### 3. Split, Drift Et Feature Engineering

Notebook : `dev/03_split_drift_feature_engineering.ipynb`

Objectif :

- creer un split initial `train` / `test` ;
- sauvegarder le test brut dans `data/raw/test.csv` ;
- ne jamais utiliser ce test pendant exploration, preprocessing, selection ou
  tuning ;
- analyser le drift entre partitions ;
- verifier les risques de data leakage ;
- creer les features metier ;
- afficher l'evolution des colonnes creees ;
- fitter le preprocessor sur le train seulement ;
- transformer train et validation avec le meme preprocessor ;
- sauvegarder les datasets transformes dans `data/processed`.

### 4. Baseline Model Selection

Notebook : `dev/04_preprocessing_and_training.ipynb`

Objectif :

- charger les datasets processed ;
- entrainer plusieurs modeles sans recherche d'hyperparametres lourde ;
- comparer les baselines sur validation ;
- ajouter `LogisticRegression` comme baseline lineaire interpretable ;
- selectionner le meilleur modele selon `settings.selection_metric` ;
- produire la feature importance lorsque le modele selectionne est CatBoost.

### 5. Hyperparameter Tuning

Notebook : `dev/05_hyperparameter_tuning.ipynb`

Objectif :

- reconstruire le ranking baseline ;
- optimiser uniquement le meilleur modele retenu : CatBoost ;
- utiliser Optuna avec `roc_auc` comme metrique d'optimisation ;
- visualiser la progression des trials ;
- analyser l'importance des hyperparametres ;
- comparer baseline CatBoost vs CatBoost optimise ;
- sauvegarder le bundle optimise.

### 6. Evaluation Finale

Notebook : `dev/06_model_evaluation_and_persistence.ipynb`

Objectif :

- charger le modele optimise ;
- charger uniquement maintenant le test untouched ;
- appliquer le meme feature engineering et le meme preprocessor ;
- calculer les metriques finales ;
- analyser la matrice de confusion avec la ROC curve ;
- etudier le trade-off precision / recall selon plusieurs seuils ;
- mesurer lift, gain, accumulation, KS, Gini, calibration, Brier, ECE ;
- produire des intervalles de confiance bootstrap ;
- executer un diagnostic fairness hors entrainement.

### 7. Batch Et Realtime Inference

Notebook : `dev/07_batch_and_realtime_inference.ipynb`

Objectif :

- charger le bundle final ;
- scorer les donnees de test en batch ;
- sauvegarder un fichier `submission.csv` ;
- simuler des requetes utilisateur une par une sans API ;
- logger chaque prediction temps reel ;
- verifier que le seuil de decision versionne est bien applique.

### 8. CI/CD MLOps Pipeline

Notebook : `dev/08_end_to_end_mlops_pipeline.ipynb`

Objectif :

- executer le pipeline complet avec `CreditRiskMLOpsPipeline` ;
- rejouer lecture, split, qualite, drift, feature engineering et preprocessing ;
- entrainer les baselines et optimiser CatBoost ;
- evaluer le modele final sur le test untouched ;
- generer les rapports metier et les artefacts modele ;
- rester compatible avec une execution CI/CD ;
- retourner un resume compact des artefacts et metriques.

## Strategie Modele

La selection suit une logique realiste :

1. Entrainer des baselines simples sans tuning lourd.
2. Comparer les performances sur validation.
3. Identifier le meilleur candidat.
4. Optimiser uniquement le meilleur modele avec Optuna.
5. Evaluer une seule fois sur le test untouched.

Les candidats sont declares dans `configs/models.yaml` :

- `LogisticRegression`
- `RandomForest`
- `XGBoost`
- `CatBoost`
- `LightGBM` desactive pour le moment

CatBoost est le modele cible du projet actuel, car il gere bien les problemes
tabulaires et permet de garder un preprocessing compact sans explosion
artificielle du nombre de colonnes.

## Reproductibilite

Le projet contient plusieurs garde-fous MLOps :

| Sujet | Implementation |
| --- | --- |
| Version Python | `python = ">=3.13,<3.15"` dans `pyproject.toml` |
| Dependances | `poetry.lock` versionne |
| Configuration | `configs/settings.yaml` et `configs/models.yaml` |
| Environnements | `configs/environments/development.yaml`, `staging.yaml`, `production.yaml` |
| Tuning | `optuna_trials` centralise et surchargeable via `CRL_OPTUNA_TRIALS` |
| Promotion | `minimum_validation_roc_auc` controle le gate avant promotion du modele |
| Randomness | `random_state` centralise |
| Donnees | separation `data/raw` et `data/processed` |
| Test untouched | test charge uniquement en evaluation finale |
| Preprocessing | fit sur train, transform sur validation/test |
| Sensibles | `person_gender` et `is_female` exclus de l'entrainement |
| Model artifact | bundle modele + preprocessor + metadata |
| Validation | tests unitaires avec `pytest` |

## Artefacts Produits

Les artefacts generes localement sont separes du code :

```text
data/raw/
  loan_data.csv              # dataset source
  train.csv                  # partition brute de developpement
  test.csv                   # holdout final untouched

data/processed/
  train.csv                  # train transforme
  validation.csv             # validation transformee
  credit_risk_preprocessor.joblib

models/
  candidates/candidate_model.joblib
  best_boosting_model.joblib # bundle promu si le quality gate passe

reports/
  boosting_model_metrics.csv
  model_promotion_report.csv
  deployment_split_drift.csv
  deployment_split_drift.html
```

Les dossiers de donnees et de rapports peuvent etre ignores par Git selon la
politique du projet afin d'eviter de versionner des artefacts lourds ou generes.

## Installation

```bash
poetry install
```

Pour installer le kernel Jupyter dans l'environnement Poetry :

```bash
poetry run python -m ipykernel install --user --name credit-risk-lab
```

Selectionner un environnement :

```bash
export CRL_ENVIRONMENT=development
export CRL_ENVIRONMENT=staging
export CRL_ENVIRONMENT=production
```

## Commandes Utiles

Regenerer les notebooks canoniques :

```bash
poetry run python scripts/build_notebooks.py
```

Creer le split de deploiement :

```bash
poetry run python scripts/create_deployment_split.py
```

Executer les tests :

```bash
poetry run pytest
```

Verifier le formatage :

```bash
poetry run black --check src scripts tests
```

Lancer l'API localement :

```bash
poetry run uvicorn credit_risk_lab.interfaces.api:app --reload
```

## Evaluation Metier

Le projet ne s'arrete pas a `accuracy`, car cette metrique est souvent
insuffisante en credit risk. Les analyses finales permettent de repondre a des
questions plus utiles :

- Combien de profils haut risque sont captures ?
- Combien de profils haut risque sont rates ?
- Combien de fausses alertes sont generees ?
- Quel seuil maximise le compromis metier ?
- Le modele classe-t-il bien les dossiers du plus risque au moins risque ?
- Le top decile concentre-t-il une part importante des defauts ?
- Les probabilites sont-elles bien calibrees ?
- Les performances sont-elles stables selon les intervalles de confiance ?
- Les variables sensibles exclues presentent-elles quand meme des ecarts
  d'impact en evaluation ?

Cette logique rend le projet plus proche d'un vrai workflow de model risk et de
decision science.

## Roadmap Production

Les prochaines etapes naturelles pour aller vers une version plus production :

- ajouter un tracking MLflow complet des runs, params, metrics et artefacts ;
- brancher un model registry avec promotion `staging` vers `production` ;
- ajouter une CI/CD avec tests, lint, build Docker et validation notebooks ;
- definir des data contracts automatises ;
- automatiser le monitoring drift/performance/calibration apres deploiement ;
- formaliser une validation fairness et gouvernance metier plus stricte ;
- ajouter des tests d'integration API et batch scoring.

## Positionnement

Ce projet montre une demarche complete de Senior ML Engineer / MLOps Engineer :
partir d'un probleme credit risk, structurer le code en architecture propre,
proteger la reproductibilite, entrainer et optimiser un modele, evaluer avec des
metriques metier avancees, puis preparer l'inference et le deploiement.
