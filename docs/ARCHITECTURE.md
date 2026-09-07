# Architecture du projet

Ce projet suit une orientation Clean Architecture pour séparer les règles
métier crédit, les cas d’usage applicatifs, les détails techniques et les
points d’entrée utilisateur. Les notebooks ne portent plus la logique du
projet : ils instancient et exécutent explicitement les composants exposés par
`src/credit_risk_lab`.

## Vue d’ensemble

```text
dev/*.ipynb          scripts/*.py          FastAPI
    │                    │                   │
    └────────────── interfaces / runners ───┘
                         │
                         ▼
                 application / workflows
                         │
                         ▼
                       domain
                         ▲
                         │
                   infrastructure
```

Règle principale : les couches haut niveau ne doivent pas dépendre des détails
techniques bas niveau.

```text
domain          aucune dépendance projet externe
application     orchestre les cas d’usage
infrastructure  implémente CSV, Pandas, sklearn, MLflow, joblib, Plotly
interfaces      expose API HTTP, scripts, helpers de simulation
```

## Structure des dossiers

```text
src/credit_risk_lab/
  domain/
    entities/       objets métier et contrats de données
    services/       règles métier pures
    ports/          interfaces attendues par l’application

  application/
    workflows/      workflows prêts pour notebooks et CLI
    *.py            cas d’usage de split, training, scoring

  infrastructure/
    data_sources/   lecture/écriture CSV
    data_quality.py adaptateur Pandas des règles qualité
    feature_engineering/
    modeling/       preprocessing, wrappers ML, persistence joblib
    analytics/      drift
    evaluation/     métriques, calibration, fairness diagnostics
    visualization/  graphiques Plotly

  interfaces/
    api.py          routes FastAPI
    api_models.py   schémas HTTP Pydantic
    api_service.py  adaptation HTTP vers scoring applicatif
    api_simulation.py

  config/
    settings.py     configuration typée
```

## Domaine

Le domaine contient ce qui appartient au métier, pas aux librairies techniques.

Exemples :

- `LoanSchema` définit les colonnes attendues.
- `QualityReport` décrit le résultat du contrôle qualité.
- `credit_quality_policy.py` contient les règles d’âge, d’expérience et de
  cible binaire.
- `credit_feature_rules.py` contient les mappings métier : score bands,
  risque par motif de prêt, niveau d’éducation, risque de propriété.

Le domaine doit rester le plus stable possible. Il ne doit pas connaître
FastAPI, joblib, MLflow, Docker, notebooks, fichiers CSV ou chemins locaux.

## Application

La couche application orchestre les cas d’usage et expose aussi des composants
utilisables étape par étape depuis les notebooks.

Les workflows restent disponibles pour les scripts, la CLI, les tests
d’intégration et l’automatisation future. Les notebooks privilégient désormais
les classes et méthodes explicites afin de montrer la construction progressive
du pipeline.

Workflows disponibles :

- `run_source_quality_workflow`
- `run_split_and_drift_workflow`
- `run_feature_engineering_workflow`
- `run_training_workflow`
- `run_external_evaluation_workflow`

Cette couche décrit le scénario métier : charger, valider, splitter, entraîner,
évaluer, persister. Elle doit progressivement dépendre des ports du domaine
plutôt que d’importer directement les implémentations infrastructure.

## Infrastructure

L’infrastructure contient les détails remplaçables :

- Pandas pour manipuler les DataFrames ;
- scikit-learn pour le preprocessing ;
- XGBoost, LightGBM et CatBoost optionnel pour les modèles ;
- joblib pour les bundles locaux ;
- Plotly pour les visualisations ;
- CSV pour les datasets locaux ;
- MLflow pour le tracking.

Les règles métier utilisées par l’infrastructure doivent venir du domaine. Par
exemple, le feature engineering Pandas applique les mappings définis dans
`domain/services/credit_feature_rules.py`.

## Interfaces

Les interfaces sont les points d’entrée :

- notebooks dans `dev/` ;
- scripts dans `scripts/` ;
- API FastAPI dans `interfaces/api.py` ;
- simulation API dans `interfaces/api_simulation.py`.

Elles doivent rester fines. Leur rôle est de recevoir une demande, instancier
les composants nécessaires, appeler des méthodes publiques, puis afficher ou
retourner le résultat.

## Rôle des notebooks

Les notebooks sont réservés à l’exécution, à l’inspection et à la visualisation.
Ils ne doivent pas contenir la logique principale du projet, mais ils doivent
montrer clairement les briques utilisées.

Forme attendue :

```python
from credit_risk_lab.infrastructure.data_sources import CsvLoanDataLoader
from credit_risk_lab.infrastructure.analytics import DatasetInspector

loader = CsvLoanDataLoader(path=settings.raw_data_path)
raw_df = loader.load()

inspector = DatasetInspector(raw_df)
inspector.summary()
```

Ce qui ne doit pas revenir dans les notebooks :

- reconstruction manuelle de chemins ;
- `sys.path.insert(...)` ;
- logique de split détaillée ;
- entraînement manuel réimplémenté dans les cellules ;
- sauvegarde joblib hors repository dédié ;
- règles métier dupliquées.

Si une cellule devient longue, elle doit probablement devenir une fonction ou
une classe dans `src/credit_risk_lab`.

## Flux principaux

## Séquence des notebooks

```text
01_data_understanding.ipynb
02_data_quality.ipynb
03_split_drift_feature_engineering.ipynb
04_preprocessing_and_training.ipynb
05_model_evaluation_and_persistence.ipynb
06_inference_and_api_simulation.ipynb
```

Chaque notebook est reproductible seul à partir des données, chemins et bundles
exposés par `settings` et les repositories. La progression est conceptuelle,
pas une dépendance fragile à l’état mémoire du notebook précédent.

### Préparation du holdout et drift

```text
raw CSV
  │
  ▼
CsvLoanDataLoader
  │
  ▼
CreditRiskQualityChecker.clean()
  │
  ▼
DatasetSplitter.split()
  │
  ├── clean / split 90-10
  ├── CSVDatasetRepository.save()
  └── DriftAnalyzer.report_frame()
```

### Entraînement

```text
train.csv
  │
  ▼
LoanFeatureEngineer
  │
  ▼
CreditRiskPreprocessor.fit_transform()
  │
  ▼
BoostingModelTrainer.fit()
  │
  ▼
BestModelSelector.select()
```

### Inférence API

```text
POST /v1/predict
  │
  ▼
Pydantic validation
  │
  ▼
RawLoanScorer
  │
  ├── deterministic feature engineering
  ├── persisted preprocessor
  └── persisted model
```

## Configuration

La configuration versionnée est dans `configs/settings.yaml` et
`configs/models.yaml`. Les overrides propres à chaque runtime sont dans
`configs/environments/`.

`settings.py` centralise les chemins et expose des propriétés comme :

- `settings.raw_data_path`
- `settings.raw_train_path`
- `settings.raw_test_path`
- `settings.train_path`
- `settings.validation_path`
- `settings.model_bundle_path`
- `settings.reports_dir`

Le code et les notebooks ne doivent pas reconstruire ces chemins à la main.

La stratégie MLOps de reproductibilité et de séparation
`development / staging / production` est documentée dans
`docs/MLOPS_REPRODUCIBILITY.md`.

## Python 3.14 et CatBoost

Le projet cible actuellement :

```toml
python = ">=3.13,<3.15"
```

L’image Docker utilise Python 3.14. CatBoost est optionnel car son installation
peut échouer sous Python 3.14 selon les roues disponibles. Il est désactivé par
défaut dans `configs/models.yaml`.

## Tests et validation

Commande principale :

```bash
poetry run pytest -q
```

Les tests couvrent notamment :

- qualité des données ;
- feature engineering ;
- persistence et chargement des bundles modèle ;
- configuration des modèles ;
- entraînement ;
- API d’inférence ;
- contrôles MLOps.

La CI exécute Black et pytest sur Python 3.13 et 3.14 pour chaque branche.

## Dette technique restante

La structure Clean Architecture est en place, mais certaines dépendances doivent
encore être inversées pour aller plus loin :

- injecter les ports dans `TrainBoostingModelsUseCase` ;
- isoler davantage Pandas dans `infrastructure` ;
- éviter de versionner les artifacts générés (`models/*.joblib`,
  `reports/*.csv`, `reports/*.html`) ou les gérer avec MLflow/DVC.

Cette dette est maîtrisée : elle peut être réduite progressivement sans casser
les notebooks ni l’API.
