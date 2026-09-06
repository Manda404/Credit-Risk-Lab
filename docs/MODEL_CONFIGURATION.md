# Configuration et wrappers des modèles

Les hyperparamètres ne sont pas codés dans les wrappers. Ils sont versionnés
dans `configs/models.yaml`, validés par Pydantic, puis injectés par la factory.

```text
configs/models.yaml
        │
        ▼
load_models_config()      validation stricte
        │
        ▼
build_configured_models() sélection des modèles enabled
        │
        ├── LogisticRegressionWrapper
        ├── RandomForestWrapper
        ├── XGBoostWrapper
        ├── CatBoostWrapper
        └── LightGBMWrapper
```

## Activer ou désactiver un modèle

```yaml
models:
  catboost:
    enabled: false
    display_name: CatBoost
    early_stopping_rounds: 15
    parameters:
      iterations: 120
```

## Modifier un hyperparamètre

```yaml
models:
  lightgbm:
    enabled: true
    display_name: LightGBM
    early_stopping_rounds: 20
    parameters:
      n_estimators: 300
      learning_rate: 0.03
      num_leaves: 31
      n_jobs: -1
      verbosity: -1
```

Après une modification, relancer `scripts/prepare_deployment.py`. La
configuration doit être versionnée avec le modèle. Un nom de modèle inconnu ou
une clé de structure incorrecte provoque une erreur explicite au démarrage.

Les implémentations se trouvent dans `infrastructure/modeling/wrappers/`, avec
un fichier par algorithme. Il n’existe plus de façade intermédiaire :
`TrainBoostingModelsUseCase` appelle directement `build_configured_models()`.
