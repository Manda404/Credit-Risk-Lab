# MLOps Reproducibility

Ce projet sépare la configuration commune de la configuration propre à chaque
environnement.

## Configuration

La configuration commune est versionnée dans :

```text
configs/settings.yaml
```

Les overrides d'environnement sont versionnés dans :

```text
configs/environments/development.yaml
configs/environments/staging.yaml
configs/environments/production.yaml
```

L'environnement actif est choisi avec :

```bash
CRL_ENVIRONMENT=development
CRL_ENVIRONMENT=staging
CRL_ENVIRONMENT=production
```

`CRL_ENV` est aussi accepté comme alias pratique. Les variables `CRL_` gardent
la priorité sur les fichiers YAML afin de permettre des overrides de
déploiement.

## Séparation Des Artefacts

Chaque environnement possède ses propres chemins pour les artefacts générés :

```text
development -> data/processed, models, reports, logs, mlruns/development
staging     -> data/staging/processed, models/staging, reports/staging, logs/staging, mlruns/staging
production  -> data/production/processed, models/production, reports/production, logs/production, mlruns/production
```

Cette séparation évite qu'une expérimentation locale écrase un modèle ou un
rapport destiné à staging ou production.

## Isolation Du Test Set

Le dataset initial est splitté très tôt en deux fichiers bruts :

```text
data/raw/train.csv
data/raw/test.csv
```

`data/raw/test.csv` est réservé à l'évaluation finale du modèle et à la
simulation d'inférence. Il ne doit pas être utilisé pour :

- choisir les règles de nettoyage ;
- analyser les outliers ;
- créer ou ajuster des features métier ;
- sélectionner les hyperparamètres ;
- calibrer le seuil opérationnel.

Les notebooks d'analyse qualité et de feature engineering utilisent
`settings.raw_train_path`. Le split de développement produit ensuite :

```text
data/processed/train.csv
data/processed/validation.csv
```

Le contrôle `DataLeakageAuditor` vérifie l'absence d'overlap exact entre les
splits avant le preprocessing.

## Reproductibilité Déjà En Place

- dépendances verrouillées par `poetry.lock` ;
- version Python contrôlée par Poetry et Docker ;
- hyperparamètres versionnés dans `configs/models.yaml` ;
- `random_state` centralisé ;
- hashes SHA-256 des datasets/configs enregistrés dans le bundle modèle ;
- version runtime Python/librairies enregistrée dans le bundle ;
- environnement actif visible dans `/health` et `/ready` ;
- tests automatisés sur Python 3.13 et 3.14.

## Limites Actuelles

Cette séparation est une base solide pour un lab industrialisable, mais une
production réelle demanderait encore :

- registry modèle avec promotion formelle `development -> staging -> production` ;
- stockage distant et gouverné des artefacts ;
- versioning robuste des datasets avec DVC, Delta, LakeFS ou MLflow artifacts ;
- tracking MLflow systématique du training ;
- validation staging avant promotion production ;
- monitoring production : drift, qualité des données, latence, erreurs API,
  performance et calibration.

## Exemples

Exécuter les tests en staging :

```bash
CRL_ENVIRONMENT=staging poetry run pytest -q
```

Démarrer l'API Docker en development :

```bash
docker compose up api
```

Démarrer l'API en production exige de fournir les artefacts production attendus,
par exemple `models/production/best_boosting_model.joblib`.
