# API d’inférence et simulation de production

## Architecture

`loan_data.csv` est nettoyé puis séparé une seule fois en `train.csv` (90 %) et
`test.csv` (10 %). Le modèle est développé exclusivement à partir des 90 %. Le
simulateur lit une ligne brute de `test.csv`, retire `loan_status`, puis appelle
`POST /v1/predict`. L’API valide la requête, refait le feature engineering,
applique le préprocesseur appris et retourne la probabilité de risque.

Le champ `risk_decision` signifie dépassement du seuil de risque. Il ne constitue
pas à lui seul une décision d’octroi ou de refus.

## Seuil opérationnel

Le seuil réellement utilisé par l’API est défini dans `configs/settings.yaml` :

```yaml
decision_threshold: 0.25
```

Si `probability_of_risk >= decision_threshold`, l’API retourne
`risk_decision=1`. Le seuil sauvegardé dans le bundle reste une trace du seuil
expérimental sélectionné pendant la validation, mais il ne remplace pas la
politique opérationnelle YAML. Une variable d’environnement telle que
`CRL_DECISION_THRESHOLD=0.30` peut surcharger cette valeur au déploiement.
Toute modification doit être versionnée, testée et approuvée selon la fonction
de coût métier ; elle ne nécessite pas de réentraîner le modèle.

## Préparer les données et le modèle

```bash
PYTHONPATH=src poetry run python scripts/prepare_deployment.py
```

Cette commande recrée `data/processed/train.csv`, `test.csv` et le bundle sans
jamais entraîner sur les 10 % réservés à la simulation.

Elle calcule aussi le drift entre les deux partitions et génère :

- `reports/deployment_split_drift.csv` : PSI pour toutes les variables, KS et
  Hellinger pour les variables numériques ;
- `reports/deployment_split_drift.html` : graphique interactif PSI avec seuils
  de revue (`0.10`) et d’alerte (`0.25`).

## Tester sans Docker

Terminal 1 :

```bash
PYTHONPATH=src poetry run uvicorn credit_risk_lab.interfaces.api:app --reload
```

Documentation interactive : `http://localhost:8000/docs`. Santé : `/health`.

Terminal 2 :

```bash
PYTHONPATH=src poetry run python scripts/simulate_production.py --interval 2 --limit 20
```

Utiliser `--interval 0` pour un test rapide ou changer `--url` pour une API
déployée.

## Tester avec Docker

```bash
docker compose up --build api
docker compose --profile simulation up --build
```

Pour un cloud gratuit compatible conteneur, construire l’image à partir du
`Dockerfile`, exposer le port fourni par la variable `PORT`, conserver une seule
instance sur les petits plans, et ne jamais inclure `test.csv` dans l’image API.
Le modèle est actuellement intégré à l’image pour obtenir un déploiement
immuable. Un registre d’artefacts signé sera préférable ensuite.

## Exemple de réponse

```json
{
  "request_id": "...",
  "model_name": "LightGBM",
  "model_version": "0.2.0",
  "probability_of_risk": 0.17,
  "risk_decision": 0,
  "risk_label": "low_risk",
  "threshold": 0.25,
  "threshold_source": "configs/settings.yaml:decision_threshold",
  "scored_at_utc": "2026-07-11T12:00:00Z",
  "latency_ms": 4.2
}
```
