# Credit Risk Lab

A reproducible educational credit-risk pipeline comparing XGBoost, CatBoost, and LightGBM through a shared wrapper interface.

## What the project does

The repository implements a complete local experimental workflow:

1. load and validate the raw loan dataset;
2. flag and remove implausible age/experience records under an explicit rule;
3. inspect deterministic domain features;
4. create a clearly labelled experimental random split, or a chronological borrower-safe split when dates and identifiers exist;
5. fit imputation, scaling, and one-hot encoding on training data only;
6. train XGBoost, CatBoost, and LightGBM with early stopping;
7. plot training and validation log loss together;
8. select the candidate and threshold on validation data only;
9. evaluate only the locked winner on untouched test data;
10. save experiment reports and a reloadable model bundle.

The dataset is synthetic. Results are educational and must not be used for real lending decisions.

## Project layout


    src/credit_risk_lab/
      application/          # end-to-end use cases
      config/               # CRL-prefixed typed settings
      infrastructure/
        analytics/          # EDA and drift
        data_sources/       # CSV adapter
        evaluation/         # credit-risk metrics and lift
        feature_engineering/# deterministic domain features
        modeling/           # preprocessing, wrappers, persistence
        visualization/      # reusable Plotly figures
    dev/                    # six ordered, progressive execution notebooks
    tests/                  # unit tests
    docs/                   # architecture audit
    reports/                # generated experiment tables

## Notebooks

- `dev/01_data_understanding.ipynb`: explicit loading, column summary, target distribution, and EDA.
- `dev/02_data_quality.ipynb`: schema checks, quality report, cleaning, and clean-data inspection.
- `dev/03_split_drift_feature_engineering.ipynb`: external holdout split, persistence, drift, and deterministic features.
- `dev/04_preprocessing_and_training.ipynb`: development split, train-only preprocessing, candidate training, and selection.
- `dev/05_model_evaluation_and_persistence.ipynb`: metrics, calibration, fairness, metadata, and explicit bundle persistence.
- `dev/06_inference_and_api_simulation.ipynb`: bundle loading, raw scoring, Pydantic validation, API prediction, and simulation.

Notebooks contain explanations and orchestration only. Reusable logic belongs in `src/`.

## Installation


    poetry install

Environment variables use the `CRL_` prefix. Select the runtime environment
with `CRL_ENVIRONMENT=development`, `CRL_ENVIRONMENT=staging`, or
`CRL_ENVIRONMENT=production`.

Environment-specific overrides live in `configs/environments/`. See
[docs/MLOPS_REPRODUCIBILITY.md](docs/MLOPS_REPRODUCIBILITY.md).

## Run tests


    poetry run pytest -q

## Execute all notebooks


    poetry run python scripts/build_notebooks.py
    poetry run python -m nbconvert --to notebook --execute --inplace "dev/*.ipynb" --ExecutePreprocessor.timeout=600

## Current experimental result

On the deterministic random test split produced on 11 July 2026, LightGBM ranked first by ROC-AUC (0.9762), followed by XGBoost (0.9731) and CatBoost (0.9718). These values are not production claims: temporal validation, calibration review, fairness testing, explainability, governance, and independent model validation remain mandatory for real credit use.

`person_gender` and its derived `is_female` are excluded from training features by default (see `Settings.sensitive_columns`) and are only retained per-row in the test split for a separate fairness audit; they must never be used to train the model directly.

See [docs/ETAT_DE_L_ART.md](docs/ETAT_DE_L_ART.md) for the detailed audit and roadmap.

## Inference API and production simulation

The repository includes a validated FastAPI endpoint, a raw 90/10 external
holdout, a timed traffic simulator, and Docker packaging. See
[docs/API_DEPLOYMENT.md](docs/API_DEPLOYMENT.md) for local and container commands.

The operational risk threshold is versioned as `decision_threshold` in
`configs/settings.yaml` (currently `0.25`) and is returned by every prediction
response. It can be overridden at deployment with `CRL_DECISION_THRESHOLD`.

Candidate model activation and hyperparameters are versioned separately in
`configs/models.yaml`. See [docs/MODEL_CONFIGURATION.md](docs/MODEL_CONFIGURATION.md).
