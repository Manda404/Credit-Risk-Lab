"""Generate progressive execution-lab notebooks for Credit Risk Lab."""

from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
DEV = ROOT / "dev"


def markdown(text: str):
    """Create a stripped Markdown cell."""
    return nbf.v4.new_markdown_cell(text.strip())


def code(text: str):
    """Create a stripped Python cell."""
    return nbf.v4.new_code_cell(text.strip())


def notebook(title: str, objective: str, cells: list):
    """Build one documented Python 3 notebook."""
    nb = nbf.v4.new_notebook()
    nb.metadata.kernelspec = {
        "display_name": ".venv (Python 3.14.1)",
        "language": "python",
        "name": "python3",
    }
    nb.metadata.language_info = {"name": "python", "version": "3.14"}
    nb.cells = [markdown(f"# {title}\n\n**Objectif :** {objective}")] + cells
    return nb


setup = code(
    """
import pandas as pd
import plotly.express as px

from credit_risk_lab.config.settings import settings

print(f"Project root: {settings.project_root}")
print(f"Environment: {settings.environment}")
"""
)


data_understanding = notebook(
    "01 - Data Understanding",
    "charger le dataset initial, créer immédiatement un train/test brut, puis inspecter uniquement le train.",
    [
        setup,
        markdown("## 1. Data source"),
        code(
            """
from credit_risk_lab.infrastructure.data_sources import CsvLoanDataLoader

loader = CsvLoanDataLoader(
    path=settings.raw_data_path,
    sep=settings.raw_data_sep,
    encoding=settings.raw_data_encoding,
)

raw_df = loader.load()
raw_df.head()
"""
        ),
        markdown("## 2. Initial raw train/test split"),
        code(
            """
from credit_risk_lab.application import DatasetSplitter, SplitConfig
from credit_risk_lab.infrastructure.data_sources import CSVDatasetRepository

initial_split_config = SplitConfig(
    test_size=0.10,
    random_state=settings.random_state,
    stratify=True,
)
initial_splitter = DatasetSplitter(initial_split_config)

raw_train_df, raw_test_df = initial_splitter.split(raw_df)
split_summary = initial_splitter.summary(raw_train_df, raw_test_df)

CSVDatasetRepository.save(raw_train_df, settings.raw_train_path)
CSVDatasetRepository.save(raw_test_df, settings.raw_test_path)

split_summary
"""
        ),
        markdown("## 3. Initial leakage check"),
        code(
            """
from credit_risk_lab.infrastructure.analytics import DataLeakageAuditor

leakage_auditor = DataLeakageAuditor(target_column=settings.target_column)
leakage_auditor.row_overlap_report(
    raw_train_df,
    raw_test_df,
    holdout_name="raw_test",
)
"""
        ),
        markdown("## 4. Train-only dataset inspection"),
        code(
            """
from credit_risk_lab.infrastructure.analytics import DatasetInspector

inspector = DatasetInspector(raw_train_df)
summary = inspector.summary()
summary
"""
        ),
        markdown("## 5. Train-only column summary"),
        code(
            """
column_summary = inspector.column_summary(sample_size=2)
column_summary
"""
        ),
        markdown("## 6. Train-only target distribution"),
        code(
            """
target_distribution = inspector.target_distribution(settings.target_column)
display(target_distribution)

from credit_risk_lab.infrastructure.visualization import plot_target_distribution

plot_target_distribution(target_distribution).show()
"""
        ),
    ],
)


data_quality = notebook(
    "02 - Data Quality",
    "appliquer les règles qualité et les diagnostics colonnes uniquement sur le train brut.",
    [
        setup,
        markdown("## 1. Load raw train data"),
        code(
            """
from credit_risk_lab.infrastructure.data_sources import CsvLoanDataLoader

loader = CsvLoanDataLoader(path=settings.raw_train_path)
raw_train_df = loader.load()
raw_train_df.shape
"""
        ),
        markdown("## 2. Quality checker"),
        code(
            """
from credit_risk_lab.domain.entities import LoanSchema
from credit_risk_lab.infrastructure import CreditRiskQualityChecker

quality_checker = CreditRiskQualityChecker(schema=LoanSchema())
quality_report = quality_checker.validate(raw_train_df)
quality_report
"""
        ),
        markdown("## 3. Clean implausible rows"),
        code(
            """
clean_train_df = quality_checker.clean(raw_train_df)

pd.DataFrame(
    [
        {"dataset": "raw_train", "rows": len(raw_train_df), "duplicates": raw_train_df.duplicated().sum()},
        {"dataset": "clean_train", "rows": len(clean_train_df), "duplicates": clean_train_df.duplicated().sum()},
    ]
)
"""
        ),
        markdown("## 4. Inspect clean data"),
        code(
            """
from credit_risk_lab.infrastructure.analytics import DatasetInspector

clean_inspector = DatasetInspector(clean_train_df)
display(clean_inspector.summary())
clean_inspector.target_distribution(settings.target_column)
"""
        ),
        markdown("## 5. Numeric outlier diagnostics"),
        code(
            """
from credit_risk_lab.infrastructure.analytics import ColumnDiagnostics

diagnostics = ColumnDiagnostics(clean_train_df, target_column=settings.target_column)
numeric_outliers = diagnostics.numeric_outlier_report()
numeric_outliers
"""
        ),
        markdown("## 6. Numeric outlier visuals"),
        code(
            """
from credit_risk_lab.infrastructure.visualization import DataQualityVisualizer

visualizer = DataQualityVisualizer(
    frame=clean_train_df,
    target_column=settings.target_column,
)

visualizer.numeric_outlier_overview(
    numeric_outliers["feature"].head(6).tolist(),
).show()
"""
        ),
        markdown("## 7. Categorical diagnostics"),
        code(
            """
categorical_profile = diagnostics.categorical_profile(rare_threshold=0.01)
categorical_profile
"""
        ),
        markdown("## 8. Categorical visuals"),
        code(
            """
visualizer.categorical_feature_overview(
    categorical_profile["feature"].head(4).tolist(),
).show()
"""
        ),
    ],
)


split_drift = notebook(
    "03 - Split, Drift, and Feature Engineering",
    "créer un split de développement depuis le train brut, vérifier le leakage, analyser le drift et créer les features métier sans toucher au test final.",
    [
        setup,
        markdown("## 1. Load raw train and keep final test untouched"),
        code(
            """
from credit_risk_lab.infrastructure import CreditRiskQualityChecker
from credit_risk_lab.infrastructure.data_sources import CsvLoanDataLoader

raw_train_df = CsvLoanDataLoader(path=settings.raw_train_path).load()
raw_test_path = settings.raw_test_path

clean_train_df = CreditRiskQualityChecker().clean(raw_train_df)

{
    "clean_train_rows": len(clean_train_df),
    "reserved_final_test_path": str(raw_test_path),
}
"""
        ),
        markdown("## 2. Development train/validation split"),
        code(
            """
from credit_risk_lab.application import DatasetSplitter, SplitConfig

split_config = SplitConfig(
    test_size=settings.validation_size,
    random_state=settings.random_state,
    stratify=True,
)
splitter = DatasetSplitter(split_config)

model_train_df, validation_df = splitter.split(clean_train_df)
split_summary = splitter.summary(model_train_df, validation_df)
split_summary
"""
        ),
        markdown("## 3. Data leakage checks before preprocessing"),
        code(
            """
from credit_risk_lab.infrastructure.analytics import DataLeakageAuditor

leakage_auditor = DataLeakageAuditor(target_column=settings.target_column)
overlap_report = leakage_auditor.row_overlap_report(
    model_train_df,
    validation_df,
    holdout_name="validation",
)
target_report = leakage_auditor.target_leakage_report(
    model_train_df.drop(columns=[settings.target_column])
)

display(overlap_report)
target_report
"""
        ),
        markdown("## 4. Persist development artifacts"),
        code(
            """
from credit_risk_lab.infrastructure.data_sources import CSVDatasetRepository

train_path = CSVDatasetRepository.save(model_train_df, settings.train_path)
validation_path = CSVDatasetRepository.save(validation_df, settings.validation_path)

{"train_path": str(train_path), "validation_path": str(validation_path)}
"""
        ),
        markdown("## 5. Train vs validation drift analysis"),
        code(
            """
from credit_risk_lab.infrastructure.analytics import DriftAnalyzer

drift_analyzer = DriftAnalyzer(bins=10)
monitored_features = [
    column for column in model_train_df.columns if column != settings.target_column
]
drift_report = drift_analyzer.report_frame(
    model_train_df,
    validation_df,
    features=monitored_features,
)
drift_report.head(10)
"""
        ),
        markdown("## 6. Drift visualisation"),
        code(
            """
from credit_risk_lab.infrastructure.visualization import plot_drift_summary

plot_drift_summary(drift_report).show()
"""
        ),
        markdown("## 7. Business feature engineering on train only"),
        code(
            """
from credit_risk_lab.infrastructure.feature_engineering import LoanFeatureEngineer

feature_engineer = LoanFeatureEngineer()
featured_train_df = feature_engineer.transform(model_train_df)
created_features = [
    column for column in featured_train_df.columns if column not in model_train_df.columns
]

display(featured_train_df.head())
pd.DataFrame({"created_feature": created_features})
"""
        ),
    ],
)


training = notebook(
    "04 - Preprocessing and Training",
    "préparer les matrices sans fuite de données, entraîner les candidats et sélectionner le meilleur modèle.",
    [
        setup,
        markdown("## 1. Load training partition"),
        code(
            """
from credit_risk_lab.infrastructure.data_sources import CsvLoanDataLoader

train_loader = CsvLoanDataLoader(path=settings.train_path)
train_raw = train_loader.load()
train_raw.head()
"""
        ),
        markdown("## 2. Feature engineering"),
        code(
            """
from credit_risk_lab.infrastructure.feature_engineering import LoanFeatureEngineer

feature_engineer = LoanFeatureEngineer()
train_features = feature_engineer.transform(train_raw)
train_features.shape
"""
        ),
        markdown("## 3. Development split"),
        code(
            """
from credit_risk_lab.application import three_way_stratified_split

split = three_way_stratified_split(train_features)

pd.DataFrame(
    [
        {"split": "train", "rows": len(split.y_train), "positive_rate": split.y_train.mean()},
        {"split": "validation", "rows": len(split.y_validation), "positive_rate": split.y_validation.mean()},
        {"split": "test", "rows": len(split.y_test), "positive_rate": split.y_test.mean()},
    ]
)
"""
        ),
        markdown("## 4. Sensitive columns excluded from training features"),
        code(
            """
sensitive_columns = [
    column for column in settings.sensitive_columns if column in split.x_train
]

x_train = split.x_train.drop(columns=sensitive_columns)
x_validation = split.x_validation.drop(columns=sensitive_columns)
x_test = split.x_test.drop(columns=sensitive_columns)
sensitive_test = split.x_test[sensitive_columns].copy()

{"excluded": sensitive_columns, "training_columns": x_train.shape[1]}
"""
        ),
        markdown("## 5. Fit preprocessing on train only"),
        code(
            """
from credit_risk_lab.infrastructure.modeling import CreditRiskPreprocessor

preprocessor = CreditRiskPreprocessor()
x_train_t = preprocessor.fit_transform(x_train)
x_validation_t = preprocessor.transform(x_validation)
x_test_t = preprocessor.transform(x_test)

{
    "train_matrix": x_train_t.shape,
    "validation_matrix": x_validation_t.shape,
    "test_matrix": x_test_t.shape,
    "first_output_features": preprocessor.feature_names_[:20],
}
"""
        ),
        markdown("## 6. Inspect configured model candidates"),
        code(
            """
from credit_risk_lab.infrastructure.modeling import build_configured_models, load_models_config

model_config = load_models_config()
configured_models = build_configured_models(
    random_state=settings.random_state,
    config=model_config,
)

pd.DataFrame(
    [
        {
            "model": model.name,
            "parameters": model.parameters,
            "early_stopping": model.early_stopping_rounds,
        }
        for model in configured_models
    ]
)
"""
        ),
        markdown("## 7. Train candidates"),
        code(
            """
from credit_risk_lab.infrastructure.modeling import BoostingModelTrainer

trainer = BoostingModelTrainer(random_state=settings.random_state)
training_results = trainer.fit(
    x_train_t,
    split.y_train,
    x_validation_t,
    split.y_validation,
)

validation_metrics = trainer.results_frame(training_results)
validation_metrics.round(4)
"""
        ),
        markdown("## 8. Select best model"),
        code(
            """
from credit_risk_lab.infrastructure.modeling import BestModelSelector

selector = BestModelSelector(metric=settings.selection_metric)
best_result = selector.select(training_results)

{
    "selected_model": best_result.model_name,
    "selection_metric": settings.selection_metric,
    "threshold": best_result.threshold,
}
"""
        ),
        markdown("## 9. Training curves"),
        code(
            """
from credit_risk_lab.infrastructure.visualization import plot_learning_curves, plot_model_comparison

histories = {
    result.model_name: result.history for result in training_results
}

plot_model_comparison(validation_metrics).show()
plot_learning_curves(histories).show()
"""
        ),
    ],
)


evaluation = notebook(
    "05 - Model Evaluation and Persistence",
    "évaluer le modèle sélectionné, produire calibration/fairness, puis sauvegarder explicitement le bundle.",
    [
        setup,
        markdown("## 1. Load training partition"),
        code(
            """
from credit_risk_lab.infrastructure.data_sources import CsvLoanDataLoader

train_raw = CsvLoanDataLoader(path=settings.train_path).load()
train_raw.shape
"""
        ),
        markdown("## 2. Rebuild features and development split"),
        code(
            """
from credit_risk_lab.application import three_way_stratified_split
from credit_risk_lab.infrastructure.feature_engineering import LoanFeatureEngineer

feature_engineer = LoanFeatureEngineer()
train_features = feature_engineer.transform(train_raw)
split = three_way_stratified_split(train_features)

sensitive_columns = [
    column for column in settings.sensitive_columns if column in split.x_train
]
x_train = split.x_train.drop(columns=sensitive_columns)
x_validation = split.x_validation.drop(columns=sensitive_columns)
x_test = split.x_test.drop(columns=sensitive_columns)
sensitive_test = split.x_test[sensitive_columns].copy()
"""
        ),
        markdown("## 3. Preprocess and train candidates"),
        code(
            """
from credit_risk_lab.infrastructure.modeling import (
    BestModelSelector,
    BoostingModelTrainer,
    CreditRiskPreprocessor,
)

preprocessor = CreditRiskPreprocessor()
x_train_t = preprocessor.fit_transform(x_train)
x_validation_t = preprocessor.transform(x_validation)
x_test_t = preprocessor.transform(x_test)

trainer = BoostingModelTrainer(random_state=settings.random_state)
training_results = trainer.fit(
    x_train_t,
    split.y_train,
    x_validation_t,
    split.y_validation,
)
best_result = BestModelSelector(metric=settings.selection_metric).select(training_results)
best_result.model_name
"""
        ),
        markdown("## 4. Evaluate on isolated internal test"),
        code(
            """
from credit_risk_lab.infrastructure.evaluation import CreditRiskModelEvaluator

evaluator = CreditRiskModelEvaluator()
test_probabilities = best_result.model.predict_proba(x_test_t)
test_metrics = evaluator.metrics_frame(
    best_result.model_name,
    split.y_test,
    test_probabilities,
    best_result.threshold,
)
test_metrics.round(4)
"""
        ),
        markdown("## 5. Calibration"),
        code(
            """
from credit_risk_lab.infrastructure.evaluation import CalibrationEvaluator
from credit_risk_lab.infrastructure.visualization import plot_calibration

calibration = CalibrationEvaluator(bins=10).evaluate(
    split.y_test,
    test_probabilities,
)

display(calibration.round(4))
plot_calibration(calibration, best_result.model_name).show()
"""
        ),
        markdown("## 6. Fairness diagnostics"),
        code(
            """
from credit_risk_lab.infrastructure.evaluation import FairnessEvaluator

fairness = FairnessEvaluator(min_group_size=30).evaluate(
    split.y_test,
    test_probabilities,
    sensitive_test,
    best_result.threshold,
)
fairness
"""
        ),
        markdown("## 7. Build metadata"),
        code(
            """
from credit_risk_lab.application.workflows import current_git_commit
from credit_risk_lab.infrastructure.modeling import sha256_file

metadata = {
    "model_name": best_result.model_name,
    "test_metrics": test_metrics.iloc[0].to_dict(),
    "selection_metric": settings.selection_metric,
    "split_strategy": settings.split_strategy,
    "training_dataset_sha256": sha256_file(settings.train_path),
    "external_test_dataset_sha256": sha256_file(settings.raw_test_path),
    "models_config_sha256": sha256_file(settings.models_config_path),
    "git_commit": current_git_commit(),
    "target_definition": "loan_status=1 is the synthetic positive risk class",
}

metadata
"""
        ),
        markdown("## 8. Save model bundle explicitly"),
        code(
            """
from credit_risk_lab.infrastructure.modeling import JoblibModelBundleRepository

repository = JoblibModelBundleRepository()
bundle_path = repository.save(
    settings.model_bundle_path,
    model=best_result.model,
    preprocessor=preprocessor.transformer,
    threshold=best_result.threshold,
    metadata=metadata,
)

bundle_path
"""
        ),
    ],
)


inference = notebook(
    "06 - Inference and API Simulation",
    "charger le bundle, scorer des dossiers bruts, puis vérifier le comportement API in-process.",
    [
        setup,
        markdown("## 1. Load external holdout"),
        code(
            """
from credit_risk_lab.infrastructure.data_sources import CsvLoanDataLoader

holdout_df = CsvLoanDataLoader(path=settings.raw_test_path).load()
holdout_df.head()
"""
        ),
        markdown("## 2. Load model bundle"),
        code(
            """
from credit_risk_lab.infrastructure.modeling import JoblibModelBundleRepository

repository = JoblibModelBundleRepository()
bundle = repository.load(settings.model_bundle_path)
bundle["metadata"]
"""
        ),
        markdown("## 3. Score raw applications"),
        code(
            """
from credit_risk_lab.application import RawLoanScorer

scorer = RawLoanScorer(bundle, threshold=settings.decision_threshold)
scoring_result = scorer.score(holdout_df.head(20))

pd.DataFrame(
    {
        "probability_of_risk": scoring_result.probabilities,
        "risk_decision": scoring_result.decisions,
    }
).head()
"""
        ),
        markdown("## 4. Validate one API payload"),
        code(
            """
from credit_risk_lab.interfaces.api_models import LoanApplication

payload = holdout_df.drop(columns=[settings.target_column]).iloc[0].to_dict()
application = LoanApplication.model_validate(payload)
application
"""
        ),
        markdown("## 5. Predict through API service"),
        code(
            """
from credit_risk_lab.interfaces.api_service import predict_application

response = predict_application(application, request_id="notebook-demo")
response
"""
        ),
        markdown("## 6. In-process API simulation"),
        code(
            """
from credit_risk_lab.interfaces.api_simulation import run_api_simulation

simulation = run_api_simulation(limit=20)
display(simulation.health)
display(simulation.responses.head())
{"invalid_status_code": simulation.invalid_status_code}
"""
        ),
        markdown("## 7. Simulation distribution"),
        code(
            """
px.histogram(
    simulation.responses,
    x="probability_of_risk",
    nbins=20,
    title="Simulated API risk probabilities",
    template="plotly_white",
).show()
"""
        ),
    ],
)


NOTEBOOKS = {
    "01_data_understanding.ipynb": data_understanding,
    "02_data_quality.ipynb": data_quality,
    "03_split_drift_feature_engineering.ipynb": split_drift,
    "04_preprocessing_and_training.ipynb": training,
    "05_model_evaluation_and_persistence.ipynb": evaluation,
    "06_inference_and_api_simulation.ipynb": inference,
}


def main() -> None:
    """Replace notebooks with the canonical progressive execution sequence."""
    DEV.mkdir(exist_ok=True)
    for old in DEV.glob("*.ipynb"):
        old.unlink()
    for filename, nb in NOTEBOOKS.items():
        nbf.write(nb, DEV / filename)
        print(f"Wrote {filename}")


if __name__ == "__main__":
    main()
