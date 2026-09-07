"""Generate progressive execution-lab notebooks for Credit Risk Lab."""

from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS_DIR = ROOT / "notebooks"


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
        markdown("## 4. Train vs validation drift analysis"),
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
        markdown("## 5. Drift visualisation"),
        code(
            """
from credit_risk_lab.infrastructure.visualization import plot_drift_summary

plot_drift_summary(drift_report).show()
"""
        ),
        markdown("## 6. Business feature engineering on train and validation"),
        code(
            """
from credit_risk_lab.infrastructure.analytics import FeatureEngineeringReport
from credit_risk_lab.infrastructure.feature_engineering import LoanFeatureEngineer

feature_engineer = LoanFeatureEngineer()
featured_train_df = feature_engineer.transform(model_train_df)
featured_validation_df = feature_engineer.transform(validation_df)
feature_report = FeatureEngineeringReport(model_train_df, featured_train_df)
created_features = feature_report.created_columns()

print(f"Columns before feature engineering: {model_train_df.shape[1]}")
print(f"Columns after feature engineering: {featured_train_df.shape[1]}")
print(f"New columns created: {len(created_features)}")
print(f"Columns removed: {len(feature_report.removed_columns())}")
print(f"Retained columns modified: {len(feature_report.changed_columns())}")
print(f"Validation received the same column schema: {featured_train_df.columns.equals(featured_validation_df.columns)}")

display(feature_report.summary())
display(feature_report.created_columns_frame())
featured_train_df.head()
"""
        ),
        markdown("## 7. Fit preprocessing on train and transform validation"),
        code(
            """
from credit_risk_lab.infrastructure.modeling import CreditRiskPreprocessor

sensitive_columns = [
    column for column in settings.sensitive_columns if column in featured_train_df
]
target_column = settings.target_column

x_train = featured_train_df.drop(columns=[target_column, *sensitive_columns])
y_train = featured_train_df[target_column].reset_index(drop=True)
x_validation = featured_validation_df.drop(columns=[target_column, *sensitive_columns])
y_validation = featured_validation_df[target_column].reset_index(drop=True)

preprocessor = CreditRiskPreprocessor()
processed_train_features = preprocessor.fit_transform_frame(x_train)
processed_validation_features = preprocessor.transform_frame(x_validation)

processed_train_df = processed_train_features.assign(**{target_column: y_train.to_numpy()})
processed_validation_df = processed_validation_features.assign(
    **{target_column: y_validation.to_numpy()}
)

print(f"Columns before preprocessing: {x_train.shape[1]}")
print(f"Columns after preprocessing: {processed_train_features.shape[1]}")
print(f"Sensitive columns excluded: {sensitive_columns}")
print(f"Target excluded from features: {target_column}")
print(f"Preprocessor output features: {len(preprocessor.feature_names_)}")
print(f"Validation output columns match train: {processed_validation_features.columns.equals(processed_train_features.columns)}")

display(
    pd.DataFrame(
        [
            {"step": "after_feature_engineering", "train_columns": x_train.shape[1], "validation_columns": x_validation.shape[1]},
            {"step": "after_preprocessing", "train_columns": processed_train_features.shape[1], "validation_columns": processed_validation_features.shape[1]},
        ]
    )
)
display(pd.DataFrame({"model_feature": preprocessor.feature_names_}))
"""
        ),
        markdown("## 8. Persist transformed development artifacts"),
        code(
            """
from credit_risk_lab.infrastructure.data_sources import CSVDatasetRepository

train_path = CSVDatasetRepository.save(processed_train_df, settings.train_path)
validation_path = CSVDatasetRepository.save(
    processed_validation_df,
    settings.validation_path,
)
preprocessor_path = preprocessor.save(settings.preprocessing_artifact_path)

{
    "processed_train_path": str(train_path),
    "processed_validation_path": str(validation_path),
    "preprocessor_path": str(preprocessor_path),
    "reserved_final_test_path": str(settings.raw_test_path),
}
"""
        ),
    ],
)


training = notebook(
    "04 - Preprocessing and Training",
    "charger les artefacts transformés par le notebook 03, vérifier le contrat de preprocessing, entraîner les candidats et sélectionner le meilleur modèle.",
    [
        setup,
        markdown("## 1. Load processed train and validation partitions"),
        code(
            """
from credit_risk_lab.infrastructure.data_sources import CSVDataSourceConfig, CSVDatasetRepository
from credit_risk_lab.infrastructure.modeling import CreditRiskPreprocessor

processed_train_df = CSVDatasetRepository(
    CSVDataSourceConfig(path=settings.train_path)
).load()
processed_validation_df = CSVDatasetRepository(
    CSVDataSourceConfig(path=settings.validation_path)
).load()
preprocessor = CreditRiskPreprocessor.load(settings.preprocessing_artifact_path)

print(f"Processed train path: {settings.train_path}")
print(f"Processed validation path: {settings.validation_path}")
print(f"Preprocessor artifact path: {settings.preprocessing_artifact_path}")
print(f"Processed train shape: {processed_train_df.shape}")
print(f"Processed validation shape: {processed_validation_df.shape}")
print(f"Preprocessor output features: {len(preprocessor.feature_names_)}")

{
    "processed_train": processed_train_df.shape,
    "processed_validation": processed_validation_df.shape,
    "preprocessor_features": len(preprocessor.feature_names_),
}
"""
        ),
        markdown("## 2. Build train and validation matrices"),
        code(
            """
target_column = settings.target_column

x_train = processed_train_df.drop(columns=[target_column])
y_train = processed_train_df[target_column]
x_validation = processed_validation_df.drop(columns=[target_column])
y_validation = processed_validation_df[target_column]

train_schema_matches_preprocessor = x_train.columns.tolist() == preprocessor.feature_names_
validation_schema_matches_train = x_validation.columns.tolist() == x_train.columns.tolist()

if not train_schema_matches_preprocessor:
    raise ValueError("Processed train columns do not match the saved preprocessor features")
if not validation_schema_matches_train:
    raise ValueError("Processed validation columns do not match processed train columns")

print(f"Target column excluded from model features: {target_column}")
print(f"Train feature columns: {x_train.shape[1]}")
print(f"Validation feature columns: {x_validation.shape[1]}")
print(f"Train schema matches saved preprocessor: {train_schema_matches_preprocessor}")
print(f"Validation schema matches train: {validation_schema_matches_train}")
print(f"Final test remains untouched at: {settings.raw_test_path}")

split_summary = pd.DataFrame(
    [
        {"split": "train", "rows": len(y_train), "positive_rate": y_train.mean()},
        {
            "split": "validation",
            "rows": len(y_validation),
            "positive_rate": y_validation.mean(),
        },
    ]
)

display(split_summary)
pd.DataFrame({"model_feature": x_train.columns})
"""
        ),
        markdown("## 3. Inspect configured model candidates"),
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
        markdown("## 4. Train candidates"),
        code(
            """
from credit_risk_lab.infrastructure.modeling import BoostingModelTrainer

trainer = BoostingModelTrainer(random_state=settings.random_state)
training_results = trainer.fit(
    x_train,
    y_train,
    x_validation,
    y_validation,
)

validation_metrics = trainer.results_frame(training_results)
validation_metrics.round(4)
"""
        ),
        markdown("## 5. Select best model"),
        code(
            """
from credit_risk_lab.infrastructure.modeling import BestModelSelector

selector = BestModelSelector(metric=settings.selection_metric)
best_result = selector.select(training_results)

print(f"Selected model: {best_result.model_name}")
print(f"Selection metric: {settings.selection_metric}")
print(f"Selected threshold: {best_result.threshold:.4f}")
"""
        ),
        markdown("## 6. Training curves"),
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
        markdown("## 7. Selected model feature importance"),
        code(
            """
from credit_risk_lab.infrastructure.modeling import CatBoostFeatureImportanceAnalyzer
from credit_risk_lab.infrastructure.visualization import plot_feature_importance

top_n_features = 20

if best_result.model_name == "CatBoost":
    baseline_importance = CatBoostFeatureImportanceAnalyzer(
        best_result.model,
        feature_names=x_train.columns.tolist(),
    ).importance_frame(top_n=top_n_features)
    display(baseline_importance.round(4))
    plot_feature_importance(
        baseline_importance,
        title=f"Top {top_n_features} CatBoost baseline feature importances",
    ).show()
else:
    print(f"Selected baseline model is {best_result.model_name}; CatBoost importance skipped.")
"""
        ),
    ],
)


tuning = notebook(
    "05 - Hyperparameter Tuning",
    "optimiser CatBoost avec Optuna sur train/validation, sans toucher au test final.",
    [
        setup,
        markdown("## 1. Load processed development partitions"),
        code(
            """
from credit_risk_lab.infrastructure.data_sources import CSVDataSourceConfig, CSVDatasetRepository
from credit_risk_lab.infrastructure.modeling import CreditRiskPreprocessor

processed_train_df = CSVDatasetRepository(
    CSVDataSourceConfig(path=settings.train_path)
).load()
processed_validation_df = CSVDatasetRepository(
    CSVDataSourceConfig(path=settings.validation_path)
).load()
preprocessor = CreditRiskPreprocessor.load(settings.preprocessing_artifact_path)

{
    "processed_train": processed_train_df.shape,
    "processed_validation": processed_validation_df.shape,
    "preprocessor_features": len(preprocessor.feature_names_),
    "reserved_final_test_path": str(settings.raw_test_path),
}
"""
        ),
        markdown("## 2. Build train and validation matrices"),
        code(
            """
target_column = settings.target_column

x_train = processed_train_df.drop(columns=[target_column])
y_train = processed_train_df[target_column]
x_validation = processed_validation_df.drop(columns=[target_column])
y_validation = processed_validation_df[target_column]

pd.DataFrame(
    [
        {"split": "train", "rows": len(y_train), "positive_rate": y_train.mean()},
        {
            "split": "validation",
            "rows": len(y_validation),
            "positive_rate": y_validation.mean(),
        },
    ]
)
"""
        ),
        markdown("## 3. Rebuild baseline ranking"),
        code(
            """
from credit_risk_lab.infrastructure.modeling import (
    BestModelSelector,
    BoostingModelTrainer,
)

trainer = BoostingModelTrainer(random_state=settings.random_state)
baseline_results = trainer.fit(
    x_train,
    y_train,
    x_validation,
    y_validation,
)
baseline_metrics = trainer.results_frame(baseline_results)
best_baseline = BestModelSelector(metric=settings.selection_metric).select(
    baseline_results
)

display(baseline_metrics.round(4))
{
    "best_baseline_model": best_baseline.model_name,
    "model_selected_for_optuna": "CatBoost",
}
"""
        ),
        markdown("## 4. Baseline metric visuals"),
        code(
            """
from credit_risk_lab.infrastructure.visualization import plot_model_comparison

catboost_baseline_metrics = baseline_metrics[
    baseline_metrics["model"].eq("CatBoost")
].copy()

plot_model_comparison(baseline_metrics).show()
display(catboost_baseline_metrics.round(4))
"""
        ),
        markdown("## 5. Configure CatBoost Optuna search"),
        code(
            """
model_to_optimize = "CatBoost"
n_trials = settings.optuna_trials
optimization_metric = "roc_auc"

if best_baseline.model_name != model_to_optimize:
    print(
        f"Baseline winner is {best_baseline.model_name}, "
        f"but this project decision is to optimize {model_to_optimize}."
    )

{
    "model_to_optimize": model_to_optimize,
    "n_trials": n_trials,
    "optimization_metric": optimization_metric,
    "test_policy": "data/raw/test.csv remains untouched",
}
"""
        ),
        markdown("## 6. Run Optuna search on CatBoost"),
        code(
            """
from credit_risk_lab.infrastructure.modeling import CatBoostOptunaTuner

catboost_tuner = CatBoostOptunaTuner(
    random_state=settings.random_state,
    metric=optimization_metric,
    n_trials=n_trials,
)
tuning_result = catboost_tuner.tune(
    x_train=x_train,
    y_train=y_train,
    x_validation=x_validation,
    y_validation=y_validation,
)

display(tuning_result.trials.head(10))
pd.DataFrame(
    [
        {
            "model": tuning_result.model_name,
            "best_value": tuning_result.best_value,
            "threshold": tuning_result.threshold,
            **tuning_result.metrics,
        }
    ]
).round(4)
"""
        ),
        markdown("## 7. Analyze tuning improvement"),
        code(
            """
from credit_risk_lab.infrastructure.visualization import (
    plot_metric_improvement,
    plot_optuna_param_importance,
    plot_optuna_parameter_slices,
    plot_optuna_trials,
)

tuned_metrics = pd.DataFrame(
    [
        {
            "model": tuning_result.model_name,
            "threshold": tuning_result.threshold,
            **tuning_result.metrics,
        }
    ]
)

plot_optuna_trials(tuning_result.trials, metric=optimization_metric).show()
plot_optuna_param_importance(
    tuning_result.study,
    metric=optimization_metric,
).show()
plot_optuna_parameter_slices(
    tuning_result.trials,
    metric=optimization_metric,
    parameters=[
        "iterations",
        "depth",
        "learning_rate",
        "l2_leaf_reg",
        "random_strength",
        "auto_class_weights",
    ],
).show()
plot_metric_improvement(
    baseline_metrics,
    tuned_metrics,
    model_name="CatBoost",
).show()

comparison = pd.concat(
    [
        catboost_baseline_metrics.assign(stage="baseline"),
        tuned_metrics.assign(stage="optuna_tuned"),
    ],
    ignore_index=True,
)
display(comparison.round(4))
"""
        ),
        markdown("## 8. Optimized CatBoost feature importance"),
        code(
            """
from credit_risk_lab.infrastructure.modeling import CatBoostFeatureImportanceAnalyzer
from credit_risk_lab.infrastructure.visualization import plot_feature_importance

top_n_features = 20
optimized_importance = CatBoostFeatureImportanceAnalyzer(
    tuning_result.model,
    feature_names=x_train.columns.tolist(),
).importance_frame(top_n=top_n_features)

display(optimized_importance.round(4))
plot_feature_importance(
    optimized_importance,
    title=f"Top {top_n_features} optimized CatBoost feature importances",
).show()
"""
        ),
        markdown("## 9. Save optimized CatBoost bundle"),
        code(
            """
from credit_risk_lab.application.workflows import current_git_commit
from credit_risk_lab.infrastructure.modeling import JoblibModelBundleRepository, sha256_file

metadata = {
    "model_name": tuning_result.model_name,
    "validation_metrics": tuning_result.metrics,
    "best_hyperparameters": tuning_result.best_parameters,
    "optimization_engine": "optuna",
    "optimization_metric": optimization_metric,
    "optuna_best_value": tuning_result.best_value,
    "optuna_trials": n_trials,
    "selection_metric": optimization_metric,
    "split_strategy": "raw_train_to_processed_train_validation_then_external_test",
    "training_dataset_sha256": sha256_file(settings.train_path),
    "validation_dataset_sha256": sha256_file(settings.validation_path),
    "preprocessing_artifact_sha256": sha256_file(settings.preprocessing_artifact_path),
    "models_config_sha256": sha256_file(settings.models_config_path),
    "git_commit": current_git_commit(),
    "target_definition": "loan_status=1 is the synthetic positive risk class",
}

bundle_path = JoblibModelBundleRepository().save(
    settings.model_bundle_path,
    model=tuning_result.model,
    preprocessor=preprocessor.transformer,
    threshold=tuning_result.threshold,
    metadata=metadata,
)

{
    "selected_tuned_model": tuning_result.model_name,
    "threshold": tuning_result.threshold,
    "parameters": tuning_result.best_parameters,
    "bundle_path": str(bundle_path),
}
"""
        ),
    ],
)


evaluation = notebook(
    "06 - Model Evaluation and Persistence",
    "évaluer le modèle tuné sauvegardé sur le test final untouched, puis persister les rapports finaux.",
    [
        setup,
        markdown("## 1. Load tuned model bundle"),
        code(
            """
from credit_risk_lab.infrastructure.modeling import JoblibModelBundleRepository

repository = JoblibModelBundleRepository()
bundle = repository.load(settings.model_bundle_path)
bundle["metadata"]
"""
        ),
        markdown("## 2. Load untouched external test"),
        code(
            """
from credit_risk_lab.infrastructure import CreditRiskQualityChecker
from credit_risk_lab.infrastructure.data_sources import CsvLoanDataLoader

raw_test_df = CsvLoanDataLoader(path=settings.raw_test_path).load()
clean_test_df = CreditRiskQualityChecker().clean(raw_test_df)

{
    "raw_test_path": str(settings.raw_test_path),
    "raw_test_rows": len(raw_test_df),
    "clean_test_rows": len(clean_test_df),
}
"""
        ),
        markdown("## 3. Score external test"),
        code(
            """
from credit_risk_lab.application import RawLoanScorer

scorer = RawLoanScorer(bundle)
scoring_result = scorer.score(clean_test_df)
y_test = clean_test_df[settings.target_column].astype(int)
test_probabilities = scoring_result.probabilities
sensitive_test = clean_test_df[
    [column for column in settings.sensitive_columns if column in clean_test_df]
]

{
    "model": scoring_result.model_name,
    "threshold": scoring_result.threshold,
    "scored_rows": len(test_probabilities),
}
"""
        ),
        markdown("## 4. Final test metrics"),
        code(
            """
from credit_risk_lab.infrastructure.evaluation import CreditRiskModelEvaluator

evaluator = CreditRiskModelEvaluator()
test_metrics = evaluator.metrics_frame(
    scoring_result.model_name,
    y_test,
    test_probabilities,
    scoring_result.threshold,
)
test_metrics.to_csv(settings.reports_dir / "external_test_metrics.csv", index=False)
test_metrics.round(4)
"""
        ),
        markdown("## 5. Confusion matrix and ROC-AUC"),
        code(
            """
from credit_risk_lab.infrastructure.visualization import plot_confusion_matrix_and_roc

plot_confusion_matrix_and_roc(
    y_test,
    test_probabilities,
    threshold=scoring_result.threshold,
    model_name=scoring_result.model_name,
).show()
"""
        ),
        markdown("## 6. Threshold trade-off analysis"),
        code(
            """
from credit_risk_lab.infrastructure.evaluation import (
    ThresholdAnalysisConfig,
    ThresholdAnalyzer,
)
from credit_risk_lab.infrastructure.visualization import plot_threshold_tradeoff

threshold_config = ThresholdAnalysisConfig(
    start=0.02,
    stop=0.42,
    step=0.04,
)
threshold_grid = ThresholdAnalyzer(threshold_config).grid(y_test, test_probabilities)
threshold_grid.to_csv(
    settings.reports_dir / "external_test_threshold_grid.csv",
    index=False,
)

print(threshold_grid.to_string(index=False))
plot_threshold_tradeoff(
    threshold_grid,
    selected_threshold=scoring_result.threshold,
).show()
"""
        ),
        markdown("## 7. Lift, gain, and accumulation analysis"),
        code(
            """
from credit_risk_lab.infrastructure.evaluation import lift_gain_table
from credit_risk_lab.infrastructure.visualization import plot_lift_gain_accumulation

lift_gain = lift_gain_table(y_test, test_probabilities, bins=10)
lift_gain.to_csv(settings.reports_dir / "external_test_lift_gain.csv", index=False)

display(lift_gain.round(4))
plot_lift_gain_accumulation(lift_gain).show()
"""
        ),
        markdown("## 8. Bootstrap confidence intervals"),
        code(
            """
from credit_risk_lab.infrastructure.evaluation import bootstrap_metric_intervals
from credit_risk_lab.infrastructure.visualization import plot_metric_confidence_intervals

metric_intervals = bootstrap_metric_intervals(
    y_test,
    test_probabilities,
    scoring_result.threshold,
    n_bootstrap=300,
    confidence_level=0.95,
    random_state=settings.random_state,
)
metric_intervals.to_csv(
    settings.reports_dir / "external_test_metric_intervals.csv",
    index=False,
)

display(metric_intervals.round(4))
plot_metric_confidence_intervals(metric_intervals).show()
"""
        ),
        markdown("## 9. Calibration"),
        code(
            """
from credit_risk_lab.infrastructure.evaluation import CalibrationEvaluator
from credit_risk_lab.infrastructure.visualization import plot_calibration

calibration = CalibrationEvaluator(bins=10).evaluate(
    y_test,
    test_probabilities,
)
calibration.to_csv(settings.reports_dir / "external_test_calibration.csv", index=False)

display(calibration.round(4))
plot_calibration(calibration, scoring_result.model_name).show()
"""
        ),
        markdown("## 10. Fairness diagnostics"),
        code(
            """
from credit_risk_lab.infrastructure.evaluation import FairnessEvaluator

fairness = FairnessEvaluator(min_group_size=30).evaluate(
    y_test,
    test_probabilities,
    sensitive_test,
    scoring_result.threshold,
)
fairness.to_csv(settings.reports_dir / "external_test_fairness.csv", index=False)
fairness
"""
        ),
        markdown("## 11. Persist final metadata report"),
        code(
            """
from credit_risk_lab.infrastructure.modeling import sha256_file

final_metadata = {
    **bundle["metadata"],
    "external_test_metrics": test_metrics.iloc[0].to_dict(),
    "external_test_dataset_sha256": sha256_file(settings.raw_test_path),
}

pd.Series(final_metadata, name="value").to_frame()
"""
        ),
    ],
)


inference = notebook(
    "07 - Batch and Realtime Inference",
    "charger le bundle final, réaliser une inférence batch sur le test untouched, puis simuler des prédictions temps réel sans API.",
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
        markdown("## 3. Configure raw scorer and input validator"),
        code(
            """
from credit_risk_lab.application import InferenceInputValidator, RawLoanScorer

scorer = RawLoanScorer(bundle, threshold=settings.decision_threshold)
input_validator = InferenceInputValidator()

{
    "model_name": bundle["metadata"].get("model_name"),
    "serving_threshold": scorer.model_scorer.threshold,
    "test_rows_available": len(holdout_df),
}
"""
        ),
        markdown("## 4. Validate batch input sample"),
        code(
            """
batch_limit = 100
batch_input = holdout_df.head(batch_limit)
batch_validation = input_validator.validate(batch_input)

print(f"Rows received: {len(batch_input)}")
print(f"Rows accepted: {batch_validation.accepted_rows}")
print(f"Rows rejected: {batch_validation.rejected_count}")
print(f"Rows with warnings: {batch_validation.warning_count}")

print("Rejected rows")
display(batch_validation.rejected_rows.head(10))

print("Warning rows")
display(batch_validation.warning_rows.head(10))
"""
        ),
        markdown("## 5. Batch inference on valid test rows"),
        code(
            """
from credit_risk_lab.application import BatchInferenceRunner

submission_path = settings.reports_dir / "submission.csv"

batch_runner = BatchInferenceRunner(scorer, validator=input_validator)
batch_result = batch_runner.predict(
    batch_input,
    output_path=submission_path,
)

print(f"Batch rows scored: {len(batch_result.submission)}")
print(f"Batch rows rejected: {len(batch_result.rejected_rows)}")
print(f"Batch rows with warnings: {len(batch_result.warning_rows)}")
print(f"Submission saved to: {batch_result.output_path}")

display(batch_result.submission.head(10))
"""
        ),
        markdown("## 6. Batch inference risk distribution"),
        code(
            """
px.histogram(
    batch_result.submission,
    x="probability_of_risk",
    color="risk_band",
    nbins=30,
    title="Batch inference - predicted risk probabilities",
    template="plotly_white",
).show()
"""
        ),
        markdown("## 7. Realtime inference simulation without API"),
        code(
            """
from credit_risk_lab.application import RealtimeInferenceSimulator

realtime_limit = 5
simulator = RealtimeInferenceSimulator(
    scorer,
    validator=input_validator,
    min_pause_seconds=1,
    max_pause_seconds=5,
)

realtime_events = simulator.stream(
    holdout_df,
    limit=realtime_limit,
    request_prefix="loan-request",
    print_events=True,
)

display(realtime_events)
"""
        ),
        markdown("## 8. Realtime decision timeline"),
        code(
            """
fig = px.line(
    realtime_events,
    x="request_id",
    y="probability_of_risk",
    markers=True,
    title="Realtime simulation - request-level risk score",
    template="plotly_white",
)
fig.add_hline(
    y=settings.decision_threshold,
    line_dash="dash",
    line_color="black",
    annotation_text=f"threshold={settings.decision_threshold:.3f}",
)
fig.show()
"""
        ),
    ],
)


pipeline_summary = notebook(
    "08 - CI/CD MLOps Pipeline",
    "exécuter le pipeline training/evaluation depuis le dataset brut jusqu'aux artefacts et métriques finales utilisables en CI/CD.",
    [
        setup,
        markdown("## 1. Configure pipeline run"),
        code(
            """
from credit_risk_lab.application.workflows import CreditRiskMLOpsPipeline

pipeline = CreditRiskMLOpsPipeline(
    project_settings=settings,
    optuna_trials=settings.optuna_trials,
)

print(f"Environment: {settings.environment}")
print(f"Selection metric: {settings.selection_metric}")
print(f"Optuna trials: {settings.optuna_trials}")
print(f"Minimum validation ROC-AUC: {settings.minimum_validation_roc_auc}")
print(f"Decision threshold: {settings.decision_threshold}")
print(f"Raw dataset: {settings.raw_data_path}")
"""
        ),
        markdown("## 2. Run full pipeline"),
        code(
            """
pipeline_result = pipeline.run()
"""
        ),
        markdown("## 3. Pipeline summary"),
        code(
            """
display(pipeline_result.summary)
"""
        ),
        markdown("## 4. Produced artifacts"),
        code(
            """
display(pipeline_result.artifacts)
display(pipeline_result.promotion_report)
"""
        ),
        markdown("## 5. Baseline model ranking"),
        code(
            """
display(pipeline_result.baseline_metrics)
"""
        ),
        markdown("## 6. Tuned CatBoost validation metrics"),
        code(
            """
display(pipeline_result.tuned_metrics)
"""
        ),
        markdown("## 7. Final untouched test metrics"),
        code(
            """
display(pipeline_result.final_metrics)
"""
        ),
        markdown("## 8. Business diagnostics"),
        code(
            """
display(pipeline_result.threshold_grid)
display(pipeline_result.lift_gain.head(10))
display(pipeline_result.confidence_intervals.round(4))
"""
        ),
        markdown("## 9. Governance diagnostics"),
        code(
            """
print("Calibration")
display(pipeline_result.calibration.round(4))

print("Fairness diagnostics")
display(pipeline_result.fairness)
"""
        ),
    ],
)


NOTEBOOKS = {
    "01_data_understanding.ipynb": data_understanding,
    "02_data_quality.ipynb": data_quality,
    "03_split_drift_feature_engineering.ipynb": split_drift,
    "04_preprocessing_and_training.ipynb": training,
    "05_hyperparameter_tuning.ipynb": tuning,
    "06_model_evaluation_and_persistence.ipynb": evaluation,
    "07_batch_and_realtime_inference.ipynb": inference,
    "08_end_to_end_mlops_pipeline.ipynb": pipeline_summary,
}


def main() -> None:
    """Replace notebooks with the canonical progressive execution sequence."""
    NOTEBOOKS_DIR.mkdir(exist_ok=True)
    for old in NOTEBOOKS_DIR.glob("*.ipynb"):
        old.unlink()
    for filename, nb in NOTEBOOKS.items():
        nbf.write(nb, NOTEBOOKS_DIR / filename)
        print(f"Wrote {filename}")


if __name__ == "__main__":
    main()
