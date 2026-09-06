"""Generate the concise end-to-end notebook path for Credit Risk Lab."""

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
    nb.metadata.kernelspec = {"display_name": "Python 3", "language": "python", "name": "python3"}
    nb.metadata.language_info = {"name": "python", "version": "3.11"}
    nb.cells = [markdown(f"# {title}\n\n**Objective:** {objective}")] + cells
    return nb


setup = code("""
from pathlib import Path
import os, sys
import pandas as pd
import plotly.express as px

_cwd = Path.cwd().resolve()
ROOT = _cwd.parent if _cwd.name == "dev" else _cwd
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("MPLCONFIGDIR", "/tmp/credit-risk-lab-matplotlib")

from credit_risk_lab.config.settings import settings
print(f"Project root: {settings.project_root}")
""")


source_quality = notebook(
    "01 — Data Source, Quality, and EDA",
    "Make the raw source explicit, validate it, clean implausible rows, and understand the target.",
    [
        setup,
        markdown("## 1. Explicit CSV configuration\n\nThe path comes from `configs/settings.yaml`; the repository receives it explicitly with parsing options."),
        code("""
from credit_risk_lab.infrastructure.data_sources import CSVDataSourceConfig, CSVDatasetRepository

raw_source = CSVDataSourceConfig(
    path=settings.raw_data_path,
    sep=settings.raw_data_sep,
    encoding=settings.raw_data_encoding,
)
repository = CSVDatasetRepository(raw_source)
print({
    "configured_path": str(settings.raw_data_path_config),
    "resolved_path": str(repository.csv_path),
    "read_options": raw_source.read_kwargs(),
})
raw_df = repository.load()
raw_df.head()
"""),
        markdown("## 2. Structural and business quality\n\nThe quality gate reports problems without silently modifying the source."),
        code("""
import pandas as pd
from credit_risk_lab.infrastructure.data_quality import build_quality_report, clean_implausible_rows

quality = build_quality_report(raw_df)
clean_df = clean_implausible_rows(raw_df)
pd.DataFrame([{**quality.as_dict(), "clean_rows": len(clean_df), "removed_rows": len(raw_df)-len(clean_df)}])
"""),
        markdown("## 3. Target and numerical ranges\n\nThe positive class is the synthetic risk class. Its exact business definition must be replaced before real use."),
        code("""
import plotly.express as px

target = raw_df[settings.target_column].value_counts().rename_axis("class").reset_index(name="rows")
target["rate"] = target["rows"] / len(raw_df)
display(target)
px.bar(target, x="class", y="rows", text="rate", title="Target distribution", template="plotly_white").show()
raw_df.select_dtypes("number").agg(["min", "median", "max"]).T
"""),
        markdown("## Conclusion\n\nThe source path, parsing options, quality problems, cleaning impact, and target balance are now explicit."),
    ],
)


split_drift = notebook(
    "02 — External 90/10 Split and Drift",
    "Create the production-simulation holdout, prove its isolation, and visualize train/test drift.",
    [
        setup,
        markdown("## 1. Create and persist the external split\n\n`test.csv` remains raw so API simulation exercises feature engineering at inference time."),
        code("""
from credit_risk_lab.application import create_deployment_split
from credit_risk_lab.infrastructure.data_sources import CSVDataSourceConfig, CSVDatasetRepository

raw_source = CSVDataSourceConfig(path=settings.raw_data_path, sep=settings.raw_data_sep, encoding=settings.raw_data_encoding)
raw_df = CSVDatasetRepository(raw_source).load()
split_result = create_deployment_split(raw_df, test_size=0.10)
split_result
"""),
        markdown("## 2. Load both persisted partitions explicitly"),
        code("""
train_df = CSVDatasetRepository(CSVDataSourceConfig(path=settings.train_path)).load()
test_df = CSVDatasetRepository(CSVDataSourceConfig(path=settings.test_path)).load()
pd.DataFrame({
    "sample": ["train", "external_test"],
    "rows": [len(train_df), len(test_df)],
    "positive_rate": [train_df[settings.target_column].mean(), test_df[settings.target_column].mean()],
})
"""),
        markdown("## 3. Complete drift report\n\nNumerical features receive PSI, KS, and Hellinger; categorical features receive PSI over the category union."),
        code("""
from credit_risk_lab.infrastructure.analytics import DriftAnalyzer

features = [c for c in train_df.columns if c != settings.target_column]
drift_report = DriftAnalyzer(bins=10).report_frame(train_df, test_df, features=features)
drift_report
"""),
        markdown("## 4. Global PSI visual and detailed distributions"),
        code("""
from credit_risk_lab.infrastructure.visualization import (
    plot_categorical_distribution, plot_drift_summary, plot_numeric_distribution,
)

plot_drift_summary(drift_report).show()
for feature in drift_report.query("type == 'numeric'")["feature"].head(3):
    plot_numeric_distribution(train_df, test_df, feature).show()
for feature in drift_report.query("type == 'categorical'")["feature"].head(2):
    plot_categorical_distribution(train_df, test_df, feature).show()
"""),
        markdown("## 5. Persist drift artifacts"),
        code("""
settings.reports_dir.mkdir(parents=True, exist_ok=True)
drift_report.to_csv(settings.drift_report_path, index=False)
plot_drift_summary(drift_report).write_html(settings.drift_summary_plot_path, include_plotlyjs="cdn")
{"csv": str(settings.drift_report_path), "html": str(settings.drift_summary_plot_path)}
"""),
    ],
)


features = notebook(
    "03 — Feature Engineering and Preprocessing",
    "Run deterministic domain features and demonstrate leakage-safe learned preprocessing on training data only.",
    [
        setup,
        markdown("## 1. Load the 90% training partition"),
        code("""
from credit_risk_lab.infrastructure.data_sources import CSVDataSourceConfig, CSVDatasetRepository

train_raw = CSVDatasetRepository(CSVDataSourceConfig(path=settings.train_path)).load()
train_raw.shape
"""),
        markdown("## 2. Deterministic feature engineering\n\nThese transformations learn no dataset statistics and are reused by the API."),
        code("""
from credit_risk_lab.infrastructure.feature_engineering import LoanFeatureEngineer

engineer = LoanFeatureEngineer()
train_features = engineer.transform(train_raw)
new_features = [c for c in train_features if c not in train_raw]
pd.DataFrame({"feature": new_features, "missing": train_features[new_features].isna().sum().values})
"""),
        markdown("## 3. Leakage-safe preprocessing\n\nThe preprocessor is fitted only on a training subset and then reused unchanged."),
        code("""
from credit_risk_lab.application import three_way_stratified_split
from credit_risk_lab.infrastructure.modeling import build_preprocessor

split = three_way_stratified_split(train_features)
sensitive = [c for c in settings.sensitive_columns if c in split.x_train]
x_train = split.x_train.drop(columns=sensitive)
x_validation = split.x_validation.drop(columns=sensitive)
preprocessor = build_preprocessor(x_train)
train_matrix = preprocessor.fit_transform(x_train)
validation_matrix = preprocessor.transform(x_validation)
{"train_matrix": train_matrix.shape, "validation_matrix": validation_matrix.shape,
 "output_features": preprocessor.get_feature_names_out()[:20].tolist()}
"""),
        markdown("## Conclusion\n\nThe API repeats deterministic feature engineering but never refits the persisted learned preprocessor."),
    ],
)


training = notebook(
    "04 — Configured Model Training and Persistence",
    "Load models from YAML, compare all enabled candidates on validation, test only the winner, and persist lineage.",
    [
        setup,
        markdown("## 1. Inspect `configs/models.yaml` and instantiate candidates"),
        code("""
from credit_risk_lab.infrastructure.modeling import build_configured_models, load_models_config

models_config = load_models_config()
models = build_configured_models(random_state=settings.random_state, config=models_config)
pd.DataFrame([{"model": m.name, "parameters": m.parameters, "early_stopping": m.early_stopping_rounds} for m in models])
"""),
        markdown("## 2. Load and engineer only the 90% training partition"),
        code("""
from credit_risk_lab.infrastructure.data_sources import CSVDataSourceConfig, CSVDatasetRepository
from credit_risk_lab.infrastructure.feature_engineering import LoanFeatureEngineer

train_raw = CSVDatasetRepository(CSVDataSourceConfig(path=settings.train_path)).load()
train_features = LoanFeatureEngineer().transform(train_raw)
"""),
        markdown("## 3. Train, select on validation, and test the locked winner"),
        code("""
from credit_risk_lab.application import TrainBoostingModelsUseCase

result = TrainBoostingModelsUseCase().execute(train_features)
display(result.metrics.round(4).rename_axis("validation candidates"))
display(result.test_metrics.round(4).rename_axis("internal final test of winner"))
"""),
        markdown("## 4. Learning curves and validation comparison"),
        code("""
from credit_risk_lab.infrastructure.visualization import plot_learning_curves, plot_model_comparison

plot_learning_curves(result.histories).show()
plot_model_comparison(result.metrics).show()
"""),
        markdown("## 5. Persist the selected bundle with lineage"),
        code("""
import subprocess
from credit_risk_lab.infrastructure.modeling import save_model_bundle, sha256_file

winner = result.selected_model_name
winner_row = result.test_metrics.iloc[0]
try:
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
except Exception:
    commit = "unavailable"
save_model_bundle(
    settings.model_bundle_path,
    model=result.models[winner], preprocessor=result.preprocessor,
    threshold=float(winner_row["threshold"]),
    metadata={
        "model_name": winner,
        "test_metrics": winner_row.to_dict(),
        "selection_metric": settings.selection_metric,
        "split_strategy": result.split_strategy,
        "training_dataset_sha256": sha256_file(settings.train_path),
        "external_test_dataset_sha256": sha256_file(settings.test_path),
        "models_config_sha256": sha256_file(settings.models_config_path),
        "git_commit": commit,
        "target_definition": "loan_status=1 is the synthetic positive risk class",
    },
)
settings.model_bundle_path
"""),
    ],
)


evaluation = notebook(
    "05 — External Evaluation, Calibration, and Fairness",
    "Evaluate the persisted winner once on the untouched 10% raw holdout and audit probability quality and groups.",
    [
        setup,
        markdown("## 1. Score the untouched raw external test through the real inference pipeline"),
        code("""
from credit_risk_lab.application import RawLoanScorer
from credit_risk_lab.infrastructure.data_sources import CSVDataSourceConfig, CSVDatasetRepository
from credit_risk_lab.infrastructure.modeling import load_model_bundle

test_raw = CSVDatasetRepository(CSVDataSourceConfig(path=settings.test_path)).load()
bundle = load_model_bundle(settings.model_bundle_path)
scorer = RawLoanScorer(bundle, threshold=settings.decision_threshold)
scored = scorer.score(test_raw.drop(columns=[settings.target_column]))
{"rows": len(scored.probabilities), "model": scored.model_name, "serving_threshold": scored.threshold}
"""),
        markdown("## 2. External metrics at the operational YAML threshold"),
        code("""
from credit_risk_lab.infrastructure.evaluation import classification_metrics

y_external = test_raw[settings.target_column].astype(int)
external_metrics = classification_metrics(y_external, scored.probabilities, settings.decision_threshold)
pd.Series(external_metrics).sort_index().to_frame("value")
"""),
        markdown("## 3. Calibration table and curve"),
        code("""
from credit_risk_lab.infrastructure.evaluation import calibration_table
from credit_risk_lab.infrastructure.visualization import plot_calibration

calibration = calibration_table(y_external, scored.probabilities, bins=10)
display(calibration.round(4))
plot_calibration(calibration, scored.model_name).show()
"""),
        markdown("## 4. Group diagnostics\n\nSensitive variables were excluded from training and are used here only for governance."),
        code("""
from credit_risk_lab.infrastructure.evaluation import fairness_report

sensitive = test_raw[[c for c in settings.sensitive_columns if c in test_raw]]
fairness = fairness_report(y_external, scored.probabilities, sensitive, settings.decision_threshold)
fairness
"""),
        markdown("## 5. Persist external evaluation"),
        code("""
external_path = settings.reports_dir / "external_test_metrics.csv"
pd.DataFrame([{"model": scored.model_name, "threshold": settings.decision_threshold, **external_metrics}]).to_csv(external_path, index=False)
external_path
"""),
    ],
)


api_simulation = notebook(
    "06 — API Inference and Production Simulation",
    "Exercise health, readiness, validation, feature engineering, prediction, and repeated raw-row requests without starting an external server.",
    [
        setup,
        markdown("## 1. Start the FastAPI application in-process\n\n`TestClient` runs the same application used by Uvicorn and Docker."),
        code("""
from fastapi.testclient import TestClient
from credit_risk_lab.interfaces.api import app

client = TestClient(app)
client.__enter__()
{"health": client.get("/health").json(), "ready": client.get("/ready").json()}
"""),
        markdown("## 2. Send one raw application\n\nThe label is removed before the request. The API validates input, engineers features, scores, and returns audit metadata."),
        code("""
from credit_risk_lab.infrastructure.data_sources import CSVDataSourceConfig, CSVDatasetRepository

test_df = CSVDatasetRepository(CSVDataSourceConfig(path=settings.test_path)).load()
row = test_df.iloc[0]
payload = row.drop(labels=[settings.target_column]).to_dict()
response = client.post("/v1/predict", json=payload, headers={"X-Request-ID": "notebook-example-1"})
response.status_code, response.json(), {"expected_label": int(row[settings.target_column])}
"""),
        markdown("## 3. Simulate a short production stream"),
        code("""
responses = []
for index, row in test_df.head(20).iterrows():
    payload = row.drop(labels=[settings.target_column]).to_dict()
    http = client.post("/v1/predict", json=payload)
    body = http.json()
    responses.append({
        "row": int(index), "http_status": http.status_code,
        "expected": int(row[settings.target_column]),
        "probability": body["probability_of_risk"],
        "decision": body["risk_decision"], "latency_ms": body["latency_ms"],
    })
simulation = pd.DataFrame(responses)
simulation
"""),
        markdown("## 4. Inspect simulation behavior"),
        code("""
display(simulation.describe())
px.histogram(simulation, x="probability", color="expected", nbins=20, title="Simulated API risk probabilities").show()
"""),
        markdown("## 5. Validate a bad request and close the client"),
        code("""
invalid = payload.copy()
invalid["person_age"] = 20
invalid["person_emp_exp"] = 10
invalid_response = client.post("/v1/predict", json=invalid)
client.__exit__(None, None, None)
{"status": invalid_response.status_code, "detail": invalid_response.json()}
"""),
    ],
)


NOTEBOOKS = {
    "01_data_source_quality_and_eda.ipynb": source_quality,
    "02_external_split_and_drift.ipynb": split_drift,
    "03_feature_engineering_and_preprocessing.ipynb": features,
    "04_configured_training_and_persistence.ipynb": training,
    "05_external_evaluation_and_fairness.ipynb": evaluation,
    "06_api_inference_and_simulation.ipynb": api_simulation,
}


def main() -> None:
    """Replace obsolete notebooks with the canonical executable sequence."""
    DEV.mkdir(exist_ok=True)
    for old in DEV.glob("*.ipynb"):
        old.unlink()
    for filename, nb in NOTEBOOKS.items():
        nbf.write(nb, DEV / filename)
        print(f"Wrote {filename}")


if __name__ == "__main__":
    main()
