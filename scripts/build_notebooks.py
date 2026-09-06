"""Generate execution-only notebooks for Credit Risk Lab."""

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
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    nb.metadata.language_info = {"name": "python", "version": "3.14"}
    nb.cells = [markdown(f"# {title}\n\n**Objective:** {objective}")] + cells
    return nb


setup = code(
    """
import pandas as pd
import plotly.express as px

from credit_risk_lab.config.settings import settings
print(f"Project root: {settings.project_root}")
"""
)


source_quality = notebook(
    "01 - Data Source, Quality, and EDA",
    "Execute the packaged source-quality workflow and inspect its outputs.",
    [
        setup,
        markdown("## Run Workflow"),
        code(
            """
from credit_risk_lab.application.workflows import run_source_quality_workflow

result = run_source_quality_workflow()
result.summary
"""
        ),
        markdown("## Target Distribution"),
        code(
            """
target = result.raw[settings.target_column].value_counts().rename_axis("class").reset_index(name="rows")
target["rate"] = target["rows"] / len(result.raw)
display(target)
px.bar(target, x="class", y="rows", text="rate", title="Target distribution", template="plotly_white").show()
"""
        ),
        markdown("## Numeric Ranges"),
        code(
            """
result.raw.select_dtypes("number").agg(["min", "median", "max"]).T
"""
        ),
    ],
)


split_drift = notebook(
    "02 - External Split and Drift",
    "Execute the packaged external-holdout and drift workflow.",
    [
        setup,
        markdown("## Run Workflow"),
        code(
            """
from credit_risk_lab.application.workflows import run_split_and_drift_workflow

result = run_split_and_drift_workflow()
display(result.split_summary)
result.drift_report
"""
        ),
        markdown("## Drift Visuals"),
        code(
            """
from credit_risk_lab.infrastructure.visualization import (
    plot_categorical_distribution,
    plot_drift_summary,
    plot_numeric_distribution,
)

plot_drift_summary(result.drift_report).show()
for feature in result.drift_report.query("type == 'numeric'")["feature"].head(3):
    plot_numeric_distribution(result.train, result.external_test, feature).show()
for feature in result.drift_report.query("type == 'categorical'")["feature"].head(2):
    plot_categorical_distribution(result.train, result.external_test, feature).show()
"""
        ),
    ],
)


features = notebook(
    "03 - Feature Engineering and Preprocessing",
    "Execute deterministic feature engineering and train-only preprocessing checks.",
    [
        setup,
        markdown("## Run Workflow"),
        code(
            """
from credit_risk_lab.application.workflows import run_feature_engineering_workflow

result = run_feature_engineering_workflow()
display(result.new_features)
result.preprocessing_summary
"""
        ),
    ],
)


training = notebook(
    "04 - Configured Training and Persistence",
    "Execute configured model training and persist the selected bundle.",
    [
        setup,
        markdown("## Run Workflow"),
        code(
            """
from credit_risk_lab.application.workflows import run_training_workflow

result = run_training_workflow()
display(result.models)
display(result.validation_metrics.round(4).rename_axis("validation candidates"))
display(result.test_metrics.round(4).rename_axis("internal final test of winner"))
result.model_bundle_path
"""
        ),
        markdown("## Training Curves"),
        code(
            """
from credit_risk_lab.infrastructure.visualization import plot_learning_curves, plot_model_comparison

plot_learning_curves(result.histories).show()
plot_model_comparison(result.validation_metrics).show()
"""
        ),
    ],
)


evaluation = notebook(
    "05 - External Evaluation, Calibration, and Fairness",
    "Execute external holdout evaluation through the persisted inference bundle.",
    [
        setup,
        markdown("## Run Workflow"),
        code(
            """
from credit_risk_lab.application.workflows import run_external_evaluation_workflow

result = run_external_evaluation_workflow()
display(result.metrics.round(4))
display(result.calibration.round(4))
result.fairness
"""
        ),
        markdown("## Calibration Curve"),
        code(
            """
from credit_risk_lab.infrastructure.visualization import plot_calibration

plot_calibration(result.calibration, result.metrics.iloc[0]["model"]).show()
"""
        ),
    ],
)


api_simulation = notebook(
    "06 - API Inference and Production Simulation",
    "Execute API health, validation, and repeated prediction checks in process.",
    [
        setup,
        markdown("## Run Workflow"),
        code(
            """
from credit_risk_lab.interfaces.api_simulation import run_api_simulation

result = run_api_simulation(limit=20)
display(result.health)
display(result.responses)
{"invalid_status_code": result.invalid_status_code}
"""
        ),
        markdown("## Simulation Behavior"),
        code(
            """
display(result.responses.describe())
px.histogram(
    result.responses,
    x="probability_of_risk",
    nbins=20,
    title="Simulated API risk probabilities",
).show()
"""
        ),
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
    """Replace notebooks with the canonical execution-only sequence."""
    DEV.mkdir(exist_ok=True)
    for old in DEV.glob("*.ipynb"):
        old.unlink()
    for filename, nb in NOTEBOOKS.items():
        nbf.write(nb, DEV / filename)
        print(f"Wrote {filename}")


if __name__ == "__main__":
    main()
