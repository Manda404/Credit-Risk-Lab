import numpy as np
import pandas as pd
import pytest

from credit_risk_lab.application import DatasetSplitter, SplitConfig
from credit_risk_lab.infrastructure import CreditRiskQualityChecker
from credit_risk_lab.infrastructure.analytics import ColumnDiagnostics, DatasetInspector
from credit_risk_lab.infrastructure.evaluation import (
    CalibrationEvaluator,
    CreditRiskModelEvaluator,
    FairnessEvaluator,
)
from credit_risk_lab.infrastructure.modeling import (
    BestModelSelector,
    CandidateTrainingResult,
    CreditRiskPreprocessor,
)
from credit_risk_lab.infrastructure.visualization import plot_target_distribution
from credit_risk_lab.infrastructure.visualization import (
    DataQualityVisualizer,
    plot_categorical_feature_overview,
    plot_numeric_outlier_overview,
)


def test_dataset_inspector_exposes_progressive_diagnostics(credit_risk_sample):
    inspector = DatasetInspector(credit_risk_sample)

    assert inspector.summary().rows == len(credit_risk_sample)
    assert "credit_score" in inspector.schema()["column"].tolist()
    column_summary = inspector.column_summary(sample_size=2)
    assert {"total_rows", "all_values_unique", "examples"}.issubset(
        column_summary.columns
    )
    assert column_summary["examples"].map(lambda values: len(values) <= 2).all()
    assert inspector.missing_values().empty
    assert set(inspector.target_distribution("loan_status")["class"]) == {0, 1}
    assert "credit_score" in inspector.numeric_profile().index


def test_target_distribution_visual_uses_bar_and_pie_with_class_labels(
    credit_risk_sample,
):
    target_distribution = DatasetInspector(credit_risk_sample).target_distribution(
        "loan_status"
    )

    figure = plot_target_distribution(target_distribution)

    assert [trace.type for trace in figure.data] == ["bar", "pie"]
    assert list(figure.data[0].x) == ["0 = Low risk", "1 = High risk"]
    assert list(figure.data[1].labels) == ["0 = Low risk", "1 = High risk"]


def test_column_diagnostics_reports_numeric_outliers_and_categorical_quality(
    credit_risk_sample,
):
    frame = credit_risk_sample.copy()
    frame.loc[len(frame)] = frame.iloc[0]
    frame.loc[len(frame) - 1, "person_income"] = 2_000_000
    diagnostics = ColumnDiagnostics(frame, target_column="loan_status")

    numeric_report = diagnostics.numeric_outlier_report()
    categorical_profile = diagnostics.categorical_profile(rare_threshold=0.2)

    assert "loan_status" not in diagnostics.numeric_features()
    assert "person_income" in numeric_report["feature"].tolist()
    assert numeric_report["outlier_count"].max() > 0
    assert "person_gender" in categorical_profile["feature"].tolist()
    assert {"top_category", "top_rate", "rare_category_count"}.issubset(
        categorical_profile.columns
    )


def test_column_diagnostic_visuals_render_numeric_and_categorical_traces(
    credit_risk_sample,
):
    diagnostics = ColumnDiagnostics(credit_risk_sample, target_column="loan_status")
    numeric_features = diagnostics.numeric_features()[:2]
    categorical_features = diagnostics.categorical_features()[:2]

    numeric_figure = plot_numeric_outlier_overview(
        credit_risk_sample,
        numeric_features,
        target_column="loan_status",
    )
    categorical_figure = plot_categorical_feature_overview(
        credit_risk_sample,
        categorical_features,
        target_column="loan_status",
    )

    assert {"histogram", "box"}.issubset({trace.type for trace in numeric_figure.data})
    histogram_colors = {
        trace.marker.color for trace in numeric_figure.data if trace.type == "histogram"
    }
    box_names = [trace.name for trace in numeric_figure.data if trace.type == "box"]
    assert len(histogram_colors) == 2
    assert set(box_names) == {"loan_status=0", "loan_status=1"}
    assert all(trace.type == "bar" for trace in categorical_figure.data)


def test_data_quality_visualizer_wraps_numeric_and_categorical_views(
    credit_risk_sample,
):
    diagnostics = ColumnDiagnostics(credit_risk_sample, target_column="loan_status")
    visualizer = DataQualityVisualizer(
        credit_risk_sample,
        target_column="loan_status",
    )

    numeric_figure = visualizer.numeric_outlier_overview(
        diagnostics.numeric_features()[:1]
    )
    categorical_figure = visualizer.categorical_feature_overview(
        diagnostics.categorical_features()[:1]
    )

    assert {"histogram", "box"}.issubset({trace.type for trace in numeric_figure.data})
    assert all(trace.type == "bar" for trace in categorical_figure.data)


def test_quality_checker_validates_and_cleans(invalid_credit_risk_rows):
    checker = CreditRiskQualityChecker()

    report = checker.validate(invalid_credit_risk_rows)

    assert report.invalid_age_rows == 1
    assert report.invalid_experience_rows == 1
    assert checker.clean(invalid_credit_risk_rows).empty


def test_dataset_splitter_returns_stratified_holdout(credit_risk_sample):
    frame = pd.concat([credit_risk_sample] * 10, ignore_index=True)
    splitter = DatasetSplitter(SplitConfig(test_size=0.2, random_state=7))

    train, holdout = splitter.split(frame)
    summary = splitter.summary(train, holdout)

    assert len(train) + len(holdout) == len(frame)
    assert len(holdout) == 10
    assert summary["positive_rate"].between(0, 1).all()


def test_credit_risk_preprocessor_requires_fit_and_tracks_features(
    credit_risk_sample,
):
    features = credit_risk_sample.drop(columns=["loan_status"])
    preprocessor = CreditRiskPreprocessor()

    with pytest.raises(RuntimeError, match="fitted"):
        preprocessor.transform(features)

    matrix = preprocessor.fit_transform(features)

    assert matrix.shape[0] == len(features)
    assert preprocessor.feature_names_
    assert preprocessor.transform(features).shape == matrix.shape


def test_evaluators_expose_metrics_calibration_and_fairness():
    y_true = np.array([0, 0, 1, 1, 0, 1])
    probabilities = np.array([0.1, 0.3, 0.7, 0.9, 0.2, 0.8])
    sensitive = pd.DataFrame({"person_gender": ["F", "M", "F", "M", "F", "M"]})

    evaluator = CreditRiskModelEvaluator()
    threshold = evaluator.optimal_threshold(y_true, probabilities)
    metrics = evaluator.metrics_frame("demo", y_true, probabilities, threshold)
    calibration = CalibrationEvaluator(bins=3).evaluate(y_true, probabilities)
    fairness = FairnessEvaluator(min_group_size=1).evaluate(
        y_true, probabilities, sensitive, threshold
    )

    assert metrics.loc[0, "model"] == "demo"
    assert {"roc_auc", "f1", "threshold"}.issubset(metrics.columns)
    assert not calibration.empty
    assert set(fairness["attribute"]) == {"person_gender"}


def test_best_model_selector_uses_requested_metric():
    weak = CandidateTrainingResult("weak", object(), 0.5, {"roc_auc": 0.6}, {})
    strong = CandidateTrainingResult("strong", object(), 0.4, {"roc_auc": 0.9}, {})

    selected = BestModelSelector(metric="roc_auc").select([weak, strong])

    assert selected.model_name == "strong"
