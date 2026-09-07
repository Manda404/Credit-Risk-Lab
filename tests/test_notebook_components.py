import numpy as np
import optuna
import pandas as pd
import pytest

from credit_risk_lab.application import DatasetSplitter, SplitConfig
from credit_risk_lab.infrastructure import CreditRiskQualityChecker
from credit_risk_lab.infrastructure.analytics import (
    ColumnDiagnostics,
    DatasetInspector,
    FeatureEngineeringReport,
)
from credit_risk_lab.infrastructure.evaluation import (
    CalibrationEvaluator,
    CreditRiskModelEvaluator,
    FairnessEvaluator,
)
from credit_risk_lab.infrastructure.modeling import (
    BestModelSelector,
    CatBoostFeatureImportanceAnalyzer,
    CatBoostOptunaTuner,
    CandidateTrainingResult,
    CreditRiskPreprocessor,
    ModelHyperparameterTuner,
)
from credit_risk_lab.infrastructure.visualization import plot_target_distribution
from credit_risk_lab.infrastructure.visualization import (
    DataQualityVisualizer,
    plot_categorical_feature_overview,
    plot_confusion_matrix_and_roc,
    plot_feature_importance,
    plot_lift_gain_accumulation,
    plot_metric_confidence_intervals,
    plot_metric_improvement,
    plot_numeric_outlier_overview,
    plot_optuna_param_importance,
    plot_optuna_parameter_slices,
    plot_optuna_trials,
    plot_threshold_tradeoff,
)
from credit_risk_lab.application.workflows import CreditRiskMLOpsPipeline


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


def test_end_to_end_pipeline_can_be_configured_without_running():
    pipeline = CreditRiskMLOpsPipeline()
    smoke_pipeline = CreditRiskMLOpsPipeline(optuna_trials=1)

    assert pipeline.optuna_trials >= 1
    assert smoke_pipeline.optuna_trials == 1


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


def test_feature_engineering_report_tracks_column_evolution():
    before = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    after = pd.DataFrame({"a": [2, 3], "c": [10.0, 20.0]})
    report = FeatureEngineeringReport(before, after)

    assert report.created_columns() == ["c"]
    assert report.removed_columns() == ["b"]
    assert report.changed_columns() == ["a"]
    assert report.summary().set_index("metric").loc["created_columns", "value"] == 1
    assert report.created_columns_frame().loc[0, "created_column"] == "c"


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


def test_tuning_visuals_compare_metrics_and_trials():
    baseline = pd.DataFrame(
        [
            {
                "model": "CatBoost",
                "roc_auc": 0.80,
                "pr_auc": 0.70,
                "ks": 0.45,
                "f1": 0.60,
                "mcc": 0.50,
                "cohen_kappa": 0.48,
            }
        ]
    )
    tuned = pd.DataFrame(
        [
            {
                "model": "CatBoost",
                "roc_auc": 0.84,
                "pr_auc": 0.73,
                "ks": 0.50,
                "f1": 0.63,
                "mcc": 0.54,
                "cohen_kappa": 0.51,
            }
        ]
    )
    trials = pd.DataFrame(
        {
            "trial": [0, 1, 2],
            "value": [0.80, 0.82, 0.81],
            "state": ["COMPLETE"] * 3,
            "depth": [4, 6, 5],
            "learning_rate": [0.02, 0.05, 0.03],
        }
    )

    improvement = plot_metric_improvement(baseline, tuned, model_name="CatBoost")
    trial_progress = plot_optuna_trials(trials, metric="roc_auc")
    parameter_slices = plot_optuna_parameter_slices(trials, metric="roc_auc")

    study = optuna.create_study(direction="maximize")

    def objective(trial):
        depth = trial.suggest_int("depth", 3, 5)
        learning_rate = trial.suggest_float("learning_rate", 0.01, 0.1)
        return depth * learning_rate

    study.optimize(objective, n_trials=3)
    importances = plot_optuna_param_importance(study, metric="roc_auc")

    assert {trace.type for trace in improvement.data} == {"bar"}
    assert [trace.type for trace in trial_progress.data] == ["scatter", "scatter"]
    assert all(trace.type == "scatter" for trace in parameter_slices.data)
    assert {trace.type for trace in importances.data} == {"bar"}


def test_catboost_feature_importance_is_normalized_and_plotted():
    class FakeCatBoost:
        feature_importances_ = np.array([2.0, 1.0, 1.0])

    importance = CatBoostFeatureImportanceAnalyzer(
        FakeCatBoost(),
        feature_names=["income", "loan_amount", "credit_score"],
    ).importance_frame(top_n=2)
    figure = plot_feature_importance(importance, title="Top features")

    assert importance["feature"].tolist() == ["income", "loan_amount"]
    assert importance["importance_pct"].round(2).tolist() == [50.0, 25.0]
    assert {trace.type for trace in figure.data} == {"bar"}


def test_final_evaluation_visuals_render_scoring_diagnostics():
    target = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    probabilities = np.array([0.1, 0.9, 0.2, 0.8, 0.35, 0.7, 0.4, 0.6])
    lift_gain = pd.DataFrame(
        {
            "decile": [1, 2, 3, 4],
            "lift": [2.0, 1.5, 0.5, 0.0],
            "sample_fraction": [0.25, 0.50, 0.75, 1.0],
            "cumulative_capture_rate": [0.5, 0.75, 1.0, 1.0],
            "cumulative_lift": [2.0, 1.5, 1.33, 1.0],
        }
    )
    intervals = pd.DataFrame(
        {
            "metric": ["roc_auc", "f1"],
            "estimate": [0.95, 0.88],
            "lower": [0.90, 0.80],
            "upper": [1.0, 0.95],
        }
    )
    threshold_grid = pd.DataFrame(
        {
            "threshold": [0.2, 0.4],
            "precision": [0.7, 0.8],
            "recall": [0.9, 0.75],
            "alert_rate": [0.4, 0.3],
            "approval_rate": [0.6, 0.7],
            "high_risk_missed": [2, 4],
            "false_alerts": [8, 5],
        }
    )

    decision = plot_confusion_matrix_and_roc(
        target,
        probabilities,
        threshold=0.5,
        model_name="CatBoost",
    )
    scoring = plot_lift_gain_accumulation(lift_gain)
    confidence = plot_metric_confidence_intervals(intervals)
    threshold = plot_threshold_tradeoff(threshold_grid, selected_threshold=0.3)

    assert [trace.type for trace in decision.data] == ["heatmap", "scatter", "scatter"]
    assert decision.data[0].text == (["TN<br>4", "FP<br>0"], ["FN<br>0", "TP<br>4"])
    assert {"scatter", "bar"}.issubset({trace.type for trace in scoring.data})
    assert [trace.type for trace in confidence.data] == ["scatter"]
    assert {"scatter", "bar"}.issubset({trace.type for trace in threshold.data})


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
    tmp_path,
):
    features = credit_risk_sample.drop(columns=["loan_status"])
    preprocessor = CreditRiskPreprocessor()

    with pytest.raises(RuntimeError, match="fitted"):
        preprocessor.transform(features)

    matrix = preprocessor.fit_transform(features)

    assert matrix.shape[0] == len(features)
    assert preprocessor.feature_names_
    assert preprocessor.transform(features).shape == matrix.shape
    assert (
        preprocessor.transform_frame(features).columns.tolist()
        == preprocessor.feature_names_
    )

    artifact_path = preprocessor.save(tmp_path / "preprocessor.joblib")
    loaded = CreditRiskPreprocessor.load(artifact_path)

    assert (
        loaded.transform_frame(features).shape
        == preprocessor.transform_frame(features).shape
    )


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


def test_model_hyperparameter_tuner_returns_sorted_results():
    x_train = pd.DataFrame({"feature": [0, 1, 2, 3, 4, 5]})
    y_train = pd.Series([0, 0, 0, 1, 1, 1])
    x_validation = pd.DataFrame({"feature": [1, 4]})
    y_validation = pd.Series([0, 1])
    tuner = ModelHyperparameterTuner(metric="roc_auc")

    results = tuner.tune(
        model_names=["LogisticRegression"],
        parameter_grids={"LogisticRegression": [{"C": [0.5, 1.0]}]},
        x_train=x_train,
        y_train=y_train,
        x_validation=x_validation,
        y_validation=y_validation,
    )
    results_frame = tuner.results_frame(results)
    best = tuner.select_best(results)

    assert len(results) == 2
    assert results_frame["roc_auc"].is_monotonic_decreasing
    assert best.model_name == "LogisticRegression"


def test_catboost_optuna_tuner_runs_one_trial():
    x_train = pd.DataFrame({"feature": [0, 1, 2, 3, 4, 5, 6, 7]})
    y_train = pd.Series([0, 0, 0, 0, 1, 1, 1, 1])
    x_validation = pd.DataFrame({"feature": [1, 6, 2, 7]})
    y_validation = pd.Series([0, 1, 0, 1])
    tuner = CatBoostOptunaTuner(n_trials=1)

    result = tuner.tune(x_train, y_train, x_validation, y_validation)

    assert tuner.metric == "roc_auc"
    assert result.model_name == "CatBoost"
    assert result.best_parameters
    assert result.trials.shape[0] == 1
    assert 0 <= result.threshold <= 1
