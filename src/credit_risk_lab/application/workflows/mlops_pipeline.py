"""End-to-end MLOps pipeline orchestration."""

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from typing import Any

import numpy as np
import pandas as pd

from credit_risk_lab.application import (
    DatasetSplitter,
    RawLoanScorer,
    SplitConfig,
)
from credit_risk_lab.config.settings import Settings, settings
from credit_risk_lab.infrastructure import CreditRiskQualityChecker
from credit_risk_lab.infrastructure.analytics import DataLeakageAuditor, DriftAnalyzer
from credit_risk_lab.infrastructure.data_sources import (
    CSVDatasetRepository,
    CsvLoanDataLoader,
)
from credit_risk_lab.infrastructure.evaluation import (
    CalibrationEvaluator,
    CreditRiskModelEvaluator,
    FairnessEvaluator,
    ThresholdAnalysisConfig,
    ThresholdAnalyzer,
    bootstrap_metric_intervals,
    lift_gain_table,
)
from credit_risk_lab.infrastructure.feature_engineering import LoanFeatureEngineer
from credit_risk_lab.infrastructure.modeling import (
    BestModelSelector,
    BoostingModelTrainer,
    CatBoostOptunaTuner,
    CreditRiskPreprocessor,
    JoblibModelBundleRepository,
    sha256_file,
)
from credit_risk_lab.shared.logging import setup_logger

from ._shared import current_git_commit


@dataclass(frozen=True)
class CreditRiskMLOpsPipelineResult:
    """Structured outputs produced by the end-to-end pipeline."""

    summary: pd.DataFrame
    artifacts: pd.DataFrame
    baseline_metrics: pd.DataFrame
    tuned_metrics: pd.DataFrame
    final_metrics: pd.DataFrame
    threshold_grid: pd.DataFrame
    lift_gain: pd.DataFrame
    calibration: pd.DataFrame
    confidence_intervals: pd.DataFrame
    fairness: pd.DataFrame
    promotion_report: pd.DataFrame


class CreditRiskMLOpsPipeline:
    """Run the CI/CD-oriented credit-risk training and evaluation flow."""

    def __init__(
        self,
        *,
        project_settings: Settings = settings,
        optuna_trials: int | None = None,
    ):
        self.settings = project_settings
        self.optuna_trials = (
            self.settings.optuna_trials if optuna_trials is None else optuna_trials
        )
        self.logger = setup_logger("CreditRiskMLOpsPipeline")

    def run(self) -> CreditRiskMLOpsPipelineResult:
        """Execute the full local MLOps pipeline and return auditable outputs."""
        self._ensure_output_directories()
        raw_df = self._load_raw_dataset()
        raw_train, raw_test, holdout_summary = self._create_or_load_external_holdout(
            raw_df
        )
        clean_train = self._validate_and_clean(raw_train)
        train, validation, split_summary = self._create_development_split(clean_train)
        leakage_report = self._leakage_check(train, validation)
        drift_report = self._drift_check(train, validation)
        processed = self._feature_engineer_and_preprocess(train, validation)
        baseline = self._train_baselines(processed)
        tuned = self._tune_catboost(processed)
        candidate_bundle_path = self._save_model_bundle(
            self.settings.candidate_model_bundle_path,
            tuned,
            processed,
            artifact_role="candidate",
        )
        evaluation = self._evaluate_on_external_test(
            raw_test,
            tuned.threshold,
            bundle_path=candidate_bundle_path,
        )
        promotion = self._promotion_gate(
            tuned=tuned,
            processed=processed,
            evaluation=evaluation,
        )

        summary = self._summary_frame(
            holdout_summary=holdout_summary,
            split_summary=split_summary,
            leakage_report=leakage_report,
            drift_report=drift_report,
            baseline=baseline,
            tuned=tuned,
            evaluation=evaluation,
            promotion=promotion,
        )
        artifacts = self._artifact_frame(
            candidate_bundle_path=candidate_bundle_path,
            promoted_bundle_path=promotion["promoted_bundle_path"],
        )

        return CreditRiskMLOpsPipelineResult(
            summary=summary,
            artifacts=artifacts,
            baseline_metrics=baseline["metrics"],
            tuned_metrics=pd.DataFrame(
                [
                    {
                        "model": tuned.model_name,
                        "threshold": tuned.threshold,
                        **tuned.metrics,
                    }
                ]
            ),
            final_metrics=evaluation["metrics"],
            threshold_grid=evaluation["threshold_grid"],
            lift_gain=evaluation["lift_gain"],
            calibration=evaluation["calibration"],
            confidence_intervals=evaluation["confidence_intervals"],
            fairness=evaluation["fairness"],
            promotion_report=promotion["report"],
        )

    def _ensure_output_directories(self) -> None:
        for path in (
            self.settings.raw_dir,
            self.settings.processed_dir,
            self.settings.models_dir,
            self.settings.reports_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)

    def _load_raw_dataset(self) -> pd.DataFrame:
        self.logger.info("Loading raw dataset from {}", self.settings.raw_data_path)
        return CsvLoanDataLoader(
            path=self.settings.raw_data_path,
            sep=self.settings.raw_data_sep,
            encoding=self.settings.raw_data_encoding,
        ).load()

    def _create_or_load_external_holdout(
        self,
        raw_df: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        split_config = SplitConfig(
            test_size=0.10,
            random_state=self.settings.random_state,
            stratify=True,
        )
        if self._split_manifest_is_current(split_config):
            raw_train = CsvLoanDataLoader(path=self.settings.raw_train_path).load()
            raw_test = CsvLoanDataLoader(path=self.settings.raw_test_path).load()
            summary = DatasetSplitter(split_config).summary(raw_train, raw_test)
            self.logger.info(
                "External holdout reused from manifest: train={} test={}",
                len(raw_train),
                len(raw_test),
            )
            return raw_train, raw_test, summary

        splitter = DatasetSplitter(split_config)
        raw_train, raw_test = splitter.split(raw_df)
        CSVDatasetRepository.save(raw_train, self.settings.raw_train_path)
        CSVDatasetRepository.save(raw_test, self.settings.raw_test_path)
        self._write_split_manifest(raw_df, raw_train, raw_test, split_config)
        summary = splitter.summary(raw_train, raw_test)
        self.logger.info(
            "External holdout created: train={} test={}", len(raw_train), len(raw_test)
        )
        return raw_train, raw_test, summary

    def _split_manifest_is_current(self, split_config: SplitConfig) -> bool:
        paths = [
            self.settings.raw_train_path,
            self.settings.raw_test_path,
            self.settings.raw_split_manifest_path,
        ]
        if not all(path.exists() for path in paths):
            return False
        try:
            manifest = json.loads(
                self.settings.raw_split_manifest_path.read_text(encoding="utf-8")
            )
        except json.JSONDecodeError:
            return False
        expected = {
            "source_sha256": sha256_file(self.settings.raw_data_path),
            "test_size": split_config.test_size,
            "random_state": split_config.random_state,
            "stratify": split_config.stratify,
            "target_column": split_config.target_column,
            "train_sha256": sha256_file(self.settings.raw_train_path),
            "test_sha256": sha256_file(self.settings.raw_test_path),
        }
        split = manifest.get("split", {})
        hashes = manifest.get("hashes", {})
        return (
            hashes.get("source_sha256") == expected["source_sha256"]
            and hashes.get("train_sha256") == expected["train_sha256"]
            and hashes.get("test_sha256") == expected["test_sha256"]
            and split.get("test_size") == expected["test_size"]
            and split.get("random_state") == expected["random_state"]
            and split.get("stratify") == expected["stratify"]
            and split.get("target_column") == expected["target_column"]
        )

    def _write_split_manifest(
        self,
        source: pd.DataFrame,
        train: pd.DataFrame,
        test: pd.DataFrame,
        split_config: SplitConfig,
    ) -> None:
        manifest = {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "environment": self.settings.environment,
            "source": {
                "path": str(self.settings.raw_data_path),
                "rows": len(source),
                "columns": len(source.columns),
            },
            "split": {
                "strategy": "stratified_random_holdout",
                "test_size": split_config.test_size,
                "random_state": split_config.random_state,
                "stratify": split_config.stratify,
                "target_column": split_config.target_column,
            },
            "outputs": {
                "train_path": str(self.settings.raw_train_path),
                "test_path": str(self.settings.raw_test_path),
                "train_rows": len(train),
                "test_rows": len(test),
                "train_positive_rate": float(train[self.settings.target_column].mean()),
                "test_positive_rate": float(test[self.settings.target_column].mean()),
            },
            "hashes": {
                "source_sha256": sha256_file(self.settings.raw_data_path),
                "train_sha256": sha256_file(self.settings.raw_train_path),
                "test_sha256": sha256_file(self.settings.raw_test_path),
            },
        }
        self.settings.raw_split_manifest_path.parent.mkdir(parents=True, exist_ok=True)
        self.settings.raw_split_manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    def _validate_and_clean(self, raw_train: pd.DataFrame) -> pd.DataFrame:
        checker = CreditRiskQualityChecker()
        report = checker.validate(raw_train)
        clean_train = checker.clean(raw_train)
        self.logger.info(
            "Quality check completed: rows={} invalid_age={} invalid_experience={} clean_rows={}",
            report.rows,
            report.invalid_age_rows,
            report.invalid_experience_rows,
            len(clean_train),
        )
        return clean_train

    def _create_development_split(
        self,
        clean_train: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        splitter = DatasetSplitter(
            SplitConfig(
                test_size=self.settings.validation_size,
                random_state=self.settings.random_state,
                stratify=True,
            )
        )
        train, validation = splitter.split(clean_train)
        summary = splitter.summary(train, validation)
        self.logger.info(
            "Development split created: train={} validation={}",
            len(train),
            len(validation),
        )
        return train, validation, summary

    def _leakage_check(
        self, train: pd.DataFrame, validation: pd.DataFrame
    ) -> pd.DataFrame:
        auditor = DataLeakageAuditor(target_column=self.settings.target_column)
        return auditor.row_overlap_report(train, validation, holdout_name="validation")

    def _drift_check(
        self, train: pd.DataFrame, validation: pd.DataFrame
    ) -> pd.DataFrame:
        features = [
            column for column in train.columns if column != self.settings.target_column
        ]
        report = DriftAnalyzer(bins=10).report_frame(
            train, validation, features=features
        )
        report.to_csv(self.settings.drift_report_path, index=False)
        return report

    def _feature_engineer_and_preprocess(
        self,
        train: pd.DataFrame,
        validation: pd.DataFrame,
    ) -> dict[str, Any]:
        feature_engineer = LoanFeatureEngineer()
        featured_train = feature_engineer.transform(train)
        featured_validation = feature_engineer.transform(validation)
        sensitive_columns = [
            column
            for column in self.settings.sensitive_columns
            if column in featured_train
        ]
        drop_columns = [self.settings.target_column, *sensitive_columns]

        y_train = featured_train[self.settings.target_column].reset_index(drop=True)
        y_validation = featured_validation[self.settings.target_column].reset_index(
            drop=True
        )
        x_train = featured_train.drop(columns=drop_columns)
        x_validation = featured_validation.drop(columns=drop_columns)

        preprocessor = CreditRiskPreprocessor()
        processed_train_features = preprocessor.fit_transform_frame(x_train)
        processed_validation_features = preprocessor.transform_frame(x_validation)
        processed_train = processed_train_features.assign(
            **{self.settings.target_column: y_train.to_numpy()}
        )
        processed_validation = processed_validation_features.assign(
            **{self.settings.target_column: y_validation.to_numpy()}
        )
        CSVDatasetRepository.save(processed_train, self.settings.train_path)
        CSVDatasetRepository.save(processed_validation, self.settings.validation_path)
        preprocessor_path = preprocessor.save(self.settings.preprocessing_artifact_path)
        self.logger.info("Preprocessor saved to {}", preprocessor_path)

        return {
            "train": processed_train,
            "validation": processed_validation,
            "x_train": processed_train.drop(columns=[self.settings.target_column]),
            "y_train": processed_train[self.settings.target_column],
            "x_validation": processed_validation.drop(
                columns=[self.settings.target_column]
            ),
            "y_validation": processed_validation[self.settings.target_column],
            "preprocessor": preprocessor,
            "sensitive_columns": sensitive_columns,
        }

    def _train_baselines(self, processed: dict[str, Any]) -> dict[str, Any]:
        trainer = BoostingModelTrainer(random_state=self.settings.random_state)
        results = trainer.fit(
            processed["x_train"],
            processed["y_train"],
            processed["x_validation"],
            processed["y_validation"],
        )
        metrics = trainer.results_frame(results)
        best = BestModelSelector(metric=self.settings.selection_metric).select(results)
        metrics.to_csv(self.settings.metrics_report_path, index=False)
        return {"results": results, "metrics": metrics, "best": best}

    def _tune_catboost(self, processed: dict[str, Any]):
        tuner = CatBoostOptunaTuner(
            random_state=self.settings.random_state,
            metric=self.settings.selection_metric,
            n_trials=self.optuna_trials,
        )
        tuned = tuner.tune(
            processed["x_train"],
            processed["y_train"],
            processed["x_validation"],
            processed["y_validation"],
        )
        tuned_metrics = pd.DataFrame(
            [{"model": tuned.model_name, "threshold": tuned.threshold, **tuned.metrics}]
        )
        tuned_metrics.to_csv(
            self.settings.reports_dir / "catboost_tuned_metrics.csv", index=False
        )
        tuned.trials.to_csv(
            self.settings.reports_dir / "catboost_optuna_trials.csv", index=False
        )
        return tuned

    def _save_model_bundle(
        self, path, tuned, processed: dict[str, Any], *, artifact_role: str
    ):
        repository = JoblibModelBundleRepository()
        bundle_path = repository.save(
            path,
            model=tuned.model,
            preprocessor=processed["preprocessor"].transformer,
            threshold=float(tuned.threshold),
            metadata={
                "model_name": tuned.model_name,
                "artifact_role": artifact_role,
                "validation_metrics": tuned.metrics,
                "selection_metric": self.settings.selection_metric,
                "optuna_best_value": tuned.best_value,
                "optuna_best_parameters": tuned.best_parameters,
                "split_strategy": "end_to_end_pipeline_raw_train_validation_test",
                "training_dataset_sha256": sha256_file(self.settings.train_path),
                "validation_dataset_sha256": sha256_file(self.settings.validation_path),
                "preprocessing_artifact_sha256": sha256_file(
                    self.settings.preprocessing_artifact_path
                ),
                "models_config_sha256": sha256_file(self.settings.models_config_path),
                "git_commit": current_git_commit(),
                "target_definition": "loan_status=1 is the synthetic positive risk class",
            },
        )
        self.logger.info("Model bundle saved to {}", bundle_path)
        return bundle_path

    def _evaluate_on_external_test(
        self, raw_test: pd.DataFrame, threshold: float, *, bundle_path
    ) -> dict[str, pd.DataFrame]:
        scorer = RawLoanScorer(
            JoblibModelBundleRepository().load(bundle_path),
            threshold=threshold,
        )
        scoring = scorer.score(raw_test)
        probabilities = np.asarray(scoring.probabilities)
        y_test = raw_test[self.settings.target_column].astype(int)
        evaluator = CreditRiskModelEvaluator()
        metrics = pd.DataFrame(
            [
                {
                    "model": scoring.model_name,
                    "threshold": scoring.threshold,
                    **evaluator.metrics(y_test, probabilities, scoring.threshold),
                }
            ]
        )
        threshold_grid = ThresholdAnalyzer(
            ThresholdAnalysisConfig(start=0.02, stop=0.42, step=0.04)
        ).grid(y_test, probabilities)
        lift_gain = lift_gain_table(y_test, probabilities, bins=10)
        calibration = CalibrationEvaluator(bins=10).evaluate(y_test, probabilities)
        confidence_intervals = bootstrap_metric_intervals(
            y_test,
            probabilities,
            scoring.threshold,
            n_bootstrap=100,
            random_state=self.settings.random_state,
        )
        sensitive = raw_test[
            [column for column in self.settings.sensitive_columns if column in raw_test]
        ]
        fairness = FairnessEvaluator(min_group_size=30).evaluate(
            y_test,
            probabilities,
            sensitive,
            scoring.threshold,
        )
        metrics.to_csv(
            self.settings.reports_dir / "external_test_metrics.csv", index=False
        )
        threshold_grid.to_csv(
            self.settings.reports_dir / "external_test_threshold_grid.csv", index=False
        )
        lift_gain.to_csv(
            self.settings.reports_dir / "external_test_lift_gain.csv", index=False
        )
        calibration.to_csv(
            self.settings.reports_dir / "external_test_calibration.csv", index=False
        )
        confidence_intervals.to_csv(
            self.settings.reports_dir / "external_test_confidence_intervals.csv",
            index=False,
        )
        fairness.to_csv(
            self.settings.reports_dir / "external_test_fairness.csv", index=False
        )
        return {
            "metrics": metrics,
            "threshold_grid": threshold_grid,
            "lift_gain": lift_gain,
            "calibration": calibration,
            "confidence_intervals": confidence_intervals,
            "fairness": fairness,
        }

    def _promotion_gate(
        self,
        *,
        tuned,
        processed: dict[str, Any],
        evaluation: dict[str, pd.DataFrame],
    ) -> dict[str, Any]:
        validation_roc_auc = float(tuned.metrics["roc_auc"])
        minimum = float(self.settings.minimum_validation_roc_auc)
        approved = validation_roc_auc >= minimum
        reason = (
            f"validation_roc_auc {validation_roc_auc:.6f} >= {minimum:.6f}"
            if approved
            else f"validation_roc_auc {validation_roc_auc:.6f} < {minimum:.6f}"
        )
        promoted_path = None
        if approved:
            promoted_path = self._save_model_bundle(
                self.settings.model_bundle_path,
                tuned,
                processed,
                artifact_role="promoted",
            )
            self.logger.info("Model promotion approved: {}", reason)
        else:
            self.logger.warning("Model promotion rejected: {}", reason)

        final_metrics = evaluation["metrics"].iloc[0].to_dict()
        report = pd.DataFrame(
            [
                {
                    "promotion_status": "approved" if approved else "rejected",
                    "promotion_reason": reason,
                    "gate_metric": "validation_roc_auc",
                    "gate_value": validation_roc_auc,
                    "gate_threshold": minimum,
                    "candidate_bundle_path": str(
                        self.settings.candidate_model_bundle_path
                    ),
                    "promoted_bundle_path": (
                        str(self.settings.model_bundle_path) if promoted_path else ""
                    ),
                    "reported_external_test_roc_auc": final_metrics.get("roc_auc"),
                    "reported_external_test_pr_auc": final_metrics.get("pr_auc"),
                }
            ]
        )
        report.to_csv(
            self.settings.reports_dir / "model_promotion_report.csv", index=False
        )
        return {
            "approved": approved,
            "promoted_bundle_path": promoted_path,
            "report": report,
        }

    def _summary_frame(
        self,
        *,
        holdout_summary: pd.DataFrame,
        split_summary: pd.DataFrame,
        leakage_report: pd.DataFrame,
        drift_report: pd.DataFrame,
        baseline: dict[str, Any],
        tuned,
        evaluation: dict[str, pd.DataFrame],
        promotion: dict[str, Any],
    ) -> pd.DataFrame:
        final = evaluation["metrics"].iloc[0]
        baseline_best = baseline["metrics"].iloc[0]
        return pd.DataFrame(
            [
                {"step": "environment", "value": self.settings.environment},
                {
                    "step": "raw_train_rows",
                    "value": int(holdout_summary.iloc[0]["rows"]),
                },
                {
                    "step": "raw_test_rows",
                    "value": int(holdout_summary.iloc[1]["rows"]),
                },
                {
                    "step": "development_train_rows",
                    "value": int(split_summary.iloc[0]["rows"]),
                },
                {
                    "step": "development_validation_rows",
                    "value": int(split_summary.iloc[1]["rows"]),
                },
                {
                    "step": "row_overlap_validation",
                    "value": self._first_available(leakage_report),
                },
                {
                    "step": "max_validation_psi",
                    "value": round(float(drift_report["psi"].max()), 6),
                },
                {"step": "best_baseline_model", "value": baseline_best["model"]},
                {
                    "step": "best_baseline_roc_auc",
                    "value": round(float(baseline_best["roc_auc"]), 6),
                },
                {"step": "tuned_model", "value": tuned.model_name},
                {
                    "step": "tuned_validation_roc_auc",
                    "value": round(float(tuned.metrics["roc_auc"]), 6),
                },
                {
                    "step": "final_test_roc_auc",
                    "value": round(float(final["roc_auc"]), 6),
                },
                {
                    "step": "final_test_pr_auc",
                    "value": round(float(final["pr_auc"]), 6),
                },
                {
                    "step": "decision_threshold",
                    "value": round(float(final["threshold"]), 6),
                },
                {
                    "step": "promotion_status",
                    "value": promotion["report"].iloc[0]["promotion_status"],
                },
                {
                    "step": "promotion_reason",
                    "value": promotion["report"].iloc[0]["promotion_reason"],
                },
            ]
        )

    def _artifact_frame(
        self, *, candidate_bundle_path, promoted_bundle_path
    ) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "artifact": "raw_split_manifest",
                    "path": str(self.settings.raw_split_manifest_path),
                },
                {"artifact": "raw_train", "path": str(self.settings.raw_train_path)},
                {
                    "artifact": "raw_test_untouched",
                    "path": str(self.settings.raw_test_path),
                },
                {"artifact": "processed_train", "path": str(self.settings.train_path)},
                {
                    "artifact": "processed_validation",
                    "path": str(self.settings.validation_path),
                },
                {
                    "artifact": "preprocessor",
                    "path": str(self.settings.preprocessing_artifact_path),
                },
                {
                    "artifact": "candidate_model_bundle",
                    "path": str(candidate_bundle_path),
                },
                {
                    "artifact": "promoted_model_bundle",
                    "path": str(promoted_bundle_path or ""),
                },
                {
                    "artifact": "promotion_report",
                    "path": str(
                        self.settings.reports_dir / "model_promotion_report.csv"
                    ),
                },
            ]
        )

    @staticmethod
    def _first_available(frame: pd.DataFrame):
        if frame.empty:
            return "not_available"
        for column in ("overlap_count", "overlap_rows", "rows", "value"):
            if column in frame.columns:
                return frame.iloc[0][column]
        return "available"
