"""Notebook- and CLI-friendly application workflows."""

from ._shared import current_git_commit
from .evaluation import ExternalEvaluationResult, run_external_evaluation_workflow
from .feature_engineering import (
    FeatureEngineeringResult,
    run_feature_engineering_workflow,
)
from .mlops_pipeline import CreditRiskMLOpsPipeline, CreditRiskMLOpsPipelineResult
from .source_quality import SourceQualityResult, run_source_quality_workflow
from .split_and_drift import SplitAndDriftResult, run_split_and_drift_workflow
from .training import TrainingWorkflowResult, run_training_workflow

__all__ = [
    "ExternalEvaluationResult",
    "CreditRiskMLOpsPipeline",
    "CreditRiskMLOpsPipelineResult",
    "FeatureEngineeringResult",
    "SourceQualityResult",
    "SplitAndDriftResult",
    "TrainingWorkflowResult",
    "current_git_commit",
    "run_external_evaluation_workflow",
    "run_feature_engineering_workflow",
    "run_source_quality_workflow",
    "run_split_and_drift_workflow",
    "run_training_workflow",
]
