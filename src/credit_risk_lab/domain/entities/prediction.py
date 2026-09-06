"""Prediction domain entities."""

from dataclasses import dataclass


@dataclass(frozen=True)
class RiskPrediction:
    """One auditable binary risk decision."""

    probability_of_risk: float
    risk_decision: int
    risk_label: str
    threshold: float
    model_name: str
