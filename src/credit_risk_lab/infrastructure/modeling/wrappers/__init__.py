"""Individually maintained model wrappers sharing one small contract."""

from .base import ModelWrapper
from .logistic_regression import LogisticRegressionWrapper
from .random_forest import RandomForestWrapper
from .xgboost_model import XGBoostWrapper
from .lightgbm_model import LightGBMWrapper

try:
    from .catboost_model import CatBoostWrapper
except ModuleNotFoundError:
    CatBoostWrapper = None  # type: ignore[assignment]

__all__ = [
    "ModelWrapper",
    "LogisticRegressionWrapper",
    "RandomForestWrapper",
    "XGBoostWrapper",
    "CatBoostWrapper",
    "LightGBMWrapper",
]
