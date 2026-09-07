"""FastAPI delivery adapter for credit-risk inference.

This package exposes the trained model through HTTP while keeping the ML logic
inside the application and infrastructure layers. The public import
``credit_risk_lab.interfaces.api:app`` is kept stable for Uvicorn, Docker, and
tests.
"""

from .main import app, create_app

__all__ = ["app", "create_app"]
