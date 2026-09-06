"""
Unified Loguru Logger Configuration — FINAL VERSION
===================================================

This logger configuration combines:
- the clean formatting of your first logger
- the production-grade structure of the second logger
- one straightforward project-wide logging setup

It ensures:
-----------
✔ Single global Loguru configuration (no duplicate handlers)
✔ Pretty colored console logs for development
✔ Rotating log file with retention + compression
✔ Bound contextual metadata (module name OR custom logger_name)
✔ Standardized logs across entire ML pipeline

Usage:
------
from credit_risk_lab.shared.logging import setup_logger

logger = setup_logger(__name__)
logger.info("Training started")
logger.error("Missing feature detected")
logger.debug("Preview of transformed dataset...")
"""

import sys
from pathlib import Path
from loguru import logger
from credit_risk_lab.config.settings import settings

# Avoid re-configuring loguru
IS_LOGURU_CONFIGURED = False


def setup_logger(
    name: str | None = None,
    *,
    log_name: str = settings.log_file,
    level: str = settings.log_level,
):
    """
    Create or retrieve a Loguru logger bound with contextual metadata.

    Parameters
    ----------
    name : str or None
        Usually __name__ of the calling module.
        Appears in logs under the `module=` metadata.
        Directory where log files will be stored.

    log_name : str
        Name of the rotating log file. Defaults to `settings.log_file`.

    level : str
        Logging level (DEBUG, INFO, WARNING, ERROR). Defaults to `settings.log_level`.

    Why this implementation?
    -------------------------
    - global logger is configured ONCE (console + file handlers)
    - modules only "bind" metadata → no duplicated handlers
    - uniform visual formatting across all layers (infra, app, domain)
    """

    global IS_LOGURU_CONFIGURED

    # Resolve log directory
    Path(settings.logs_dir).mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------
    # STEP 1 — Configure Loguru ONCE globally
    # ---------------------------------------------------------
    if not IS_LOGURU_CONFIGURED:
        logger.remove()  # Remove default handler

        # -----------------------
        # Console Handler (pretty)
        # -----------------------
        logger.add(
            sys.stdout,
            level=level,
            colorize=True,
            format=(
                "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
                "<level>{level: <8}</level> | "
                "<magenta>{extra[module]}</magenta> | "
                "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
                "<level>{message}</level>"
            ),
        )

        # -----------------------
        # File Handler (production)
        # -----------------------
        logger.add(
            f"{settings.logs_dir}/{log_name}",
            rotation="20 MB",
            retention="30 days",
            compression="zip",
            encoding="utf-8",
            level=level,
            format=(
                "{time:YYYY-MM-DD HH:mm:ss} | "
                "{level: <8} | "
                "{extra[module]} | "
                "{name}:{function}:{line} - {message}"
            ),
        )

        IS_LOGURU_CONFIGURED = True

    # ---------------------------------------------------------
    # STEP 2 — Bind contextual metadata for this module
    # ---------------------------------------------------------
    return logger.bind(module=name or "--")
