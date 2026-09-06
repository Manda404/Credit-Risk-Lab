"""Leakage-aware dataset splitting strategies.

Random stratification is kept only for synthetic teaching datasets.  Real
credit datasets should use :func:`three_way_temporal_split`, which preserves
chronology and can keep every borrower in a single partition.
"""

from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split

from credit_risk_lab.config.settings import settings


@dataclass
class ThreeWaySplit:
    x_train: pd.DataFrame
    x_validation: pd.DataFrame
    x_test: pd.DataFrame
    y_train: pd.Series
    y_validation: pd.Series
    y_test: pd.Series


def three_way_stratified_split(
    frame: pd.DataFrame,
    target_column: str = settings.target_column,
    random_state: int = settings.random_state,
) -> ThreeWaySplit:
    """Create the canonical 60/20/20 stratified development split."""
    x = frame.drop(columns=[target_column])
    y = frame[target_column].astype(int)
    x_train_val, x_test, y_train_val, y_test = train_test_split(
        x, y, test_size=settings.test_size, stratify=y, random_state=random_state
    )
    relative_validation = settings.validation_size / (1 - settings.test_size)
    x_train, x_validation, y_train, y_validation = train_test_split(
        x_train_val,
        y_train_val,
        test_size=relative_validation,
        stratify=y_train_val,
        random_state=random_state,
    )
    return ThreeWaySplit(x_train, x_validation, x_test, y_train, y_validation, y_test)


def three_way_temporal_split(
    frame: pd.DataFrame,
    *,
    date_column: str,
    target_column: str = settings.target_column,
    group_column: str | None = None,
) -> ThreeWaySplit:
    """Split chronologically into train/validation/test partitions.

    Parameters
    ----------
    frame:
        Modeling table containing a decision timestamp and binary target.
    date_column:
        Timestamp known at application time. Invalid or missing dates fail
        loudly rather than being silently assigned to a partition.
    group_column:
        Optional borrower identifier. If supplied, a borrower is assigned
        according to its earliest decision and may never cross partitions.

    Notes
    -----
    The configured validation and test fractions refer to the full dataset.
    This is an out-of-time development split, not a substitute for additional
    cohort backtesting or independent model validation.
    """
    required = {date_column, target_column}
    if group_column:
        required.add(group_column)
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Temporal split requires columns: {missing}")
    work = frame.copy()
    work[date_column] = pd.to_datetime(
        work[date_column], errors="coerce", utc=True, format="mixed"
    )
    if work[date_column].isna().any():
        raise ValueError(f"{date_column} contains missing or invalid timestamps")
    if group_column:
        if work[group_column].isna().any():
            raise ValueError(f"{group_column} contains missing borrower identifiers")
        first_date = work.groupby(group_column)[date_column].transform("min")
        work = work.assign(_split_date=first_date)
    else:
        work = work.assign(_split_date=work[date_column])
    work = work.sort_values(["_split_date", date_column], kind="stable")
    keys = (
        work[group_column].drop_duplicates() if group_column else work.index.to_series()
    )
    n_keys = len(keys)
    train_end = int(n_keys * (1 - settings.validation_size - settings.test_size))
    validation_end = int(n_keys * (1 - settings.test_size))
    if train_end < 1 or validation_end <= train_end or validation_end >= n_keys:
        raise ValueError("Dataset is too small for the configured temporal split")
    train_keys = set(keys.iloc[:train_end])
    validation_keys = set(keys.iloc[train_end:validation_end])
    if group_column:
        train = work[work[group_column].isin(train_keys)]
        validation = work[work[group_column].isin(validation_keys)]
        test = work[~work[group_column].isin(train_keys | validation_keys)]
    else:
        train, validation, test = (
            work.iloc[:train_end],
            work.iloc[train_end:validation_end],
            work.iloc[validation_end:],
        )

    def unpack(part: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        y = part[target_column].astype(int)
        x = part.drop(columns=[target_column, "_split_date"])
        return x, y

    x_train, y_train = unpack(train)
    x_validation, y_validation = unpack(validation)
    x_test, y_test = unpack(test)
    return ThreeWaySplit(x_train, x_validation, x_test, y_train, y_validation, y_test)
