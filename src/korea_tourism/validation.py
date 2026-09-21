"""Fail-fast data contracts shared by ingestion and evaluation."""

from __future__ import annotations

import numpy as np
import pandas as pd


def validate_monthly(df: pd.DataFrame, name: str, *, contiguous: bool = True) -> None:
    if df.empty or "date" not in df:
        raise ValueError(f"{name}: expected a nonempty table with a date column")
    dates = df["date"]
    if not pd.api.types.is_datetime64_any_dtype(dates) or dates.isna().any():
        raise ValueError(f"{name}: invalid monthly dates")
    if not dates.is_unique:
        raise ValueError(f"{name}: duplicate months")
    if not dates.is_monotonic_increasing or not dates.dt.is_month_start.all():
        raise ValueError(f"{name}: dates must be ordered month starts")
    if contiguous and list(dates) != list(pd.date_range(dates.min(), dates.max(), freq="MS")):
        raise ValueError(f"{name}: missing months; row shifts would not represent calendar lags")


def validate_numeric(df: pd.DataFrame, columns: list[str], name: str) -> None:
    missing = set(columns) - set(df.columns)
    if missing:
        raise ValueError(f"{name}: missing columns {sorted(missing)}")
    values = df[columns].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError(f"{name}: expected finite, nonnegative values in {columns}")
