"""Small, fixed candidate set; all learned preprocessing stays inside fit."""

from __future__ import annotations

import numpy as np
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

LAGS = (1, 2, 3, 12)
CALENDAR = ["month_sin", "month_cos"]
FEATURE_GROUPS = {
    "history": ["visitors"],
    "history_fx": ["visitors", "usd_krw", "jpy_krw_100"],
    "history_interest": ["visitors", "google_search_index", "youtube_search_index", "k_index"],
    "all_signals": [
        "visitors",
        "usd_krw",
        "jpy_krw_100",
        "google_search_index",
        "youtube_search_index",
        "k_index",
    ],
}
MODEL_NAMES = ("ridge_log", "random_forest_log", "hist_gradient_boosting_log")
BASELINES = {"last_month": "visitors_lag1", "seasonal_naive": "visitors_lag12"}


def feature_columns(group: str) -> list[str]:
    """Explicit allowlist: arbitrary columns named '*_lag1' cannot enter a model."""
    return [f"{source}_lag{lag}" for source in FEATURE_GROUPS[group] for lag in LAGS] + CALENDAR


def make_model(name: str, seed: int = 42) -> TransformedTargetRegressor:
    if name == "ridge_log":
        regressor = make_pipeline(StandardScaler(), Ridge(alpha=10.0))
    elif name == "random_forest_log":
        regressor = RandomForestRegressor(
            n_estimators=150,
            min_samples_leaf=2,
            random_state=seed,
            n_jobs=1,
        )
    elif name == "hist_gradient_boosting_log":
        regressor = HistGradientBoostingRegressor(
            learning_rate=0.04,
            max_iter=150,
            max_leaf_nodes=15,
            min_samples_leaf=20,
            l2_regularization=0.01,
            early_stopping=False,
            random_state=seed,
        )
    else:
        raise ValueError(f"Unknown model: {name}")
    return TransformedTargetRegressor(
        regressor=regressor,
        func=np.log1p,
        inverse_func=np.expm1,
    )
