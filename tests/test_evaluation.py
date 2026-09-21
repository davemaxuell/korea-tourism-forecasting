from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from korea_tourism import evaluation
from korea_tourism.evaluation import (
    SCENARIOS,
    Scenario,
    aggregate_metrics,
    evaluate_window,
    load_model_data,
    monthly_splits,
    score_predictions,
    select_candidate,
)
from korea_tourism.models import FEATURE_GROUPS, MODEL_NAMES, feature_columns, make_model


def test_metrics_pool_errors_and_handle_zero_totals():
    result = score_predictions(np.array([0, 100]), np.array([10, 80]))
    assert result["mae"] == 15
    assert result["wape_pct"] == 30
    assert result["rmse"] == pytest.approx(np.sqrt(250))
    assert result["bias"] == -5
    assert np.isnan(score_predictions(np.array([0]), np.array([1]))["wape_pct"])


@pytest.mark.parametrize(
    "actual,predicted", [([], []), ([1], [1, 2]), ([1], [np.nan]), ([1], [-1]), ([np.inf], [1])]
)
def test_metrics_reject_invalid_inputs(actual, predicted):
    with pytest.raises(ValueError):
        score_predictions(np.array(actual), np.array(predicted))


def test_forecast_features_are_explicit_and_lagged():
    for group in FEATURE_GROUPS:
        columns = feature_columns(group)
        assert "visitors" not in columns
        assert "covid_period" not in columns
        assert "unknown_lag1" not in columns
        assert all("_lag" in c or c in ("month_sin", "month_cos") for c in columns)
    assert len(feature_columns("history")) == 6
    assert len(feature_columns("all_signals")) == 26


@pytest.mark.integration
@pytest.mark.parametrize("scenario", SCENARIOS)
def test_development_and_holdout_are_disjoint(rebuilt_root, scenario):
    df = load_model_data(rebuilt_root, scenario)
    split = len(df) - scenario.holdout_months
    dev = list(monthly_splits(df, scenario.initial_train_size, split))
    held = list(monthly_splits(df, split, len(df)))
    assert dev[-1][1]["date"].max() < held[0][1]["date"].min()
    assert len(held) == scenario.holdout_months
    for train, test in dev + held:
        assert train["date"].max() < test["date"].min()
        assert len(test) == 1


@pytest.mark.parametrize("start,stop", [(0, 10), (20, 10), (2, 100)])
def test_split_boundaries(monthly_data, start, stop):
    with pytest.raises(ValueError):
        list(monthly_splits(monthly_data, start, stop))


def test_candidate_selection_rejects_holdout():
    with pytest.raises(ValueError, match="development"):
        select_candidate(pd.DataFrame({"split": ["holdout"]}))


def test_candidate_selection_uses_wape_and_deterministic_ties():
    frame = pd.DataFrame(
        {
            "scenario": ["test"] * 3,
            "split": ["development"] * 3,
            "model": ["c", "b", "a"],
            "feature_set": ["history"] * 3,
            "wape_pct": [20, 10, 10],
            "rmse": [1, 5, 5],
        }
    )
    assert select_candidate(frame) == ("a", "history")


def test_future_changes_cannot_affect_earlier_predictions(monthly_data, monkeypatch):
    # A fitted deterministic estimator exposes any accidental access to future labels.
    class MeanModel:
        def fit(self, x, y):
            self.value = float(y.mean()) + float(x.iloc[:, 0].mean())
            return self

        def predict(self, x):
            return np.repeat(self.value, len(x))

    monkeypatch.setattr(evaluation, "make_model", lambda *_: MeanModel())
    frame = monthly_data.iloc[12:].reset_index(drop=True)
    original = evaluate_window(frame, "test", "development", 20, 23, 42)
    changed = frame.copy()
    changed.loc[23:, changed.columns != "date"] = 999999
    repeated = evaluate_window(changed, "test", "development", 20, 23, 42)
    pd.testing.assert_frame_equal(original, repeated)
    # The current target is scoring-only, never a feature or training label.
    changed = frame.copy()
    changed.loc[20, "visitors"] = 999999
    current = evaluate_window(changed, "test", "development", 20, 21, 42)
    np.testing.assert_allclose(original.iloc[:14]["prediction"], current["prediction"])


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_real_models_fit_and_produce_finite_predictions(monthly_data, name):
    from threadpoolctl import threadpool_limits

    df = monthly_data.iloc[12:]
    features = feature_columns("history")
    with threadpool_limits(limits=1):
        model = make_model(name)
        model.fit(df.iloc[:40][features], df.iloc[:40]["visitors"])
        result = model.predict(df.iloc[40:][features])
    assert np.isfinite(result).all()
    assert (result >= 0).all()


def test_corrupted_lags_fail_before_training(tmp_path, monthly_data):
    output = tmp_path / "data/processed"
    output.mkdir(parents=True)
    monthly_data.loc[20, "visitors_lag1"] += 1
    monthly_data.to_csv(output / "tourism_features_monthly_with_lags.csv", index=False)
    with pytest.raises(ValueError, match="Misaligned lag"):
        load_model_data(tmp_path, Scenario("test", None, 24, 8))


def test_duplicate_predictions_cannot_bias_metrics():
    row = {
        "scenario": "test",
        "split": "development",
        "model": "seasonal_naive",
        "feature_set": "benchmark",
        "date": pd.Timestamp("2020-01-01"),
        "actual": 10,
        "prediction": 12,
    }
    with pytest.raises(ValueError, match="Duplicate forecast"):
        aggregate_metrics(pd.DataFrame([row, row]))
