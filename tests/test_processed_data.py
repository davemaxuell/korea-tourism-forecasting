from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.train_baseline import KNOWN_FUTURE_COLUMNS, select_forecast_features

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"


def read_processed(name: str) -> pd.DataFrame:
    path = PROCESSED / name
    assert path.exists(), f"Missing processed dataset: {path}"
    return pd.read_csv(path, parse_dates=["date"])


def test_feature_table_has_contiguous_months() -> None:
    df = read_processed("tourism_features_monthly.csv")
    assert not df.empty
    assert df["date"].is_unique
    expected = pd.date_range(df["date"].min(), df["date"].max(), freq="MS")
    assert df["date"].tolist() == list(expected)


def test_forecast_features_do_not_use_same_month_observed_signals() -> None:
    df = read_processed("tourism_features_monthly_with_lags.csv")
    features = select_forecast_features(df)

    forbidden = {
        "usd_krw",
        "jpy_krw_100",
        "google_search_index",
        "youtube_search_index",
        "k_index",
        "visitors",
    }
    assert forbidden.isdisjoint(features)
    assert all(
        feature.endswith(("_lag1", "_lag2", "_lag3", "_lag12")) or feature in KNOWN_FUTURE_COLUMNS
        for feature in features
    )


def test_lag_columns_are_aligned() -> None:
    df = read_processed("tourism_features_monthly_with_lags.csv")
    pd.testing.assert_series_equal(
        df["visitors_lag1"].iloc[1:].reset_index(drop=True),
        df["visitors"].shift(1).iloc[1:].reset_index(drop=True),
        check_names=False,
        check_dtype=False,
    )
    pd.testing.assert_series_equal(
        df["visitors_lag12"].iloc[12:].reset_index(drop=True),
        df["visitors"].shift(12).iloc[12:].reset_index(drop=True),
        check_names=False,
        check_dtype=False,
    )
    pd.testing.assert_series_equal(
        df["usd_krw_lag3"].iloc[3:].reset_index(drop=True),
        df["usd_krw"].shift(3).iloc[3:].reset_index(drop=True),
        check_names=False,
        check_dtype=False,
    )


def test_exchange_rate_schema_uses_krw_rates() -> None:
    df = read_processed("exchange_rates_monthly.csv")
    assert {"date", "usd_krw", "jpy_krw_100"}.issubset(df.columns)
    assert "usd_jpy" not in df.columns
    assert "usd_cny" not in df.columns
    assert df[["usd_krw", "jpy_krw_100"]].notna().all().all()


def test_model_ready_dataset_has_no_missing_current_values() -> None:
    df = read_processed("tourism_features_monthly.csv")
    current_columns = [column for column in df.columns if column != "date"]
    assert df[current_columns].notna().all().all()
