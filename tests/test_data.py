from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from korea_tourism.data import (
    add_lag_features,
    build_exchange_rates,
    build_foreign_visitors,
    normalize_month,
    parse_korean_month,
    read_google_trends_file,
)
from korea_tourism.validation import validate_monthly, validate_numeric

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.integration
def test_raw_rebuild_matches_committed_tables(rebuilt_root):
    for path in (ROOT / "data/processed").glob("*.csv"):
        expected = pd.read_csv(path)
        actual = pd.read_csv(rebuilt_root / "data/processed" / path.name)
        pd.testing.assert_frame_equal(actual, expected, rtol=1e-12, atol=1e-12)
    df = pd.read_csv(rebuilt_root / "data/processed/tourism_features_monthly.csv")
    assert len(df) == 129
    assert df["date"].iloc[[0, -1]].tolist() == ["2015-01-01", "2025-09-01"]
    assert "covid_period" not in df


@pytest.mark.parametrize("value,expected", [("2015년 1월", "2015-01-01"), ("합계", None)])
def test_korean_month(value, expected):
    assert parse_korean_month(value) == expected


def test_invalid_korean_month():
    with pytest.raises(ValueError, match="Invalid Korean month"):
        parse_korean_month("2015년 13월")


def test_normalize_month():
    assert normalize_month(pd.Series(["2020-01", "2020-02-29"])).tolist() == [
        pd.Timestamp("2020-01-01"),
        pd.Timestamp("2020-02-01"),
    ]
    with pytest.raises(ValueError):
        normalize_month(pd.Series(["bad-date"]))


@pytest.mark.parametrize("values", [["<1", "20"], ["0", "100"]])
def test_trends_censored_values(tmp_path, values):
    path = tmp_path / "trend.csv"
    path.write_text(f"Category: all\n\nMonth,value\n2020-01,{values[0]}\n2020-02,{values[1]}\n")
    df = read_google_trends_file(path)
    assert df["value"].tolist() == [0, int(values[1])]


@pytest.mark.parametrize("value", ["bad", "", "-1", "101"])
def test_trends_rejects_corrupt_values(tmp_path, value):
    path = tmp_path / "trend.csv"
    path.write_text(f"Category: all\n\nMonth,value\n2020-01,{value}\n")
    with pytest.raises(ValueError):
        read_google_trends_file(path)


@pytest.mark.parametrize(
    "dates",
    [
        ["2020-01-01", "2020-01-01"],
        ["2020-01-01", "2020-03-01"],
        ["2020-02-01", "2020-01-01"],
        ["2020-01-02", "2020-02-01"],
    ],
)
def test_invalid_monthly_grids(dates):
    with pytest.raises(ValueError):
        validate_monthly(pd.DataFrame({"date": pd.to_datetime(dates)}), "test")


@pytest.mark.parametrize("value", [np.nan, np.inf, -1])
def test_invalid_numeric_data(value):
    with pytest.raises(ValueError):
        validate_numeric(pd.DataFrame({"visitors": [value]}), ["visitors"], "test")


def test_every_lag_is_calendar_aligned(monthly_data):
    for column in monthly_data:
        if "_lag" in column:
            source, lag = column.rsplit("_lag", 1)
            pd.testing.assert_series_equal(
                monthly_data[column], monthly_data[source].shift(int(lag)), check_names=False
            )


@pytest.mark.parametrize("lag", [0, -1, 1.5])
def test_nonpositive_or_fractional_lags_rejected(monthly_data, lag):
    with pytest.raises(ValueError, match="positive integers"):
        add_lag_features(monthly_data, ["visitors"], (lag,))


def test_lags_reject_gaps_and_missing_sources(monthly_data):
    with pytest.raises(ValueError, match="missing months"):
        add_lag_features(monthly_data.drop(index=10), ["visitors"])
    with pytest.raises(ValueError, match="Missing lag source"):
        add_lag_features(monthly_data, ["unknown"])


def test_conflicting_visitor_months_are_not_silently_deduplicated(tmp_path, monkeypatch):
    raw = tmp_path / "data/raw/foreign_visitors"
    raw.mkdir(parents=True)
    (raw / "a.xls").touch()
    (raw / "b.xls").touch()
    monkeypatch.setattr(
        pd, "read_excel", lambda _: pd.DataFrame({"purpose": ["관광"], "2020년 1월": [100]})
    )
    with pytest.raises(ValueError, match="duplicate months"):
        build_foreign_visitors(tmp_path)


def test_exchange_rates_require_unambiguous_schema(tmp_path):
    raw = tmp_path / "data/raw/exchange_rates"
    raw.mkdir(parents=True)
    pd.DataFrame({"date": ["2020-01"], "usd_jpy": [100]}).to_csv(
        raw / "krw_exchange_rates_monthly.csv", index=False
    )
    with pytest.raises(ValueError, match="explicit USD/KRW"):
        build_exchange_rates(tmp_path)
