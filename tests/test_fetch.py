from __future__ import annotations

from unittest.mock import Mock

import pandas as pd
import pytest
import requests

from korea_tourism import fetch


@pytest.mark.parametrize(
    "start,end", [("202013", "202101"), ("202101", "202001"), ("2020-01", "202101")]
)
def test_invalid_ranges(start, end):
    with pytest.raises(ValueError):
        fetch.validate_range(start, end)


def test_api_failures_do_not_expose_key(monkeypatch):
    def fail(*args, **kwargs):
        raise requests.HTTPError("https://ecos.bok.or.kr/secret-key/")

    monkeypatch.setattr(fetch.requests, "get", fail)
    with pytest.raises(RuntimeError) as error:
        fetch.fetch_currency("secret-key", "0000001", "202001", "202001")
    assert "secret-key" not in str(error.value)
    assert error.value.__suppress_context__


@pytest.mark.parametrize(
    "payload", [{"RESULT": {"CODE": "ERROR"}}, {"StatisticSearch": {"row": []}}]
)
def test_api_errors_and_empty_responses(monkeypatch, payload):
    response = Mock()
    response.json.return_value = payload
    monkeypatch.setattr(fetch.requests, "get", lambda *a, **kw: response)
    with pytest.raises((RuntimeError, ValueError)):
        fetch.fetch_currency("fake", "0000001", "202001", "202001")


def test_valid_response_and_partial_coverage(monkeypatch):
    response = Mock()
    response.json.return_value = {
        "StatisticSearch": {"row": [{"TIME": "202001", "DATA_VALUE": "1100"}]}
    }
    monkeypatch.setattr(fetch.requests, "get", lambda *a, **kw: response)
    result = fetch.fetch_currency("fake", "0000001", "202001", "202001")
    assert result["value"].tolist() == [1100]
    with pytest.raises(ValueError, match="requested range"):
        fetch.fetch_currency("fake", "0000001", "202001", "202002")


def test_fetch_stages_data_without_overwriting_raw_inputs(tmp_path, monkeypatch):
    monkeypatch.setenv("ECOS_API_KEY", "fake")
    monkeypatch.setattr(
        fetch,
        "fetch_currency",
        lambda *a: pd.DataFrame({"date": pd.to_datetime(["2020-01-01"]), "value": [1100]}),
    )
    output = fetch.fetch_exchange_rates(tmp_path, "202001", "202001")
    assert output == tmp_path / "data/external/krw_exchange_rates_monthly.csv"
    assert output.exists()
    assert not (tmp_path / "data/raw").exists()
