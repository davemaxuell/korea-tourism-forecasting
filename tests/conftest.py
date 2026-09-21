from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from korea_tourism.data import add_calendar_features, add_lag_features, build_feature_table
from korea_tourism.models import FEATURE_GROUPS

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def rebuilt_root(tmp_path_factory):
    root = tmp_path_factory.mktemp("rebuild")
    shutil.copytree(ROOT / "data/raw", root / "data/raw")
    build_feature_table(root)
    return root


@pytest.fixture
def monthly_data():
    dates = pd.date_range("2015-01-01", periods=72, freq="MS")
    df = pd.DataFrame({"date": dates})
    for index, source in enumerate(FEATURE_GROUPS["all_signals"]):
        df[source] = 100 + index * 10 + np.arange(len(dates)) * 2.0
    return add_lag_features(add_calendar_features(df), FEATURE_GROUPS["all_signals"], (1, 2, 3, 12))
