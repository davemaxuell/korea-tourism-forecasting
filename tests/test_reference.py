"""Verify that the published evidence is internally consistent, without refitting."""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from korea_tourism.evaluation import aggregate_metrics, select_candidate
from korea_tourism.reporting import sha256

ROOT = Path(__file__).resolve().parents[1]


def test_published_metrics_and_selection_follow_predictions():
    predictions = pd.read_csv(
        ROOT / "reports/predictions.csv", parse_dates=["date", "train_start", "train_end"]
    )
    assert (predictions["train_end"] < predictions["date"]).all()
    actual = aggregate_metrics(predictions)
    published = pd.read_csv(ROOT / "reports/metrics.csv")
    # CSVs retain 10 significant digits. Million-scale forecasts can round by
    # fractions of an arrival; cancellation in signed bias needs an absolute tolerance.
    pd.testing.assert_frame_equal(actual, published, check_dtype=False, rtol=1e-8, atol=1e-3)
    selection = pd.read_csv(ROOT / "reports/selection.csv")
    for chosen in selection.itertuples():
        development = actual.loc[
            (actual["scenario"] == chosen.scenario) & (actual["split"] == "development")
        ]
        assert select_candidate(development) == (chosen.model, chosen.feature_set)
        rows = predictions.loc[predictions["scenario"] == chosen.scenario]
        assert (
            rows.loc[rows["split"] == "development", "date"].max()
            < rows.loc[rows["split"] == "holdout", "date"].min()
        )
        dates_by_candidate = rows.groupby(["split", "model", "feature_set"])["date"].apply(tuple)
        for split in ("development", "holdout"):
            assert dates_by_candidate.loc[split].nunique() == 1
    assert np.isfinite(predictions["prediction"]).all()


def test_published_manifest_matches_inputs_code_and_artifacts():
    manifest = json.loads((ROOT / "reports/run_manifest.json").read_text(encoding="utf-8"))
    for group, base in (
        ("inputs_sha256", ROOT),
        ("code_sha256", ROOT),
        ("artifacts_sha256", ROOT / "reports"),
    ):
        assert manifest[group], f"Empty provenance section: {group}"
        for relative, expected in manifest[group].items():
            assert sha256(base / relative) == expected, (
                f"Stale {relative}; regenerate the experiment"
            )


def test_local_documentation_links_exist():
    paths = [ROOT / "README.md", ROOT / "CONTRIBUTING.md", ROOT / "data/README.md"]
    paths += list((ROOT / "docs").glob("*.md")) + list((ROOT / "reports").glob("*.md"))
    for path in paths:
        for target in re.findall(r"\]\(([^)]+)\)", path.read_text(encoding="utf-8")):
            if "://" not in target and not target.startswith("#"):
                assert (path.parent / target.split("#")[0]).exists(), (
                    f"Broken link in {path.name}: {target}"
                )
