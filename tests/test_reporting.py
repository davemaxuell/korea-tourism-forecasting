from __future__ import annotations

import json

import numpy as np
import pandas as pd

from korea_tourism import evaluation
from korea_tourism.evaluation import Scenario, run_evaluation, score_predictions
from korea_tourism.reporting import sha256


def test_end_to_end_artifacts_are_auditable(rebuilt_root, monkeypatch):
    class MeanModel:
        def fit(self, x, y):
            self.mean = float(y.mean())
            return self

        def predict(self, x):
            return np.repeat(self.mean, len(x))

    monkeypatch.setattr(evaluation, "make_model", lambda *_: MeanModel())
    monkeypatch.setattr(
        evaluation,
        "SCENARIOS",
        (
            Scenario("full_period", None, 114, 2),
            Scenario("recovery_period", "2022-07-01", 36, 2),
        ),
    )
    metrics = run_evaluation(rebuilt_root)
    output = rebuilt_root / "reports"
    predictions = pd.read_csv(output / "predictions.csv")
    assert len(predictions) == 6 * 14
    assert len(metrics) == 2 * 2 * 14
    for (scenario, split, model, group), rows in predictions.groupby(
        ["scenario", "split", "model", "feature_set"]
    ):
        expected = score_predictions(rows["actual"].to_numpy(), rows["prediction"].to_numpy())
        metric = metrics.loc[
            (metrics["scenario"] == scenario)
            & (metrics["split"] == split)
            & (metrics["model"] == model)
            & (metrics["feature_set"] == group)
        ].iloc[0]
        assert np.isclose(expected["wape_pct"], metric.wape_pct)
    manifest = json.loads((output / "run_manifest.json").read_text(encoding="utf-8"))
    for name, digest in manifest["artifacts_sha256"].items():
        assert sha256(output / name) == digest
    for name, digest in manifest["inputs_sha256"].items():
        assert sha256(rebuilt_root / name) == digest
    report = (output / "model_results.md").read_text(encoding="utf-8")
    assert "Candidates selected before holdout" in report
    assert "Feature ablation" in report
