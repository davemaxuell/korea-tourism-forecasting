"""Monthly expanding-window evaluation with selection before the final holdout."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from korea_tourism.models import (
    BASELINES,
    FEATURE_GROUPS,
    MODEL_NAMES,
    feature_columns,
    make_model,
)
from korea_tourism.validation import validate_monthly, validate_numeric


@dataclass(frozen=True)
class Scenario:
    name: str
    start_date: str | None
    initial_train_size: int
    holdout_months: int

    def __post_init__(self) -> None:
        if self.initial_train_size < 2 or self.holdout_months < 1:
            raise ValueError("Scenarios need at least two training rows and one holdout month")


SCENARIOS = (
    Scenario("full_period", None, initial_train_size=60, holdout_months=24),
    Scenario("recovery_period", "2022-07-01", initial_train_size=24, holdout_months=8),
)


def score_predictions(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    actual, predicted = np.asarray(actual, dtype=float), np.asarray(predicted, dtype=float)
    if actual.ndim != 1 or actual.shape != predicted.shape or len(actual) == 0:
        raise ValueError("Metrics require matching nonempty one-dimensional arrays")
    if not np.isfinite(actual).all() or not np.isfinite(predicted).all():
        raise ValueError("Metrics do not accept missing or infinite predictions")
    if (actual < 0).any() or (predicted < 0).any():
        raise ValueError("Visitor counts and predictions must be nonnegative")
    error = actual - predicted
    denominator = np.abs(actual).sum()
    return {
        "mae": float(np.abs(error).mean()),
        "rmse": float(np.sqrt(np.square(error).mean())),
        "wape_pct": float(100 * np.abs(error).sum() / denominator) if denominator else np.nan,
        "bias": float((predicted - actual).mean()),
    }


def load_model_data(root: Path, scenario: Scenario) -> pd.DataFrame:
    path = root / "data/processed/tourism_features_monthly_with_lags.csv"
    df = pd.read_csv(path, parse_dates=["date"])
    validate_monthly(df, str(path))
    required = feature_columns("all_signals")
    missing = set(required + ["visitors"]) - set(df)
    if missing:
        raise ValueError(f"Missing model columns: {sorted(missing)}. Rebuild the dataset.")
    # Verify lags against their source series, including the expected warm-up nulls.
    for column in required:
        if "_lag" in column:
            source, lag = column.rsplit("_lag", 1)
            if source not in df or not np.allclose(
                df[column],
                df[source].shift(int(lag)),
                equal_nan=True,
            ):
                raise ValueError(f"Misaligned lag: {column}")
    # Only the deterministic 12-month warm-up is dropped; interior missing values fail.
    df = df.iloc[12:].copy()
    if scenario.start_date:
        df = df.loc[df["date"] >= pd.Timestamp(scenario.start_date)].copy()
    validate_monthly(df, scenario.name)
    validate_numeric(df, ["visitors"] + [c for c in required if "_lag" in c], scenario.name)
    if not np.isfinite(df[required].to_numpy()).all():
        raise ValueError("Nonfinite calendar features")
    month = df["date"].dt.month
    if not np.allclose(df["month_sin"], np.sin(2 * np.pi * month / 12)) or not np.allclose(
        df["month_cos"], np.cos(2 * np.pi * month / 12)
    ):
        raise ValueError("Invalid calendar features")
    if len(df) <= scenario.initial_train_size + scenario.holdout_months:
        raise ValueError(f"{scenario.name}: insufficient rows for development and holdout")
    return df.reset_index(drop=True)


def monthly_splits(df: pd.DataFrame, start: int, stop: int):
    """Yield one forecast month at a time; training labels end strictly before it."""
    validate_monthly(df, "evaluation input")
    if not 2 <= start < stop <= len(df):
        raise ValueError("Invalid expanding-window boundaries")
    for position in range(start, stop):
        yield df.iloc[:position], df.iloc[position : position + 1]


def evaluate_window(
    df: pd.DataFrame,
    scenario: str,
    split: str,
    start: int,
    stop: int,
    seed: int,
) -> pd.DataFrame:
    rows = []
    with threadpool_limits(limits=1):
        for train, test in monthly_splits(df, start, stop):
            common = {
                "scenario": scenario,
                "split": split,
                "date": test["date"].iloc[0],
                "train_start": train["date"].iloc[0],
                "train_end": train["date"].iloc[-1],
                "n_train": len(train),
                "actual": float(test["visitors"].iloc[0]),
            }
            for name, column in BASELINES.items():
                rows.append(
                    {
                        **common,
                        "model": name,
                        "feature_set": "benchmark",
                        "prediction": float(test[column].iloc[0]),
                    }
                )
            for group in FEATURE_GROUPS:
                columns = feature_columns(group)
                for name in MODEL_NAMES:
                    model = make_model(name, seed)
                    model.fit(train[columns], train["visitors"])
                    prediction = float(model.predict(test[columns])[0])
                    if not np.isfinite(prediction):
                        raise ValueError(f"{name}/{group}: nonfinite forecast")
                    rows.append(
                        {
                            **common,
                            "model": name,
                            "feature_set": group,
                            "prediction": max(0.0, prediction),
                        }
                    )
    return pd.DataFrame(rows)


def aggregate_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    """Pool monthly errors, rather than averaging incompatible fold percentages."""
    keys = ["scenario", "split", "model", "feature_set"]
    if predictions.duplicated(keys + ["date"]).any():
        raise ValueError("Duplicate forecast months would double-count errors")
    rows = []
    for identity, group in predictions.groupby(keys, sort=True):
        rows.append(
            {
                **dict(zip(keys, identity, strict=True)),
                "start": group["date"].min().strftime("%Y-%m-%d"),
                "end": group["date"].max().strftime("%Y-%m-%d"),
                "n_months": len(group),
                **score_predictions(group["actual"].to_numpy(), group["prediction"].to_numpy()),
            }
        )
    result = pd.DataFrame(rows)
    benchmarks = result.loc[result["model"] == "seasonal_naive", ["scenario", "split", "mae"]]
    benchmarks = benchmarks.rename(columns={"mae": "seasonal_naive_mae"})
    result = result.merge(benchmarks, on=["scenario", "split"], validate="many_to_one")
    result["skill_vs_seasonal_pct"] = 100 * (
        1 - result["mae"] / result["seasonal_naive_mae"].replace(0, np.nan)
    )
    return result.sort_values(keys).reset_index(drop=True)


def select_candidate(development_metrics: pd.DataFrame) -> tuple[str, str]:
    if set(development_metrics["split"]) != {"development"}:
        raise ValueError("Candidate selection may only use development results")
    if development_metrics["scenario"].nunique() != 1:
        raise ValueError("Select candidates separately for each scenario")
    valid = development_metrics.dropna(subset=["wape_pct"])
    if valid.empty:
        raise ValueError("Cannot select a candidate with undefined WAPE")
    winner = valid.sort_values(["wape_pct", "rmse", "model", "feature_set"]).iloc[0]
    return str(winner["model"]), str(winner["feature_set"])


def run_evaluation(root: Path, seed: int = 42) -> pd.DataFrame:
    from korea_tourism.reporting import write_artifacts

    predictions = []
    selections = []
    for scenario in SCENARIOS:
        df = load_model_data(root, scenario)
        holdout_start = len(df) - scenario.holdout_months
        print(
            f"{scenario.name}: development {holdout_start - scenario.initial_train_size} months; "
            f"holdout {scenario.holdout_months} months",
            flush=True,
        )
        development = evaluate_window(
            df,
            scenario.name,
            "development",
            scenario.initial_train_size,
            holdout_start,
            seed,
        )
        model, group = select_candidate(aggregate_metrics(development))
        selections.append({"scenario": scenario.name, "model": model, "feature_set": group})
        print(f"  Selected before holdout: {model} / {group}", flush=True)
        holdout = evaluate_window(df, scenario.name, "holdout", holdout_start, len(df), seed)
        predictions.extend([development, holdout])
    all_predictions = pd.concat(predictions, ignore_index=True)
    metrics = aggregate_metrics(all_predictions)
    selection = pd.DataFrame(selections)
    write_artifacts(root, all_predictions, metrics, selection, seed, [asdict(s) for s in SCENARIOS])
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run_evaluation(args.root.resolve(), args.seed)
