"""Train leakage-safe monthly visitor forecasting baselines.

The script evaluates models with:

- seasonal naive forecasts,
- log-target regressors,
- chronological holdout tests,
- expanding-window rolling backtests,
- full-period and post-COVID scenarios,
- permutation feature importance for the best holdout model.
"""

from __future__ import annotations

import argparse
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

try:
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from sklearn.compose import TransformedTargetRegressor
    from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
    from sklearn.inspection import permutation_importance
    from sklearn.linear_model import Ridge
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    try:
        from sklearn.metrics import root_mean_squared_error
    except ImportError:  # scikit-learn < 1.4
        root_mean_squared_error = None
except ImportError as exc:  # pragma: no cover - friendly CLI failure
    raise SystemExit(
        "Missing modeling dependency. Install dependencies with "
        "`pip install -r requirements.txt` and run this script again."
    ) from exc


TARGET_COL = "visitors"
KNOWN_FUTURE_COLUMNS = ["month_sin", "month_cos", "covid_period"]


@dataclass(frozen=True)
class Scenario:
    name: str
    start_date: str | None
    initial_train_size: int
    horizon: int
    step: int


SCENARIOS = [
    Scenario("full_period", None, initial_train_size=60, horizon=12, step=12),
    Scenario("post_covid", "2022-07-01", initial_train_size=24, horizon=6, step=6),
]


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def mape(y_true: pd.Series | np.ndarray, y_pred: pd.Series | np.ndarray) -> float:
    true = np.asarray(y_true, dtype="float64")
    pred = np.asarray(y_pred, dtype="float64")
    mask = np.isfinite(true) & np.isfinite(pred) & (true != 0)
    if not mask.any():
        return np.nan
    return float(np.mean(np.abs((true[mask] - pred[mask]) / true[mask])) * 100)


def wape(y_true: pd.Series | np.ndarray, y_pred: pd.Series | np.ndarray) -> float:
    true = np.asarray(y_true, dtype="float64")
    pred = np.asarray(y_pred, dtype="float64")
    mask = np.isfinite(true) & np.isfinite(pred)
    denominator = np.sum(np.abs(true[mask]))
    if denominator == 0:
        return np.nan
    return float(np.sum(np.abs(true[mask] - pred[mask])) / denominator * 100)


def rmse(y_true: pd.Series | np.ndarray, y_pred: pd.Series | np.ndarray) -> float:
    if root_mean_squared_error is not None:
        return float(root_mean_squared_error(y_true, y_pred))
    return float(mean_squared_error(y_true, y_pred) ** 0.5)


def evaluate(
    scenario: str,
    model: str,
    fold: str,
    train: pd.DataFrame,
    test: pd.DataFrame,
    y_pred: pd.Series | np.ndarray,
) -> dict[str, float | int | str]:
    y_true = test[TARGET_COL]
    return {
        "scenario": scenario,
        "model": model,
        "fold": fold,
        "train_start": train["date"].min().strftime("%Y-%m-%d"),
        "train_end": train["date"].max().strftime("%Y-%m-%d"),
        "test_start": test["date"].min().strftime("%Y-%m-%d"),
        "test_end": test["date"].max().strftime("%Y-%m-%d"),
        "n_train": len(train),
        "n_test": len(test),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": rmse(y_true, y_pred),
        "mape": mape(y_true, y_pred),
        "wape": wape(y_true, y_pred),
        "r2": float(r2_score(y_true, y_pred)) if len(test) > 1 else np.nan,
    }


def make_log_model(regressor) -> TransformedTargetRegressor:
    return TransformedTargetRegressor(
        regressor=regressor,
        func=np.log1p,
        inverse_func=np.expm1,
        check_inverse=False,
    )


def model_factories(random_state: int) -> dict[str, Callable[[], TransformedTargetRegressor]]:
    return {
        "ridge_regression_log": lambda: make_log_model(
            make_pipeline(StandardScaler(), Ridge(alpha=10.0))
        ),
        "random_forest_log": lambda: make_log_model(
            RandomForestRegressor(
                n_estimators=300,
                random_state=random_state,
                min_samples_leaf=2,
                n_jobs=-1,
            )
        ),
        "hist_gradient_boosting_log": lambda: make_log_model(
            HistGradientBoostingRegressor(
                learning_rate=0.04,
                max_iter=300,
                max_leaf_nodes=15,
                l2_regularization=0.01,
                random_state=random_state,
            )
        ),
    }


def select_forecast_features(df: pd.DataFrame) -> list[str]:
    lag_pattern = re.compile(r"_lag\d+$")
    lag_columns = [column for column in df.columns if lag_pattern.search(column)]
    feature_columns = lag_columns + [
        column for column in KNOWN_FUTURE_COLUMNS if column in df.columns
    ]
    feature_columns = [
        column for column in feature_columns if pd.api.types.is_numeric_dtype(df[column])
    ]
    if not feature_columns:
        raise ValueError("No leakage-safe forecast features found. Rebuild the dataset first.")
    return feature_columns


def load_model_data(root: Path, scenario: Scenario) -> tuple[pd.DataFrame, list[str]]:
    data_path = root / "data" / "processed" / "tourism_features_monthly_with_lags.csv"
    if not data_path.exists():
        raise FileNotFoundError("Run `python src/build_dataset.py` before training models.")

    df = pd.read_csv(data_path, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    feature_columns = select_forecast_features(df)
    if scenario.start_date:
        df = df[df["date"] >= pd.Timestamp(scenario.start_date)].copy()

    model_df = (
        df.dropna(subset=feature_columns + [TARGET_COL]).sort_values("date").reset_index(drop=True)
    )
    minimum_rows = scenario.initial_train_size + scenario.horizon
    if len(model_df) < minimum_rows:
        raise ValueError(
            f"Scenario {scenario.name} needs at least {minimum_rows} rows after lag creation, "
            f"but only {len(model_df)} are available."
        )
    return model_df, feature_columns


def predict_models(
    train: pd.DataFrame,
    test: pd.DataFrame,
    feature_columns: list[str],
    random_state: int,
) -> dict[str, np.ndarray]:
    predictions = {"seasonal_naive": test["visitors_lag12"].to_numpy()}
    x_train = train[feature_columns]
    y_train = train[TARGET_COL]
    x_test = test[feature_columns]

    for name, factory in model_factories(random_state).items():
        model = factory()
        model.fit(x_train, y_train)
        predictions[name] = np.maximum(model.predict(x_test), 0)
    return predictions


def holdout_split(model_df: pd.DataFrame, test_size: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    split_index = int(len(model_df) * (1 - test_size))
    train = model_df.iloc[:split_index].copy()
    test = model_df.iloc[split_index:].copy()
    if train.empty or test.empty:
        raise ValueError("Train/test split produced an empty partition.")
    return train, test


def evaluate_holdout(
    scenario: Scenario,
    model_df: pd.DataFrame,
    feature_columns: list[str],
    test_size: float,
    random_state: int,
) -> tuple[list[dict[str, float | int | str]], pd.DataFrame]:
    train, test = holdout_split(model_df, test_size)
    predictions = predict_models(train, test, feature_columns, random_state)

    metrics = []
    prediction_rows = test[["date", TARGET_COL]].copy()
    prediction_rows.insert(0, "scenario", scenario.name)
    for model_name, pred in predictions.items():
        metrics.append(evaluate(scenario.name, model_name, "holdout", train, test, pred))
        prediction_rows[f"{model_name}_prediction"] = pred
    return metrics, prediction_rows


def evaluate_rolling_backtest(
    scenario: Scenario,
    model_df: pd.DataFrame,
    feature_columns: list[str],
    random_state: int,
) -> tuple[list[dict[str, float | int | str]], pd.DataFrame]:
    metrics = []
    prediction_frames = []
    fold_number = 1

    train_end = scenario.initial_train_size
    while train_end < len(model_df):
        test_end = min(train_end + scenario.horizon, len(model_df))
        if test_end <= train_end:
            break

        train = model_df.iloc[:train_end].copy()
        test = model_df.iloc[train_end:test_end].copy()
        predictions = predict_models(train, test, feature_columns, random_state + fold_number)
        fold_name = f"fold_{fold_number:02d}"

        fold_predictions = test[["date", TARGET_COL]].copy()
        fold_predictions.insert(0, "fold", fold_name)
        fold_predictions.insert(0, "scenario", scenario.name)
        for model_name, pred in predictions.items():
            metrics.append(evaluate(scenario.name, model_name, fold_name, train, test, pred))
            fold_predictions[f"{model_name}_prediction"] = pred
        prediction_frames.append(fold_predictions)

        fold_number += 1
        train_end += scenario.step

    if not metrics:
        raise ValueError(f"No rolling folds were produced for scenario {scenario.name}.")

    return metrics, pd.concat(prediction_frames, ignore_index=True)


def summarize_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    return (
        metrics.groupby(["scenario", "model"], as_index=False)
        .agg(
            folds=("fold", "nunique"),
            mae=("mae", "mean"),
            rmse=("rmse", "mean"),
            mape=("mape", "mean"),
            wape=("wape", "mean"),
            r2=("r2", "mean"),
        )
        .sort_values(["scenario", "rmse"])
        .reset_index(drop=True)
    )


def fit_best_model_and_importance(
    scenario: Scenario,
    model_df: pd.DataFrame,
    feature_columns: list[str],
    holdout_metrics: list[dict[str, float | int | str]],
    test_size: float,
    random_state: int,
) -> pd.DataFrame:
    train, test = holdout_split(model_df, test_size)
    candidate_metrics = [
        row
        for row in holdout_metrics
        if row["scenario"] == scenario.name and row["model"] != "seasonal_naive"
    ]
    best_metric = min(candidate_metrics, key=lambda row: row["rmse"])
    best_model_name = str(best_metric["model"])

    model = model_factories(random_state)[best_model_name]()
    model.fit(train[feature_columns], train[TARGET_COL])
    result = permutation_importance(
        model,
        test[feature_columns],
        test[TARGET_COL],
        scoring="neg_mean_absolute_error",
        n_repeats=8,
        random_state=random_state,
        n_jobs=1,
    )

    importance = pd.DataFrame(
        {
            "scenario": scenario.name,
            "model": best_model_name,
            "feature": feature_columns,
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        }
    )
    return importance.sort_values("importance_mean", ascending=False).reset_index(drop=True)


def plot_holdout_predictions(
    root: Path,
    scenario_name: str,
    predictions: pd.DataFrame,
) -> None:
    figure_dir = root / "outputs" / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    scenario_predictions = predictions[predictions["scenario"] == scenario_name].copy()
    plt.figure(figsize=(12, 5))
    plt.plot(
        scenario_predictions["date"], scenario_predictions[TARGET_COL], label="Actual", linewidth=2
    )

    prediction_columns = [
        column for column in scenario_predictions.columns if column.endswith("_prediction")
    ]
    for column in prediction_columns:
        label = column.replace("_prediction", "")
        plt.plot(
            scenario_predictions["date"], scenario_predictions[column], label=label, alpha=0.85
        )

    plt.title(f"Monthly Foreign Visitors Holdout Forecast: {scenario_name}")
    plt.xlabel("Date")
    plt.ylabel("Visitors")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figure_dir / f"holdout_predictions_{scenario_name}.png", dpi=200)
    plt.close()


def train_baselines(root: Path, test_size: float, random_state: int) -> None:
    output_dir = root / "outputs" / "models"
    figure_dir = root / "outputs" / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    all_feature_rows = []
    all_holdout_metrics = []
    all_holdout_predictions = []
    all_rolling_metrics = []
    all_rolling_predictions = []
    all_importance = []

    for scenario in SCENARIOS:
        model_df, feature_columns = load_model_data(root, scenario)
        all_feature_rows.extend(
            {"scenario": scenario.name, "feature": feature} for feature in feature_columns
        )

        holdout_metrics, holdout_predictions = evaluate_holdout(
            scenario, model_df, feature_columns, test_size, random_state
        )
        rolling_metrics, rolling_predictions = evaluate_rolling_backtest(
            scenario, model_df, feature_columns, random_state
        )
        importance = fit_best_model_and_importance(
            scenario, model_df, feature_columns, holdout_metrics, test_size, random_state
        )

        all_holdout_metrics.extend(holdout_metrics)
        all_holdout_predictions.append(holdout_predictions)
        all_rolling_metrics.extend(rolling_metrics)
        all_rolling_predictions.append(rolling_predictions)
        all_importance.append(importance)

    feature_summary = pd.DataFrame(all_feature_rows)
    holdout_metrics_df = pd.DataFrame(all_holdout_metrics).sort_values(["scenario", "rmse"])
    holdout_predictions_df = pd.concat(all_holdout_predictions, ignore_index=True)
    rolling_metrics_df = pd.DataFrame(all_rolling_metrics).sort_values(["scenario", "fold", "rmse"])
    rolling_summary_df = summarize_metrics(rolling_metrics_df)
    rolling_predictions_df = pd.concat(all_rolling_predictions, ignore_index=True)
    importance_df = pd.concat(all_importance, ignore_index=True)

    feature_summary.to_csv(output_dir / "baseline_features.csv", index=False, encoding="utf-8-sig")
    holdout_metrics_df.to_csv(output_dir / "holdout_metrics.csv", index=False, encoding="utf-8-sig")
    holdout_predictions_df.to_csv(
        output_dir / "holdout_predictions.csv", index=False, encoding="utf-8-sig"
    )
    rolling_metrics_df.to_csv(
        output_dir / "rolling_backtest_metrics.csv", index=False, encoding="utf-8-sig"
    )
    rolling_summary_df.to_csv(
        output_dir / "rolling_backtest_summary.csv", index=False, encoding="utf-8-sig"
    )
    rolling_predictions_df.to_csv(
        output_dir / "rolling_backtest_predictions.csv", index=False, encoding="utf-8-sig"
    )
    importance_df.to_csv(
        output_dir / "permutation_importance.csv", index=False, encoding="utf-8-sig"
    )

    for scenario in SCENARIOS:
        plot_holdout_predictions(root, scenario.name, holdout_predictions_df)

    # Backward-compatible filenames for earlier README references.
    holdout_metrics_df.to_csv(
        output_dir / "baseline_metrics.csv", index=False, encoding="utf-8-sig"
    )
    holdout_predictions_df.to_csv(
        output_dir / "baseline_predictions.csv", index=False, encoding="utf-8-sig"
    )

    print("\nHoldout metrics")
    print(
        holdout_metrics_df[["scenario", "model", "mae", "rmse", "mape", "wape", "r2"]]
        .round(3)
        .to_string(index=False)
    )
    print("\nRolling backtest summary")
    print(
        rolling_summary_df[["scenario", "model", "folds", "mae", "rmse", "mape", "wape", "r2"]]
        .round(3)
        .to_string(index=False)
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=project_root(), help="Project root.")
    parser.add_argument(
        "--test-size", type=float, default=0.2, help="Chronological holdout split fraction."
    )
    parser.add_argument(
        "--random-state", type=int, default=42, help="Random seed for stochastic models."
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_baselines(args.root.resolve(), args.test_size, args.random_state)


if __name__ == "__main__":
    main()
