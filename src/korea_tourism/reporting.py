"""Generate reviewable tables, figures, and a hash-based experiment manifest."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import FuncFormatter

from korea_tourism.models import FEATURE_GROUPS, MODEL_NAMES, feature_columns, make_model

COLORS = {"actual": "#152b3c", "selected": "#007f86", "seasonal": "#b86d36", "last": "#89949c"}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def experiment_manifest(root: Path, seed: int, scenarios: list[dict]) -> dict:
    input_paths = sorted((root / "data/raw").rglob("*"))
    input_paths += sorted((root / "data/processed").glob("*.csv"))
    package_path = Path(__file__).resolve().parent
    code_hashes = {
        f"src/korea_tourism/{p.name}": sha256(p) for p in sorted(package_path.glob("*.py"))
    }
    for name in ("pyproject.toml", "uv.lock"):
        path = package_path.parents[1] / name
        if path.is_file():
            code_hashes[name] = sha256(path)
    return {
        "schema_version": 1,
        "protocol": "expanding_window_one_month_ahead",
        "availability_assumption": "all previous-month observations available; no vintage data",
        "selection": "minimum pooled development WAPE; holdout excluded from selection",
        "seed": seed,
        "scenarios": scenarios,
        "features": {group: feature_columns(group) for group in FEATURE_GROUPS},
        "model_parameters": {
            name: {key: repr(value) for key, value in make_model(name, seed).get_params().items()}
            for name in MODEL_NAMES
        },
        "python": platform.python_version(),
        "platform": platform.system(),
        "packages": {
            name: importlib.metadata.version(name)
            for name in (
                "korea-tourism-forecasting",
                "numpy",
                "pandas",
                "scikit-learn",
                "matplotlib",
                "xlrd",
                "threadpoolctl",
            )
        },
        "inputs_sha256": {
            p.relative_to(root).as_posix(): sha256(p) for p in input_paths if p.is_file()
        },
        "code_sha256": code_hashes,
    }


def markdown_table(metrics: pd.DataFrame) -> str:
    lines = [
        "| Model | Features | Months | WAPE | MAE | RMSE | Skill vs seasonal |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in metrics.sort_values(["wape_pct", "model", "feature_set"]).itertuples():
        lines.append(
            f"| `{row.model}` | `{row.feature_set}` | {row.n_months} | {row.wape_pct:.2f}% "
            f"| {row.mae:,.0f} | {row.rmse:,.0f} | {row.skill_vs_seasonal_pct:+.2f}% |"
        )
    return "\n".join(lines)


def plot_results(root: Path, predictions: pd.DataFrame, selection: pd.DataFrame) -> None:
    output = root / "reports/figures"
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "#d2d9dd",
            "axes.labelcolor": "#384b58",
            "xtick.color": "#536470",
            "ytick.color": "#536470",
            "figure.facecolor": "#ffffff",
            "axes.titleweight": "bold",
            "savefig.facecolor": "#ffffff",
        }
    )
    fig, axes = plt.subplots(2, 1, figsize=(12, 8.5), constrained_layout=True)
    for ax, chosen in zip(axes, selection.itertuples(), strict=True):
        subset = predictions.loc[
            (predictions["scenario"] == chosen.scenario) & (predictions["split"] == "holdout")
        ]
        actual = subset.drop_duplicates("date").sort_values("date")
        selected = subset.loc[
            (subset["model"] == chosen.model) & (subset["feature_set"] == chosen.feature_set)
        ]
        ax.plot(
            actual["date"],
            actual["actual"],
            color=COLORS["actual"],
            lw=2.5,
            label="Observed arrivals",
        )
        ax.plot(
            selected["date"],
            selected["prediction"],
            color=COLORS["selected"],
            lw=2,
            marker="o",
            ms=3.5,
            label=f"Development-selected: {chosen.model} / {chosen.feature_set}",
        )
        if chosen.model != "seasonal_naive":
            naive = subset.loc[subset["model"] == "seasonal_naive"]
            ax.plot(
                naive["date"],
                naive["prediction"],
                color=COLORS["seasonal"],
                lw=1.5,
                ls="--",
                label="Seasonal naive",
            )
        ax.set_title(chosen.scenario.replace("_", " ").title(), loc="left", pad=14)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value / 1e6:.1f}m"))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=4, maxticks=8))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        ax.set_ylabel("Tourism-purpose arrivals")
        ax.grid(axis="y", color="#e7ecef", lw=0.7)
        ax.set_axisbelow(True)
        ax.legend(loc="upper left", fontsize=8, frameon=False)
    fig.suptitle(
        "Korea tourism · one month ahead", x=0.01, ha="left", fontsize=20, fontweight="bold"
    )
    fig.savefig(output / "holdout_forecasts.png", dpi=180)
    plt.close(fig)

    data = pd.read_csv(root / "data/processed/foreign_visitors_monthly.csv", parse_dates=["date"])
    fig, ax = plt.subplots(figsize=(12, 3.3), constrained_layout=True)
    ax.plot(data["date"], data["visitors"], color=COLORS["actual"], lw=1.8)
    ax.fill_between(data["date"], data["visitors"], color=COLORS["selected"], alpha=0.08)
    ax.axvspan(
        pd.Timestamp("2020-03-01"),
        pd.Timestamp("2022-06-01"),
        color="#b86d36",
        alpha=0.12,
        label="Pandemic disruption (descriptive window)",
    )
    ax.set_title("A small dataset with a major structural break", loc="left", pad=12)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value / 1e6:.1f}m"))
    ax.set_ylabel("Tourism-purpose arrivals")
    ax.grid(axis="y", color="#e7ecef", lw=0.7)
    ax.legend(frameon=False, fontsize=9, loc="lower left")
    fig.savefig(output / "arrivals_history.png", dpi=180)
    plt.close(fig)


def write_artifacts(
    root: Path,
    predictions: pd.DataFrame,
    metrics: pd.DataFrame,
    selection: pd.DataFrame,
    seed: int,
    scenarios: list[dict],
) -> None:
    output = root / "reports"
    output.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(
        output / "predictions.csv",
        index=False,
        date_format="%Y-%m-%d",
        float_format="%.10g",
        lineterminator="\n",
    )
    metrics.to_csv(output / "metrics.csv", index=False, float_format="%.10g", lineterminator="\n")
    selection.to_csv(output / "selection.csv", index=False, lineterminator="\n")
    plot_results(root, predictions, selection)
    lines = [
        "# Evaluation results",
        "",
        "<!-- Generated by korea-tourism reproduce; do not hand-edit. -->",
        "",
        f"Seed: `{seed}`. Monthly expanding-window refits; pooled errors in arrival counts.",
        "",
        "Selection uses development WAPE only. Every candidate's holdout result is shown for transparency; "
        "the lowest holdout error is not a new selection decision. Positive skill means lower MAE than seasonal naive.",
        "",
        "These are retrospective one-step experiments. Previous-month releases are assumed available; "
        "historical data vintages and actual publication delays are not modeled. This is not a live or multi-step forecast.",
        "",
        "![Holdout forecasts](figures/holdout_forecasts.png)",
        "",
        "## Candidates selected before holdout",
        "",
        "| Scenario | Candidate | Development WAPE | Holdout WAPE | Holdout seasonal WAPE |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for chosen in selection.itertuples():
        candidate = metrics.loc[
            (metrics["scenario"] == chosen.scenario)
            & (metrics["model"] == chosen.model)
            & (metrics["feature_set"] == chosen.feature_set)
        ]
        dev = candidate.loc[candidate["split"] == "development"].iloc[0]
        test = candidate.loc[candidate["split"] == "holdout"].iloc[0]
        naive = metrics.loc[
            (metrics["scenario"] == chosen.scenario)
            & (metrics["split"] == "holdout")
            & (metrics["model"] == "seasonal_naive")
        ].iloc[0]
        lines.append(
            f"| {chosen.scenario} | `{chosen.model}` / `{chosen.feature_set}` "
            f"| {dev.wape_pct:.2f}% | {test.wape_pct:.2f}% | {naive.wape_pct:.2f}% |"
        )
    lines += [
        "",
        "## Feature ablation",
        "",
        "Same estimator, dates, and refit schedule. Positive WAPE change means added signals hurt; "
        "negative means they helped on this holdout. These comparisons are descriptive, not significance tests.",
        "",
        "| Scenario | Model | History WAPE | + FX change (pp) | + Interest change (pp) | + Both change (pp) |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    held = metrics.loc[metrics["split"] == "holdout"]
    for (scenario, model), group in held.loc[held["feature_set"] != "benchmark"].groupby(
        ["scenario", "model"]
    ):
        values = group.set_index("feature_set")["wape_pct"]
        base = values["history"]
        lines.append(
            f"| {scenario} | `{model}` | {base:.2f}% | {values['history_fx'] - base:+.2f} "
            f"| {values['history_interest'] - base:+.2f} | {values['all_signals'] - base:+.2f} |"
        )
    for (scenario, split), group in metrics.groupby(["scenario", "split"], sort=False):
        first = group.iloc[0]
        lines += [
            "",
            f"## {scenario} · {split}",
            "",
            f"{first['start']} through {first['end']} ({first['n_months']} monthly forecasts).",
            "",
            markdown_table(group),
        ]
    lines += [
        "",
        "## Interpretation boundaries",
        "",
        "- Only 129 raw months, including an extreme pandemic break; recovery development has seven forecasts.",
        "- Search indices are retrospective, normalized, correlated proxies. Export vintages and keyword selection can bias backtests.",
        "- Log-target inversion is not a bias-corrected estimate of the conditional mean; uncertainty intervals are not estimated.",
        "- Comparing scenarios does not isolate a causal recovery effect: they have different training and test dates.",
        "- Histogram boosting retains a minimum leaf size of 20. With fewer than 40 recovery training rows, "
        "it cannot split and predicts a constant within each origin; identical ablation scores follow from this constraint.",
        "- No claim of causal K-culture impact, production readiness, or guaranteed future accuracy.",
        "",
        "Audit files: [monthly predictions](predictions.csv), [metrics](metrics.csv), "
        "[selection](selection.csv), [run manifest](run_manifest.json).",
        "",
    ]
    (output / "model_results.md").write_text("\n".join(lines), encoding="utf-8", newline="\n")
    manifest = experiment_manifest(root, seed, scenarios)
    artifacts = [
        "predictions.csv",
        "metrics.csv",
        "selection.csv",
        "model_results.md",
        "figures/holdout_forecasts.png",
        "figures/arrivals_history.png",
    ]
    manifest["artifacts_sha256"] = {name: sha256(output / name) for name in artifacts}
    (output / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n"
    )
    print(f"Wrote evaluation tables, figures, and manifest to {output}", flush=True)
