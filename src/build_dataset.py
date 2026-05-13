"""Build the public monthly tourism analysis datasets.

This script turns the raw visitor, exchange-rate, and search-trend files in
``data/raw`` into clean CSV files in ``data/processed``.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

try:
    import numpy as np
    import pandas as pd
except ImportError as exc:  # pragma: no cover - friendly CLI failure
    raise SystemExit(
        "Missing dependency: pandas. Install the project dependencies with "
        "`pip install -r requirements.txt` and run this script again."
    ) from exc


GROUP_MAP = {
    "blackpink": "kpop",
    "bts": "kpop",
    "demon_hunters": "kpop",
    "k_pop": "kpop",
    "korean_pop": "kpop",
    "kpop": "kpop",
    "k_food": "kfood",
    "korean_food": "kfood",
    "kfood": "kfood",
    "k_culture": "kculture",
    "korean_culture": "kculture",
    "kculture": "kculture",
    "k_drama": "kdrama",
    "korean_drama": "kdrama",
    "kdrama": "kdrama",
}


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def ensure_dirs(root: Path) -> None:
    for folder in [
        root / "data" / "processed",
        root / "outputs" / "figures",
        root / "outputs" / "models",
    ]:
        folder.mkdir(parents=True, exist_ok=True)


def normalize_month(values: pd.Series) -> pd.Series:
    """Return timestamps normalized to the first day of each month."""
    dates = pd.to_datetime(values.astype(str).str.strip(), errors="coerce")
    return dates.dt.to_period("M").dt.to_timestamp()


def normalize_filename(path: Path) -> str:
    return re.sub(r"[^a-z0-9]+", "_", path.stem.lower()).strip("_")


def extract_group(path: Path) -> str | None:
    name = normalize_filename(path)
    for key, group in sorted(GROUP_MAP.items(), key=lambda item: len(item[0]), reverse=True):
        if key in name:
            return group
    return None


def write_monthly_csv(df: pd.DataFrame, path: Path) -> None:
    out = df.copy()
    if "date" in out.columns:
        out["date"] = pd.to_datetime(out["date"]).dt.strftime("%Y-%m-%d")
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False, encoding="utf-8-sig")
    try:
        display_path = path.relative_to(project_root())
    except ValueError:
        display_path = path
    print(f"Wrote {display_path} ({len(out):,} rows)")


def read_google_trends_file(path: Path) -> pd.DataFrame:
    """Read a two-column Google Trends CSV exported with metadata rows."""
    df = pd.read_csv(path, skiprows=2)
    if df.shape[1] < 2:
        raise ValueError(f"Expected at least two columns in {path}")

    df = df.iloc[:, :2].copy()
    df.columns = ["date", "value"]
    df["date"] = normalize_month(df["date"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce").fillna(0)
    df.loc[df["value"] < 1, "value"] = 0
    return df.dropna(subset=["date"])


def build_k_culture_index(root: Path) -> pd.DataFrame:
    trend_dirs = [
        root / "data" / "raw" / "k_culture" / "web_search",
        root / "data" / "raw" / "k_culture" / "youtube_search",
    ]

    grouped: dict[str, list[pd.DataFrame]] = {}
    for trend_dir in trend_dirs:
        for path in sorted(trend_dir.glob("*.csv")):
            group = extract_group(path)
            if group is None:
                print(f"Skipping unrecognized K-culture file: {path}")
                continue
            grouped.setdefault(group, []).append(read_google_trends_file(path))

    if not grouped:
        raise FileNotFoundError("No K-culture Google Trends CSV files were found.")

    monthly_frames = []
    for group, frames in sorted(grouped.items()):
        merged = pd.concat(frames, ignore_index=True)
        monthly = merged.groupby("date", as_index=False)["value"].mean()
        monthly = monthly.rename(columns={"value": f"{group}_mean"})
        monthly_frames.append(monthly)

    result = monthly_frames[0]
    for frame in monthly_frames[1:]:
        result = result.merge(frame, on="date", how="outer")

    mean_cols = [column for column in result.columns if column.endswith("_mean")]
    result["k_index"] = result[mean_cols].mean(axis=1)
    result = result.sort_values("date").reset_index(drop=True)

    write_monthly_csv(result, root / "data" / "processed" / "k_culture_interest_monthly.csv")
    return result


def parse_korean_month(value: object) -> str | None:
    cleaned = re.sub(r"\s+", "", str(value))
    match = re.search(r"(\d{4})년(\d{1,2})월", cleaned)
    if not match:
        return None
    return f"{match.group(1)}-{int(match.group(2)):02d}-01"


def build_foreign_visitors(root: Path) -> pd.DataFrame:
    raw_dir = root / "data" / "raw" / "foreign_visitors"
    files = sorted(raw_dir.glob("*.xls"))
    if not files:
        raise FileNotFoundError(f"No visitor Excel files found in {raw_dir}")

    frames = []
    for path in files:
        df = pd.read_excel(path)
        purpose_column = df.columns[0]
        tourism = df[df[purpose_column].astype(str).str.strip() == "관광"].copy()
        tourism = tourism.drop(columns=[purpose_column])
        long = tourism.melt(var_name="raw_month", value_name="visitors")
        long["date"] = pd.to_datetime(long["raw_month"].map(parse_korean_month), errors="coerce")
        long["visitors"] = (
            long["visitors"].astype(str).str.replace(",", "", regex=False).str.strip()
        )
        long["visitors"] = pd.to_numeric(long["visitors"], errors="coerce")
        frames.append(long[["date", "visitors"]])

    result = pd.concat(frames, ignore_index=True)
    result = result.dropna(subset=["date", "visitors"])
    result["visitors"] = result["visitors"].astype(int)
    result = result.drop_duplicates(subset=["date"]).sort_values("date").reset_index(drop=True)

    write_monthly_csv(result, root / "data" / "processed" / "foreign_visitors_monthly.csv")
    return result


def build_exchange_rates(root: Path) -> pd.DataFrame:
    raw_dir = root / "data" / "raw" / "exchange_rates"
    krw_path = raw_dir / "krw_exchange_rates_monthly.csv"
    fallback_path = raw_dir / "monthly_exchange_rates.csv"
    path = krw_path if krw_path.exists() else fallback_path
    if not path.exists():
        raise FileNotFoundError(f"Missing exchange-rate file: {path}")

    df = pd.read_csv(path)
    date_column = "date" if "date" in df.columns else "year_month"
    df["date"] = normalize_month(df[date_column])

    rename_map = {"jpy_krw": "jpy_krw_100"}
    df = df.rename(columns=rename_map)

    candidate_columns = ["usd_krw", "jpy_krw_100", "cny_krw", "usd_jpy", "usd_cny"]
    keep_columns = ["date"]
    for column in candidate_columns:
        if column not in df.columns:
            continue
        coverage = pd.to_numeric(df[column], errors="coerce").notna().mean()
        if column == "cny_krw" and coverage < 0.8:
            print(f"Skipping {column}: only {coverage:.1%} coverage in raw data.")
            continue
        keep_columns.append(column)

    result = df[keep_columns].copy()
    for column in keep_columns[1:]:
        result[column] = pd.to_numeric(result[column], errors="coerce")

    result = result.dropna(subset=["date"])
    result = result.drop_duplicates(subset=["date"]).sort_values("date").reset_index(drop=True)

    write_monthly_csv(result, root / "data" / "processed" / "exchange_rates_monthly.csv")
    return result


def build_keyword_search_totals(root: Path) -> pd.DataFrame:
    raw_dir = root / "data" / "raw" / "search_trends"
    sources = {
        "google_search_index": raw_dir / "google_k_trends_keywords_monthly.csv",
        "youtube_search_index": raw_dir / "youtube_k_trends_keywords_monthly.csv",
    }

    frames = []
    for output_column, path in sources.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing keyword search file: {path}")
        df = pd.read_csv(path)
        df["date"] = normalize_month(df["Month"])
        df["search_index"] = pd.to_numeric(df["SearchIndex"], errors="coerce").fillna(0)
        monthly = df.groupby("date", as_index=False)["search_index"].sum()
        frames.append(monthly.rename(columns={"search_index": output_column}))

    result = frames[0]
    for frame in frames[1:]:
        result = result.merge(frame, on="date", how="outer")

    result = result.sort_values("date").reset_index(drop=True)
    write_monthly_csv(result, root / "data" / "processed" / "search_trends_monthly.csv")
    return result


def add_lag_features(
    df: pd.DataFrame, columns: list[str], lags: tuple[int, ...] = (1, 2, 3)
) -> pd.DataFrame:
    result = df.sort_values("date").copy()
    for column in columns:
        if column not in result.columns:
            continue
        for lag in lags:
            result[f"{column}_lag{lag}"] = result[column].shift(lag)
    return result


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    month = result["date"].dt.month
    result["month_sin"] = np.sin(2 * np.pi * month / 12)
    result["month_cos"] = np.cos(2 * np.pi * month / 12)
    result["covid_period"] = (
        (result["date"] >= pd.Timestamp("2020-03-01"))
        & (result["date"] <= pd.Timestamp("2022-06-01"))
    ).astype(int)
    return result


def build_feature_table(root: Path) -> pd.DataFrame:
    visitors = build_foreign_visitors(root)
    exchange_rates = build_exchange_rates(root)
    search_totals = build_keyword_search_totals(root)
    k_culture = build_k_culture_index(root)

    result = exchange_rates.merge(search_totals, on="date", how="inner")
    result = result.merge(k_culture, on="date", how="left")
    result = result.merge(visitors, on="date", how="inner")
    result = result.sort_values("date").reset_index(drop=True)
    result = add_calendar_features(result)

    write_monthly_csv(result, root / "data" / "processed" / "tourism_features_monthly.csv")

    lag_columns = [
        column
        for column in [
            "usd_krw",
            "jpy_krw_100",
            "cny_krw",
            "usd_jpy",
            "usd_cny",
            "google_search_index",
            "youtube_search_index",
            "k_index",
            "visitors",
        ]
        if column in result.columns
    ]
    lagged = add_lag_features(result, lag_columns, lags=(1, 2, 3, 12))
    write_monthly_csv(
        lagged, root / "data" / "processed" / "tourism_features_monthly_with_lags.csv"
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=project_root(),
        help="Project root. Defaults to the parent of the src directory.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    ensure_dirs(root)
    build_feature_table(root)


if __name__ == "__main__":
    main()
