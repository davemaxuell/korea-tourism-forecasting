"""Fetch monthly exchange rates from the Bank of Korea ECOS API.

Set ECOS_API_KEY in your environment before running this script.
"""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

from korea_tourism.validation import validate_monthly, validate_numeric

try:
    import pandas as pd
    import requests
except ImportError as exc:  # pragma: no cover - friendly CLI failure
    raise SystemExit(
        "Missing dependency. Install the project dependencies with "
        "`pip install -r requirements.txt` and run this script again."
    ) from exc


STAT_CODE = "731Y004"
CYCLE = "M"
CURRENCY_CODES = {
    "usd_krw": "0000001",
    "jpy_krw_100": "0000002",
    "cny_krw": "0000027",
}


def project_root() -> Path:
    return Path.cwd()


def validate_range(start: str, end: str) -> None:
    if not all(re.fullmatch(r"\d{4}(0[1-9]|1[0-2])", value) for value in (start, end)):
        raise ValueError("Dates must be valid YYYYMM months")
    if start > end:
        raise ValueError("Start month must not be after end month")


def fetch_currency(api_key: str, item_code: str, start: str, end: str) -> pd.DataFrame:
    validate_range(start, end)
    url = (
        "https://ecos.bok.or.kr/api/StatisticSearch/"
        f"{api_key}/json/kr/1/10000/{STAT_CODE}/{CYCLE}/{start}/{end}/{item_code}"
    )
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        data = response.json()
    except (requests.RequestException, ValueError):
        # ECOS embeds the secret in the URL; never surface exception URLs or API payloads.
        raise RuntimeError("ECOS request failed; check connectivity and API credentials") from None

    if not isinstance(data, dict) or not isinstance(data.get("StatisticSearch"), dict):
        raise RuntimeError("ECOS returned an API error; check credentials and requested series")

    rows = data["StatisticSearch"].get("row", [])
    if not rows:
        raise ValueError("ECOS returned no rows for the requested currency")

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["TIME"], format="%Y%m").dt.to_period("M").dt.to_timestamp()
    df["value"] = pd.to_numeric(df["DATA_VALUE"], errors="raise")
    result = df[["date", "value"]].sort_values("date").reset_index(drop=True)
    validate_monthly(result, "ECOS response")
    validate_numeric(result, ["value"], "ECOS response")
    expected = pd.date_range(
        pd.to_datetime(start, format="%Y%m"), pd.to_datetime(end, format="%Y%m"), freq="MS"
    )
    if result["date"].tolist() != list(expected) or (result["value"] <= 0).any():
        raise ValueError("ECOS response must cover the requested range with positive rates")
    return result


def fetch_exchange_rates(root: Path, start: str, end: str) -> Path:
    validate_range(start, end)
    api_key = os.environ.get("ECOS_API_KEY")
    if not api_key:
        raise SystemExit("Set ECOS_API_KEY before fetching exchange rates.")

    frames = []
    for name, item_code in CURRENCY_CODES.items():
        frame = fetch_currency(api_key, item_code, start, end).rename(columns={"value": name})
        frames.append(frame)

    result = frames[0]
    for frame in frames[1:]:
        result = result.merge(frame, on="date", how="outer")

    result = result.sort_values("date").reset_index(drop=True)
    result["date"] = result["date"].dt.strftime("%Y-%m-%d")

    # Stage fetched data separately so reproducing this research never overwrites its inputs.
    output_path = root / "data" / "external" / "krw_exchange_rates_monthly.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False, encoding="utf-8-sig", lineterminator="\n")
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default="201501", help="Start month in YYYYMM format.")
    parser.add_argument("--end", default="202509", help="End month in YYYYMM format.")
    parser.add_argument("--root", type=Path, default=project_root(), help="Project root.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = fetch_exchange_rates(args.root.resolve(), args.start, args.end)
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
