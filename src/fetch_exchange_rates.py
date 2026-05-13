"""Fetch monthly exchange rates from the Bank of Korea ECOS API.

Set ECOS_API_KEY in your environment before running this script.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

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
    return Path(__file__).resolve().parents[1]


def fetch_currency(api_key: str, item_code: str, start: str, end: str) -> pd.DataFrame:
    url = (
        "https://ecos.bok.or.kr/api/StatisticSearch/"
        f"{api_key}/json/kr/1/10000/{STAT_CODE}/{CYCLE}/{start}/{end}/{item_code}"
    )
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    data = response.json()

    if "StatisticSearch" not in data:
        raise RuntimeError(f"ECOS API error: {data.get('RESULT', data)}")

    rows = data["StatisticSearch"].get("row", [])
    if not rows:
        return pd.DataFrame(columns=["date", "value"])

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["TIME"], format="%Y%m").dt.to_period("M").dt.to_timestamp()
    df["value"] = pd.to_numeric(df["DATA_VALUE"], errors="coerce")
    return df[["date", "value"]]


def fetch_exchange_rates(root: Path, start: str, end: str) -> Path:
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

    output_path = root / "data" / "raw" / "exchange_rates" / "krw_exchange_rates_monthly.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False, encoding="utf-8-sig")
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default="201501", help="Start month in YYYYMM format.")
    parser.add_argument("--end", default="202512", help="End month in YYYYMM format.")
    parser.add_argument("--root", type=Path, default=project_root(), help="Project root.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = fetch_exchange_rates(args.root.resolve(), args.start, args.end)
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
