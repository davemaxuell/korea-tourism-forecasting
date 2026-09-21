"""Command-line entry point for the reproducible research workflow."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("build", "evaluate", "reproduce", "fetch"):
        command = commands.add_parser(name)
        command.add_argument("--root", type=Path, default=Path.cwd())
        if name in ("evaluate", "reproduce"):
            command.add_argument("--seed", type=int, default=42)
        if name == "fetch":
            command.add_argument("--start", default="201501")
            command.add_argument("--end", default="202509")
    args = parser.parse_args()
    root = args.root.resolve()
    if args.command in ("build", "reproduce"):
        from korea_tourism.data import build_feature_table

        build_feature_table(root)
    if args.command in ("evaluate", "reproduce"):
        from korea_tourism.evaluation import run_evaluation

        run_evaluation(root, args.seed)
    if args.command == "fetch":
        from korea_tourism.fetch import fetch_exchange_rates

        print(fetch_exchange_rates(root, args.start, args.end))


if __name__ == "__main__":
    main()
