# Contributing

Keep the analysis small, reproducible, and honest about its evidence. Prefer changes that strengthen a clear research question over adding unused dependencies or model families.

## Setup and checks

```bash
uv sync --frozen --extra dev
uv run --frozen ruff check src tests
uv run --frozen ruff format --check src tests
uv run --frozen pytest --cov=korea_tourism --cov-report=term-missing
```

Without uv, activate a Python 3.11–3.13 virtual environment and run `python -m pip install -e ".[dev]"`. The checked-in lock and Python 3.12 are the reference environment. `uv run --frozen pre-commit install` enables local lint/format hooks.

Tests rebuild the bundled raw data in a temporary workspace; they do not need a network connection or API key. The suite checks data contracts, all lags, selection boundaries, future-data isolation, metric edge cases, real estimator fitting, API error handling, and generated artifacts.

## Changing the experiment

1. Explain the hypothesis and information available at forecast time.
2. Add a regression test for changes to parsing, temporal alignment, metrics, or evaluation.
3. Keep feature lists explicit and fit learned preprocessing within each training window.
4. Run `uv run --frozen korea-tourism reproduce` to update processed data, report tables, figures, and manifest together.
5. Review every data/report diff and update methodology when the protocol changes.

Do not tune on the published holdout and then describe it as untouched. Report disappointing results and strong naive benchmarks. New model choices informed by public holdout scores need additional evaluation data.

Dependencies live in `pyproject.toml`. After intentional dependency changes, run `uv lock`, sync, test, and regenerate the reference artifacts. Avoid hand-editing `uv.lock` or generated reports. Patch-level numerical differences across platforms are possible; compare numeric results with tolerances rather than claiming universal byte-identical model outputs.

## Data and secrets

Preserve source files and document units, collection settings, and reuse terms. Reject malformed inputs rather than silently filling them. Optional refreshed downloads belong in `data/external/` until reviewed. Never include real API keys, `.env`, local environments, or trained binary model files in a pull request.

See [review checklist](docs/REVIEW_CHECKLIST.md) before publishing experiment changes.
