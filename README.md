# K-Culture, Exchange Rates, and Foreign Tourism in Korea

This project analyzes how global K-culture interest and exchange-rate movements relate to monthly foreign tourist arrivals in Korea from 2015 to 2025.

The repository now has a clean public workflow:

```text
data/raw/          Source files used by the reproducible pipeline
data/processed/    Clean monthly datasets generated from raw inputs
src/               Reproducible data and baseline-model scripts
outputs/           Generated figures and model outputs
docs/              Notes for publishing and project maintenance
```

## Research Questions

- Does global interest in K-pop, K-drama, K-food, and K-culture move with foreign tourist arrivals?
- Do exchange-rate changes have immediate or lagged relationships with tourism demand?
- Which signals are useful for simple monthly visitor forecasting?

## Data Sources

- Foreign visitor arrivals: Korea tourism immigration statistics, monthly tourism-purpose arrivals.
- K-culture interest: Google Trends exports for Web Search and YouTube Search.
- Keyword search interest: monthly Google/YouTube Trends keyword-index tables. These are normalized indices, not absolute search counts.
- Exchange rates: monthly KRW-denominated exchange rates. The clean baseline currently uses USD/KRW and JPY/KRW because CNY/KRW coverage is too sparse in the available ECOS export.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Optional notebook/deep-learning dependencies are listed separately:

```bash
pip install -r requirements-optional.txt
```

## Rebuild Data

```bash
python src/build_dataset.py
```

This creates:

- `data/processed/foreign_visitors_monthly.csv`
- `data/processed/exchange_rates_monthly.csv`
- `data/processed/search_trends_monthly.csv`
- `data/processed/k_culture_interest_monthly.csv`
- `data/processed/tourism_features_monthly.csv`
- `data/processed/tourism_features_monthly_with_lags.csv`

## Train Baseline Models

```bash
python src/train_baseline.py
```

The baseline uses only lagged observed predictors plus known calendar indicators, so it is a leakage-safe forecasting baseline rather than an after-the-fact explanatory regression.

The training script evaluates:

- Seasonal naive forecasts using the same month last year.
- Log-target ridge regression.
- Log-target random forest.
- Log-target histogram gradient boosting.
- Chronological holdout tests.
- Expanding-window rolling backtests.
- Full-period and post-COVID scenarios.
- Permutation feature importance for the best holdout model.

Main outputs:

- `outputs/models/holdout_metrics.csv`
- `outputs/models/rolling_backtest_summary.csv`
- `outputs/models/rolling_backtest_metrics.csv`
- `outputs/models/permutation_importance.csv`
- `outputs/figures/holdout_predictions_full_period.png`
- `outputs/figures/holdout_predictions_post_covid.png`

Generated outputs are ignored by Git by default.

## Development Checks

Install development tools:

```bash
pip install -r requirements-dev.txt
```

Run the public-repo checks:

```bash
ruff check src tests
ruff format --check src tests
python src/build_dataset.py
pytest
```

You can also enable local pre-commit checks:

```bash
pre-commit install
```

## Optional ECOS Exchange-Rate Fetch

Create a local `.env` or set the environment variable manually:

```bash
export ECOS_API_KEY=your_key_here
python src/fetch_exchange_rates.py --start 201501 --end 202512
```

Do not commit `.env` or API keys. The repository includes `.env.example` only.

## Notes

The original exploratory notebooks and Korean-named working folders are preserved locally, but `.gitignore` excludes them from the public repository. The public GitHub version should focus on `src/`, `data/raw/`, `data/processed/`, `docs/`, and the root documentation files.

Additional project documentation:

- `docs/METHODOLOGY.md` explains the modeling and leakage-control choices.
- `docs/DATA_SOURCES.md` documents source families and reuse notes.
- `docs/PUBLICATION_CHECKLIST.md` lists final checks before publishing.
- `reports/model_results.md` keeps a compact, committed summary of baseline results.
