# Model Results

Generated with:

```bash
python src/build_dataset.py
python src/train_baseline.py
```

## Holdout Results

| Scenario | Best model by WAPE | WAPE | RMSE | Notes |
| --- | --- | ---: | ---: | --- |
| Full period | `random_forest_log` | 14.69% | 226,768 | Best chronological holdout model for 2023-10 through 2025-09. |
| Post-COVID | `seasonal_naive` | 14.16% | 203,732 | Same-month-last-year benchmark remains hard to beat on the short recovery-period holdout. |

## Rolling Backtest Results

| Scenario | Best model by WAPE | WAPE | RMSE | Notes |
| --- | --- | ---: | ---: | --- |
| Full period | `hist_gradient_boosting_log` | 37.15% | 153,249 | Best expanding-window average RMSE/WAPE among model-based methods. |
| Post-COVID | `random_forest_log` | 15.51% | 216,678 | Best rolling recovery-period model. |

## Interpretation

The strongest full-period holdout model is the log-target random forest, while rolling backtests favor histogram gradient boosting on the full period. In post-COVID holdout, the seasonal naive benchmark is competitive, which means any production-style forecast should keep seasonal naive as a required benchmark.

Permutation importance for the full-period random forest identifies `visitors_lag1` as the dominant predictor, followed by calendar seasonality and `visitors_lag12`. This suggests recent tourism momentum and recurring seasonal patterns explain more short-term variation than exchange-rate or search-interest signals in the current baseline.
