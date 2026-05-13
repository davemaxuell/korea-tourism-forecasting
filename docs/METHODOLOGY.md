# Methodology Notes

## Periodic Dataset

The clean public workflow uses monthly observations. All dates are normalized to the first day of each month before merging.

The generated feature table covers `2015-01-01` through `2025-09-01`, the intersection of visitor, exchange-rate, search-interest, and K-culture-index data.

## Forecasting Rules

Forecasting models should not use same-month observed predictors such as current-month search indices or current-month exchange rates unless the use case explicitly assumes those values are known before predicting visitor arrivals.

The baseline script therefore uses:

- Lagged exchange-rate features.
- Lagged search-interest features.
- Lagged K-culture-interest features.
- Lagged visitor counts.
- Known calendar features, including month seasonality and a historical COVID-period indicator.

The current baseline system evaluates a seasonal naive benchmark and three log-target regressors:

- `seasonal_naive`: predicts the same month from the previous year.
- `ridge_regression_log`: ridge regression trained on `log1p(visitors)`.
- `random_forest_log`: random forest trained on `log1p(visitors)`.
- `hist_gradient_boosting_log`: histogram gradient boosting trained on `log1p(visitors)`.

Metrics are reported for a chronological holdout split and expanding-window rolling backtests. WAPE is included because MAPE can become unstable during COVID-era months with unusually low visitor counts.

## Google Trends

Google Trends values are normalized indices, not absolute search counts. The project therefore calls these features search-interest indices rather than search volume.

Summed keyword indices are useful as a broad signal, but they should be interpreted cautiously because Trends values from separate pulls are not automatically on the same absolute scale.

## Exchange Rates

The public baseline uses KRW-denominated exchange rates where coverage is sufficient. USD/KRW and JPY/KRW are included. CNY/KRW is excluded from the clean baseline when coverage is too sparse.

## COVID Period

The full 2015-2025 period contains a major structural break from COVID travel restrictions. Models should report this limitation and, where relevant, compare full-period results with post-restriction results.

The baseline script reports both:

- `full_period`: all available monthly rows after lag creation.
- `post_covid`: rows from `2022-07-01` onward.
