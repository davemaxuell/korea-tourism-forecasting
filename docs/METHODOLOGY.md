# Methodology

## Task and information boundary

Predict tourism-purpose foreign arrivals for month **t**, using observed series through **t − 1** and the known calendar month. Dates are monthly labels, not release timestamps. This is a retrospective one-month-ahead protocol: earlier held-out observations become training data only after their month has passed. It is not a 24-month forecast made at a single origin.

The source files do not record release dates or historical vintages. We assume previous-month values are available. Lagging prevents same-month feature use, but cannot remove look-ahead introduced by retrospective revisions, full-window Trends normalization, or hindsight-driven keyword selection. A production claim would require vintage snapshots and a release-aware feature store.

## Dataset construction

1. Read exactly one `관광` (tourism) row per visitor workbook; parse Korean month headings and integer arrival counts.
2. Read USD/KRW and KRW per 100 JPY. The legacy `jpy_krw` header is renamed under the original project's unit assumption. Do not substitute USD/JPY cross-rates.
3. Sum keyword-level indices separately for Web and YouTube, requiring a constant keyword basket and unique month/group/keyword records.
4. Average individual Trends exports within each of four K-culture groups, then take the equally weighted mean of the four groups. Exports within each group must cover identical months. `<1` is mapped to zero, preserving the original convention; malformed or missing numeric values fail.
5. Left-join all sources onto the visitor months with one-to-one joins. Missing predictor coverage fails instead of silently shortening the target period.
6. Validate contiguous month starts and generate 1-, 2-, 3-, and 12-month lags. Only the first 12 warm-up months are removed from model evaluation. Missing values inside the evaluation span fail.

There is no interpolation, backward filling, target imputation, or learned preprocessing across the entire dataset. The original raw files remain unchanged.

## Feature ablation

| Feature set | Observed series, each lagged by 1, 2, 3, 12 months | Total features including calendar |
| --- | --- | ---: |
| `history` | Visitor arrivals | 6 |
| `history_fx` | Arrivals, USD/KRW, KRW per 100 JPY | 14 |
| `history_interest` | Arrivals, Google keyword index, YouTube keyword index, K-culture composite | 18 |
| `all_signals` | All six series above | 26 |

All sets include sine/cosine month seasonality. No retrospective pandemic flag is treated as known future information. Calendar features are recomputed against dates during model-data validation. Lag columns must match source-series shifts; arbitrary numeric columns cannot enter via naming conventions.

## Models and selection

Two untrained benchmarks predict the previous month's count and the same month last year. Learned candidates are:

- Ridge: standardized inputs, alpha 10.
- Random forest: 150 trees, minimum leaf size 2.
- Histogram gradient boosting: 150 iterations, learning rate 0.04, at most 15 leaf nodes, minimum leaf size 20, L2 0.01, early stopping disabled. With fewer than 40 training rows in the recovery scenario, this fixed configuration cannot split and produces a constant prediction per origin. The report retains this limitation instead of retuning after inspecting holdout results.

Each learned candidate uses `log1p(visitors)` and `expm1` inversion, with predictions clipped at zero. The inverse transform is not corrected for retransformation bias; forecasts are not guaranteed to estimate the conditional mean. Hyperparameters are fixed, without a tuning search. Seed 42 and one numerical thread are used for the reference run.

Each scenario reserves its final months before evaluation. Expanding-window development forecasts select a **model plus feature set**, including the simple benchmarks, using pooled WAPE. Ties use RMSE then lexicographic model/feature names. Selection is frozen before the holdout is evaluated. The candidate is refitted every month using all earlier scenario rows.

| Scenario | Available model rows | Initial training rows | Development origins | Holdout origins |
| --- | ---: | ---: | ---: | ---: |
| Full period | 117 | 60 | 33 | 24 |
| Recovery, Jul 2022 onward | 39 | 24 | 7 | 8 |

Recovery features can use pre-July-2022 lagged observations; only training targets are restricted by the scenario boundary. The July 2022 boundary is an analysis choice, not a claim that all restrictions ended on that date. The scenarios have different holdouts and should not be compared as a controlled causal experiment.

All candidates' holdout scores are disclosed for transparency, but their ranking does not replace the development selection. Development is small, particularly in the recovery scenario; repeated inspection of this public holdout can influence subsequent research decisions. Future model changes need a new evaluation period or nested time-series validation.

## Metrics

For actual arrivals `y` and predictions `p`:

- `MAE = mean(abs(y − p))`, in arrivals.
- `RMSE = sqrt(mean((y − p)^2))`, in arrivals.
- `WAPE = 100 × sum(abs(y − p)) / sum(abs(y))`, percent.
- `bias = mean(p − y)`, positive for overprediction.
- `skill = 100 × (1 − MAE_model / MAE_seasonal)`, percent improvement over seasonal naive.

Metrics pool all evaluated months, rather than averaging fold percentages or fold RMSE. WAPE is undefined when total actual arrivals are zero; seasonal skill is undefined when the seasonal benchmark has zero error. Undefined values are recorded as empty CSV values, not favorable scores. Invalid, negative, or nonfinite predictions fail. MAPE is omitted because near-zero pandemic arrivals make per-month relative errors misleading. WAPE still gives higher-volume months greater influence.

## Limits and next experiments

The dataset has 129 months, correlated features, a pandemic break, and only seven recovery development origins. No uncertainty intervals or significance tests are claimed. Global interest is not market-specific intent to travel; pooled arrival counts hide origin-country differences. Keywords such as contemporary entertainment titles create possible hindsight selection bias. Summed separately normalized Trends pulls are an exploratory proxy, not a calibrated measurement scale.

Useful extensions would collect archived release vintages, test longer publication delays, predefine keyword baskets, evaluate source-country arrivals, and validate forecast intervals on more origins. These should precede adding large model families or presenting the analysis as a production service.

## References

- [scikit-learn: preventing data leakage](https://scikit-learn.org/stable/common_pitfalls.html#data-leakage) supports fitting transformations on training data only.
- [scikit-learn: time-series cross-validation](https://scikit-learn.org/stable/modules/cross_validation.html#time-series-split) explains chronological evaluation.
- [Google Trends data FAQ](https://support.google.com/trends/answer/4365533) describes sampled, normalized search-interest data.
