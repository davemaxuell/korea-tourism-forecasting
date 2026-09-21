# Data dictionary

All processed dates are ISO month starts (`YYYY-MM-01`). Raw files are preserved. Run `korea-tourism build` to regenerate the six processed CSVs; never edit processed tables by hand.

| Table | Grain | Rows in reference snapshot |
| --- | --- | ---: |
| `foreign_visitors_monthly.csv` | One tourism-purpose arrival count per month | 129 |
| `exchange_rates_monthly.csv` | One pair of KRW-denominated rates per month | 130 |
| `search_trends_monthly.csv` | Two summed keyword-interest indices per month | 130 |
| `k_culture_interest_monthly.csv` | Four group means and their composite per month | 129 |
| `tourism_features_monthly.csv` | All current signals and calendar terms on visitor months | 129 |
| `tourism_features_monthly_with_lags.csv` | Current signals plus explicit calendar-month lags | 129 |

| Column | Meaning / unit |
| --- | --- |
| `date` | Observation month; **not** a release timestamp |
| `visitors` | Nonnegative integer foreign arrivals for tourism purpose, not all foreign arrivals |
| `usd_krw` | KRW per US dollar |
| `jpy_krw_100` | KRW per 100 Japanese yen; legacy raw `jpy_krw` unit assumption retained |
| `google_search_index` | Sum of fixed-basket Web keyword indices; not search volume |
| `youtube_search_index` | Sum of fixed-basket YouTube keyword indices |
| `kpop_mean`, `kdrama_mean`, `kfood_mean`, `kculture_mean` | Arithmetic mean of the group's individual Web/YouTube export indices |
| `k_index` | Equally weighted mean of the four group means |
| `month_sin`, `month_cos` | `sin(2π × month / 12)` and `cos(2π × month / 12)` |
| `<series>_lagN` | Value exactly N calendar months earlier, N ∈ {1, 2, 3, 12} |

The six lagged series are arrivals, USD/KRW, KRW per 100 JPY, Google index, YouTube index, and the K-culture composite. Nulls are expected only in the initial lag warm-up. Current observed signals are retained for auditing and lag verification but never used directly as forecast features.

`raw/foreign_visitors/` holds the workbooks; `raw/k_culture/` separates Web and YouTube exports; `raw/search_trends/` holds long keyword tables; `raw/exchange_rates/` retains the original exchange tables. `external/` is an ignored staging area for optional downloads.

See [source provenance](../docs/DATA_SOURCES.md) for attribution caveats and [methodology](../docs/METHODOLOGY.md) for aggregation details.
