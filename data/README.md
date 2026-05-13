# Data Dictionary

## Raw Data

`data/raw/foreign_visitors/`

- Source Excel files for monthly foreign arrivals by purpose.
- The reproducible pipeline filters the tourism-purpose row only.

`data/raw/k_culture/`

- Individual Google Trends exports for K-pop, K-drama, K-food, and K-culture keywords.
- Web Search and YouTube Search exports are kept in separate folders.

`data/raw/search_trends/`

- Keyword-level monthly Google/YouTube Trends tables.
- These are summed into monthly search-interest index features. Google Trends values are normalized indices, not absolute search counts.

`data/raw/exchange_rates/`

- Monthly exchange-rate data used by the modeling feature table.

## Processed Data

`foreign_visitors_monthly.csv`

- `date`: Month start date.
- `visitors`: Foreign arrivals for tourism purpose.

`exchange_rates_monthly.csv`

- `date`: Month start date.
- `usd_krw`: Korean won per US dollar.
- `jpy_krw_100`: Korean won per 100 Japanese yen, when available.
- `cny_krw`: Korean won per Chinese yuan, only included when coverage is sufficient.

`search_trends_monthly.csv`

- `date`: Month start date.
- `google_search_index`: Sum of keyword-level Google Trends indices.
- `youtube_search_index`: Sum of keyword-level YouTube Trends indices.

`k_culture_interest_monthly.csv`

- `kpop_mean`, `kdrama_mean`, `kfood_mean`, `kculture_mean`: Group-level mean Trends indices.
- `k_index`: Average of available group means.

`tourism_features_monthly.csv`

- Model-ready monthly table combining exchange rates, search-interest indices, K-culture interest, visitors, and known calendar indicators.

`tourism_features_monthly_with_lags.csv`

- Same as `tourism_features_monthly.csv`, plus 1-, 2-, 3-, and 12-month lag features for key predictors. Forecast models should use lagged observed predictors only.
