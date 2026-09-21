# Data sources and provenance

The bundled files reproduce the analysis without credentials or network calls. Attribution below comes from the original project and file contents; not every file's collection history can be independently established. Hashes identify the exact files used, but do not certify provider authenticity or licensing.

| Input | Coverage in bundled files | Attribution | Provenance limits |
| --- | --- | --- | --- |
| Four tourism-purpose `.xls` workbooks | Jan 2015–Sep 2025 | Korea Tourism Organization / [Tourism DataLab](https://datalab.visitkorea.or.kr/) | Exact download URLs, retrieval dates, and revision vintages are not recorded. |
| 22 individual Trends CSV exports | Jan 2015–Sep 2025 | [Google Trends](https://trends.google.com/trends/), Web and YouTube; export headings indicate worldwide scope | Retrieval timestamps and original query URLs are absent. Historical values can reflect retrospective normalization. |
| Google and YouTube keyword tables | Jan 2015–Oct 2025 | Google Trends according to original documentation | Original extraction code, retrieval dates, and scaling across separate pulls are not supplied. |
| `krw_exchange_rates_monthly.csv` | Jan 2015–Oct 2025 | Bank of Korea [ECOS](https://ecos.bok.or.kr/api/#/) according to original documentation | Rounded values and precomputed lag columns are present. Original API responses and retrieval timestamp are absent; unit mapping is inherited. |
| `monthly_exchange_rates.csv` | Legacy alternate table | Exact provider not established in repository | Preserved for provenance; excluded from the pipeline because it includes cross-rates with different units. |

The target period is anchored to the visitor data. October 2025 values from longer predictor files are outside the modeling period. CNY/KRW is sparse and excluded by a fixed schema, not by a threshold learned from future coverage. Supplied raw exchange-rate lag columns are ignored; all lags are regenerated after monthly alignment.

## Google Trends interpretation

Values are normalized indices from 0 to 100, not absolute search counts. The [provider FAQ](https://support.google.com/trends/answer/4365533) explains sampling and normalization. The pipeline sums keyword indices and averages grouped exports as exploratory proxies. It does not claim that separate searches or Web/YouTube scales are directly comparable in absolute demand.

Export `<1` values are mapped to zero, following the original project. This slightly understates low interest. Other malformed values raise errors. The K-pop group includes artist and entertainment-title searches; hindsight in choosing those terms is a limitation.

## Refreshing exchange rates

The optional downloader requests ECOS statistical table `731Y004`, monthly cycle, and the currency item codes retained from the original implementation. The API integration is unit-tested with simulated responses. Live retrieval is optional and requires a user's API key; the bundled analysis does not depend on live service availability.

Set `ECOS_API_KEY` in the environment, then run:

```bash
korea-tourism fetch --start 201501 --end 202509
```

Results go to ignored `data/external/krw_exchange_rates_monthly.csv`. Review provider metadata, units, completeness, and revisions before intentionally replacing research inputs. API error messages suppress request URLs because ECOS places the key in the URL path. `.env` files are not loaded automatically.

## Reuse and improvements

The MIT license covers project code. Provider data may have different terms; no blanket dataset license is asserted here. A stronger future collection record would include source URL, retrieval UTC time, exact query parameters, units, license or terms reference, and unmodified response hash for each source.
