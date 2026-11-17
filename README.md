# Kenya Food Price Intelligence

*Transforming food price data into actionable intelligence.*

A local analytics platform built with Streamlit. Upload your own CSV or Excel file and get profiling, cleaning, statistics, trends, food-price analysis, forecasting and downloadable reports. It is specialised for food-market price data (for example WFP / HDX Kenya price files) but works on any tabular dataset.

The app ships with **no data**. Until you upload a file it shows an empty state, and every number it displays is calculated from your active dataset.

## Who it is for

Analysts, researchers, NGOs, county and national agencies, traders and journalists who need to understand how food prices differ across markets, move over time and behave seasonally, without writing code for each new file.

**Business value:** faster price monitoring, early sight of volatile commodities and expensive markets, and repeatable reports with the calculations behind each finding.

## Features

| Area | What you get |
|---|---|
| Upload | CSV, XLSX, XLS; drag and drop; size and type validation; worksheet picker; encoding detection; HXL tag-row handling; originals saved unmodified |
| Profiling and cleaning | Column type detection, missing values, duplicates, IQR outliers, data-driven cleaning recommendations you opt into, cleaned CSV and Excel export |
| Explorer | Text search, column selection, filters, sorting, pagination, grouping and comparison, filtered download |
| Statistical analysis | Mean, median, mode, SD, variance, quartiles, IQR, frequency tables, histograms, box plots, scatter, Pearson and Spearman correlation, normality test, confidence intervals, t-test / ANOVA, OLS regression, plain-language explanations |
| Trend analysis | Daily, weekly, monthly, annual aggregation, moving averages, percent change, CAGR, seasonality, additive decomposition, year-on-year comparison |
| Food price intelligence | Automatic column role detection (commodity, market, county, country, date, price, currency, unit, observation type) with manual override; market and county rankings; commodity comparison via rebased price index; volatility; seasonality. Analyses only appear if the columns support them |
| Forecasting | Naive, seasonal naive, moving average, exponential smoothing (Holt-Winters, damped), ARIMA (AIC-selected); chronological train/test split; MAE, RMSE, MAPE; 95% intervals and interval coverage |
| Reports | Dataset description, methodology, data quality, statistics, charts, findings, implications, recommendations, limitations. Download as HTML (print to PDF from the browser) and Excel |
| Privacy | Local only, no public URLs for uploaded data, delete-all control, metadata log stores file names and sizes only |

## Technologies

Python 3, Streamlit, pandas, NumPy, Plotly, scikit-learn, SciPy, statsmodels, openpyxl, xlrd, SQLite, pytest.

## Installation

### Windows (PowerShell)

```powershell
cd kenya-food-price-intelligence
py -3 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

If PowerShell blocks activation, run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once.

### macOS / Linux

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

The app opens at http://localhost:8501. Python 3.10 or newer is recommended.

## Using the platform

1. **Data Upload**: drop a file, pick a worksheet if prompted, review the preview, select *Use this dataset*.
2. Review the profile, then open the **Cleaning** tab. Tick only the operations you want. Cleaning always starts from your original upload, and *Reset to original* undoes it.
3. **Executive Overview** summarises the data. **Data Explorer** lets you slice it.
4. **Statistical Analysis** and **Trend Analysis** let you choose columns. Trend and forecast pages explain when data is unsuitable instead of forcing a result.
5. **Food Price Intelligence**: check the detected columns, choose a commodity, then use the tabs. Rankings compare one commodity at a time because prices of different foods are not comparable.
6. **Forecasting**: choose date and value columns, optionally filter to one commodity or market, pick a horizon, and run. Needs at least 24 periods.
7. **Reports**: generate, preview, and download.

## Tests

```bash
python -m pytest -q
```

87 automated tests cover uploads, CSV and Excel parsing, cleaning, missing values, duplicates, statistics, correlation, time series, food-price analysis, forecasting, reports and exports, and headless rendering of every page. Synthetic data exists only inside `tests/`.

## Project structure

```
app.py                  entry point, navigation, theme
config.py               limits, paths, colours
src/                    data_loader, data_cleaner, data_profiler, analytics_engine,
                        statistical_analysis, time_series, forecasting, food_price_analysis,
                        visualization, report_generator, export_manager, metadata_store, state, ui
views/                  one module per page (named views/ because Streamlit auto-discovers a pages/ folder)
data/uploads|processed  originals and cleaned exports (git-ignored)
tests/                  pytest suite
```

## Limitations

- **Price trends are not household affordability.** Affordability needs income or expenditure data, which price files lack. The app states this wherever relevant.
- No PDF generator: reports download as HTML (use Print to PDF) and Excel. Charts in the HTML report load Plotly from a CDN.
- Forecasts extrapolate history; they cannot anticipate weather, policy or supply shocks. ARIMA is non-seasonal; seasonal behaviour is handled by exponential smoothing and seasonal naive. Intervals for the simple models are approximations.
- Seasonal indices are averages across years and ignore trend; use decomposition for a trend-adjusted view.
- Column-role detection is name- and type-based. Check it, and override it when it is wrong.
- Not tested in a real browser here: layout and styling were verified by headless rendering only. The Google Fonts import needs internet access; a system font is used otherwise.
- Single-user local design. For online deployment, add authentication, per-user storage and a TLS proxy.
