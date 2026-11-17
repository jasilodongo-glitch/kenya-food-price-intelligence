"""Analytical report builder. Every statement is generated from computed results."""
from __future__ import annotations

import html
from dataclasses import dataclass, field
from datetime import datetime

import pandas as pd
import plotly.graph_objects as go

from src import analytics_engine, data_profiler as dp, export_manager, food_price_analysis as fpa
from src import statistical_analysis as sa, visualization as viz
from src.forecasting import ForecastResult
from src.time_series import TimeSeriesError


@dataclass
class Section:
    title: str
    paragraphs: list[str] = field(default_factory=list)
    tables: dict[str, pd.DataFrame] = field(default_factory=dict)
    figures: list[go.Figure] = field(default_factory=list)


@dataclass
class Report:
    title: str
    generated_at: str
    sections: list[Section]


def build_report(df: pd.DataFrame, dataset_name: str, title: str | None = None, food_roles: dict | None = None,
                 forecast: ForecastResult | None = None, forecast_label: str = "") -> Report:
    profile = dp.profile_dataset(df)
    kinds = profile.kinds
    cols = dp.columns_by_kind(df, kinds)
    nums = cols.get("numeric", [])
    sections: list[Section] = []
    findings: list[str] = []
    limitations = ["Findings describe patterns in the supplied data only and do not establish causation.",
                   "Results depend on the completeness and accuracy of the uploaded dataset."]

    sections.append(Section("Dataset description", [
        f"Source file: {dataset_name}. {profile.rows:,} records and {profile.columns} variables "
        f"({len(nums)} numeric, {len(cols.get('categorical', []))} categorical, {len(cols.get('datetime', []))} date).",
        *([f"Dates in '{profile.date_column}' span {profile.date_range[0]:%d %b %Y} to {profile.date_range[1]:%d %b %Y}."] if profile.date_range else []),
    ], tables={"Columns": dp.column_profile(df, kinds)}))

    sections.append(Section("Methodology", [
        "Column types were inferred from values. Descriptive statistics use pandas; outliers use the 1.5 x IQR rule; correlations are Pearson.",
        "Time-series and forecast figures (where present) use monthly aggregation, chronological train/test evaluation and 95% intervals.",
    ]))

    miss = dp.missing_summary(df)
    out = dp.outlier_summary(df, nums)
    dq = Section("Data-quality findings", [
        f"Overall missing cells: {profile.missing_pct:.2f}%. Exact duplicate rows: {profile.duplicates:,}."],
        tables={"Missing values": miss[miss["missing"] > 0]} if (miss["missing"] > 0).any() else {})
    if not out.empty:
        dq.tables["Outliers (IQR rule)"] = out
    if (miss["missing"] > 0).any():
        dq.figures.append(viz.missing_bar(miss))
    sections.append(dq)
    findings.extend(analytics_engine.auto_observations(df, profile))

    stat = Section("Statistical summaries")
    if nums:
        stat.tables["Numeric summary"] = sa.describe_numeric(df, nums)
        if len(nums) >= 2:
            corr = sa.correlation_matrix(df, nums)
            top = sa.top_correlations(corr, 5)
            stat.tables["Strongest correlations"] = top
            stat.figures.append(viz.corr_heatmap(corr))
            for _, r in top.head(3).iterrows():
                findings.append(f"'{r.variable_a}' and '{r.variable_b}' show a {sa.describe_strength(r.correlation)} correlation (r = {r.correlation:.2f}). Correlation is not causation.")
        stat.figures.append(viz.histogram(df[nums[0]], nums[0]))
    else:
        stat.paragraphs.append("No numeric columns were detected, so numeric summaries are not available.")
    sections.append(stat)

    if food_roles and food_roles.get("price"):
        sections.append(_food_section(df, food_roles, findings, limitations))

    if forecast is not None:
        best = forecast.metrics.iloc[0]
        fs = Section(f"Forecast: {forecast_label}", [
            f"{best['model']} achieved the lowest test error (RMSE {best['RMSE']:.3g}, MAE {best['MAE']:.3g}"
            + (f", MAPE {best['MAPE (%)']:.1f}%" if pd.notna(best['MAPE (%)']) else "") + ").",
            *forecast.notes], tables={"Model comparison": forecast.metrics})
        fs.figures.append(viz.forecast_chart(forecast.train, forecast.test, forecast.future[forecast.best_model], forecast.best_model))
        sections.append(fs)
        findings.append(f"Best-performing forecast model for {forecast_label}: {best['model']} (RMSE {best['RMSE']:.3g}).")
        limitations.append("Forecasts extrapolate historical patterns and cannot anticipate weather shocks, policy changes or supply disruptions.")

    sections.append(Section("Major findings", findings))
    implications = []
    if profile.missing_pct > 5:
        implications.append("Missing data is material; conclusions drawn from affected columns may be less reliable.")
    if profile.duplicates:
        implications.append("Duplicate records may inflate counts and averages until they are removed.")
    if not implications:
        implications.append("No major data-quality barriers were found for the analyses performed.")
    sections.append(Section("Business implications", implications))
    recs = []
    if profile.duplicates:
        recs.append("Review and remove duplicate rows before relying on aggregates.")
    if profile.missing_pct > 0:
        recs.append("Investigate the source of missing values, starting with the columns listed under data quality.")
    if not out.empty and out["outliers"].sum() > 0:
        recs.append("Verify flagged outliers against source records before excluding or keeping them.")
    recs.append("Refresh the analysis whenever new data is added.")
    sections.append(Section("Recommendations", recs))
    sections.append(Section("Limitations", limitations))
    return Report(title or f"Analysis report: {dataset_name}", datetime.now().strftime("%d %b %Y %H:%M"), sections)


def _food_section(df, roles, findings, limitations) -> Section:
    sec = Section("Food price intelligence")
    try:
        f = fpa.standardise(df, roles)
    except fpa.FoodPriceError as exc:
        sec.paragraphs.append(str(exc))
        return sec
    sec.paragraphs.append(fpa.AFFORDABILITY_NOTE)
    limitations.append(fpa.AFFORDABILITY_NOTE)
    commodity = None
    if "commodity" in f:
        commodity = f["commodity"].value_counts().index[0]
        sec.paragraphs.append(f"Commodity-specific results use '{commodity}', the most frequently recorded commodity.")
    for by in ("market", "county"):
        if by in f:
            try:
                rk, notes = fpa.location_ranking(f, by, commodity)
                sec.tables[f"{by.title()} ranking"] = rk
                sec.paragraphs.extend(notes)
                hi, lo = rk.iloc[0], rk.iloc[-1]
                findings.append(f"For {commodity or 'prices'}, {hi[by]} has the highest average price ({hi['mean']:.2f}) and {lo[by]} the lowest ({lo['mean']:.2f}), a gap of {(hi['mean'] / lo['mean'] - 1) * 100:.0f}%.")
                sec.figures.append(viz.bar(rk.head(15), by, "mean", f"Average price by {by}", horizontal=True))
            except fpa.FoodPriceError as exc:
                sec.paragraphs.append(f"{by.title()} ranking not produced: {exc}")
    if {"commodity", "date"} <= set(f.columns):
        try:
            vol = fpa.volatility_table(f)
            sec.tables["Commodity volatility"] = vol
            findings.append(f"{vol.iloc[0]['commodity']} is the most volatile commodity (monthly volatility {vol.iloc[0]['monthly_volatility_pct']:.1f}%).")
        except (fpa.FoodPriceError, TimeSeriesError) as exc:
            sec.paragraphs.append(f"Volatility not produced: {exc}")
    return sec


def _fmt(df: pd.DataFrame) -> str:
    return df.head(200).to_html(index=False, classes="tbl", border=0, float_format=lambda v: f"{v:,.3f}", na_rep="")


def to_html(report: Report) -> bytes:
    parts = [f"<!doctype html><html lang='en'><head><meta charset='utf-8'><title>{html.escape(report.title)}</title>",
             "<style>body{font-family:'IBM Plex Sans',Arial,sans-serif;max-width:960px;margin:40px auto;padding:0 20px;color:#17211C;line-height:1.55}"
             "h1,h2{font-family:Georgia,serif;color:#1F5F46}h2{border-bottom:1px solid #DCE3DB;padding-bottom:4px;margin-top:36px}"
             ".tbl{border-collapse:collapse;font-size:13px;margin:12px 0}.tbl th{background:#E3EEE7;text-align:left}.tbl td,.tbl th{padding:4px 10px;border-bottom:1px solid #DCE3DB}"
             "@media print{.pb{page-break-before:always}}</style></head><body>",
             f"<h1>{html.escape(report.title)}</h1><p>Generated {report.generated_at}. Open in a browser and use Print to save as PDF. Charts load from the Plotly CDN.</p>"]
    first_fig = True
    for sec in report.sections:
        parts.append(f"<h2>{html.escape(sec.title)}</h2>")
        if len(sec.paragraphs) > 1 and sec.title in {"Major findings", "Business implications", "Recommendations", "Limitations"}:
            parts.append("<ul>" + "".join(f"<li>{html.escape(p)}</li>" for p in sec.paragraphs) + "</ul>")
        else:
            parts.extend(f"<p>{html.escape(p)}</p>" for p in sec.paragraphs)
        for name, t in sec.tables.items():
            parts.append(f"<h3>{html.escape(name)}</h3>{_fmt(t)}")
        for fig in sec.figures:
            parts.append(fig.to_html(full_html=False, include_plotlyjs="cdn" if first_fig else False))
            first_fig = False
    parts.append("</body></html>")
    return "".join(parts).encode("utf-8")


def to_excel(report: Report) -> bytes:
    summary = pd.DataFrame({"section": [s.title for s in report.sections for _ in (s.paragraphs or [""])],
                            "text": [p for s in report.sections for p in (s.paragraphs or [""])]})
    sheets = {"Summary": summary}
    for sec in report.sections:
        for name, t in sec.tables.items():
            key = f"{sec.title[:12]}-{name}"[:31]
            while key in sheets:
                key = key[:-1] + "_"
            sheets[key] = t
    return export_manager.to_excel_bytes(sheets)
