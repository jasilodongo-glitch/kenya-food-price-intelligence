"""Plotly figure builders with a consistent visual system."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

import config

C = config.COLORS
FONT = "IBM Plex Sans, system-ui, sans-serif"


def style(fig: go.Figure, title: str | None = None, height: int = 420) -> go.Figure:
    fig.update_layout(
        title=dict(text=title, x=0, font=dict(family="Fraunces, Georgia, serif", size=18, color=C["ink"])) if title else None,
        font=dict(family=FONT, color=C["ink"], size=13), height=height, margin=dict(l=10, r=10, t=56 if title else 20, b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", colorway=config.SERIES_COLORS,
        legend=dict(orientation="h", y=-0.18, x=0), hoverlabel=dict(font_family=FONT),
    )
    fig.update_xaxes(showgrid=False, linecolor=C["line"], zeroline=False)
    fig.update_yaxes(gridcolor=C["line"], zeroline=False)
    return fig


def histogram(s: pd.Series, name: str, bins: int = 30) -> go.Figure:
    fig = px.histogram(pd.to_numeric(s, errors="coerce").dropna(), nbins=bins, color_discrete_sequence=[C["green"]])
    fig.update_layout(showlegend=False, xaxis_title=name, yaxis_title="Count", bargap=0.04)
    return style(fig, f"Distribution of {name}")


def boxplot(df: pd.DataFrame, value: str, group: str | None = None, max_groups: int = 15) -> go.Figure:
    d = df[[value] + ([group] if group else [])].copy()
    d[value] = pd.to_numeric(d[value], errors="coerce")
    if group:
        top = d[group].value_counts().head(max_groups).index
        d = d[d[group].isin(top)]
    fig = px.box(d, y=value, x=group, color_discrete_sequence=[C["green"]], points="outliers")
    return style(fig, f"{value}" + (f" by {group}" if group else ""))


def scatter(df: pd.DataFrame, x: str, y: str, color: str | None = None, trend: bool = True) -> go.Figure:
    d = df[[x, y] + ([color] if color else [])].copy()
    d[x], d[y] = pd.to_numeric(d[x], errors="coerce"), pd.to_numeric(d[y], errors="coerce")
    d = d.dropna(subset=[x, y])
    if len(d) > 20000:
        d = d.sample(20000, random_state=0)
    fig = px.scatter(d, x=x, y=y, color=color, opacity=0.65, color_discrete_sequence=config.SERIES_COLORS)
    if trend and len(d) > 2 and d[x].nunique() > 1:
        k, b = np.polyfit(d[x], d[y], 1)
        xs = np.array([d[x].min(), d[x].max()])
        fig.add_trace(go.Scatter(x=xs, y=k * xs + b, mode="lines", name="Linear fit", line=dict(color=C["maize"], width=3)))
    return style(fig, f"{y} vs {x}")


def corr_heatmap(corr: pd.DataFrame) -> go.Figure:
    fig = go.Figure(go.Heatmap(z=corr.values, x=corr.columns, y=corr.index, zmin=-1, zmax=1, text=np.round(corr.values, 2),
                               texttemplate="%{text}", colorscale=[[0, C["clay"]], [0.5, "#FFFFFF"], [1, C["green"]]]))
    fig.update_yaxes(autorange="reversed")
    return style(fig, "Correlation matrix", height=max(380, 60 * len(corr)))


def bar(df: pd.DataFrame, x: str, y: str, title: str, horizontal: bool = False, color: str | None = None) -> go.Figure:
    if horizontal:
        fig = px.bar(df, x=y, y=x, orientation="h", color_discrete_sequence=[color or C["green"]])
        fig.update_yaxes(autorange="reversed")
    else:
        fig = px.bar(df, x=x, y=y, color_discrete_sequence=[color or C["green"]])
    return style(fig, title, height=max(360, 26 * len(df)) if horizontal else 400)


def line_series(s: pd.Series, title: str, ma: pd.Series | None = None, ma_label: str = "Moving average", ylabel: str = "") -> go.Figure:
    fig = go.Figure(go.Scatter(x=s.index, y=s.values, mode="lines", name=s.name or "Value", line=dict(color=C["green"], width=2)))
    if ma is not None:
        fig.add_trace(go.Scatter(x=ma.index, y=ma.values, mode="lines", name=ma_label, line=dict(color=C["maize"], width=3)))
    fig.update_layout(yaxis_title=ylabel)
    return style(fig, title)


def multi_line(wide: pd.DataFrame, title: str, ylabel: str = "") -> go.Figure:
    fig = go.Figure()
    for i, c in enumerate(wide.columns):
        fig.add_trace(go.Scatter(x=wide.index, y=wide[c], mode="lines", name=str(c), line=dict(color=config.SERIES_COLORS[i % 8], width=2)))
    fig.update_layout(yaxis_title=ylabel)
    return style(fig, title)


def seasonal_bar(profile: pd.DataFrame) -> go.Figure:
    names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    colors = [C["clay"] if v > 100 else C["green"] for v in profile["index"]]
    fig = go.Figure(go.Bar(x=[names[m - 1] for m in profile["month"]], y=profile["index"], marker_color=colors))
    fig.add_hline(y=100, line_dash="dot", line_color=C["muted"])
    fig.update_layout(yaxis_title="Seasonal index (100 = average)", showlegend=False)
    return style(fig, "Average price by calendar month")


def month_year_heatmap(table: pd.DataFrame) -> go.Figure:
    names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    fig = go.Figure(go.Heatmap(z=table.values, x=[names[m - 1] for m in table.columns], y=[str(y) for y in table.index],
                               colorscale=[[0, "#F2F6F1"], [1, C["green"]]], colorbar=dict(title="Avg")))
    return style(fig, "Average value by year and month", height=max(320, 28 * len(table) + 120))


def decomposition(dec: pd.DataFrame) -> go.Figure:
    from plotly.subplots import make_subplots
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.04, subplot_titles=["Observed", "Trend", "Seasonal", "Residual"])
    for i, col in enumerate(["observed", "trend", "seasonal", "residual"], 1):
        fig.add_trace(go.Scatter(x=dec.index, y=dec[col], mode="lines", line=dict(color=config.SERIES_COLORS[i - 1]), showlegend=False), row=i, col=1)
    return style(fig, "Trend decomposition (additive)", height=700)


def forecast_chart(train: pd.Series, test: pd.Series, future: pd.DataFrame, model: str, ylabel: str = "") -> go.Figure:
    fig = go.Figure()
    hist = pd.concat([train, test])
    fig.add_trace(go.Scatter(x=hist.index, y=hist.values, name="Historical", line=dict(color=C["ink"], width=2)))
    fig.add_trace(go.Scatter(x=list(future.index) + list(future.index[::-1]), y=list(future["upper"]) + list(future["lower"][::-1]),
                             fill="toself", fillcolor="rgba(217,162,27,0.20)", line=dict(width=0), name="95% interval", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=future.index, y=future["forecast"], name=f"Forecast ({model})", line=dict(color=C["maize"], width=3)))
    fig.update_layout(yaxis_title=ylabel)
    return style(fig, f"Forecast: {model}")


def backtest_chart(train: pd.Series, test: pd.Series, test_fc: dict[str, pd.DataFrame]) -> go.Figure:
    fig = go.Figure()
    tail = train.iloc[-max(len(test) * 3, 12):]
    fig.add_trace(go.Scatter(x=tail.index, y=tail.values, name="Training (tail)", line=dict(color=C["muted"], width=1.5)))
    fig.add_trace(go.Scatter(x=test.index, y=test.values, name="Actual (held out)", line=dict(color=C["ink"], width=3)))
    for i, (name, fc) in enumerate(test_fc.items()):
        fig.add_trace(go.Scatter(x=fc.index, y=fc["forecast"], name=name, line=dict(color=config.SERIES_COLORS[(i + 1) % 8], width=2, dash="dot")))
    return style(fig, "Held-out test period: actual vs model forecasts")


def missing_bar(miss: pd.DataFrame, top: int = 20) -> go.Figure:
    d = miss[miss["missing"] > 0].head(top)
    return bar(d, "column", "missing_pct", "Missing values by column (%)", horizontal=True, color=C["clay"])
