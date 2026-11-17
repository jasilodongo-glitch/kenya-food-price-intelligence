"""Descriptive and inferential statistics with plain-language interpretation."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def describe_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    rows = []
    for c in cols:
        s = pd.to_numeric(df[c], errors="coerce").dropna()
        if s.empty:
            continue
        mode = s.mode()
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        rows.append({
            "column": c, "count": int(s.count()), "mean": s.mean(), "median": s.median(),
            "mode": mode.iloc[0] if not mode.empty else np.nan,
            "std": s.std(ddof=1) if len(s) > 1 else np.nan, "variance": s.var(ddof=1) if len(s) > 1 else np.nan,
            "min": s.min(), "q1": q1, "q3": q3, "iqr": q3 - q1, "max": s.max(),
            "skew": s.skew() if len(s) > 2 else np.nan,
        })
    return pd.DataFrame(rows)


def frequency_table(s: pd.Series, top: int = 25) -> pd.DataFrame:
    vc = s.value_counts(dropna=False).head(top)
    out = vc.rename_axis("value").reset_index(name="count")
    out["percent"] = (out["count"] / len(s) * 100).round(2)
    return out


def correlation_matrix(df: pd.DataFrame, cols: list[str], method: str = "pearson") -> pd.DataFrame:
    return df[cols].apply(pd.to_numeric, errors="coerce").corr(method=method)


def top_correlations(corr: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    pairs = []
    cols = list(corr.columns)
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            v = corr.loc[a, b]
            if pd.notna(v):
                pairs.append({"variable_a": a, "variable_b": b, "correlation": float(v)})
    out = pd.DataFrame(pairs, columns=["variable_a", "variable_b", "correlation"])
    if out.empty:
        return out
    return out.reindex(out["correlation"].abs().sort_values(ascending=False).index).head(n).reset_index(drop=True)


def describe_strength(r: float) -> str:
    a = abs(r)
    word = "very weak" if a < 0.2 else "weak" if a < 0.4 else "moderate" if a < 0.6 else "strong" if a < 0.8 else "very strong"
    return f"{word} {'positive' if r > 0 else 'negative'}"


def mean_confidence_interval(s: pd.Series, level: float = 0.95) -> dict:
    s = pd.to_numeric(s, errors="coerce").dropna()
    n = len(s)
    if n < 2:
        raise ValueError("At least 2 observations are needed for a confidence interval.")
    mean, sem = s.mean(), stats.sem(s)
    lo, hi = stats.t.interval(level, df=n - 1, loc=mean, scale=sem) if sem > 0 else (mean, mean)
    return {"mean": float(mean), "lower": float(lo), "upper": float(hi), "level": level, "n": n}


def normality_test(s: pd.Series) -> dict:
    s = pd.to_numeric(s, errors="coerce").dropna()
    if len(s) < 8:
        raise ValueError("At least 8 observations are needed for a normality test.")
    sample = s.sample(5000, random_state=0) if len(s) > 5000 else s
    stat, p = stats.shapiro(sample)
    return {"test": "Shapiro-Wilk", "statistic": float(stat), "p_value": float(p), "n": len(sample),
            "interpretation": ("The data is not consistent with a normal distribution (p < 0.05)." if p < 0.05
                               else "No significant departure from normality was detected (p >= 0.05).")}


def compare_groups(df: pd.DataFrame, value_col: str, group_col: str, alpha: float = 0.05) -> dict:
    """Welch t-test (2 groups) or one-way ANOVA + Kruskal-Wallis (3+ groups)."""
    work = df[[value_col, group_col]].copy()
    work[value_col] = pd.to_numeric(work[value_col], errors="coerce")
    work = work.dropna()
    groups = {k: g[value_col].to_numpy() for k, g in work.groupby(group_col) if len(g) >= 2}
    if len(groups) < 2:
        raise ValueError("Need at least two groups with 2 or more observations each.")
    names = list(groups)
    if len(groups) == 2:
        stat, p = stats.ttest_ind(groups[names[0]], groups[names[1]], equal_var=False)
        test = "Welch two-sample t-test"
    else:
        stat, p = stats.f_oneway(*groups.values())
        test = "One-way ANOVA"
    extra = {}
    if len(groups) >= 3:
        h, kp = stats.kruskal(*groups.values())
        extra = {"kruskal_h": float(h), "kruskal_p": float(kp)}
    sig = bool(p < alpha)
    summary = work.groupby(group_col)[value_col].agg(["count", "mean", "median", "std"]).reset_index()
    return {"test": test, "statistic": float(stat), "p_value": float(p), "alpha": alpha, "significant": sig,
            "groups": len(groups), "summary": summary, **extra,
            "interpretation": (f"The group averages differ by more than chance alone would typically explain (p = {p:.4g}). "
                               "This shows a difference exists, not why it exists." if sig else
                               f"No statistically significant difference between group averages was found (p = {p:.4g}).")}


def ols_regression(df: pd.DataFrame, target: str, predictors: list[str]) -> dict:
    import statsmodels.api as sm
    work = df[[target] + predictors].apply(pd.to_numeric, errors="coerce").dropna()
    if len(work) < len(predictors) + 5:
        raise ValueError("Not enough complete rows for this regression.")
    X = sm.add_constant(work[predictors])
    model = sm.OLS(work[target], X).fit()
    ci = model.conf_int()
    table = pd.DataFrame({"coefficient": model.params, "std_error": model.bse, "p_value": model.pvalues,
                          "ci_lower": ci[0], "ci_upper": ci[1]}).reset_index(names="term")
    return {"table": table, "r_squared": float(model.rsquared), "adj_r_squared": float(model.rsquared_adj),
            "f_p_value": float(model.f_pvalue), "n": int(model.nobs),
            "interpretation": (f"The model explains {model.rsquared * 100:.1f}% of the variation in {target}. "
                               "Coefficients describe association in this data; they do not establish causation.")}
