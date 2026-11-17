import numpy as np
import pandas as pd
import pytest

from src import statistical_analysis as sa


@pytest.fixture
def df():
    rng = np.random.default_rng(1)
    x = rng.normal(50, 10, 400)
    return pd.DataFrame({"x": x, "y": 3 * x + 7 + rng.normal(0, 5, 400), "z": rng.normal(0, 1, 400),
                         "g": rng.choice(["a", "b"], 400)})


def test_descriptives_match_numpy():
    d = pd.DataFrame({"v": [1, 2, 2, 3, 4, 10]})
    r = sa.describe_numeric(d, ["v"]).iloc[0]
    v = d["v"].to_numpy()
    assert r["mean"] == pytest.approx(v.mean()) and r["median"] == 2.5 and r["mode"] == 2
    assert r["std"] == pytest.approx(v.std(ddof=1)) and r["variance"] == pytest.approx(v.var(ddof=1))
    assert r["q1"] == pytest.approx(np.percentile(v, 25)) and r["iqr"] == pytest.approx(r["q3"] - r["q1"])
    assert r["min"] == 1 and r["max"] == 10


def test_frequency_table():
    ft = sa.frequency_table(pd.Series(["a", "a", "b", None]))
    assert ft.set_index("value").loc["a", "count"] == 2 and ft["percent"].sum() == pytest.approx(100)


def test_correlation(df):
    corr = sa.correlation_matrix(df, ["x", "y", "z"])
    assert corr.loc["x", "y"] > 0.9 and abs(corr.loc["x", "z"]) < 0.2
    top = sa.top_correlations(corr, 1).iloc[0]
    assert {top.variable_a, top.variable_b} == {"x", "y"}
    assert sa.describe_strength(0.9) == "very strong positive" and sa.describe_strength(-0.3) == "weak negative"


def test_confidence_interval_contains_mean_and_widens(df):
    a, b = sa.mean_confidence_interval(df["x"], 0.90), sa.mean_confidence_interval(df["x"], 0.99)
    assert a["lower"] < a["mean"] < a["upper"] and (b["upper"] - b["lower"]) > (a["upper"] - a["lower"])
    with pytest.raises(ValueError):
        sa.mean_confidence_interval(pd.Series([1]))


def test_group_comparison_detects_real_difference():
    rng = np.random.default_rng(0)
    d = pd.DataFrame({"v": np.r_[rng.normal(10, 1, 60), rng.normal(14, 1, 60)], "g": ["a"] * 60 + ["b"] * 60})
    res = sa.compare_groups(d, "v", "g")
    assert res["significant"] and res["p_value"] < 0.001
    null = pd.DataFrame({"v": rng.normal(0, 1, 120), "g": ["a"] * 60 + ["b"] * 60})
    assert not sa.compare_groups(null, "v", "g", alpha=0.001)["significant"]


def test_anova_three_groups():
    rng = np.random.default_rng(3)
    d = pd.DataFrame({"v": np.r_[rng.normal(0, 1, 30), rng.normal(0, 1, 30), rng.normal(5, 1, 30)], "g": list("abc") * 30})
    d["g"] = np.repeat(list("abc"), 30)
    r = sa.compare_groups(d, "v", "g")
    assert r["test"] == "One-way ANOVA" and r["significant"] and "kruskal_p" in r


def test_ols_recovers_slope(df):
    r = sa.ols_regression(df, "y", ["x"])
    slope = r["table"].set_index("term").loc["x", "coefficient"]
    assert slope == pytest.approx(3, abs=0.2) and r["r_squared"] > 0.8
    assert "causation" in r["interpretation"]
    with pytest.raises(ValueError):
        sa.ols_regression(df.head(3), "y", ["x"])


def test_normality(df):
    assert sa.normality_test(df["x"])["p_value"] > 0.01
    assert sa.normality_test(pd.Series(np.random.default_rng(0).exponential(size=300)))["p_value"] < 0.05
    with pytest.raises(ValueError):
        sa.normality_test(pd.Series([1, 2, 3]))
