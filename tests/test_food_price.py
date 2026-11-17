import pandas as pd
import pytest

from src import food_price_analysis as fpa


def test_roles_detected_on_wfp_style_columns(food_df):
    r = fpa.detect_roles(food_df)
    assert r["price"] == "price" and r["date"] == "date" and r["commodity"] == "commodity"
    assert r["market"] == "market" and r["county"] == "admin1" and r["currency"] == "currency"
    assert r["unit"] == "unit" and r["price_type"] == "pricetype"


def test_roles_do_not_depend_on_fixed_names(food_df):
    renamed = food_df.rename(columns={"price": "Retail Cost", "commodity": "Food Item", "market": "Town", "date": "Observation Month"})
    r = fpa.detect_roles(renamed)
    assert (r["price"], r["commodity"], r["market"], r["date"]) == ("Retail Cost", "Food Item", "Town", "Observation Month")


def test_usd_price_not_preferred_over_local(food_df):
    df = food_df.assign(usdprice=food_df["price"] / 130)
    assert fpa.detect_roles(df)["price"] == "price"


def test_unsupported_data_activates_nothing():
    df = pd.DataFrame({"a": ["x", "y"], "b": ["p", "q"]})
    roles = fpa.detect_roles(df)
    assert not any(fpa.available_analyses(roles).values())
    with pytest.raises(fpa.FoodPriceError):
        fpa.standardise(df, roles)


def test_availability_follows_columns(food_df):
    roles = fpa.detect_roles(food_df)
    assert all(fpa.available_analyses(roles).values())
    roles["date"] = None
    a = fpa.available_analyses(roles)
    assert a["market"] and not a["volatility"] and not a["seasonality"]


def test_market_ranking(food_df):
    f = fpa.standardise(food_df, fpa.detect_roles(food_df))
    rk, _ = fpa.location_ranking(f, "market", "Maize")
    assert list(rk["market"]) == ["Beta", "Alpha", "Gamma"]            # factors 1.15 > 1.00 > 0.90
    assert rk["rank"].tolist() == [1, 2, 3] and rk["vs_average_pct"].iloc[0] > 0
    manual = f[(f.commodity == "Maize") & (f.market == "Beta")]["price"].mean()
    assert rk.set_index("market").loc["Beta", "mean"] == pytest.approx(manual)


def test_ranking_refuses_to_mix_commodities(food_df):
    f = fpa.standardise(food_df, fpa.detect_roles(food_df))
    with pytest.raises(fpa.FoodPriceError, match="single commodity"):
        fpa.location_ranking(f, "market", None)


def test_county_ranking(food_df):
    f = fpa.standardise(food_df, fpa.detect_roles(food_df))
    rk, _ = fpa.location_ranking(f, "county", "Beans")
    assert set(rk["county"]) == {"North", "South"} and rk.iloc[0]["county"] == "South"


def test_volatility_ranks_seasonal_commodity_higher(food_df):
    f = fpa.standardise(food_df, fpa.detect_roles(food_df))
    vol = fpa.volatility_table(f)
    assert vol.iloc[0]["commodity"] == "Maize" and vol["monthly_volatility_pct"].gt(0).all()
    with pytest.raises(fpa.FoodPriceError):
        fpa.volatility_table(f, min_months=500)


def test_price_index_rebased_to_100(food_df):
    f = fpa.standardise(food_df, fpa.detect_roles(food_df))
    idx = fpa.price_index(f, ["Maize", "Beans"])
    assert idx.iloc[0].tolist() == pytest.approx([100, 100])


def test_dominant_unit_filter_flags_mixed_units(food_df):
    mixed = food_df.copy()
    mixed.loc[mixed.index[:10], "unit"] = "90KG"
    f = fpa.standardise(mixed, fpa.detect_roles(mixed))
    sub, notes = fpa.commodity_subset(f, "Maize")
    assert set(sub["unit"]) == {"KG"} and any("most common" in n for n in notes)


def test_latest_change_table(food_df):
    f = fpa.standardise(food_df, fpa.detect_roles(food_df))
    t = fpa.latest_change_table(f).set_index("commodity")
    assert set(t.index) == {"Maize", "Beans"} and t["change_12m_pct"].notna().all()


def test_affordability_caveat_present():
    assert "income" in fpa.AFFORDABILITY_NOTE.lower() and "not household affordability" in fpa.AFFORDABILITY_NOTE.lower()
