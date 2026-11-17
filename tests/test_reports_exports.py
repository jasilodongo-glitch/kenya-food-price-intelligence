import io

import openpyxl
import pandas as pd
import pytest

from src import export_manager as em, food_price_analysis as fpa, forecasting as fc, metadata_store as ms
from src import report_generator as rg, time_series as ts


def test_report_sections_and_food_findings(food_df):
    rep = rg.build_report(food_df, "synthetic.csv", food_roles=fpa.detect_roles(food_df))
    titles = [s.title for s in rep.sections]
    for t in ["Dataset description", "Methodology", "Data-quality findings", "Statistical summaries", "Food price intelligence",
              "Major findings", "Business implications", "Recommendations", "Limitations"]:
        assert t in titles
    findings = next(s for s in rep.sections if s.title == "Major findings").paragraphs
    assert any("highest average price" in f for f in findings)
    assert any(f"{len(food_df):,} records" in f for f in findings)
    assert any("affordab" in p.lower() for p in next(s for s in rep.sections if s.title == "Limitations").paragraphs)


def test_report_traceable_numbers(food_df):
    rep = rg.build_report(food_df, "s.csv", food_roles=fpa.detect_roles(food_df))
    f = fpa.standardise(food_df, fpa.detect_roles(food_df))
    commodity = f["commodity"].value_counts().index[0]
    rk, _ = fpa.location_ranking(f, "market", commodity)
    text = " ".join(next(s for s in rep.sections if s.title == "Major findings").paragraphs)
    assert f"{rk.iloc[0]['mean']:.2f}" in text


def test_report_without_food_columns_or_numbers():
    rep = rg.build_report(pd.DataFrame({"a": ["x", "y", "z"], "b": ["p", "q", "r"]}), "t.csv")
    assert any("No numeric columns" in p for s in rep.sections for p in s.paragraphs)


def test_report_includes_forecast_and_exports(food_df):
    s = ts.prepare_series(food_df[food_df.commodity == "Maize"], "date", "price", "MS")
    res = fc.run_forecast(s, 6, ["Naive", "Exponential smoothing"], 12)
    rep = rg.build_report(food_df, "s.csv", forecast=res, forecast_label="Maize")
    html = rg.to_html(rep).decode()
    assert "<h1>" in html and "Forecast: Maize" in html and "plotly" in html.lower()
    wb = openpyxl.load_workbook(io.BytesIO(rg.to_excel(rep)))
    assert "Summary" in wb.sheetnames and len(wb.sheetnames) > 3


def test_html_escapes_user_content():
    rep = rg.build_report(pd.DataFrame({"a": [1, 2, 3]}), "<script>alert(1)</script>.csv", title="<b>x</b>")
    html = rg.to_html(rep).decode()
    assert "<script>alert(1)</script>" not in html and "&lt;b&gt;x&lt;/b&gt;" in html


def test_exports_roundtrip(tmp_path, food_df):
    assert pd.read_csv(io.BytesIO(em.to_csv_bytes(food_df))).shape == food_df.shape
    back = pd.read_excel(io.BytesIO(em.to_excel_bytes({"Data": food_df})))
    assert back.shape == food_df.shape
    p = em.save_processed(food_df, "../weird name.csv", tmp_path)
    assert p.parent == tmp_path and p.name.endswith("_cleaned.csv")


def test_delete_controls(tmp_path):
    (tmp_path / "a.csv").write_text("x")
    (tmp_path / ".gitkeep").write_text("")
    assert len(em.stored_files([tmp_path])) == 1
    assert em.delete_stored_files([tmp_path]) == 1 and (tmp_path / ".gitkeep").exists()


def test_metadata_store_holds_no_data(tmp_path):
    db = tmp_path / "m.sqlite3"
    ms.record_dataset("a.csv", "abc", None, 10, 3, 999, db)
    h = ms.history(db)
    assert h[0]["filename"] == "a.csv" and "price" not in str(h)
    ms.clear_history(db)
    assert ms.history(db) == []
