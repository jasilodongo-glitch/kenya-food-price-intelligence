import io

import pandas as pd
import pytest

import config
from src import data_loader as dl


def test_csv_loads(csv_bytes, food_df):
    res = dl.load_dataset(csv_bytes, "prices.csv")
    assert res.df.shape == food_df.shape
    assert res.sheet is None and len(res.file_hash) == 64


def test_xlsx_sheet_selection(xlsx_bytes, food_df):
    assert dl.excel_sheet_names(xlsx_bytes, ".xlsx") == ["Notes", "Prices"]
    assert dl.load_dataset(xlsx_bytes, "p.xlsx", "Prices").df.shape == food_df.shape
    assert dl.load_dataset(xlsx_bytes, "p.xlsx").sheet == "Notes"       # default = first sheet
    assert dl.load_dataset(xlsx_bytes, "p.xlsx", "Missing").sheet == "Notes"


def test_semicolon_csv_and_latin1():
    data = "a;b\n1;café\n2;thé".encode("latin-1")
    res = dl.load_dataset(data, "x.csv")
    assert list(res.df.columns) == ["a", "b"] and res.df["b"].iloc[0] == "café"


def test_rejects_bad_extension():
    with pytest.raises(dl.DataLoadError, match="Unsupported"):
        dl.load_dataset(b"hello", "evil.exe")


def test_rejects_empty_and_header_only():
    with pytest.raises(dl.DataLoadError, match="empty"):
        dl.load_dataset(b"", "a.csv")
    with pytest.raises(dl.DataLoadError):
        dl.load_dataset(b"a,b\n", "a.csv")


def test_rejects_oversize(monkeypatch):
    monkeypatch.setattr(config, "MAX_UPLOAD_MB", 0.0001)
    with pytest.raises(dl.DataLoadError, match="limit"):
        dl.load_dataset(b"a,b\n" + b"1,2\n" * 500, "a.csv")


def test_corrupt_excel_gives_friendly_error():
    with pytest.raises(dl.DataLoadError):
        dl.load_dataset(b"not really excel", "broken.xlsx")


def test_duplicate_column_names_are_made_unique():
    res = dl.load_dataset(b"a,a,b\n1,2,3\n", "d.csv")
    assert len(set(res.df.columns)) == 3


def test_hxl_row_removed():
    res = dl.load_dataset(b"date,price\n#date,#value\n2020-01-01,5\n2020-02-01,6\n", "h.csv")
    assert len(res.df) == 2 and any("HXL" in n for n in res.notes)


def test_safe_filename_blocks_traversal():
    assert dl.safe_filename("../../etc/passwd") == "passwd"
    assert "/" not in dl.safe_filename("a/b\\c d.csv")


def test_original_saved_unmodified_and_write_once(tmp_path, csv_bytes):
    p = dl.save_original(csv_bytes, "../../x.csv", tmp_path)
    assert p.parent == tmp_path and p.read_bytes() == csv_bytes
    mtime = p.stat().st_mtime_ns
    assert dl.save_original(csv_bytes, "../../x.csv", tmp_path) == p
    assert p.stat().st_mtime_ns == mtime
