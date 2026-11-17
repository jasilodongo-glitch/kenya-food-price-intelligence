"""Safe loading of CSV / Excel files. Original uploads are never modified."""
from __future__ import annotations

import hashlib
import io
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import BinaryIO

import pandas as pd

import config


class DataLoadError(Exception):
    """Raised with a user-readable message when a file cannot be processed."""


@dataclass
class LoadResult:
    df: pd.DataFrame
    filename: str
    file_hash: str
    sheet: str | None = None
    size_bytes: int = 0
    notes: list[str] = field(default_factory=list)


def safe_filename(name: str) -> str:
    """Strip any path components and unsafe characters from a user-supplied name."""
    base = Path(name).name
    base = re.sub(r"[^A-Za-z0-9._-]+", "_", base).strip("._")
    return base or "dataset"


def validate_file(filename: str, size_bytes: int) -> str:
    """Validate extension and size. Returns the lower-cased extension."""
    ext = Path(filename).suffix.lower()
    if ext not in config.ALLOWED_EXTENSIONS:
        allowed = ", ".join(sorted(config.ALLOWED_EXTENSIONS))
        raise DataLoadError(f"Unsupported file type '{ext or 'none'}'. Upload one of: {allowed}.")
    if size_bytes == 0:
        raise DataLoadError("The file is empty.")
    if size_bytes > config.MAX_UPLOAD_MB * 1024 * 1024:
        raise DataLoadError(f"The file is {size_bytes / 1024 / 1024:.1f} MB; the limit is {config.MAX_UPLOAD_MB} MB.")
    return ext


def _read_bytes(source: BinaryIO | bytes) -> bytes:
    if isinstance(source, (bytes, bytearray)):
        return bytes(source)
    data = source.read()
    try:
        source.seek(0)
    except Exception:  # noqa: BLE001 - not all streams are seekable
        pass
    return data


def excel_sheet_names(data: bytes, ext: str) -> list[str]:
    try:
        return list(pd.ExcelFile(io.BytesIO(data), engine="openpyxl" if ext == ".xlsx" else "xlrd").sheet_names)
    except Exception as exc:  # noqa: BLE001
        raise DataLoadError(f"Could not read the Excel workbook: {exc}") from exc


def _read_csv(data: bytes) -> tuple[pd.DataFrame, list[str]]:
    notes: list[str] = []
    last_error: Exception | None = None
    for enc in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
        try:
            df = pd.read_csv(io.BytesIO(data), sep=None, engine="python", encoding=enc)
            if enc not in ("utf-8-sig", "utf-8"):
                notes.append(f"File decoded as {enc}; check special characters.")
            return df, notes
        except UnicodeDecodeError as exc:
            last_error = exc
        except (pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
            raise DataLoadError(f"The CSV could not be parsed: {exc}") from exc
    raise DataLoadError(f"Could not decode the CSV file: {last_error}")


def load_dataset(source: BinaryIO | bytes, filename: str, sheet: str | None = None) -> LoadResult:
    """Parse an uploaded file into a DataFrame (in memory only)."""
    data = _read_bytes(source)
    ext = validate_file(filename, len(data))
    notes: list[str] = []
    try:
        if ext == ".csv":
            df, notes = _read_csv(data)
            used_sheet = None
        else:
            engine = "openpyxl" if ext == ".xlsx" else "xlrd"
            names = excel_sheet_names(data, ext)
            used_sheet = sheet if sheet in names else names[0]
            df = pd.read_excel(io.BytesIO(data), sheet_name=used_sheet, engine=engine)
    except DataLoadError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise DataLoadError(f"The file could not be read: {exc}") from exc

    if df.empty or df.shape[1] == 0:
        raise DataLoadError("The file was read but contains no data rows or columns.")
    if len(df) > config.MAX_ROWS:
        raise DataLoadError(f"The dataset has {len(df):,} rows; the limit is {config.MAX_ROWS:,}.")
    df.columns = [str(c).strip() for c in df.columns]
    df, hxl_removed = _drop_hxl_row(df)
    if hxl_removed:
        notes.append("A HXL tag row (cells starting with '#') beneath the header was removed.")
    if df.columns.duplicated().any():
        df.columns = _dedupe(list(df.columns))
        notes.append("Duplicate column names were renamed to keep them unique.")
    return LoadResult(df=df, filename=safe_filename(filename), file_hash=hashlib.sha256(data).hexdigest(),
                      sheet=used_sheet, size_bytes=len(data), notes=notes)


def _drop_hxl_row(df: pd.DataFrame) -> tuple[pd.DataFrame, bool]:
    """Humanitarian Data Exchange files carry a row of '#tag' labels under the header."""
    if df.empty:
        return df, False
    first = df.iloc[0].dropna().astype(str)
    if len(first) >= 2 and first.str.startswith("#").mean() >= 0.8:
        return df.iloc[1:].reset_index(drop=True), True
    return df, False


def _dedupe(cols: list[str]) -> list[str]:
    seen: dict[str, int] = {}
    out = []
    for c in cols:
        if c in seen:
            seen[c] += 1
            out.append(f"{c}_{seen[c]}")
        else:
            seen[c] = 0
            out.append(c)
    return out


def save_original(data: bytes, filename: str, directory: Path | None = None) -> Path:
    """Persist the untouched original bytes (write-once, named by content hash)."""
    directory = directory or config.UPLOAD_DIR
    directory.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(data).hexdigest()[:12]
    target = directory / f"{digest}_{safe_filename(filename)}"
    if not target.exists():
        target.write_bytes(data)
    return target
