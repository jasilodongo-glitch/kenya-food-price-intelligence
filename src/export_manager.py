"""Exports and data-retention controls."""
from __future__ import annotations

import io
from pathlib import Path

import pandas as pd

import config


def to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8-sig")


def to_excel_bytes(sheets: dict[str, pd.DataFrame]) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        for name, frame in sheets.items():
            safe = "".join(ch for ch in name if ch not in '[]:*?/\\')[:31] or "Sheet"
            frame.to_excel(writer, sheet_name=safe, index=False)
    return buf.getvalue()


def save_processed(df: pd.DataFrame, name: str, directory: Path | None = None) -> Path:
    from src.data_loader import safe_filename
    directory = directory or config.PROCESSED_DIR
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / f"{Path(safe_filename(name)).stem}_cleaned.csv"
    df.to_csv(target, index=False)
    return target


def stored_files(directories: list[Path] | None = None) -> list[dict]:
    dirs = directories or [config.UPLOAD_DIR, config.PROCESSED_DIR, config.OUTPUT_DIR]
    out = []
    for d in dirs:
        if d.exists():
            for p in sorted(d.iterdir()):
                if p.is_file() and p.name != ".gitkeep":
                    out.append({"folder": d.name, "file": p.name, "size_kb": round(p.stat().st_size / 1024, 1), "path": str(p)})
    return out


def delete_stored_files(directories: list[Path] | None = None) -> int:
    n = 0
    for item in stored_files(directories):
        Path(item["path"]).unlink(missing_ok=True)
        n += 1
    return n
