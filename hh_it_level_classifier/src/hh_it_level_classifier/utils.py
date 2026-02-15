from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def try_read_csv_encoding(path: Path) -> list[str]:
    # For hh.csv most commonly: utf-8 or cp1251
    return ["utf-8", "cp1251"]


def normalize_text(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"\s+", " ", s)
    return s


def extract_first_int(text: str) -> int | None:
    m = re.search(r"(\d+)", text)
    if not m:
        return None
    return int(m.group(1))


def safe_float(x: Any) -> float | None:
    try:
        if x is None:
            return None
        if isinstance(x, float):
            if np.isnan(x):
                return None
            return float(x)
        if isinstance(x, int):
            return float(x)
        s = str(x).strip()
        if s == "" or s.lower() == "nan":
            return None
        return float(s)
    except Exception:
        return None
