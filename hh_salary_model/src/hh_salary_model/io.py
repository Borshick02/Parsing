from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np



def load_x(path: Path) -> np.ndarray:
    x = np.load(path)
    return np.asarray(x, dtype=np.float64)

def predictions_to_list(predictions: np.ndarray) -> list[float]:
    

    flat = np.asarray(predictions).reshape(-1)
    return [float(v) for v in flat]


def dump_predictions(
    predictions: list[float],
    output_path: Optional[Path] = None,
) -> str:
   

    text = json.dumps(predictions, ensure_ascii=False)
    if output_path is not None:
        output_path.write_text(text, encoding="utf-8")
    return text
