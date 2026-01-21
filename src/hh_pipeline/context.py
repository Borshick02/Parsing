from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass(slots=True)
class PipelineContext:

    input_path: Path
    output_dir: Path

    df: pd.DataFrame | None = None
    x: np.ndarray | None = None
    y: np.ndarray | None = None

    meta: dict[str, Any] = field(default_factory=dict)
