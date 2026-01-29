from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np


@dataclass(slots=True)
class LinearRegressionModel:
    coef: np.ndarray
    intercept: float
    feature_names: Optional[list[str]] = None

    @classmethod
    def fit(
        cls,
        x: np.ndarray,
        y: np.ndarray,
        feature_names: Optional[Iterable[str]] = None,
    ) -> "LinearRegressionModel":
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).reshape(-1)

        if x.ndim != 2:
            raise ValueError(f"x must be 2D array, got shape={x.shape}")
        if y.ndim != 1:
            raise ValueError(f"y must be 1D array, got shape={y.shape}")
        if x.shape[0] != y.shape[0]:
            raise ValueError("x and y must have the same number of rows")

        # y = X@coef + intercept
        ones = np.ones((x.shape[0], 1), dtype=np.float64)
        x_aug = np.hstack([x, ones])

        weights, *_ = np.linalg.lstsq(x_aug, y, rcond=None)
        coef = np.asarray(weights[:-1], dtype=np.float64)
        intercept = float(weights[-1])

        names_list: Optional[list[str]] = None
        if feature_names is not None:
            names_list = list(feature_names)

        return cls(coef=coef, intercept=intercept, feature_names=names_list)

    def predict(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)

        if x.ndim != 2:
            raise ValueError(f"x must be 2D array, got shape={x.shape}")
        if x.shape[1] != self.coef.shape[0]:
            raise ValueError(
                "Feature count mismatch: "
                f"x has {x.shape[1]} features, model expects {self.coef.shape[0]}"
            )

        return x @ self.coef + self.intercept

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "coef": np.asarray(self.coef, dtype=np.float64),
            "intercept": np.asarray(self.intercept, dtype=np.float64),
        }
        if self.feature_names:
            payload["feature_names"] = np.asarray(self.feature_names, dtype=object)

        np.savez_compressed(path, **payload)

    @classmethod
    def load(cls, path: Path) -> "LinearRegressionModel":
        data = np.load(path, allow_pickle=True)

        coef = np.asarray(data["coef"], dtype=np.float64).reshape(-1)
        intercept = float(np.asarray(data["intercept"], dtype=np.float64))

        feature_names: Optional[list[str]] = None
        if "feature_names" in data.files:
            names = data["feature_names"].tolist()
            if names:
                feature_names = [str(x) for x in names]

        return cls(coef=coef, intercept=intercept, feature_names=feature_names)
