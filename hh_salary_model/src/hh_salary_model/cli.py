from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from hh_salary_model.model import LinearRegressionModel


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="hh-salary-model",
        description="Predict salaries (RUB) from x_data.npy using saved linear regression weights.",
    )
    parser.add_argument("x_path", type=Path, help="Path to x_data.npy")
    parser.add_argument(
        "--model",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "resources" / "model.npz",
        help="Path to model weights (.npz). Default: resources/model.npz",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output file (preds.json / preds.txt). If omitted, prints JSON list to stdout.",
    )
    return parser.parse_args()


def _save_predictions(pred: np.ndarray, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    suffix = output_path.suffix.lower()
    pred = pred.astype(np.float64)

    if suffix == ".json":
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(pred.tolist(), f, ensure_ascii=False, indent=2)
        return

    # txt/csv: one value per line
    np.savetxt(output_path, pred, fmt="%.6f")


def main() -> int:
    args = _parse_args()

    if not args.x_path.exists():
        raise FileNotFoundError(f"x_data.npy not found: {args.x_path}")

    if not args.model.exists():
        raise FileNotFoundError(
            f"Model weights not found: {args.model}. "
            f"Train the model first: python scripts/train.py X.npy Y.npy --out resources/model.npz"
        )

    x = np.load(args.x_path)
    model = LinearRegressionModel.load(args.model)
    pred = model.predict(x)

    if args.output is None:
        # Требование задания: вернуть список float (в рублях).
        # Печатаем JSON-массив float в stdout.
        print(json.dumps(pred.tolist(), ensure_ascii=False))
        return 0

    _save_predictions(pred, args.output)
    print(f"Saved {len(pred)} predictions to: {args.output}")
    return 0
