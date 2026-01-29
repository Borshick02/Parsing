from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from hh_salary_model.model import LinearRegressionModel  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a linear regression model")
    parser.add_argument("x_path", type=Path, help="Path to x_data.npy")
    parser.add_argument("y_path", type=Path, help="Path to y_data.npy")
    parser.add_argument(
        "--out",
        type=Path,
        default=PROJECT_ROOT / "resources" / "model.npz",
        help="Where to save weights (default: resources/model.npz)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    x = np.load(args.x_path)
    y = np.load(args.y_path)

    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).reshape(-1)

    model = LinearRegressionModel.fit(x, y)
    model.save(args.out)

    print(f"Saved weights to: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
