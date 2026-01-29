from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from hh_pipeline import PipelineConfig, run_pipeline


def parse_args() -> argparse.Namespace:
    
    parser = argparse.ArgumentParser(
        prog="app",
        description=(
            "Preprocess hh.csv and save x_data.npy and y_data.npy next to it."
        ),
    )
    parser.add_argument(
        "csv_path",
        type=Path,
        help="Path to hh.csv",
    )
    parser.add_argument(
        "--target",
        dest="target_column",
        default="ЗП",
        help="Target column name (default: 'ЗП').",
    )
    return parser.parse_args()

def save_predictions(pred: np.ndarray, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    suffix = output_path.suffix.lower()
    pred = pred.astype(float)

    if suffix == ".json":
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(pred.tolist(), f, ensure_ascii=False, indent=2)
        return

    # txt/csv: one value per line
    np.savetxt(output_path, pred, fmt="%.6f")

def main() -> None:
    args = parse_args()

    config = PipelineConfig(target_column=args.target_column)
    ctx = run_pipeline(args.csv_path, config=config)

    print("Done.")
    print(f"Saved: {ctx.meta['x_path']}")
    print(f"Saved: {ctx.meta['y_path']}")


if __name__ == "__main__":
    main()
