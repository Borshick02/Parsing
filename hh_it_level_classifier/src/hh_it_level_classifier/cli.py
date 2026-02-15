from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from hh_it_level_classifier.evaluate import evaluate_and_save_report
from hh_it_level_classifier.train import train_and_save_model
from hh_it_level_classifier.utils import ensure_dir


Mode = Literal["prepare", "train", "evaluate"]


@dataclass(frozen=True, slots=True)
class CliArgs:
    csv_path: Path
    mode: Mode
    target_column: str
    chunksize: int
    limit_rows: int | None


def _parse_args() -> CliArgs:
    parser = argparse.ArgumentParser(
        prog="app",
        description=(
            "PoC: classify IT developer level (junior/middle/senior) from hh.csv."
        ),
    )
    parser.add_argument("csv_path", type=Path, help="Path to hh.csv")

    parser.add_argument(
        "--mode",
        choices=["prepare", "train", "evaluate"],
        default="train",
        help="What to do: prepare | train | evaluate (default: train).",
    )
    parser.add_argument(
        "--target",
        dest="target_column",
        default="ЗП",
        help="Salary column name hint (default: 'ЗП').",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=50_000,
        help="CSV reading chunk size for large files (default: 50000).",
    )
    parser.add_argument(
        "--limit-rows",
        type=int,
        default=None,
        help="Optional: limit processed rows for faster PoC debugging.",
    )

    ns = parser.parse_args()
    return CliArgs(
        csv_path=ns.csv_path,
        mode=ns.mode,
        target_column=ns.target_column,
        chunksize=ns.chunksize,
        limit_rows=ns.limit_rows,
    )


def main() -> int:
    args = _parse_args()

    project_root = Path(__file__).resolve().parents[2]  # hh_it_level_classifier/
    resources_dir = project_root / "resources"
    reports_dir = project_root / "reports"

    ensure_dir(resources_dir)
    ensure_dir(reports_dir)

    if args.mode == "prepare":
        # prepare includes: build dataset + class balance plot + meta.json
        evaluate_and_save_report(
            csv_path=args.csv_path,
            model_path=None,
            reports_dir=reports_dir,
            resources_dir=resources_dir,
            target_column_hint=args.target_column,
            chunksize=args.chunksize,
            limit_rows=args.limit_rows,
            only_prepare=True,
        )
        print("Done: prepared dataset stats + class balance plot.")
        return 0

    if args.mode == "train":
        train_and_save_model(
            csv_path=args.csv_path,
            resources_dir=resources_dir,
            reports_dir=reports_dir,
            target_column_hint=args.target_column,
            chunksize=args.chunksize,
            limit_rows=args.limit_rows,
        )
        print("Done: trained model and saved to resources/.")
        return 0

    if args.mode == "evaluate":
        evaluate_and_save_report(
            csv_path=args.csv_path,
            model_path=resources_dir / "model.joblib",
            reports_dir=reports_dir,
            resources_dir=resources_dir,
            target_column_hint=args.target_column,
            chunksize=args.chunksize,
            limit_rows=args.limit_rows,
            only_prepare=False,
        )
        print("Done: evaluation report saved to reports/.")
        return 0

    raise ValueError(f"Unknown mode: {args.mode}")
