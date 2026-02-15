from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import classification_report

from hh_it_level_classifier.dataset import build_dataframe, class_counts
from hh_it_level_classifier.features import ensure_columns, get_feature_spec
from hh_it_level_classifier.utils import ensure_dir, write_json


def save_class_balance_plot(counts: dict[str, int], out_path: Path) -> None:
    ensure_dir(out_path.parent)

    labels = list(counts.keys())
    values = [counts[k] for k in labels]

    plt.figure(figsize=(8, 4))
    plt.bar(labels, values)
    plt.title("Class balance (junior/middle/senior)")
    plt.xlabel("Level")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def _maybe_reason_counts(df: pd.DataFrame) -> dict[str, int] | None:
    # Optional: store why the label was assigned (keyword vs experience fallback)
    if "label_reason" not in df.columns:
        return None

    s = df["label_reason"].astype(str)
    vc = s.value_counts(dropna=False)
    return {str(k): int(v) for k, v in vc.items()}


def save_report_text(
    model: Any,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    out_path: Path,
) -> dict[str, float]:
    ensure_dir(out_path.parent)

    y_pred = model.predict(x_test)
    report_dict = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    report_text = classification_report(y_test, y_pred, zero_division=0)

    with out_path.open("w", encoding="utf-8") as f:
        f.write(report_text)
        f.write("\n")

    summary = {
        "accuracy": float(report_dict.get("accuracy", 0.0)),
        "macro_f1": float(report_dict.get("macro avg", {}).get("f1-score", 0.0)),
        "weighted_f1": float(report_dict.get("weighted avg", {}).get("f1-score", 0.0)),
    }
    return summary


def evaluate_and_save_report(
    csv_path: Path,
    model_path: Path | None,
    reports_dir: Path,
    resources_dir: Path,
    target_column_hint: str,
    chunksize: int,
    limit_rows: int | None,
    only_prepare: bool,
) -> None:
    df = build_dataframe(
        csv_path=csv_path,
        target_column_hint=target_column_hint,
        chunksize=chunksize,
        limit_rows=limit_rows,
    )

    counts = class_counts(df)
    save_class_balance_plot(counts, reports_dir / "class_balance.png")

    meta: dict[str, Any] = {
        "rows_total_after_filter": int(df.shape[0]),
        "class_counts": counts,
        "prepared_only": bool(only_prepare),
    }

    reason_counts = _maybe_reason_counts(df)
    if reason_counts is not None:
        meta["label_reason_counts"] = reason_counts

    write_json(resources_dir / "meta.json", meta)

    if only_prepare:
        return

    if model_path is None or not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    model = joblib.load(model_path)

    spec = get_feature_spec()
    df = ensure_columns(df, spec)

    x = df[spec.numeric + spec.categorical + [spec.text]]
    y = df["label"].astype(str)

    # PoC evaluation: simple holdout (last 20%)
    n = len(df)
    if n < 100:
        raise ValueError("Too few rows to evaluate reliably. Increase data or remove --limit-rows.")
    split = int(n * 0.8)

    x_test = x.iloc[split:].copy()
    y_test = y.iloc[split:].copy()

    summary = save_report_text(model, x_test, y_test, reports_dir / "classification_report.txt")

    meta["report_summary"] = summary
    write_json(resources_dir / "meta.json", meta)
