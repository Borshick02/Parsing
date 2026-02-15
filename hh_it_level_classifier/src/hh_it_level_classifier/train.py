from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from hh_it_level_classifier.dataset import build_dataframe, class_counts
from hh_it_level_classifier.evaluate import save_class_balance_plot, save_report_text
from hh_it_level_classifier.features import build_preprocessor, ensure_columns, get_feature_spec
from hh_it_level_classifier.utils import write_json


def _maybe_reason_counts(df: pd.DataFrame) -> dict[str, int] | None:
    if "label_reason" not in df.columns:
        return None

    s = df["label_reason"].astype(str)
    vc = s.value_counts(dropna=False)
    return {str(k): int(v) for k, v in vc.items()}


def train_and_save_model(
    csv_path: Path,
    resources_dir: Path,
    reports_dir: Path,
    target_column_hint: str,
    chunksize: int,
    limit_rows: int | None,
) -> None:
    df = build_dataframe(
        csv_path=csv_path,
        target_column_hint=target_column_hint,
        chunksize=chunksize,
        limit_rows=limit_rows,
    )
    if df.empty:
        raise ValueError("No rows after filtering IT resumes and labeling.")

    spec = get_feature_spec()
    df = ensure_columns(df, spec)

    x = df[spec.numeric + spec.categorical + [spec.text]]
    y = df["label"].astype(str)

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    preprocessor = build_preprocessor(spec)

    # PoC baseline classifier:
    # Use class_weight='balanced' because junior/middle/senior are typically imbalanced.
    clf = LogisticRegression(
        max_iter=2000,
        solver="lbfgs",
        class_weight="balanced",
    )

    model = Pipeline(
        steps=[
            ("prep", preprocessor),
            ("clf", clf),
        ]
    )

    model.fit(x_train, y_train)

    model_path = resources_dir / "model.joblib"
    joblib.dump(model, model_path)

    counts = class_counts(df)
    save_class_balance_plot(counts, reports_dir / "class_balance.png")
    report = save_report_text(model, x_test, y_test, reports_dir / "classification_report.txt")

    meta: dict[str, Any] = {
        "rows_total_after_filter": int(df.shape[0]),
        "class_counts": counts,
        "feature_spec": {
            "numeric": spec.numeric,
            "categorical": spec.categorical,
            "text": spec.text,
        },
        "model_path": str(model_path),
        "train_test_split": {"test_size": 0.2, "random_state": 42, "stratify": True},
        "report_summary": report,
        "notes": {
            "poc_labeling": (
                "title keywords (token-based; phrases: team lead/tech lead => senior); "
                "fallback by experience: <2 junior, 2-6 middle, >=6 senior"
            ),
            "class_weight": "balanced",
            "model": "LogisticRegression + TF-IDF + OneHot + numeric scaling",
        },
    }

    reason_counts = _maybe_reason_counts(df)
    if reason_counts is not None:
        meta["label_reason_counts"] = reason_counts

    write_json(resources_dir / "meta.json", meta)
