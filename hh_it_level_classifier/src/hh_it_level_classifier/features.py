from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer


@dataclass(frozen=True, slots=True)
class FeatureSpec:
    numeric: list[str]
    categorical: list[str]
    text: str


def get_feature_spec() -> FeatureSpec:
    return FeatureSpec(
        numeric=["age", "salary_rub", "exp_years"],
        categorical=["city", "employment", "schedule"],
        text="skills_text",
    )

def select_text_column(x):
    """
    Convert a single-column DataFrame/array to a 1D array of strings
    for TfidfVectorizer.
    """
    return x.squeeze().astype(str)

def build_preprocessor(spec: FeatureSpec) -> ColumnTransformer:
    # numeric pipeline
    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler(with_mean=False)),
        ]
    )

    # categorical pipeline
    cat_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    # text pipeline: extract single column from DataFrame -> series -> tfidf
    text_pipe = Pipeline(
        steps=[
            (
                "selector",
               FunctionTransformer(
                   select_text_column,
                   validate=False,
               ),
            ),
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=30_000,
                    ngram_range=(1, 2),
                    min_df=2,
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, spec.numeric),
            ("cat", cat_pipe, spec.categorical),
            ("txt", text_pipe, [spec.text]),
        ],
        remainder="drop",
        sparse_threshold=0.3,
    )
    return preprocessor


def ensure_columns(df: pd.DataFrame, spec: FeatureSpec) -> pd.DataFrame:
    df = df.copy()
    for col in spec.numeric + spec.categorical + [spec.text]:
        if col not in df.columns:
            df[col] = np.nan
    return df
