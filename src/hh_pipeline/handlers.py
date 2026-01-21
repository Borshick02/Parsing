from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from .context import PipelineContext


class Handler(ABC):
    def __init__(self) -> None:
        self._next: Handler | None = None

    def set_next(self, handler: "Handler") -> "Handler":
        self._next = handler
        return handler

    def handle(self, ctx: PipelineContext) -> PipelineContext:
        ctx = self._handle(ctx)
        if self._next is None:
            return ctx
        return self._next.handle(ctx)

    @abstractmethod
    def _handle(self, ctx: PipelineContext) -> PipelineContext:
        raise NotImplementedError


class LoadCsvChunksHandler(Handler):
    def __init__(self, *, chunksize: int = 100_000) -> None:
        super().__init__()
        self._chunksize = chunksize

    def _handle(self, ctx: PipelineContext) -> PipelineContext:
        if not ctx.input_path.exists():
            raise FileNotFoundError(f"Input file not found: {ctx.input_path}")

        desired_cols = {
            "Пол, возраст",
            "ЗП",
            "Город",
            "Занятость",
            "График",
            "Авто",
        }

        def keep_col(col_name: str) -> bool:
            return col_name in desired_cols or col_name.startswith("Unnamed")

        for encoding in ("utf-8", "cp1251"):
            try:
                chunk_iter = pd.read_csv(
                    ctx.input_path,
                    sep=",",
                    engine="python",
                    chunksize=self._chunksize,
                    usecols=keep_col,
                    encoding=encoding,
                )
                first = next(chunk_iter) 
                ctx.meta["encoding_used"] = encoding
                ctx.meta["chunks"] = self._chain_first(first, chunk_iter)
                return ctx
            except UnicodeDecodeError:
                continue

        raise UnicodeDecodeError(
            "Failed to decode CSV using utf-8 and cp1251",
            b"",
            0,
            1,
            "decode error",
        )

    @staticmethod
    def _chain_first(
        first: pd.DataFrame, rest: Iterable[pd.DataFrame]
    ) -> Iterable[pd.DataFrame]:
        yield first
        yield from rest


@dataclass(slots=True)
class _CategoryEncoder:
    mapping: dict[str, int]

    @classmethod
    def new(cls) -> "_CategoryEncoder":
        return cls(mapping={})

    def encode(self, values: pd.Series) -> np.ndarray:
        series = values.astype("string").fillna("unknown")
        out = np.empty(len(series), dtype=np.int32)

        next_id = len(self.mapping)
        for i, val in enumerate(series):
            key = str(val)
            if key not in self.mapping:
                self.mapping[key] = next_id
                next_id += 1
            out[i] = self.mapping[key]

        return out


class HhPreprocessHandler(Handler):
    _CONTROL_CHARS_RE = re.compile(r"[\t\r\n]+")
    _MULTI_SPACE_RE = re.compile(r"\s+")

    def __init__(self, *, target_column: str = "ЗП") -> None:
        super().__init__()
        self._target_column = target_column

        self._city = _CategoryEncoder.new()
        self._employment = _CategoryEncoder.new()
        self._schedule = _CategoryEncoder.new()
        self._auto = _CategoryEncoder.new()

    def _handle(self, ctx: PipelineContext) -> PipelineContext:
        chunks = ctx.meta.get("chunks")
        if chunks is None:
            raise ValueError("Chunks iterator not found in ctx.meta['chunks']")

        x_parts: list[np.ndarray] = []
        y_parts: list[np.ndarray] = []

        rows_in = 0
        rows_out = 0

        for chunk in chunks:
            rows_in += int(chunk.shape[0])
            chunk = self._drop_unnamed_index_col(chunk)

            if self._target_column not in chunk.columns:
                raise ValueError(
                    f"Target column '{self._target_column}' not found. "
                    f"Available columns: {list(chunk.columns)}"
                )

            # --- target: salary
            y = self._parse_salary(chunk[self._target_column])
            keep_salary = ~np.isnan(y)
            if keep_salary.sum() == 0:
                continue

            sub = chunk.loc[keep_salary].copy()
            y = y[keep_salary]

            # --- gender/age extraction
            gender, age = self._parse_gender_age(sub.get("Пол, возраст"))
            keep_features = ~np.isnan(gender) & ~np.isnan(age)
            if keep_features.sum() == 0:
                continue

            idx = np.nonzero(keep_features)[0]
            sub = sub.iloc[idx].copy()
            y = y[keep_features]
            gender = gender[keep_features]
            age = age[keep_features]

            # --- sanitize categoricals 
            city = self._sanitize_text_series(sub.get("Город"))
            employment = self._sanitize_text_series(sub.get("Занятость"))
            schedule = self._sanitize_text_series(sub.get("График"))
            auto = self._sanitize_text_series(sub.get("Авто"))

            # reduce city cardinality: take "first token" before comma
            city = city.str.split(",").str[0].fillna("unknown")

            city_code = self._city.encode(city)
            employment_code = self._employment.encode(employment)
            schedule_code = self._schedule.encode(schedule)
            auto_code = self._auto.encode(auto)

            x = np.column_stack(
                [
                    gender.astype(np.float32),
                    age.astype(np.float32),
                    city_code.astype(np.float32),
                    employment_code.astype(np.float32),
                    schedule_code.astype(np.float32),
                    auto_code.astype(np.float32),
                ]
            )

            x_parts.append(x)
            y_parts.append(y.astype(np.float32))
            rows_out += int(x.shape[0])

        if rows_out == 0:
            raise ValueError("No valid rows after preprocessing (empty output)")

        ctx.x = np.vstack(x_parts)
        ctx.y = np.concatenate(y_parts)

        ctx.meta["rows_in"] = rows_in
        ctx.meta["rows_out"] = rows_out
        ctx.meta["target_column"] = self._target_column
        ctx.meta["feature_names"] = [
            "gender",
            "age",
            "city_code",
            "employment_code",
            "schedule_code",
            "auto_code",
        ]
        ctx.meta["encoding"] = {
            "categoricals": "ordinal (incremental mapping)",
            "gender": "binary (man=1, woman=0)",
        }
        ctx.meta["encoders"] = {
            "city_mapping": self._city.mapping,
            "employment_mapping": self._employment.mapping,
            "schedule_mapping": self._schedule.mapping,
            "auto_mapping": self._auto.mapping,
        }
        return ctx

    @staticmethod
    def _drop_unnamed_index_col(df: pd.DataFrame) -> pd.DataFrame:
        unnamed = [c for c in df.columns if str(c).startswith("Unnamed")]
        return df.drop(columns=unnamed) if unnamed else df

    def _sanitize_text_series(self, series: pd.Series | None) -> pd.Series:
        if series is None:
            return pd.Series([], dtype="string")

        s = series.astype("string").fillna("unknown")

        # BOM, NBSP, control chars, etc.
        s = s.str.replace("\ufeff", "", regex=False) 
        s = s.str.replace("\xa0", " ", regex=False)   
        s = s.str.replace(self._CONTROL_CHARS_RE, " ", regex=True)
        s = s.str.replace(self._MULTI_SPACE_RE, " ", regex=True)
        s = s.str.strip()

        s = s.replace("", "unknown")
        return s

    @staticmethod
    def _parse_salary(series: pd.Series) -> np.ndarray:
        s = series.astype("string").fillna("")
        s = s.str.replace("\ufeff", "", regex=False).str.replace("\xa0", " ", regex=False)
        digits = s.str.replace(r"\D+", "", regex=True)
        digits = digits.replace("", pd.NA)
        return digits.astype("float64").to_numpy()

    @staticmethod
    def _parse_gender_age(series: pd.Series | None) -> tuple[np.ndarray, np.ndarray]:
        if series is None:
            return (
                np.full(0, np.nan, dtype=np.float64),
                np.full(0, np.nan, dtype=np.float64),
            )

        s = series.astype("string").fillna("")
        s = s.str.replace("\ufeff", "", regex=False).str.replace("\xa0", " ", regex=False)

        gender = np.where(s.str.contains("Мужчина", na=False), 1.0, np.nan)
        gender = np.where(s.str.contains("Женщина", na=False), 0.0, gender)

        age = s.str.extract(r"(\d+)\s*год", expand=False)
        age = age.fillna(s.str.extract(r"(\d+)\s*лет", expand=False))
        age_num = pd.to_numeric(age, errors="coerce").to_numpy(dtype=np.float64)

        return gender.astype(np.float64), age_num


class FinalizeFeaturesHandler(Handler):

    def __init__(self, *, drop_duplicates: bool, scale_age_minmax: bool) -> None:
        super().__init__()
        self._drop_duplicates = drop_duplicates
        self._scale_age_minmax = scale_age_minmax

    def _handle(self, ctx: PipelineContext) -> PipelineContext:
        if ctx.x is None or ctx.y is None:
            raise ValueError("FinalizeFeaturesHandler requires prepared X and y")

        x = ctx.x
        y = ctx.y

        if self._drop_duplicates:
            df = pd.DataFrame(x, columns=ctx.meta.get("feature_names", None))
            df["__y__"] = y
            before = df.shape[0]
            df = df.drop_duplicates()
            after = df.shape[0]

            y = df["__y__"].to_numpy(dtype=np.float32)
            x = df.drop(columns="__y__").to_numpy(dtype=np.float32)

            ctx.meta["dropped_duplicates"] = int(before - after)

        if self._scale_age_minmax:
            age = x[:, 1].astype(np.float32)
            min_v = float(np.min(age))
            max_v = float(np.max(age))

            if max_v > min_v:
                x[:, 1] = (age - min_v) / (max_v - min_v)
                ctx.meta["age_scaling"] = {"type": "minmax", "min": min_v, "max": max_v}
            else:
                ctx.meta["age_scaling"] = {"type": "minmax", "min": min_v, "max": max_v}

        ctx.x = x
        ctx.y = y
        return ctx


class SaveArtifactsHandler(Handler):

    def _handle(self, ctx: PipelineContext) -> PipelineContext:
        if ctx.x is None or ctx.y is None:
            raise ValueError("X/y arrays are not ready to be saved")

        ctx.output_dir.mkdir(parents=True, exist_ok=True)

        x_path = ctx.output_dir / "x_data.npy"
        y_path = ctx.output_dir / "y_data.npy"
        meta_path = ctx.output_dir / "meta.json"

        np.save(x_path, ctx.x)
        np.save(y_path, ctx.y)

        meta_to_save = dict(ctx.meta)
        meta_to_save.pop("chunks", None)


        with meta_path.open("w", encoding="utf-8") as f:
           json.dump(meta_to_save, f, ensure_ascii=False, indent=2)

        ctx.meta["x_path"] = str(x_path)
        ctx.meta["y_path"] = str(y_path)
        ctx.meta["meta_path"] = str(meta_path)
        return ctx
