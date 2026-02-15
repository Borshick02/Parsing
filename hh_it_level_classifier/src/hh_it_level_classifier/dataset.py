from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator
from dataclasses import asdict


import numpy as np
import pandas as pd

from hh_it_level_classifier.labels import infer_level
from hh_it_level_classifier.utils import normalize_text, try_read_csv_encoding


IT_KEYWORDS = [
    "разработчик",
    "developer",
    "программист",
    "software",
    "backend",
    "frontend",
    "fullstack",
    "full stack",
    "python",
    "java",
    "javascript",
    "js ",
    "golang",
    "go ",
    "c#",
    "c++",
    "ios",
    "android",
    "qa",
    "тестировщик",
    "devops",
    "data engineer",
    "ml",
    "machine learning",
]


@dataclass(frozen=True, slots=True)
class PreparedRow:
    age: float | None
    salary_rub: float | None
    exp_years: float | None
    city: str | None
    position_text: str
    employment: str | None
    schedule: str | None
    skills_text: str
    label: str


def _pick_first_existing(columns: Iterable[str], candidates: list[str]) -> str | None:
    cols = set(columns)
    for c in candidates:
        if c in cols:
            return c
    return None


def _parse_gender_age(s: str | None) -> tuple[str | None, float | None]:
    """
    Example: "Мужчина ,  42 года , родился ..."
    """
    if not s:
        return None, None
    t = normalize_text(s)

    gender: str | None = None
    if "мужчина" in t:
        gender = "male"
    elif "женщина" in t:
        gender = "female"

    m = re.search(r"(\d+)\s*год", t)
    age = float(m.group(1)) if m else None
    return gender, age


def _parse_salary_rub(x: object) -> float | None:
    """
    Examples:
      "27 000 руб."
      "120000"
    PoC: we only parse numbers and assume RUB.
    """
    if x is None:
        return None
    s = str(x)
    s = s.replace("\xa0", " ").strip()
    if s == "" or s.lower() == "nan":
        return None

    digits = re.findall(r"\d+", s)
    if not digits:
        return None
    value = float("".join(digits))
    return value


def _parse_experience_years(x: object) -> float | None:
    """
    Experience column often contains: "Опыт работы 6 лет 1 месяц..."
    We parse years + months from the text.
    """
    if x is None:
        return None
    s = str(x)
    s = s.replace("\xa0", " ").strip()
    if s == "" or s.lower() == "nan":
        return None

    t = normalize_text(s)

    y = 0
    m = 0

    my = re.search(r"(\d+)\s*лет", t)
    if my:
        y = int(my.group(1))
    mm = re.search(r"(\d+)\s*месяц", t)
    if mm:
        m = int(mm.group(1))

    if y == 0 and m == 0:
        return None
    return float(y) + float(m) / 12.0


def _is_it_resume(position_text: str) -> bool:
    t = normalize_text(position_text)
    return any(k in t for k in IT_KEYWORDS)


def iter_prepared_rows(
    csv_path: Path,
    target_column_hint: str,
    chunksize: int,
    limit_rows: int | None,
) -> Iterator[PreparedRow]:
    encodings = try_read_csv_encoding(csv_path)
    last_error: Exception | None = None

    for enc in encodings:
        try:
            # We read header to know columns
            head = pd.read_csv(csv_path, nrows=0, encoding=enc)
            columns = head.columns.tolist()

            col_gender_age = _pick_first_existing(columns, ["Пол, возраст", "Пол, Возраст", "Пол, возраст "])
            col_salary = _pick_first_existing(columns, [target_column_hint, "ЗП", "Зарплата", "salary"])
            col_city = _pick_first_existing(columns, ["Город", "Город ", "city"])
            col_employment = _pick_first_existing(columns, ["Занятость", "employment"])
            col_schedule = _pick_first_existing(columns, ["График", "schedule"])
            col_exp = _pick_first_existing(columns, ["Опыт (двойное нажатие для полной версии)", "Опыт работы", "Опыт", "experience"])
            col_pos_want = _pick_first_existing(columns, ["Ищет работу на должность:", "Ищет работу на должность", "Должность", "position"])
            col_pos_last = _pick_first_existing(columns, ["Последеняя/нынешняя должность", "Последняя/нынешняя должность", "Последняя должность"])

            # Optional skills column (may not exist)
            col_skills = _pick_first_existing(columns, ["Ключевые навыки", "Навыки", "skills"])

            usecols = [c for c in [col_gender_age, col_salary, col_city, col_employment, col_schedule, col_exp, col_pos_want, col_pos_last, col_skills] if c]
            if not usecols:
                raise ValueError("Could not detect required columns in hh.csv")

            seen = 0
            for chunk in pd.read_csv(csv_path, encoding=enc, chunksize=chunksize, usecols=usecols):
                for _, row in chunk.iterrows():
                    if limit_rows is not None and seen >= limit_rows:
                        return

                    gender, age = _parse_gender_age(row.get(col_gender_age) if col_gender_age else None)
                    salary = _parse_salary_rub(row.get(col_salary) if col_salary else None)
                    city = str(row.get(col_city)).strip() if col_city and row.get(col_city) is not None else None
                    employment = str(row.get(col_employment)).strip() if col_employment and row.get(col_employment) is not None else None
                    schedule = str(row.get(col_schedule)).strip() if col_schedule and row.get(col_schedule) is not None else None
                    exp_years = _parse_experience_years(row.get(col_exp) if col_exp else None)

                    pos1 = str(row.get(col_pos_want)).strip() if col_pos_want and row.get(col_pos_want) is not None else ""
                    pos2 = str(row.get(col_pos_last)).strip() if col_pos_last and row.get(col_pos_last) is not None else ""
                    position_text = (pos1 + " " + pos2).strip()

                    if not position_text:
                        continue
                    if not _is_it_resume(position_text):
                        continue

                    skills = ""
                    if col_skills and row.get(col_skills) is not None:
                        skills = str(row.get(col_skills)).strip()

                    # Add gender word into text to help TF-IDF slightly
                    full_text = f"{position_text} {skills}".strip()

                    label_res = infer_level(position_text=position_text, exp_years=exp_years)
                    if label_res.label is None:
                        continue

                    # gender can be useful as text too, but we will encode it as categorical
                    # Here we keep it in skills_text, and store separately later if needed.
                    yield PreparedRow(
                        age=age,
                        salary_rub=salary,
                        exp_years=exp_years,
                        city=city,
                        position_text=position_text,
                        employment=employment,
                        schedule=schedule,
                        skills_text=full_text,
                        label=label_res.label,
                    )
                    seen += 1

            return

        except Exception as e:
            last_error = e
            continue

    raise RuntimeError(f"Failed to read CSV with known encodings. Last error: {last_error}")


def build_dataframe(
    csv_path: Path,
    target_column_hint: str,
    chunksize: int,
    limit_rows: int | None,
) -> pd.DataFrame:
    rows = list(
        iter_prepared_rows(
            csv_path=csv_path,
            target_column_hint=target_column_hint,
            chunksize=chunksize,
            limit_rows=limit_rows,
        )
    )
    if not rows:
        return asdict()

    df = pd.DataFrame([asdict(r) for r in rows])
    return df


def class_counts(df: pd.DataFrame) -> dict[str, int]:
    if df.empty:
        return {}
    vc = df["label"].value_counts()
    return {str(k): int(v) for k, v in vc.items()}
