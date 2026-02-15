from __future__ import annotations

from dataclasses import dataclass

from hh_it_level_classifier.utils import normalize_text


@dataclass(frozen=True, slots=True)
class LabelResult:
    label: str | None
    reason: str


# --- keyword dictionaries (PoC, but closer to real HH titles) ---

JUNIOR_KW: list[str] = [
    "junior",
    "jr",
    "джуниор",
    "джун",
    "младший",
    "начинающий",
    "стажер",
    "стажёр",
    "intern",
    "trainee",
]

MIDDLE_KW: list[str] = [
    "middle",
    "mid",
    "midlevel",
    "mid-level",
    "мидл",
    "мид",
    "миддл",
    "middle+",
    "мидл+",
    "regular",
]

SENIOR_KW: list[str] = [
    "senior",
    "sr",
    "сеньор",
    "старший",
    "ведущий",
    "главный",
    "lead",
    "тимлид",
    "архитектор",
    "architect",
    "principal",
    "staff",
]

SENIOR_PHRASES: list[str] = [
    "team lead",
    "tech lead",
]


def _tokenize(text: str) -> set[str]:
    # normalize_text() should lowercase + trim + collapse spaces, etc.
    # Tokenization is intentionally simple for robustness.
    return set(text.split())


def infer_level(
    position_text: str | None,
    exp_years: float | None,
) -> LabelResult:
    """
    PoC labeling rules:
    1) Try keywords in position text (title).
       - Multi-word phrases (e.g., "team lead") via substring search.
       - Single-word keywords via token matching.
    2) Fallback by experience:
       <2 -> junior, 2-6 -> middle, >=6 -> senior
    """
    pos = normalize_text(position_text or "")
    tokens = _tokenize(pos)

    # multi-word phrases first (more specific)
    if any(phrase in pos for phrase in SENIOR_PHRASES):
        return LabelResult(label="senior", reason="position_keyword_senior_phrase")

    # token-based keyword checks (safer than substring)
    if any(k in tokens for k in SENIOR_KW):
        return LabelResult(label="senior", reason="position_keyword_senior")

    if any(k in tokens for k in JUNIOR_KW):
        return LabelResult(label="junior", reason="position_keyword_junior")

    if any(k in tokens for k in MIDDLE_KW):
        return LabelResult(label="middle", reason="position_keyword_middle")

    # fallback: experience-based labeling
    if exp_years is None:
        return LabelResult(label=None, reason="no_keywords_and_no_experience")

    if exp_years < 2.0:
        return LabelResult(label="junior", reason="experience_lt_2")

    if exp_years < 6.0:
        return LabelResult(label="middle", reason="experience_2_6")

    return LabelResult(label="senior", reason="experience_ge_6")
