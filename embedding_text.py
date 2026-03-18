from __future__ import annotations

import math
from typing import Any


_MISSING_STRINGS = {"", "nan", "none", "null"}


def _is_missing(value: Any) -> bool:
    if value is None:
        return True

    # Handle float NaN without depending on numpy/pandas.
    if isinstance(value, float) and math.isnan(value):
        return True

    # Best-effort pandas NA support (optional).
    try:
        import pandas as pd  # type: ignore

        try:
            if pd.isna(value):
                return True
        except Exception:
            pass
    except Exception:
        pass

    # If something already came through as a stringified missing value.
    try:
        s = str(value).strip().lower()
        if s in _MISSING_STRINGS:
            return True
    except Exception:
        return True

    return False


def _clean_text(value: Any) -> str:
    if _is_missing(value):
        return ""

    try:
        s = str(value).strip()
    except Exception:
        return ""

    # Avoid "nan" artifacts even if they came in as a literal string.
    if s.strip().lower() in _MISSING_STRINGS:
        return ""

    return s


def _get_field(row: Any, key: str) -> Any:
    # Mapping / Series-like
    try:
        return row[key]
    except Exception:
        pass

    # Attribute / namedtuple-like
    try:
        return getattr(row, key)
    except Exception:
        return None


def build_embedding_text(row: Any) -> str:
    """Build the rich text used for embedding.

    Combines Title + Tags (if present) + SemanticText.

    - Safe normalization (no NaN/None artifacts)
    - Missing columns/fields are treated as empty
    - No extra spaces
    """

    title = _clean_text(_get_field(row, "Title"))

    tags = _clean_text(_get_field(row, "Tags"))
    if not tags:
        tags = _clean_text(_get_field(row, "tags"))

    semantic_text = _clean_text(_get_field(row, "SemanticText"))

    parts = [title, tags, semantic_text]
    return " ".join([p for p in parts if p])
