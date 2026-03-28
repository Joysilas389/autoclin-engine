"""
AutoClin Engine Schema Inference Engine
Automatically infers column types, detects mixed types, hidden nulls,
impossible values, and sentinel patterns.
"""
import re
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import pandas as pd
from datetime import datetime


SENTINEL_NULLS = {
    "", " ", "  ", "N/A", "n/a", "NA", "na", "NULL", "null", "None", "none",
    "MISSING", "missing", ".", "..", "---", "-", "NaN", "nan", "NaT",
    "-999", "-9999", "9999", "99999", "-1", "999", "888", "777",
    "UNK", "unk", "UNKNOWN", "unknown", "NOT AVAILABLE", "not available",
    "NOT APPLICABLE", "not applicable", "ND", "nd",
}

DATE_PATTERNS = [
    r"\d{4}-\d{2}-\d{2}",           # 2024-01-15
    r"\d{2}/\d{2}/\d{4}",           # 01/15/2024
    r"\d{2}-\d{2}-\d{4}",           # 01-15-2024
    r"\d{4}/\d{2}/\d{2}",           # 2024/01/15
    r"\d{2}\w{3}\d{4}",             # 15Jan2024 (clinical)
    r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}",  # ISO with time
]


@dataclass
class ColumnSchema:
    """Inferred schema for a single column."""
    name: str
    inferred_type: str  # numeric, categorical, datetime, boolean, free_text, identifier
    type_confidence: float = 0.0
    mixed_type: bool = False
    mixed_type_proportion: float = 0.0
    nullity_rate: float = 0.0
    hidden_null_count: int = 0
    hidden_null_patterns: list[str] = field(default_factory=list)
    unique_count: int = 0
    cardinality_ratio: float = 0.0  # unique / total
    sample_values: list[Any] = field(default_factory=list)
    impossible_values: list[Any] = field(default_factory=list)
    stats: Optional[dict] = None


@dataclass
class DatasetSchema:
    """Complete inferred schema for the dataset."""
    columns: list[ColumnSchema]
    row_count: int
    col_count: int
    overall_nullity: float
    type_distribution: dict[str, float]


class SchemaInferenceEngine:
    """
    Infers types, detects mixed types, hidden nulls, and impossible values.
    Operates on a pandas DataFrame (or chunk).
    """

    def __init__(self, sample_size: int = 5):
        self.sample_size = sample_size

    def infer(self, df: pd.DataFrame) -> DatasetSchema:
        columns = []
        for col in df.columns:
            col_schema = self._infer_column(df[col], col)
            columns.append(col_schema)

        row_count = len(df)
        col_count = len(df.columns)
        total_cells = row_count * col_count
        total_null = sum(df[c].isna().sum() for c in df.columns)
        total_hidden = sum(c.hidden_null_count for c in columns)
        overall_nullity = (total_null + total_hidden) / total_cells if total_cells > 0 else 0.0

        type_counts: dict[str, int] = {}
        for c in columns:
            type_counts[c.inferred_type] = type_counts.get(c.inferred_type, 0) + 1
        type_distribution = {t: cnt / col_count for t, cnt in type_counts.items()} if col_count > 0 else {}

        return DatasetSchema(
            columns=columns,
            row_count=row_count,
            col_count=col_count,
            overall_nullity=overall_nullity,
            type_distribution=type_distribution,
        )

    def _infer_column(self, series: pd.Series, name: str) -> ColumnSchema:
        total = len(series)
        null_count = series.isna().sum()
        nullity_rate = null_count / total if total > 0 else 0.0

        # Detect hidden nulls
        hidden_nulls = []
        hidden_count = 0
        if series.dtype == object:
            str_vals = series.dropna().astype(str)
            for val in str_vals:
                stripped = val.strip()
                if stripped in SENTINEL_NULLS:
                    hidden_count += 1
                    if stripped not in hidden_nulls:
                        hidden_nulls.append(stripped)

        non_null = series.dropna()
        unique_count = non_null.nunique()
        cardinality = unique_count / total if total > 0 else 0.0

        # Sample values
        sample = non_null.head(self.sample_size).tolist() if len(non_null) > 0 else []

        # Type inference
        inferred_type, confidence, mixed, mixed_prop = self._detect_type(series, name)

        # Impossible value detection
        impossible = self._detect_impossible(series, inferred_type, name)

        # Compute stats for numeric columns
        stats = None
        if inferred_type == "numeric":
            numeric_series = pd.to_numeric(non_null, errors="coerce").dropna()
            if len(numeric_series) > 0:
                stats = {
                    "mean": float(numeric_series.mean()),
                    "median": float(numeric_series.median()),
                    "std": float(numeric_series.std()) if len(numeric_series) > 1 else 0.0,
                    "min": float(numeric_series.min()),
                    "max": float(numeric_series.max()),
                    "q25": float(numeric_series.quantile(0.25)),
                    "q75": float(numeric_series.quantile(0.75)),
                    "skewness": float(numeric_series.skew()) if len(numeric_series) > 2 else 0.0,
                    "kurtosis": float(numeric_series.kurtosis()) if len(numeric_series) > 3 else 0.0,
                }

        return ColumnSchema(
            name=name,
            inferred_type=inferred_type,
            type_confidence=confidence,
            mixed_type=mixed,
            mixed_type_proportion=mixed_prop,
            nullity_rate=nullity_rate,
            hidden_null_count=hidden_count,
            hidden_null_patterns=hidden_nulls,
            unique_count=unique_count,
            cardinality_ratio=cardinality,
            sample_values=sample,
            impossible_values=impossible,
            stats=stats,
        )

    def _detect_type(self, series: pd.Series, name: str) -> tuple[str, float, bool, float]:
        """
        Returns (inferred_type, confidence, is_mixed, mixed_proportion).
        """
        non_null = series.dropna()
        if len(non_null) == 0:
            return "unknown", 0.0, False, 0.0

        total = len(non_null)

        # Check if boolean
        unique_vals = set(non_null.astype(str).str.strip().str.lower())
        bool_vals = {"0", "1", "true", "false", "yes", "no", "y", "n", "t", "f"}
        if unique_vals.issubset(bool_vals) and len(unique_vals) <= 4:
            return "boolean", 0.95, False, 0.0

        # Check if numeric
        numeric_coerced = pd.to_numeric(non_null, errors="coerce")
        numeric_valid = numeric_coerced.notna().sum()
        numeric_ratio = numeric_valid / total

        if numeric_ratio > 0.9:
            mixed = numeric_ratio < 1.0
            mixed_prop = 1.0 - numeric_ratio
            return "numeric", numeric_ratio, mixed, mixed_prop

        # Check if datetime
        date_matches = 0
        str_vals = non_null.astype(str)
        for pattern in DATE_PATTERNS:
            date_matches += str_vals.str.match(pattern, na=False).sum()
        date_ratio = min(date_matches / total, 1.0)
        if date_ratio > 0.7:
            return "datetime", date_ratio, date_ratio < 0.95, 1.0 - date_ratio

        # Check if identifier (high cardinality, often alphanumeric patterns)
        name_lower = name.lower()
        id_keywords = ["id", "code", "key", "number", "num", "no", "idx", "index"]
        if any(kw in name_lower for kw in id_keywords) and cardinality_high(non_null, total):
            return "identifier", 0.85, False, 0.0

        # Check if free text (high cardinality + long strings)
        if series.dtype == object:
            avg_len = str_vals.str.len().mean()
            if avg_len > 50 and non_null.nunique() / total > 0.8:
                return "free_text", 0.8, False, 0.0

        # Default: categorical
        return "categorical", 0.8, False, 0.0

    def _detect_impossible(self, series: pd.Series, dtype: str, name: str) -> list[Any]:
        """Detect impossible values based on type and name heuristics."""
        impossible = []
        if dtype != "numeric":
            return impossible

        name_lower = name.lower()
        numeric = pd.to_numeric(series, errors="coerce").dropna()
        if len(numeric) == 0:
            return impossible

        # Age checks
        if "age" in name_lower:
            mask = (numeric < 0) | (numeric > 150)
            impossible.extend(numeric[mask].tolist()[:10])

        # Heart rate
        if any(kw in name_lower for kw in ["hr", "heart_rate", "heartrate", "pulse"]):
            mask = (numeric < 0) | (numeric > 400)
            impossible.extend(numeric[mask].tolist()[:10])

        # Blood pressure
        if any(kw in name_lower for kw in ["sbp", "systolic", "sys_bp", "dbp", "diastolic", "dia_bp", "bp"]):
            mask = (numeric < 0) | (numeric > 400)
            impossible.extend(numeric[mask].tolist()[:10])

        # Temperature (C or F)
        if any(kw in name_lower for kw in ["temp", "temperature"]):
            mask = (numeric < 20) | (numeric > 115)  # covers both C and F broadly
            impossible.extend(numeric[mask].tolist()[:10])

        # Weight
        if any(kw in name_lower for kw in ["weight", "wt", "mass"]):
            mask = (numeric < 0) | (numeric > 700)
            impossible.extend(numeric[mask].tolist()[:10])

        # Height (cm)
        if any(kw in name_lower for kw in ["height", "ht", "stature"]):
            mask = (numeric < 0) | (numeric > 300)
            impossible.extend(numeric[mask].tolist()[:10])

        # Generic: negative values in columns that should be positive
        positive_keywords = ["count", "dose", "volume", "duration", "length", "width"]
        if any(kw in name_lower for kw in positive_keywords):
            mask = numeric < 0
            impossible.extend(numeric[mask].tolist()[:10])

        return impossible


def cardinality_high(series: pd.Series, total: int) -> bool:
    return series.nunique() / total > 0.8 if total > 0 else False
