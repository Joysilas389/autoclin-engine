"""
AutoClin Engine Preprocessing Module
Type coercion, unit harmonization, date normalization, and encoding.
"""
import re
import numpy as np
import pandas as pd
from typing import Optional


# Known unit conversion factors: (from_unit, to_unit) -> multiplier
UNIT_CONVERSIONS = {
    ("mmol/L", "mg/dL", "glucose"): 18.0,
    ("mg/dL", "mmol/L", "glucose"): 1.0 / 18.0,
    ("g/L", "g/dL", "hemoglobin"): 0.1,
    ("g/dL", "g/L", "hemoglobin"): 10.0,
    ("kg", "lb", "weight"): 2.20462,
    ("lb", "kg", "weight"): 0.453592,
    ("cm", "in", "height"): 0.393701,
    ("in", "cm", "height"): 2.54,
    ("°F", "°C", "temperature"): None,  # special formula
    ("°C", "°F", "temperature"): None,
}


class TypeCoercer:
    """Coerce columns to their inferred types with error trapping."""

    def coerce_numeric(self, series: pd.Series) -> pd.Series:
        return pd.to_numeric(series, errors="coerce")

    def coerce_datetime(self, series: pd.Series) -> pd.Series:
        return pd.to_datetime(series, errors="coerce", infer_datetime_format=True)

    def coerce_boolean(self, series: pd.Series) -> pd.Series:
        mapping = {"true": True, "false": False, "yes": True, "no": False,
                   "1": True, "0": False, "y": True, "n": False, "t": True, "f": False}
        return series.astype(str).str.strip().str.lower().map(mapping)

    def normalize_categorical(self, series: pd.Series) -> pd.Series:
        """Standardize categorical values: strip whitespace, normalize case."""
        return series.astype(str).str.strip().str.lower().replace({"nan": np.nan, "none": np.nan})


class UnitHarmonizer:
    """Detect and correct unit mismatches in clinical variables."""

    def __init__(self, reference_ranges: Optional[dict] = None):
        self.ref = reference_ranges or {}

    def detect_unit_mismatch(self, series: pd.Series, expected_range: dict) -> dict:
        """
        Check if a numeric column has values suggesting mixed units.
        Returns: {mismatch: bool, likely_factor: float, affected_indices: list}
        """
        numeric = pd.to_numeric(series, errors="coerce").dropna()
        if len(numeric) < 10:
            return {"mismatch": False}

        rmin, rmax = expected_range.get("min", -np.inf), expected_range.get("max", np.inf)
        in_range = (numeric >= rmin) & (numeric <= rmax)
        out_range = ~in_range
        out_count = out_range.sum()

        if out_count == 0 or out_count == len(numeric):
            return {"mismatch": False}

        out_vals = numeric[out_range]
        in_vals = numeric[in_range]

        # Check common conversion factors
        for factor in [10, 18, 0.1, 1/18, 2.54, 0.3937]:
            converted = out_vals * factor
            newly_in = (converted >= rmin) & (converted <= rmax)
            if newly_in.mean() > 0.7:
                return {
                    "mismatch": True,
                    "likely_factor": factor,
                    "affected_indices": out_vals.index.tolist(),
                    "affected_count": int(out_count),
                }

        return {"mismatch": False, "affected_count": int(out_count)}

    def harmonize(self, series: pd.Series, factor: float, indices: list) -> pd.Series:
        """Apply unit conversion to specific indices."""
        result = series.copy()
        for idx in indices:
            if idx in result.index:
                val = pd.to_numeric(result.at[idx], errors="coerce")
                if not pd.isna(val):
                    result.at[idx] = val * factor
        return result


class DateNormalizer:
    """Normalize dates to ISO-8601 format."""

    def normalize(self, series: pd.Series) -> pd.Series:
        parsed = pd.to_datetime(series, errors="coerce", infer_datetime_format=True)
        return parsed.dt.strftime("%Y-%m-%d").replace("NaT", np.nan)


class FeatureEncoder:
    """Encode features for ML: numeric scaling, categorical one-hot."""

    def encode(self, df: pd.DataFrame, profile) -> tuple[np.ndarray, list[str]]:
        """Convert DataFrame to ML-ready numeric matrix."""
        from sklearn.preprocessing import StandardScaler

        parts = []
        names = []

        for cp in profile.columns:
            col = cp.name
            if col not in df.columns:
                continue

            if cp.dtype == "numeric":
                numeric = pd.to_numeric(df[col], errors="coerce")
                median_val = numeric.median()
                numeric = numeric.fillna(median_val if not pd.isna(median_val) else 0)
                parts.append(numeric.values.reshape(-1, 1))
                names.append(col)

            elif cp.dtype == "categorical" and cp.unique_count <= 50:
                dummies = pd.get_dummies(
                    df[col].fillna("__MISSING__"), prefix=col, drop_first=True
                )
                if dummies.shape[1] <= 20:
                    parts.append(dummies.values)
                    names.extend(dummies.columns.tolist())

        if not parts:
            return np.zeros((len(df), 1)), ["__empty__"]

        X = np.hstack(parts).astype(np.float64)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        return X, names
