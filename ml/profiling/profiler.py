"""
AutoClin Engine Data Profiler
Produces a comprehensive statistical fingerprint: per-column stats,
distribution analysis, missingness mapping, correlation matrix,
and preliminary noise estimate.
"""
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import pandas as pd
from scipy import stats as sp_stats


@dataclass
class ColumnProfileResult:
    name: str
    dtype: str
    count: int
    null_count: int
    null_rate: float
    unique_count: int
    cardinality_ratio: float
    mean: Optional[float] = None
    median: Optional[float] = None
    mode: Optional[Any] = None
    std: Optional[float] = None
    iqr: Optional[float] = None
    skewness: Optional[float] = None
    kurtosis: Optional[float] = None
    min_val: Optional[float] = None
    max_val: Optional[float] = None
    q1: Optional[float] = None
    q3: Optional[float] = None
    top_values: Optional[list[dict]] = None
    normality_pvalue: Optional[float] = None
    distribution_fit: Optional[str] = None


@dataclass
class MissingnessPattern:
    pattern_type: str  # MCAR, MAR, MNAR, unknown
    columns_with_correlated_missingness: list[tuple[str, str, float]]
    monotone: bool = False


@dataclass
class DataProfileResult:
    row_count: int
    col_count: int
    overall_missingness: float
    estimated_duplicate_proportion: float
    type_distribution: dict[str, float]
    noise_estimate: float
    columns: list[ColumnProfileResult]
    missingness_pattern: MissingnessPattern
    correlation_matrix: Optional[dict] = None
    impossible_value_count: int = 0


class DataProfiler:
    def __init__(self, max_categories: int = 50, normality_sample: int = 5000):
        self.max_categories = max_categories
        self.normality_sample = normality_sample

    def profile(self, df: pd.DataFrame, schema=None, duplicate_rate: float = 0.0) -> DataProfileResult:
        col_profiles = []
        numeric_cols = []
        impossible_count = 0

        for col in df.columns:
            cp = self._profile_column(df[col], col)
            col_profiles.append(cp)
            if cp.dtype == "numeric":
                numeric_cols.append(col)

        if schema:
            for cs in schema.columns:
                impossible_count += len(cs.impossible_values)

        total_cells = len(df) * len(df.columns)
        total_nulls = sum(df[c].isna().sum() for c in df.columns)
        overall_miss = total_nulls / total_cells if total_cells > 0 else 0.0

        type_counts: dict[str, int] = {}
        for cp in col_profiles:
            type_counts[cp.dtype] = type_counts.get(cp.dtype, 0) + 1
        n_cols = len(df.columns)
        type_dist = {t: c / n_cols for t, c in type_counts.items()} if n_cols else {}

        corr_data = None
        if len(numeric_cols) >= 2:
            try:
                numeric_df = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
                corr_matrix = numeric_df.corr(method="spearman")
                corr_data = {"columns": numeric_cols, "values": corr_matrix.fillna(0).values.tolist()}
            except Exception:
                pass

        miss_pattern = self._analyze_missingness(df)
        noise_from_impossible = impossible_count / len(df) if len(df) > 0 else 0.0
        noise_from_tails = self._estimate_tail_noise(df, numeric_cols)
        noise_estimate = min(noise_from_impossible + noise_from_tails, 1.0)

        return DataProfileResult(
            row_count=len(df), col_count=len(df.columns),
            overall_missingness=overall_miss, estimated_duplicate_proportion=duplicate_rate,
            type_distribution=type_dist, noise_estimate=noise_estimate,
            columns=col_profiles, missingness_pattern=miss_pattern,
            correlation_matrix=corr_data, impossible_value_count=impossible_count,
        )

    def _profile_column(self, series: pd.Series, name: str) -> ColumnProfileResult:
        total = len(series)
        null_count = int(series.isna().sum())
        null_rate = null_count / total if total > 0 else 0.0
        non_null = series.dropna()
        unique_count = int(non_null.nunique())
        cardinality = unique_count / total if total > 0 else 0.0

        numeric = pd.to_numeric(non_null, errors="coerce")
        numeric_valid = numeric.dropna()
        is_numeric = len(numeric_valid) > 0.8 * len(non_null) if len(non_null) > 0 else False

        if is_numeric and len(numeric_valid) > 0:
            q1 = float(numeric_valid.quantile(0.25))
            q3 = float(numeric_valid.quantile(0.75))
            norm_p = None
            if 3 < len(numeric_valid) <= self.normality_sample:
                try:
                    _, norm_p = sp_stats.shapiro(
                        numeric_valid.sample(min(len(numeric_valid), self.normality_sample))
                    )
                    norm_p = float(norm_p)
                except Exception:
                    pass

            return ColumnProfileResult(
                name=name, dtype="numeric", count=total,
                null_count=null_count, null_rate=null_rate,
                unique_count=unique_count, cardinality_ratio=cardinality,
                mean=float(numeric_valid.mean()), median=float(numeric_valid.median()),
                mode=float(numeric_valid.mode().iloc[0]) if len(numeric_valid.mode()) > 0 else None,
                std=float(numeric_valid.std()) if len(numeric_valid) > 1 else 0.0,
                iqr=q3 - q1,
                skewness=float(numeric_valid.skew()) if len(numeric_valid) > 2 else None,
                kurtosis=float(numeric_valid.kurtosis()) if len(numeric_valid) > 3 else None,
                min_val=float(numeric_valid.min()), max_val=float(numeric_valid.max()),
                q1=q1, q3=q3, normality_pvalue=norm_p,
                distribution_fit="normal" if norm_p and norm_p > 0.05 else "non-normal",
            )
        else:
            top_vals = None
            if len(non_null) > 0 and unique_count <= self.max_categories * 2:
                vc = non_null.value_counts().head(self.max_categories)
                top_vals = [
                    {"value": str(v), "count": int(c), "pct": round(c / len(non_null) * 100, 2)}
                    for v, c in vc.items()
                ]
            dtype = "categorical"
            if cardinality > 0.9:
                dtype = "free_text" if non_null.astype(str).str.len().mean() > 30 else "identifier"

            return ColumnProfileResult(
                name=name, dtype=dtype, count=total,
                null_count=null_count, null_rate=null_rate,
                unique_count=unique_count, cardinality_ratio=cardinality,
                mode=str(non_null.mode().iloc[0]) if len(non_null.mode()) > 0 else None,
                top_values=top_vals,
            )

    def _analyze_missingness(self, df: pd.DataFrame) -> MissingnessPattern:
        null_matrix = df.isna().astype(int)
        correlated_pairs = []
        null_cols = [c for c in df.columns if df[c].isna().any()]
        if len(null_cols) >= 2:
            for i in range(len(null_cols)):
                for j in range(i + 1, min(len(null_cols), i + 20)):
                    try:
                        corr = null_matrix[null_cols[i]].corr(null_matrix[null_cols[j]])
                        if abs(corr) > 0.3:
                            correlated_pairs.append((null_cols[i], null_cols[j], round(corr, 3)))
                    except Exception:
                        pass

        monotone = self._check_monotone(df, null_cols)
        if len(correlated_pairs) == 0:
            pattern_type = "MCAR"
        elif monotone:
            pattern_type = "MNAR"
        else:
            pattern_type = "MAR"

        return MissingnessPattern(pattern_type=pattern_type,
                                  columns_with_correlated_missingness=correlated_pairs, monotone=monotone)

    def _check_monotone(self, df: pd.DataFrame, null_cols: list[str]) -> bool:
        if len(null_cols) < 3:
            return False
        null_rates = sorted([df[c].isna().mean() for c in null_cols])
        diffs = [null_rates[i + 1] - null_rates[i] for i in range(len(null_rates) - 1)]
        return any(d > 0.1 for d in diffs)

    def _estimate_tail_noise(self, df: pd.DataFrame, numeric_cols: list[str]) -> float:
        if not numeric_cols or len(df) == 0:
            return 0.0
        total_outlier = 0
        total_cells = 0
        for col in numeric_cols:
            numeric = pd.to_numeric(df[col], errors="coerce").dropna()
            if len(numeric) < 10:
                continue
            q1, q3 = numeric.quantile(0.25), numeric.quantile(0.75)
            iqr = q3 - q1
            if iqr == 0:
                continue
            lower, upper = q1 - 3.0 * iqr, q3 + 3.0 * iqr
            total_outlier += ((numeric < lower) | (numeric > upper)).sum()
            total_cells += len(numeric)
        return total_outlier / total_cells if total_cells > 0 else 0.0
