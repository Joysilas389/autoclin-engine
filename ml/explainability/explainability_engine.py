"""
AutoClin Engine Explainability Framework
Generates multi-layered explanations for every flagged anomaly:
- Feature-level contributions
- Counterfactual nearest-normal ("clean twin")
- Human-readable explanation cards
- Global dataset-level summary
"""
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional, Any
from scipy.spatial.distance import cdist


@dataclass
class CleanTwin:
    """Counterfactual nearest-normal record."""
    nearest_normal_idx: int
    original_values: dict[str, Any]
    clean_values: dict[str, Any]
    differing_columns: list[str]
    distance: float


@dataclass
class ExplanationCard:
    """Human-readable explanation for a single anomaly."""
    row_index: int
    summary: str
    feature_contributions: dict[str, float]
    clean_twin: Optional[CleanTwin]
    recommended_action: str
    confidence: float
    severity: str


@dataclass
class GlobalSummary:
    """Dataset-level explanation summary."""
    narrative: str
    top_drivers: list[dict]  # [{column, contribution_pct, description}]
    anomaly_type_distribution: dict[str, int]
    total_anomalies: int
    noise_pct: float


class ExplainabilityEngine:
    """
    Generates explanations at record, feature, and dataset levels.
    """

    def generate_explanation_card(
        self,
        row_idx: int,
        df: pd.DataFrame,
        anomaly_scores: np.ndarray,
        feature_contributions: np.ndarray,
        anomaly_flags: np.ndarray,
        classified_anomaly: Any,
        clinical_mappings: list = None,
    ) -> ExplanationCard:
        """Generate a full explanation card for one flagged record."""
        columns = df.columns.tolist()
        row = df.iloc[row_idx]

        # Feature contribution dict
        if feature_contributions is not None and row_idx < len(feature_contributions):
            contribs = feature_contributions[row_idx]
            contrib_dict = {
                columns[i]: round(float(contribs[i]), 4)
                for i in np.argsort(contribs)[-5:][::-1]
                if contribs[i] > 0.01
            }
        else:
            contrib_dict = {}

        # Clean twin
        clean_twin = self._find_clean_twin(
            row_idx, df, anomaly_scores, anomaly_flags
        )

        # Build summary text
        summary = self._build_summary(
            row_idx, row, classified_anomaly, contrib_dict, clinical_mappings
        )

        return ExplanationCard(
            row_index=row_idx,
            summary=summary,
            feature_contributions=contrib_dict,
            clean_twin=clean_twin,
            recommended_action=getattr(classified_anomaly, "anomaly_type", "review"),
            confidence=getattr(classified_anomaly, "confidence", 0.5),
            severity=getattr(classified_anomaly, "severity", "medium"),
        )

    def _find_clean_twin(
        self,
        row_idx: int,
        df: pd.DataFrame,
        anomaly_scores: np.ndarray,
        anomaly_flags: np.ndarray,
    ) -> Optional[CleanTwin]:
        """Find the nearest non-anomalous record (clean twin)."""
        normal_mask = ~anomaly_flags
        if not normal_mask.any():
            return None

        # Work with numeric columns only for distance computation
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == 0:
            return None

        X = df[numeric_cols].values.copy()
        # Simple imputation for NaN (mean)
        col_means = np.nanmean(X, axis=0)
        for j in range(X.shape[1]):
            mask = np.isnan(X[:, j])
            X[mask, j] = col_means[j]

        target = X[row_idx].reshape(1, -1)
        normal_indices = np.where(normal_mask)[0]
        normal_X = X[normal_indices]

        # Compute distances
        distances = cdist(target, normal_X, metric="euclidean")[0]
        nearest_idx_in_normal = np.argmin(distances)
        nearest_original_idx = int(normal_indices[nearest_idx_in_normal])

        # Compare values
        original_row = df.iloc[row_idx]
        clean_row = df.iloc[nearest_original_idx]
        all_cols = df.columns.tolist()

        original_vals = {}
        clean_vals = {}
        differing = []

        for col in all_cols:
            orig_val = original_row.get(col)
            clean_val = clean_row.get(col)
            original_vals[col] = self._safe_value(orig_val)
            clean_vals[col] = self._safe_value(clean_val)

            if self._values_differ(orig_val, clean_val):
                differing.append(col)

        return CleanTwin(
            nearest_normal_idx=nearest_original_idx,
            original_values=original_vals,
            clean_values=clean_vals,
            differing_columns=differing[:10],
            distance=float(distances[nearest_idx_in_normal]),
        )

    def _build_summary(
        self,
        row_idx: int,
        row: pd.Series,
        classified: Any,
        contrib_dict: dict,
        mappings: list = None,
    ) -> str:
        """Build a human-readable explanation summary."""
        parts = []

        # Record identifier
        patient_id = None
        visit = None
        if mappings:
            for m in mappings:
                if hasattr(m, "clinical_type"):
                    if m.clinical_type == "patient_id" and m.column in row.index:
                        patient_id = row[m.column]
                    if m.clinical_type == "visit_date" and m.column in row.index:
                        visit = row[m.column]

        if patient_id:
            parts.append(f"Record for patient {patient_id}")
            if visit:
                parts.append(f" (visit: {visit})")
        else:
            parts.append(f"Record at row {row_idx}")

        parts.append(f" was flagged as '{getattr(classified, 'anomaly_type', 'anomaly')}'")
        parts.append(f" with {getattr(classified, 'confidence', 0)*100:.0f}% confidence.")

        # Add description
        desc = getattr(classified, "description", "")
        if desc:
            parts.append(f" {desc}")

        # Top contributing features
        if contrib_dict:
            top_features = list(contrib_dict.items())[:3]
            feature_text = ", ".join(
                f"'{col}' ({pct*100:.0f}%)" for col, pct in top_features
            )
            parts.append(f" Top contributing features: {feature_text}.")

        return "".join(parts)

    def generate_global_summary(
        self,
        df: pd.DataFrame,
        anomaly_flags: np.ndarray,
        feature_contributions: np.ndarray,
        classified_anomalies: list,
        clinical_mappings: list = None,
    ) -> GlobalSummary:
        """Generate a dataset-level narrative summary."""
        columns = df.columns.tolist()
        n_total = len(df)
        n_anomalies = int(anomaly_flags.sum())
        noise_pct = n_anomalies / n_total * 100 if n_total > 0 else 0.0

        # Aggregate feature contributions across all flagged rows
        flagged_contribs = feature_contributions[anomaly_flags]
        if len(flagged_contribs) > 0:
            mean_contribs = flagged_contribs.mean(axis=0)
            total_contrib = mean_contribs.sum()
            top_indices = np.argsort(mean_contribs)[-5:][::-1]

            top_drivers = []
            for i in top_indices:
                pct = (mean_contribs[i] / total_contrib * 100) if total_contrib > 0 else 0
                col = columns[i]
                desc = self._describe_column_anomaly(df[col], anomaly_flags, col)
                top_drivers.append({
                    "column": col,
                    "contribution_pct": round(pct, 1),
                    "description": desc,
                })
        else:
            top_drivers = []

        # Anomaly type distribution
        type_dist = {}
        for ca in classified_anomalies:
            t = ca.anomaly_type
            type_dist[t] = type_dist.get(t, 0) + 1

        # Build narrative
        narrative_parts = [
            f"AutoClin Engine analyzed {n_total:,} records and flagged {n_anomalies:,} "
            f"({noise_pct:.1f}%) as potential data quality issues."
        ]

        if top_drivers:
            narrative_parts.append(
                f" The top drivers of anomaly detection were: "
            )
            for i, driver in enumerate(top_drivers[:3], 1):
                narrative_parts.append(
                    f"({i}) column '{driver['column']}' contributing "
                    f"{driver['contribution_pct']:.0f}% — {driver['description']}; "
                )

        if type_dist:
            most_common = max(type_dist, key=type_dist.get)
            narrative_parts.append(
                f"The most common anomaly type was '{most_common}' "
                f"({type_dist[most_common]} cases)."
            )

        return GlobalSummary(
            narrative="".join(narrative_parts),
            top_drivers=top_drivers,
            anomaly_type_distribution=type_dist,
            total_anomalies=n_anomalies,
            noise_pct=round(noise_pct, 2),
        )

    def _describe_column_anomaly(
        self, series: pd.Series, flags: np.ndarray, col_name: str
    ) -> str:
        """Generate a description of why a column contributes to anomalies."""
        flagged = series[flags].dropna()
        normal = series[~flags].dropna()

        if pd.api.types.is_numeric_dtype(series) and len(flagged) > 0 and len(normal) > 0:
            flagged_mean = flagged.mean()
            normal_mean = normal.mean()
            flagged_std = flagged.std()
            return (
                f"flagged values (mean={flagged_mean:.2f}, std={flagged_std:.2f}) "
                f"deviate from normal values (mean={normal_mean:.2f})"
            )
        elif len(flagged) > 0:
            return f"{len(flagged)} flagged values found"
        return "contributes to anomaly patterns"

    @staticmethod
    def _safe_value(val) -> Any:
        """Convert value to JSON-safe type."""
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return None
        if isinstance(val, (np.integer,)):
            return int(val)
        if isinstance(val, (np.floating,)):
            return float(val)
        return val

    @staticmethod
    def _values_differ(a, b) -> bool:
        """Check if two values are meaningfully different."""
        if a is None and b is None:
            return False
        if a is None or b is None:
            return True
        try:
            if isinstance(a, (int, float)) and isinstance(b, (int, float)):
                if np.isnan(a) and np.isnan(b):
                    return False
                return abs(a - b) > 1e-6
        except (TypeError, ValueError):
            pass
        return str(a) != str(b)
