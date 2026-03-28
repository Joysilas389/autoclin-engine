"""
AutoClin Engine Anomaly Taxonomy Classifier
Classifies each flagged anomaly into one of 19 categories based on
feature contributions, clinical mappings, and detection method context.
"""
from dataclasses import dataclass
from typing import Optional
import numpy as np
import pandas as pd


ANOMALY_TYPES = [
    "extreme_numeric_outlier",
    "contextual_outlier",
    "site_specific_anomaly",
    "temporal_inconsistency",
    "duplicate_near_duplicate",
    "impossible_biological_value",
    "suspicious_missingness",
    "unit_mismatch",
    "data_entry_typo",
    "category_encoding_inconsistency",
    "drift_related_anomaly",
    "cross_field_contradiction",
    "visit_sequence_inconsistency",
    "latent_cluster_isolated",
    "graph_isolated_record",
    "high_reconstruction_error",
    "distributional_contamination",
    "rare_but_plausible",
    "rare_and_implausible",
]

SEVERITY_MAP = {
    "extreme_numeric_outlier": "high",
    "contextual_outlier": "medium",
    "site_specific_anomaly": "high",
    "temporal_inconsistency": "high",
    "duplicate_near_duplicate": "medium",
    "impossible_biological_value": "critical",
    "suspicious_missingness": "medium",
    "unit_mismatch": "high",
    "data_entry_typo": "medium",
    "category_encoding_inconsistency": "low",
    "drift_related_anomaly": "medium",
    "cross_field_contradiction": "high",
    "visit_sequence_inconsistency": "high",
    "latent_cluster_isolated": "low",
    "graph_isolated_record": "medium",
    "high_reconstruction_error": "medium",
    "distributional_contamination": "medium",
    "rare_but_plausible": "low",
    "rare_and_implausible": "high",
}


@dataclass
class AnomalyClassification:
    row_index: int
    anomaly_type: str
    severity: str
    confidence: float
    flagged_columns: list[str]
    rationale: str


class AnomalyTaxonomyClassifier:
    """
    Classifies flagged rows into anomaly types using feature contributions,
    clinical context, and detection metadata.
    """

    def __init__(self, clinical_mappings=None, reference_ranges=None):
        self.clinical_mappings = clinical_mappings or []
        self.reference_ranges = reference_ranges or {}
        self._build_clinical_lookup()

    def _build_clinical_lookup(self):
        self.clinical_cols = {}
        for m in self.clinical_mappings:
            if m.clinical_type:
                self.clinical_cols[m.column_name] = {
                    "type": m.clinical_type,
                    "reference_range": m.reference_range,
                }

    def classify(
        self,
        row_index: int,
        row_data: pd.Series,
        anomaly_score: float,
        feature_contributions: dict[str, float],
        method_name: str,
        cluster_label: Optional[int] = None,
        ensemble_agreement: Optional[float] = None,
    ) -> AnomalyClassification:
        """Classify a single flagged row."""

        top_features = sorted(feature_contributions.items(), key=lambda x: abs(x[1]), reverse=True)
        flagged_cols = [f[0] for f in top_features[:5] if abs(f[1]) > 0.01]

        # Rule-based classification cascade
        anomaly_type, confidence, rationale = self._classify_rules(
            row_data, flagged_cols, feature_contributions, method_name,
            anomaly_score, cluster_label, ensemble_agreement,
        )

        severity = SEVERITY_MAP.get(anomaly_type, "medium")

        return AnomalyClassification(
            row_index=row_index,
            anomaly_type=anomaly_type,
            severity=severity,
            confidence=round(confidence, 3),
            flagged_columns=flagged_cols,
            rationale=rationale,
        )

    def _classify_rules(self, row, flagged_cols, contributions, method, score,
                        cluster_label, agreement):
        """Rule cascade to determine anomaly type."""

        # Check for impossible biological values first
        for col in flagged_cols:
            if col in self.clinical_cols:
                ref = self.clinical_cols[col].get("reference_range")
                if ref and not pd.isna(row.get(col)):
                    val = pd.to_numeric(row.get(col), errors="coerce")
                    if val is not None and not np.isnan(val):
                        if val < ref["min"] * 0.5 or val > ref["max"] * 1.5:
                            return (
                                "impossible_biological_value", 0.95,
                                f"{col}={val} outside plausible range [{ref['min']}-{ref['max']}] {ref.get('unit','')}",
                            )

        # Check for extreme numeric outlier (high single-feature contribution)
        if flagged_cols and abs(contributions.get(flagged_cols[0], 0)) > 0.6:
            col = flagged_cols[0]
            val = row.get(col)
            return (
                "extreme_numeric_outlier", min(score + 0.1, 0.99),
                f"{col}={val} is an extreme outlier based on {method} scoring",
            )

        # High reconstruction error pattern (autoencoder-specific)
        if method == "autoencoder" and len(flagged_cols) >= 3:
            return (
                "high_reconstruction_error", score,
                f"Multiple features ({', '.join(flagged_cols[:3])}) contribute to high reconstruction error",
            )

        # Cluster-isolated observation
        if cluster_label == -1:
            if score > 0.8:
                return (
                    "rare_and_implausible", score,
                    "Record is noise-labeled by density clustering with high anomaly score",
                )
            return (
                "latent_cluster_isolated", score,
                "Record is isolated from all clusters (noise point)",
            )

        # Low agreement → contextual outlier
        if agreement is not None and agreement < 0.5:
            return (
                "contextual_outlier", score * 0.8,
                "Flagged by some methods but not others; likely context-dependent anomaly",
            )

        # Default: extreme numeric outlier if score is high, else distributional contamination
        if score > 0.7:
            return (
                "extreme_numeric_outlier", score,
                f"High anomaly score ({score:.3f}) driven by {', '.join(flagged_cols[:2])}",
            )
        return (
            "distributional_contamination", score,
            f"Moderate anomaly score; may represent distributional tail",
        )

    def classify_batch(
        self,
        df: pd.DataFrame,
        flagged_indices: np.ndarray,
        anomaly_scores: np.ndarray,
        feature_contributions_list: list[dict],
        method_name: str,
        cluster_labels: Optional[np.ndarray] = None,
        agreement_scores: Optional[np.ndarray] = None,
    ) -> list[AnomalyClassification]:
        """Classify all flagged rows in batch."""
        results = []
        for i, row_idx in enumerate(flagged_indices):
            row = df.iloc[row_idx]
            contribs = feature_contributions_list[i] if i < len(feature_contributions_list) else {}
            cl = int(cluster_labels[row_idx]) if cluster_labels is not None else None
            ag = float(agreement_scores[row_idx]) if agreement_scores is not None else None

            classification = self.classify(
                row_index=int(row_idx),
                row_data=row,
                anomaly_score=float(anomaly_scores[row_idx]),
                feature_contributions=contribs,
                method_name=method_name,
                cluster_label=cl,
                ensemble_agreement=ag,
            )
            results.append(classification)
        return results
