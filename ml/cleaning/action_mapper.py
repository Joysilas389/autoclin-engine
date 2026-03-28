"""
AutoClin Engine Cleaning Action Mapper
Maps each flagged anomaly to recommended cleaning actions with severity,
confidence, and auto-clean risk assessment.
"""
from dataclasses import dataclass
from typing import Optional


ACTION_MAP = {
    "extreme_numeric_outlier": {
        "default": "winsorize",
        "alternatives": ["review", "remove_row", "impute_median"],
        "auto_clean_risk": "caution",
    },
    "contextual_outlier": {
        "default": "review",
        "alternatives": ["impute_conditional"],
        "auto_clean_risk": "manual_only",
    },
    "site_specific_anomaly": {
        "default": "escalate_data_manager",
        "alternatives": ["normalize_site_effect", "review"],
        "auto_clean_risk": "manual_only",
    },
    "temporal_inconsistency": {
        "default": "flag_for_review",
        "alternatives": ["reorder_visits", "correct_date", "remove_row"],
        "auto_clean_risk": "manual_only",
    },
    "duplicate_near_duplicate": {
        "default": "merge_keep_complete",
        "alternatives": ["review", "remove_duplicate"],
        "auto_clean_risk": "caution",
    },
    "impossible_biological_value": {
        "default": "correct_or_impute",
        "alternatives": ["convert_unit", "escalate_clinician", "remove_row"],
        "auto_clean_risk": "manual_only",
    },
    "suspicious_missingness": {
        "default": "annotate_pattern",
        "alternatives": ["impute_mice", "impute_knn", "escalate_data_manager"],
        "auto_clean_risk": "caution",
    },
    "unit_mismatch": {
        "default": "convert_unit",
        "alternatives": ["flag_for_review"],
        "auto_clean_risk": "safe",
    },
    "data_entry_typo": {
        "default": "correct_pattern",
        "alternatives": ["review", "remove_field"],
        "auto_clean_risk": "caution",
    },
    "category_encoding_inconsistency": {
        "default": "normalize_coding",
        "alternatives": ["flag_for_review"],
        "auto_clean_risk": "safe",
    },
    "drift_related_anomaly": {
        "default": "annotate_drift_window",
        "alternatives": ["split_by_period", "escalate_data_manager"],
        "auto_clean_risk": "manual_only",
    },
    "cross_field_contradiction": {
        "default": "escalate_clinician",
        "alternatives": ["remove_contradicting_field"],
        "auto_clean_risk": "manual_only",
    },
    "visit_sequence_inconsistency": {
        "default": "reorder_or_flag",
        "alternatives": ["remove_row", "impute_visit"],
        "auto_clean_risk": "manual_only",
    },
    "latent_cluster_isolated": {
        "default": "review",
        "alternatives": ["remove_if_implausible"],
        "auto_clean_risk": "manual_only",
    },
    "graph_isolated_record": {
        "default": "review_linkage",
        "alternatives": ["escalate_data_manager"],
        "auto_clean_risk": "manual_only",
    },
    "high_reconstruction_error": {
        "default": "review",
        "alternatives": ["remove_row", "impute"],
        "auto_clean_risk": "manual_only",
    },
    "distributional_contamination": {
        "default": "exclude_from_modeling",
        "alternatives": ["split_subpopulations"],
        "auto_clean_risk": "caution",
    },
    "rare_but_plausible": {
        "default": "preserve_annotate",
        "alternatives": ["sensitivity_analysis_variant"],
        "auto_clean_risk": "manual_only",
    },
    "rare_and_implausible": {
        "default": "escalate_clinician",
        "alternatives": ["remove_after_review"],
        "auto_clean_risk": "manual_only",
    },
}


@dataclass
class CleaningRecommendation:
    row_index: int
    column_name: str
    anomaly_type: str
    severity: str
    confidence: float
    recommended_action: str
    alternative_actions: list[str]
    auto_clean_risk: str  # safe, caution, manual_only
    rationale: str
    original_value: object = None
    proposed_value: object = None


class CleaningActionMapper:
    """Maps classified anomalies to cleaning recommendations."""

    def map_actions(self, classifications: list, row_data_fn=None) -> list[CleaningRecommendation]:
        """
        Args:
            classifications: List of AnomalyClassification objects.
            row_data_fn: Optional callable(row_index) -> dict of row data.
        """
        recommendations = []
        for cls in classifications:
            action_info = ACTION_MAP.get(cls.anomaly_type, {
                "default": "review",
                "alternatives": [],
                "auto_clean_risk": "manual_only",
            })

            for col in cls.flagged_columns:
                original_value = None
                if row_data_fn:
                    row = row_data_fn(cls.row_index)
                    original_value = row.get(col) if row else None

                recommendations.append(CleaningRecommendation(
                    row_index=cls.row_index,
                    column_name=col,
                    anomaly_type=cls.anomaly_type,
                    severity=cls.severity,
                    confidence=cls.confidence,
                    recommended_action=action_info["default"],
                    alternative_actions=action_info["alternatives"],
                    auto_clean_risk=action_info["auto_clean_risk"],
                    rationale=cls.rationale,
                    original_value=original_value,
                ))

        return recommendations

    def get_auto_safe_actions(self, recommendations: list[CleaningRecommendation]) -> list[CleaningRecommendation]:
        """Filter to only auto-applicable (safe) actions."""
        return [r for r in recommendations if r.auto_clean_risk == "safe"]

    def get_manual_review_actions(self, recommendations: list[CleaningRecommendation]) -> list[CleaningRecommendation]:
        """Filter to manual-only actions."""
        return [r for r in recommendations if r.auto_clean_risk == "manual_only"]
