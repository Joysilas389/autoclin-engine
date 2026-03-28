"""
AutoClin Engine Explanation Card Generator
Produces human-readable, one-paragraph summaries for each flagged record.
"""
from typing import Optional


class ExplanationCardGenerator:
    """Generates clinician-friendly explanation cards for flagged records."""

    def generate_card(
        self,
        row_index: int,
        anomaly_type: str,
        severity: str,
        confidence: float,
        flagged_columns: list[str],
        feature_contributions: dict[str, float],
        row_data: dict,
        clean_twin: Optional[dict] = None,
        clinical_context: Optional[dict] = None,
        recommended_action: str = "review",
    ) -> dict:
        """
        Generate a single explanation card.
        
        Returns:
            {
                "summary": str,  # One-paragraph natural language
                "finding": str,
                "clinical_context": str,
                "likely_cause": str,
                "confidence_label": str,
                "recommended_action": str,
                "risk_if_uncorrected": str,
            }
        """
        # Build the top contributing features description
        top_features = sorted(feature_contributions.items(), key=lambda x: abs(x[1]), reverse=True)[:3]
        feature_desc = self._describe_features(top_features, row_data)

        # Patient/record identifier
        record_id = self._extract_record_id(row_data, row_index)

        # Confidence label
        if confidence >= 0.9:
            conf_label = "Very High"
        elif confidence >= 0.7:
            conf_label = "High"
        elif confidence >= 0.5:
            conf_label = "Moderate"
        else:
            conf_label = "Low"

        # Type-specific language
        type_desc = self._type_description(anomaly_type)
        cause_desc = self._likely_cause(anomaly_type, flagged_columns, row_data)
        risk_desc = self._risk_description(anomaly_type, severity)
        action_desc = self._action_description(recommended_action)

        # Clinical context
        ctx_str = ""
        if clinical_context:
            ref = clinical_context.get("reference_range")
            if ref:
                ctx_str = f"Normal range: {ref.get('min')}-{ref.get('max')} {ref.get('unit', '')}"

        # Build summary paragraph
        summary = (
            f"Record {record_id} was flagged as a {type_desc} "
            f"({severity} severity, {conf_label} confidence). "
            f"{feature_desc} "
            f"{cause_desc} "
            f"Recommended action: {action_desc}."
        )

        return {
            "summary": summary,
            "finding": feature_desc,
            "clinical_context": ctx_str if ctx_str else "No specific clinical reference available",
            "likely_cause": cause_desc,
            "confidence_label": f"{conf_label} ({confidence*100:.0f}%)",
            "recommended_action": action_desc,
            "risk_if_uncorrected": risk_desc,
        }

    def _extract_record_id(self, row_data: dict, row_index: int) -> str:
        id_keys = ["patient_id", "subject_id", "ptid", "subjid", "record_id", "usubjid", "mrn"]
        for key in id_keys:
            if key in row_data and row_data[key] is not None:
                return f"#{row_index} (Patient {row_data[key]})"
        return f"#{row_index}"

    def _describe_features(self, top_features, row_data):
        if not top_features:
            return "No specific feature dominated the anomaly score."
        parts = []
        for feat_name, contrib in top_features:
            val = row_data.get(feat_name, "N/A")
            parts.append(f"{feat_name}={val} (contribution: {abs(contrib)*100:.0f}%)")
        return f"Key drivers: {'; '.join(parts)}."

    def _type_description(self, anomaly_type: str) -> str:
        desc_map = {
            "extreme_numeric_outlier": "extreme numeric outlier",
            "contextual_outlier": "contextual outlier",
            "site_specific_anomaly": "site-specific data anomaly",
            "temporal_inconsistency": "temporal inconsistency",
            "duplicate_near_duplicate": "duplicate or near-duplicate record",
            "impossible_biological_value": "clinically impossible value",
            "suspicious_missingness": "suspicious missingness pattern",
            "unit_mismatch": "possible unit mismatch",
            "data_entry_typo": "likely data entry error",
            "category_encoding_inconsistency": "category encoding inconsistency",
            "drift_related_anomaly": "drift-related anomaly",
            "cross_field_contradiction": "cross-field contradiction",
            "visit_sequence_inconsistency": "visit sequence inconsistency",
            "latent_cluster_isolated": "cluster-isolated observation",
            "graph_isolated_record": "graph-isolated patient record",
            "high_reconstruction_error": "high reconstruction error pattern",
            "distributional_contamination": "distributional contamination",
            "rare_but_plausible": "rare but clinically plausible case",
            "rare_and_implausible": "rare and clinically implausible case",
        }
        return desc_map.get(anomaly_type, anomaly_type.replace("_", " "))

    def _likely_cause(self, anomaly_type, flagged_cols, row_data):
        if anomaly_type == "impossible_biological_value":
            return "Likely a data entry error (e.g., extra digit, decimal point error, or wrong unit)."
        elif anomaly_type == "unit_mismatch":
            return "Values suggest a possible unit conversion error between measurement systems."
        elif anomaly_type == "data_entry_typo":
            return "Pattern suggests a typographical error during data entry."
        elif anomaly_type == "duplicate_near_duplicate":
            return "Record appears to be a duplicate or near-duplicate of another entry."
        elif anomaly_type == "temporal_inconsistency":
            return "Date or visit sequence does not align with expected timeline."
        elif anomaly_type == "rare_but_plausible":
            return "Value is statistically extreme but within the realm of clinical possibility."
        return f"Anomalous pattern detected across {', '.join(flagged_cols[:3])}."

    def _risk_description(self, anomaly_type, severity):
        if severity == "critical":
            return "Could bias safety analyses and lead to incorrect clinical conclusions."
        elif severity == "high":
            return "May affect statistical summaries and downstream modeling."
        elif severity == "medium":
            return "Could introduce minor noise in analyses."
        return "Low risk to downstream analyses, but worth documenting."

    def _action_description(self, action):
        action_map = {
            "review": "Review with source data and clinical context",
            "correct_or_impute": "Verify against source document; correct if confirmed as error",
            "remove_row": "Consider removing if confirmed as invalid",
            "winsorize": "Winsorize to a clinically plausible boundary",
            "escalate_clinician": "Escalate to clinician for adjudication",
            "preserve_annotate": "Preserve the value and annotate as a rare clinical case",
            "merge_duplicate": "Merge with the most complete version of the record",
            "normalize_coding": "Standardize to canonical category encoding",
            "convert_unit": "Apply unit conversion based on detected mismatch",
        }
        return action_map.get(action, action.replace("_", " "))
