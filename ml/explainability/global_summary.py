"""
AutoClin Engine Global Explanation Summary
Generates dataset-level narrative summarizing top anomaly drivers.
"""
import numpy as np
from typing import Optional
from collections import Counter


class GlobalSummaryGenerator:
    """Produces dataset-level anomaly driver narrative."""

    def generate(
        self,
        feature_contributions_list: list[dict[str, float]],
        classifications: list,
        feature_names: list[str],
        total_rows: int,
        clinical_mappings: Optional[list] = None,
    ) -> dict:
        """
        Returns:
            {
                "narrative": str,
                "top_drivers": [{column, contribution_pct, description}],
                "type_distribution": {anomaly_type: count},
                "severity_distribution": {severity: count},
            }
        """
        # Aggregate feature contributions across all flagged records
        feature_totals = {}
        for contribs in feature_contributions_list:
            for feat, val in contribs.items():
                feature_totals[feat] = feature_totals.get(feat, 0.0) + abs(val)

        total_score_mass = sum(feature_totals.values()) or 1.0
        top_drivers = sorted(feature_totals.items(), key=lambda x: x[1], reverse=True)[:10]

        driver_entries = []
        for col, total in top_drivers:
            pct = round(total / total_score_mass * 100, 1)
            clinical_label = self._get_clinical_label(col, clinical_mappings)
            driver_entries.append({
                "column": col,
                "contribution_pct": pct,
                "clinical_label": clinical_label,
            })

        # Type and severity distributions
        type_dist = Counter(c.anomaly_type for c in classifications)
        severity_dist = Counter(c.severity for c in classifications)

        # Build narrative
        narrative = self._build_narrative(
            driver_entries, type_dist, severity_dist,
            len(classifications), total_rows,
        )

        return {
            "narrative": narrative,
            "top_drivers": driver_entries,
            "type_distribution": dict(type_dist),
            "severity_distribution": dict(severity_dist),
        }

    def _build_narrative(self, drivers, type_dist, severity_dist,
                         total_anomalies, total_rows):
        pct = round(total_anomalies / total_rows * 100, 2) if total_rows > 0 else 0
        parts = [
            f"AutoClin Engine detected {total_anomalies} anomalies across {total_rows} records ({pct}% noise rate)."
        ]

        if drivers:
            parts.append("The top drivers of anomaly detection were:")
            for i, d in enumerate(drivers[:5], 1):
                label = d["clinical_label"] or d["column"]
                parts.append(
                    f"  ({i}) {label} contributing {d['contribution_pct']}% of total anomaly score mass"
                )

        top_type = max(type_dist, key=type_dist.get) if type_dist else "unknown"
        top_type_count = type_dist.get(top_type, 0)
        parts.append(
            f"The most common anomaly type was '{top_type.replace('_', ' ')}' "
            f"with {top_type_count} occurrences."
        )

        critical = severity_dist.get("critical", 0)
        high = severity_dist.get("high", 0)
        if critical > 0:
            parts.append(f"{critical} anomalies are classified as CRITICAL severity requiring immediate review.")
        if high > 0:
            parts.append(f"{high} additional anomalies are HIGH severity.")

        return " ".join(parts)

    def _get_clinical_label(self, col_name, clinical_mappings):
        if not clinical_mappings:
            return None
        for m in clinical_mappings:
            if m.column_name == col_name and m.clinical_type:
                return f"{col_name} ({m.clinical_type})"
        return None
