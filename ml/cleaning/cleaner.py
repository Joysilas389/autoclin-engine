"""
AutoClin Engine Data Cleaner
Applies cleaning transformations: winsorize, impute, remove, convert units,
normalize encoding. All operations are non-destructive and logged.
"""
import numpy as np
import pandas as pd
from typing import Optional
from ml.cleaning.audit_ledger import AuditLedger
from ml.cleaning.plausibility_engine import PlausibilityEngine


class DataCleaner:
    """Applies cleaning actions to a DataFrame with full audit logging."""

    def __init__(self, audit_ledger: AuditLedger,
                 plausibility_engine: Optional[PlausibilityEngine] = None):
        self.ledger = audit_ledger
        self.plaus = plausibility_engine or PlausibilityEngine()

    def apply_actions(self, df: pd.DataFrame,
                      recommendations: list, method_name: str = "auto") -> pd.DataFrame:
        """
        Apply a list of cleaning recommendations.
        Returns a copy of the DataFrame with transformations applied.
        """
        cleaned = df.copy()
        for rec in recommendations:
            action = rec.recommended_action
            fn = {
                "winsorize": self._winsorize,
                "correct_or_impute": self._impute_median,
                "impute_median": self._impute_median,
                "impute_mice": self._impute_median,  # simplified fallback
                "impute_knn": self._impute_knn,
                "remove_row": self._remove_row,
                "remove_field": self._remove_field,
                "convert_unit": self._convert_unit,
                "normalize_coding": self._normalize_coding,
                "merge_keep_complete": self._flag_only,
                "flag_for_review": self._flag_only,
                "review": self._flag_only,
                "escalate_clinician": self._flag_only,
                "escalate_data_manager": self._flag_only,
                "preserve_annotate": self._flag_only,
                "annotate_pattern": self._flag_only,
                "correct_pattern": self._correct_pattern,
            }.get(action, self._flag_only)

            cleaned = fn(cleaned, rec, method_name)
        return cleaned

    def _winsorize(self, df, rec, method):
        """Winsorize extreme values to the 1st/99th percentile."""
        col = rec.column_name
        row = rec.row_index
        if col not in df.columns or row >= len(df):
            return df

        numeric = pd.to_numeric(df[col], errors="coerce")
        p1, p99 = numeric.quantile(0.01), numeric.quantile(0.99)

        original = df.at[row, col]
        val = pd.to_numeric(original, errors="coerce")
        if pd.isna(val):
            return df

        if val < p1:
            new_val = round(float(p1), 4)
        elif val > p99:
            new_val = round(float(p99), 4)
        else:
            return df

        # Validate plausibility
        check = self.plaus.validate_correction(col, new_val)
        if not check["valid"]:
            self.ledger.log(row, col, original, None, "winsorize_rejected",
                           method, "manual_only",
                           f"Winsorized value {new_val} failed plausibility: {check['reason']}")
            return df

        df.at[row, col] = new_val
        self.ledger.log(row, col, original, new_val, "winsorize",
                       method, "caution",
                       f"Winsorized from {original} to {new_val} (1st/99th percentile)")
        return df

    def _impute_median(self, df, rec, method):
        """Impute with column median."""
        col = rec.column_name
        row = rec.row_index
        if col not in df.columns or row >= len(df):
            return df

        numeric = pd.to_numeric(df[col], errors="coerce")
        median_val = numeric.median()
        if pd.isna(median_val):
            return df

        original = df.at[row, col]
        new_val = round(float(median_val), 4)

        check = self.plaus.validate_correction(col, new_val)
        if not check["valid"]:
            self.ledger.log(row, col, original, None, "impute_rejected",
                           method, "manual_only",
                           f"Imputed value {new_val} failed plausibility: {check['reason']}")
            return df

        df.at[row, col] = new_val
        self.ledger.log(row, col, original, new_val, "impute_median",
                       method, "caution",
                       f"Imputed with median {new_val} (original: {original})")
        return df

    def _impute_knn(self, df, rec, method):
        """Impute using k-nearest neighbors average (simplified)."""
        col = rec.column_name
        row = rec.row_index
        if col not in df.columns or row >= len(df):
            return df

        numeric = pd.to_numeric(df[col], errors="coerce").dropna()
        if len(numeric) < 5:
            return self._impute_median(df, rec, method)

        val = pd.to_numeric(df.at[row, col], errors="coerce")
        original = df.at[row, col]

        # Find 5 nearest values
        diffs = (numeric - numeric.median()).abs().nsmallest(5)
        knn_mean = round(float(diffs.index.map(lambda i: numeric.loc[i]).mean()), 4)

        check = self.plaus.validate_correction(col, knn_mean)
        if not check["valid"]:
            return df

        df.at[row, col] = knn_mean
        self.ledger.log(row, col, original, knn_mean, "impute_knn",
                       method, "caution", f"KNN imputed to {knn_mean}")
        return df

    def _remove_row(self, df, rec, method):
        """Mark row for removal (set all values to NaN)."""
        row = rec.row_index
        if row >= len(df):
            return df

        original_vals = df.iloc[row].to_dict()
        for col in df.columns:
            df.at[row, col] = np.nan

        self.ledger.log(row, "__all__", str(original_vals)[:200], None,
                       "remove_row", method, "manual_only",
                       f"Row {row} marked for removal")
        return df

    def _remove_field(self, df, rec, method):
        """Set a single field to NaN."""
        col = rec.column_name
        row = rec.row_index
        if col not in df.columns or row >= len(df):
            return df

        original = df.at[row, col]
        df.at[row, col] = np.nan
        self.ledger.log(row, col, original, None, "remove_field",
                       method, "caution", f"Field cleared (was: {original})")
        return df

    def _convert_unit(self, df, rec, method):
        """Apply unit conversion (placeholder — needs unit detection context)."""
        self.ledger.log(rec.row_index, rec.column_name, df.at[rec.row_index, rec.column_name],
                       None, "convert_unit_pending", method, "safe",
                       "Unit conversion requires manual specification of source/target units")
        return df

    def _normalize_coding(self, df, rec, method):
        """Normalize categorical encoding (lowercase, strip whitespace)."""
        col = rec.column_name
        row = rec.row_index
        if col not in df.columns or row >= len(df):
            return df

        original = df.at[row, col]
        if pd.isna(original):
            return df

        new_val = str(original).strip().lower()

        # Map common variants
        sex_map = {"male": "M", "female": "F", "m": "M", "f": "F",
                   "man": "M", "woman": "F"}
        if new_val in sex_map:
            new_val = sex_map[new_val]

        if str(original) != new_val:
            df.at[row, col] = new_val
            self.ledger.log(row, col, original, new_val, "normalize_coding",
                           method, "safe",
                           f"Normalized encoding: '{original}' → '{new_val}'")
        return df

    def _correct_pattern(self, df, rec, method):
        """Attempt pattern-based correction (e.g., extra digit removal)."""
        col = rec.column_name
        row = rec.row_index
        if col not in df.columns or row >= len(df):
            return df

        original = df.at[row, col]
        val = pd.to_numeric(original, errors="coerce")
        if pd.isna(val):
            return df

        # Try dividing by 10 (common extra-digit error)
        for divisor in [10, 100]:
            candidate = val / divisor
            check = self.plaus.validate_correction(col, candidate)
            if check["valid"]:
                df.at[row, col] = round(float(candidate), 4)
                self.ledger.log(row, col, original, round(float(candidate), 4),
                               "correct_pattern", method, "caution",
                               f"Likely extra digit: {original} → {candidate} (÷{divisor})")
                return df

        self.ledger.log(row, col, original, None, "correct_pattern_failed",
                       method, "manual_only", "No pattern correction found")
        return df

    def _flag_only(self, df, rec, method):
        """Log the flag without modifying data."""
        self.ledger.log(rec.row_index, rec.column_name,
                       df.at[rec.row_index, rec.column_name] if rec.row_index < len(df) else None,
                       None, "flag_for_review", method, "manual_only",
                       rec.rationale)
        return df
