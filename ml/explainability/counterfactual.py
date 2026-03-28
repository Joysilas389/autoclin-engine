"""
AutoClin Engine Counterfactual Clean Twin Generator
For every flagged record, finds the nearest non-anomalous record
and shows what fields would need to change to become 'normal'.
"""
import numpy as np
import pandas as pd
from typing import Optional


class CounterfactualGenerator:
    """
    Generates a 'clean twin' — the nearest plausible version of a flagged record.
    """

    def generate_clean_twin(
        self,
        row_idx: int,
        df: pd.DataFrame,
        X: np.ndarray,
        anomaly_flags: np.ndarray,
        feature_names: list[str],
    ) -> dict:
        """
        Find nearest non-anomalous record and build a clean twin.
        
        Returns:
            {
                "nearest_normal_index": int,
                "clean_twin_values": {col: value},
                "differences": [{col, original, clean, magnitude}],
                "distance": float,
            }
        """
        normal_mask = ~anomaly_flags
        if not normal_mask.any():
            return {"nearest_normal_index": None, "clean_twin_values": {},
                    "differences": [], "distance": float("inf")}

        row_vec = X[row_idx]
        normal_indices = np.where(normal_mask)[0]
        normal_X = X[normal_indices]

        # Euclidean distance to all normal points
        dists = np.linalg.norm(normal_X - row_vec, axis=1)
        nearest_pos = np.argmin(dists)
        nearest_idx = int(normal_indices[nearest_pos])
        nearest_dist = float(dists[nearest_pos])

        # Build clean twin values and differences
        original_row = df.iloc[row_idx]
        normal_row = df.iloc[nearest_idx]

        clean_twin = {}
        differences = []
        for i, col in enumerate(feature_names):
            if col in df.columns:
                orig_val = original_row.get(col)
                clean_val = normal_row.get(col)
                clean_twin[col] = clean_val

                if i < X.shape[1]:
                    mag = abs(float(X[row_idx, i] - X[nearest_idx, i]))
                    if mag > 0.01:
                        differences.append({
                            "column": col,
                            "original": orig_val if not pd.isna(orig_val) else None,
                            "clean": clean_val if not pd.isna(clean_val) else None,
                            "magnitude": round(mag, 4),
                        })

        differences.sort(key=lambda d: d["magnitude"], reverse=True)

        return {
            "nearest_normal_index": nearest_idx,
            "clean_twin_values": clean_twin,
            "differences": differences[:10],  # top 10 differences
            "distance": round(nearest_dist, 4),
        }

    def generate_batch(
        self,
        flagged_indices: np.ndarray,
        df: pd.DataFrame,
        X: np.ndarray,
        anomaly_flags: np.ndarray,
        feature_names: list[str],
    ) -> list[dict]:
        """Generate clean twins for all flagged records."""
        return [
            self.generate_clean_twin(int(idx), df, X, anomaly_flags, feature_names)
            for idx in flagged_indices
        ]
