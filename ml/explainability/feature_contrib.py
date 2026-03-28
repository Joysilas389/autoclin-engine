"""
AutoClin Engine Feature Contribution Extractor
Extracts per-feature anomaly contributions for each detection method.
"""
import numpy as np
from typing import Optional


class FeatureContributionExtractor:
    """
    Extracts feature-level contributions to anomaly scores.
    Method-specific techniques per the blueprint:
    - Isolation Forest: mean split contribution per feature
    - LOF/kNN: per-feature distance contribution
    - Autoencoder/VAE: per-feature reconstruction error
    - HDBSCAN: cluster membership + outlier score
    - Robust PCA: sparse component magnitude per feature
    """

    def extract(
        self,
        method_name: str,
        X: np.ndarray,
        row_idx: int,
        detection_result,
        feature_names: list[str],
    ) -> dict[str, float]:
        """Return {feature_name: contribution_score} for a single row."""

        extractors = {
            "isolation_forest": self._isolation_forest_contrib,
            "lof": self._distance_decomposition,
            "hdbscan": self._distance_decomposition,
            "autoencoder": self._reconstruction_error_contrib,
            "robust_pca": self._sparse_component_contrib,
        }

        extractor = extractors.get(method_name, self._default_contrib)
        raw = extractor(X, row_idx, detection_result)

        # Map to feature names
        contributions = {}
        for i, name in enumerate(feature_names):
            if i < len(raw):
                contributions[name] = round(float(raw[i]), 4)

        # Normalize to sum to 1
        total = sum(abs(v) for v in contributions.values())
        if total > 0:
            contributions = {k: round(v / total, 4) for k, v in contributions.items()}

        return contributions

    def extract_batch(
        self,
        method_name: str,
        X: np.ndarray,
        flagged_indices: np.ndarray,
        detection_result,
        feature_names: list[str],
    ) -> list[dict[str, float]]:
        """Extract contributions for all flagged rows."""
        return [
            self.extract(method_name, X, int(idx), detection_result, feature_names)
            for idx in flagged_indices
        ]

    def _isolation_forest_contrib(self, X, row_idx, result) -> np.ndarray:
        """Use stored feature importances from Isolation Forest metadata."""
        metadata = result.metadata or {}
        importances = metadata.get("feature_importances")
        if importances:
            row = X[row_idx]
            # Weight global importances by how extreme each feature value is
            z_scores = np.abs(row - X.mean(axis=0)) / (X.std(axis=0) + 1e-10)
            return np.array(importances) * z_scores
        return self._default_contrib(X, row_idx, result)

    def _distance_decomposition(self, X, row_idx, result) -> np.ndarray:
        """Per-feature contribution to distance from neighbors."""
        row = X[row_idx]
        # Find k nearest neighbors
        dists = np.linalg.norm(X - row, axis=1)
        dists[row_idx] = np.inf
        k = min(20, len(X) - 1)
        nn_indices = np.argsort(dists)[:k]
        neighbors = X[nn_indices]
        mean_neighbor = neighbors.mean(axis=0)
        per_feature_dist = (row - mean_neighbor) ** 2
        return per_feature_dist

    def _reconstruction_error_contrib(self, X, row_idx, result) -> np.ndarray:
        """Per-feature reconstruction error from autoencoder metadata."""
        metadata = result.metadata or {}
        per_feature_mean = metadata.get("per_feature_error_mean")
        if per_feature_mean:
            # Scale by how much this row deviates from the mean error
            row = X[row_idx]
            global_mean = X.mean(axis=0)
            deviation = (row - global_mean) ** 2
            return deviation * np.array(per_feature_mean)
        return self._default_contrib(X, row_idx, result)

    def _sparse_component_contrib(self, X, row_idx, result) -> np.ndarray:
        """Sparse component magnitude from Robust PCA metadata."""
        metadata = result.metadata or {}
        sparse_per_feature = metadata.get("sparse_magnitude_per_feature")
        if sparse_per_feature:
            row = X[row_idx]
            deviation = np.abs(row - X.mean(axis=0))
            return deviation * np.array(sparse_per_feature)
        return self._default_contrib(X, row_idx, result)

    def _default_contrib(self, X, row_idx, result) -> np.ndarray:
        """Default: z-score based contribution."""
        row = X[row_idx]
        means = X.mean(axis=0)
        stds = X.std(axis=0) + 1e-10
        return np.abs((row - means) / stds)
