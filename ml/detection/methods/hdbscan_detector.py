"""HDBSCAN — hierarchical density-based clustering with robust noise labeling."""
import time
import numpy as np
from ml.detection.method_adapter import BaseDetector, DetectionResult


class HDBSCANDetector(BaseDetector):
    name = "hdbscan"
    explainability_prior = 0.6

    def __init__(self, min_cluster_size: int = 15, min_samples: int = 5):
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples

    def fit_score(self, X: np.ndarray, **kwargs) -> DetectionResult:
        import hdbscan

        start = time.time()
        min_cs = max(5, min(self.min_cluster_size, len(X) // 10))
        min_s = max(3, min(self.min_samples, min_cs))

        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cs,
            min_samples=min_s,
            prediction_data=True,
        )
        clusterer.fit(X)

        outlier_scores = clusterer.outlier_scores_
        if outlier_scores is None or len(outlier_scores) == 0:
            outlier_scores = np.zeros(len(X))

        labels = np.where(clusterer.labels_ == -1, -1, 1)
        duration_ms = int((time.time() - start) * 1000)

        return DetectionResult(
            method_name=self.name,
            anomaly_scores=self.normalize_scores(outlier_scores),
            labels=labels,
            cluster_labels=clusterer.labels_,
            duration_ms=duration_ms,
            params={"min_cluster_size": min_cs, "min_samples": min_s},
            metadata={
                "n_clusters": int(clusterer.labels_.max() + 1) if clusterer.labels_.max() >= 0 else 0,
                "noise_ratio": float((clusterer.labels_ == -1).mean()),
            },
        )

    def get_penalty(self, dcv: dict) -> float:
        d = dcv.get("d", 0)
        if d > 100:
            return 0.3  # Expensive in high dimensions
        if dcv.get("n", 0) < 30:
            return 0.5
        return 0.0
