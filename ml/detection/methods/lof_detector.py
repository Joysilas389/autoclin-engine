"""Local Outlier Factor — detects local density anomalies."""
import time
import numpy as np
from sklearn.neighbors import LocalOutlierFactor
from ml.detection.method_adapter import BaseDetector, DetectionResult


class LOFDetector(BaseDetector):
    name = "lof"
    explainability_prior = 0.8

    def __init__(self, n_neighbors: int = 20, contamination: float = 0.05):
        self.n_neighbors = n_neighbors
        self.contamination = contamination

    def fit_score(self, X: np.ndarray, **kwargs) -> DetectionResult:
        contamination = kwargs.get("contamination", self.contamination)
        start = time.time()

        k = min(self.n_neighbors, len(X) - 1)
        if k < 2:
            return DetectionResult(
                method_name=self.name,
                anomaly_scores=np.zeros(len(X)),
                duration_ms=0,
                params={"n_neighbors": k, "contamination": contamination},
            )

        model = LocalOutlierFactor(
            n_neighbors=k,
            contamination=contamination,
            novelty=False,
            n_jobs=-1,
        )
        labels = model.fit_predict(X)
        raw_scores = -model.negative_outlier_factor_  # Higher = more anomalous

        duration_ms = int((time.time() - start) * 1000)

        return DetectionResult(
            method_name=self.name,
            anomaly_scores=self.normalize_scores(raw_scores),
            labels=labels,
            duration_ms=duration_ms,
            params={"n_neighbors": k, "contamination": contamination},
            metadata={"lof_scores": raw_scores.tolist()[:100]},
        )

    def get_penalty(self, dcv: dict) -> float:
        n = dcv.get("n", 0)
        if n > 100_000:
            return 0.4  # O(n^2) naive; slow for large datasets
        if n < 20:
            return 0.4
        return 0.0
