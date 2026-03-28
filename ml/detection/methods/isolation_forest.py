"""Isolation Forest — fast global outlier detection in high dimensions."""
import time
import numpy as np
from sklearn.ensemble import IsolationForest
from ml.detection.method_adapter import BaseDetector, DetectionResult


class IsolationForestDetector(BaseDetector):
    name = "isolation_forest"
    explainability_prior = 0.9

    def __init__(self, n_estimators: int = 200, contamination: float = 0.05, random_state: int = 42):
        self.n_estimators = n_estimators
        self.contamination = contamination
        self.random_state = random_state

    def fit_score(self, X: np.ndarray, **kwargs) -> DetectionResult:
        contamination = kwargs.get("contamination", self.contamination)
        start = time.time()

        model = IsolationForest(
            n_estimators=self.n_estimators,
            contamination=contamination,
            random_state=self.random_state,
            n_jobs=-1,
        )
        model.fit(X)

        raw_scores = -model.decision_function(X)  # Higher = more anomalous
        labels = model.predict(X)  # -1 = anomaly, 1 = normal

        duration_ms = int((time.time() - start) * 1000)

        return DetectionResult(
            method_name=self.name,
            anomaly_scores=self.normalize_scores(raw_scores),
            labels=labels,
            duration_ms=duration_ms,
            params={"n_estimators": self.n_estimators, "contamination": contamination},
            metadata={"feature_importances": self._feature_importances(model, X)},
        )

    def _feature_importances(self, model: IsolationForest, X: np.ndarray) -> list[float]:
        """Estimate per-feature importance via mean split contribution."""
        n_features = X.shape[1]
        importances = np.zeros(n_features)
        for tree in model.estimators_:
            tree_features = tree.tree_.feature
            valid = tree_features >= 0
            for f in tree_features[valid]:
                importances[f] += 1
        total = importances.sum()
        if total > 0:
            importances = importances / total
        return importances.tolist()

    def get_penalty(self, dcv: dict) -> float:
        # Isolation Forest works well in most settings; slight penalty for very low n
        if dcv.get("n", 0) < 50:
            return 0.3
        return 0.0
