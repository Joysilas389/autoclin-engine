"""Robust PCA — detects sparse corruption via low-rank + sparse decomposition."""
import time
import numpy as np
from ml.detection.method_adapter import BaseDetector, DetectionResult


class RobustPCADetector(BaseDetector):
    name = "robust_pca"
    explainability_prior = 0.7

    def __init__(self, max_iter: int = 100, tol: float = 1e-6, random_state: int = 42):
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

    def fit_score(self, X: np.ndarray, **kwargs) -> DetectionResult:
        start = time.time()
        n_samples, n_features = X.shape

        if n_samples < 10 or n_features < 2:
            return DetectionResult(
                method_name=self.name,
                anomaly_scores=np.zeros(n_samples),
                duration_ms=0,
                params={},
                metadata={"skipped": True},
            )

        # Principal Component Pursuit via Augmented Lagrangian Method
        L, S = self._rpca_alm(X)

        # Anomaly score: L1 norm of sparse component per row
        per_feature_sparse = np.abs(S)
        row_scores = per_feature_sparse.mean(axis=1)

        duration_ms = int((time.time() - start) * 1000)

        return DetectionResult(
            method_name=self.name,
            anomaly_scores=self.normalize_scores(row_scores),
            duration_ms=duration_ms,
            params={"max_iter": self.max_iter, "tol": self.tol},
            metadata={
                "sparse_magnitude_per_feature": per_feature_sparse.mean(axis=0).tolist(),
                "low_rank_explained": float(np.linalg.norm(L) / (np.linalg.norm(X) + 1e-10)),
            },
        )

    def _rpca_alm(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Robust PCA via Augmented Lagrangian Method.
        Decomposes X = L + S where L is low-rank and S is sparse.
        """
        n, m = X.shape
        lam = 1.0 / np.sqrt(max(n, m))
        mu = n * m / (4.0 * np.abs(X).sum() + 1e-10)
        mu_inv = 1.0 / mu

        L = np.zeros_like(X)
        S = np.zeros_like(X)
        Y = np.zeros_like(X)

        for _ in range(self.max_iter):
            # Update L via SVD thresholding
            U, sigma, Vt = np.linalg.svd(X - S + mu_inv * Y, full_matrices=False)
            sigma_thresh = np.maximum(sigma - mu_inv, 0)
            L = U * sigma_thresh @ Vt

            # Update S via soft thresholding
            temp = X - L + mu_inv * Y
            S = np.sign(temp) * np.maximum(np.abs(temp) - lam * mu_inv, 0)

            # Update Lagrange multiplier
            residual = X - L - S
            Y = Y + mu * residual

            # Check convergence
            if np.linalg.norm(residual) / (np.linalg.norm(X) + 1e-10) < self.tol:
                break

        return L, S

    def get_penalty(self, dcv: dict) -> float:
        n = dcv.get("n", 0)
        d = dcv.get("d", 0)
        if n * d > 5_000_000:
            return 0.4  # SVD is expensive for very large matrices
        return 0.0
