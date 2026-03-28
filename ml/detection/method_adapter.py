"""
AutoClin Engine Method Adapter — standardized interface for all detection methods.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional
import numpy as np


@dataclass
class DetectionResult:
    method_name: str
    anomaly_scores: np.ndarray     # (n_rows,), range [0, 1]
    labels: Optional[np.ndarray] = None
    cluster_labels: Optional[np.ndarray] = None
    duration_ms: int = 0
    params: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)


class BaseDetector(ABC):
    name: str = "base"
    explainability_prior: float = 0.5

    @abstractmethod
    def fit_score(self, X: np.ndarray, **kwargs) -> DetectionResult:
        pass

    def normalize_scores(self, scores: np.ndarray) -> np.ndarray:
        if len(scores) == 0:
            return scores
        smin, smax = scores.min(), scores.max()
        if smax - smin < 1e-10:
            return np.zeros_like(scores)
        return (scores - smin) / (smax - smin)

    def is_applicable(self, dcv: dict) -> bool:
        return True

    def get_penalty(self, dcv: dict) -> float:
        return 0.0
