"""
AutoClin Engine Method Benchmarking — scores each method on 6 dimensions.
"""
from dataclasses import dataclass
import numpy as np
from typing import Optional


@dataclass
class MethodScore:
    method_name: str
    ad: float
    ss: float
    cp: float
    ex: float
    cc: float
    ndc: float
    composite: float
    anomaly_count: int
    duration_ms: int


class MethodScorer:
    def __init__(self, bootstrap_samples: int = 20):
        self.B = bootstrap_samples
        self.weights = {"ndc": 0.35, "cp": 0.25, "ex": 0.20, "cc": 0.10, "penalty": 0.10}

    def score_method(self, method_name, scores, X, detector, contamination,
                     clinical_flags=None, max_duration=1.0, dcv=None) -> MethodScore:
        threshold = np.quantile(scores, 1.0 - contamination) if contamination > 0 else 0.5
        flagged = scores > threshold
        anomaly_count = int(flagged.sum())

        ad = self._anomaly_discrimination(scores, flagged)
        ss = self._stability_score(detector, X, contamination)
        cp = self._clinical_plausibility(flagged, clinical_flags)
        ex = detector.explainability_prior
        duration = getattr(detector, '_last_duration_ms', 0)
        cc = max(0.0, 1.0 - (duration / (max_duration + 1e-10))) if max_duration > 0 else 0.5
        ndc = 0.6 * ad + 0.4 * ss
        penalty_adj = 1.0 - detector.get_penalty(dcv or {})

        composite = (self.weights["ndc"] * ndc + self.weights["cp"] * cp +
                     self.weights["ex"] * ex + self.weights["cc"] * cc +
                     self.weights["penalty"] * penalty_adj)

        return MethodScore(method_name=method_name, ad=round(ad, 4), ss=round(ss, 4),
                           cp=round(cp, 4), ex=round(ex, 4), cc=round(cc, 4),
                           ndc=round(ndc, 4), composite=round(composite, 4),
                           anomaly_count=anomaly_count, duration_ms=duration)

    def _anomaly_discrimination(self, scores, flagged):
        if flagged.sum() == 0 or flagged.sum() == len(flagged):
            return 0.0
        a_scores = scores[flagged]
        n_scores = scores[~flagged]
        if len(a_scores) == 0 or len(n_scores) == 0:
            return 0.0
        u = sum((n_scores < a).sum() for a in a_scores)
        return float(np.clip(u / (len(a_scores) * len(n_scores)), 0, 1))

    def _stability_score(self, detector, X, contamination):
        n = len(X)
        if n < 20:
            return 0.5
        B_actual = min(self.B, 8)
        rng = np.random.RandomState(42)
        flagged_sets = []
        for _ in range(B_actual):
            idx = rng.choice(n, size=n, replace=True)
            try:
                result = detector.fit_score(X[idx], contamination=contamination)
                thr = np.quantile(result.anomaly_scores, 1.0 - contamination)
                original_flagged = set(idx[np.where(result.anomaly_scores > thr)[0]])
                flagged_sets.append(original_flagged)
            except Exception:
                continue
        if len(flagged_sets) < 2:
            return 0.5
        jsum, cnt = 0.0, 0
        for i in range(len(flagged_sets)):
            for j in range(i + 1, len(flagged_sets)):
                u = len(flagged_sets[i] | flagged_sets[j])
                if u > 0:
                    jsum += len(flagged_sets[i] & flagged_sets[j]) / u
                cnt += 1
        return jsum / cnt if cnt > 0 else 0.5

    def _clinical_plausibility(self, flagged, clinical_flags):
        if clinical_flags is None or flagged.sum() == 0:
            return 0.5
        return float(clinical_flags[flagged].mean())
