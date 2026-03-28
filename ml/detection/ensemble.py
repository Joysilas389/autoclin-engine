"""
AutoClin Engine Ensemble Engine
Weighted combination of method scores with disagreement tracking.
"""
import numpy as np
from dataclasses import dataclass, field


@dataclass
class EnsembleResult:
    final_scores: np.ndarray
    anomaly_flags: np.ndarray  # boolean
    threshold: float
    method_agreement: np.ndarray  # per-row: fraction of methods agreeing on flag
    ambiguous_indices: list[int]  # rows where methods disagree
    mode: str  # single, ensemble, fallback


class EnsembleEngine:
    """Combines method results with weighted scoring and disagreement tracking."""

    def __init__(self, ambiguity_threshold: float = 0.4):
        self.ambiguity_threshold = ambiguity_threshold

    def combine(
        self,
        method_results: dict[str, np.ndarray],
        weights: dict[str, float],
        contamination: float,
        mode: str = "ensemble",
    ) -> EnsembleResult:
        methods = list(method_results.keys())
        n = len(next(iter(method_results.values())))

        # Weighted average scores
        final_scores = np.zeros(n)
        total_weight = 0.0
        for method_name, weight in weights.items():
            if method_name in method_results:
                final_scores += weight * method_results[method_name]
                total_weight += weight
        if total_weight > 0:
            final_scores /= total_weight

        # Threshold
        threshold = np.quantile(final_scores, 1.0 - contamination) if contamination > 0 else 0.5
        anomaly_flags = final_scores > threshold

        # Per-row method agreement
        per_method_flags = {}
        for method_name, scores in method_results.items():
            method_thresh = np.quantile(scores, 1.0 - contamination) if contamination > 0 else 0.5
            per_method_flags[method_name] = scores > method_thresh

        agreement = np.zeros(n)
        if len(per_method_flags) > 0:
            flag_matrix = np.column_stack(list(per_method_flags.values()))
            agreement = flag_matrix.mean(axis=1)

        # Ambiguous cases: flagged by ensemble but low agreement, or not flagged but high individual flags
        ambiguous = []
        for i in range(n):
            if anomaly_flags[i] and agreement[i] < (1.0 - self.ambiguity_threshold):
                ambiguous.append(i)
            elif not anomaly_flags[i] and agreement[i] > self.ambiguity_threshold:
                ambiguous.append(i)

        return EnsembleResult(
            final_scores=final_scores,
            anomaly_flags=anomaly_flags,
            threshold=threshold,
            method_agreement=agreement,
            ambiguous_indices=ambiguous,
            mode=mode,
        )
