"""
AutoClin Engine Method Selector
Auto-selects best method or builds weighted ensemble based on composite scores.
"""
from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass
class SelectionResult:
    mode: str  # "single", "ensemble", "fallback"
    selected_methods: list[str]
    weights: dict[str, float]  # method_name -> weight
    rationale: str


class MethodSelector:
    """
    Selection Logic (from blueprint):
    - Top method > 0.70 → single method
    - Top-3 within 0.05 of each other → weighted ensemble
    - No method > 0.50 → fallback (Isolation Forest + LOF)
    """

    def __init__(self, single_threshold: float = 0.70, ensemble_spread: float = 0.05,
                 minimum_threshold: float = 0.50):
        self.single_threshold = single_threshold
        self.ensemble_spread = ensemble_spread
        self.minimum_threshold = minimum_threshold

    def select(self, method_scores: list) -> SelectionResult:
        """
        Args:
            method_scores: List of MethodScore objects, sorted by composite desc.
        """
        if not method_scores:
            return SelectionResult(
                mode="fallback",
                selected_methods=["isolation_forest", "lof"],
                weights={"isolation_forest": 0.5, "lof": 0.5},
                rationale="No methods available; using conservative fallback.",
            )

        sorted_scores = sorted(method_scores, key=lambda m: m.composite, reverse=True)
        top = sorted_scores[0]

        # Case 1: Clear winner
        if top.composite > self.single_threshold:
            return SelectionResult(
                mode="single",
                selected_methods=[top.method_name],
                weights={top.method_name: 1.0},
                rationale=(
                    f"{top.method_name} selected as sole method with composite score "
                    f"{top.composite:.3f} (>{self.single_threshold}). "
                    f"AD={top.ad:.3f}, SS={top.ss:.3f}, CP={top.cp:.3f}."
                ),
            )

        # Case 2: Top-3 are close → ensemble
        top3 = sorted_scores[:min(3, len(sorted_scores))]
        if len(top3) >= 2:
            spread = top3[0].composite - top3[-1].composite
            if spread < self.ensemble_spread and top3[0].composite > self.minimum_threshold:
                total_score = sum(m.composite for m in top3)
                weights = {m.method_name: m.composite / total_score for m in top3}
                return SelectionResult(
                    mode="ensemble",
                    selected_methods=[m.method_name for m in top3],
                    weights=weights,
                    rationale=(
                        f"Top-{len(top3)} methods within {spread:.3f} spread. "
                        f"Ensemble: {', '.join(f'{m.method_name}({w:.2f})' for m, w in zip(top3, weights.values()))}."
                    ),
                )

        # Case 3: No method exceeds minimum → fallback
        if top.composite < self.minimum_threshold:
            return SelectionResult(
                mode="fallback",
                selected_methods=["isolation_forest", "lof"],
                weights={"isolation_forest": 0.5, "lof": 0.5},
                rationale=(
                    f"No method exceeded minimum threshold ({self.minimum_threshold}). "
                    f"Best: {top.method_name} at {top.composite:.3f}. "
                    f"Using conservative Isolation Forest + LOF fallback."
                ),
            )

        # Case 4: Top method is decent but not outstanding → single selection
        return SelectionResult(
            mode="single",
            selected_methods=[top.method_name],
            weights={top.method_name: 1.0},
            rationale=(
                f"{top.method_name} selected with composite {top.composite:.3f}. "
                f"Below ideal threshold but best available."
            ),
        )

    def compute_ensemble_scores(
        self,
        method_results: dict[str, np.ndarray],
        weights: dict[str, float],
    ) -> np.ndarray:
        """Compute weighted ensemble anomaly scores."""
        n = None
        for scores in method_results.values():
            n = len(scores)
            break
        if n is None:
            return np.array([])

        ensemble = np.zeros(n)
        total_weight = 0.0
        for method_name, weight in weights.items():
            if method_name in method_results:
                ensemble += weight * method_results[method_name]
                total_weight += weight

        if total_weight > 0:
            ensemble /= total_weight
        return ensemble
