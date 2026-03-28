"""
AutoClin Engine Pipeline Orchestrator
Runs the complete 9-phase pipeline from ingestion to reporting.
"""
import time
import logging
from dataclasses import dataclass, field
from typing import Optional, Callable

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

from ml.ingestion.schema_inference import SchemaInferenceEngine
from ml.ingestion.duplicate_detector import DuplicateDetector
from ml.profiling.profiler import DataProfiler
from ml.profiling.clinical_mapper import ClinicalFieldMapper
from ml.profiling.dcv_calculator import DCVCalculator
from ml.detection.method_adapter import DetectionResult
from ml.detection.ensemble import EnsembleEngine
from ml.detection.methods.isolation_forest import IsolationForestDetector
from ml.detection.methods.hdbscan_detector import HDBSCANDetector
from ml.detection.methods.lof_detector import LOFDetector
from ml.detection.methods.autoencoder import AutoencoderDetector
from ml.detection.methods.robust_pca import RobustPCADetector
from ml.benchmarking.scorer import MethodScorer
from ml.benchmarking.selector import MethodSelector
from ml.taxonomy.classifier import AnomalyTaxonomyClassifier
from ml.explainability.feature_contrib import FeatureContributionExtractor
from ml.explainability.counterfactual import CounterfactualGenerator
from ml.explainability.explanation_cards import ExplanationCardGenerator
from ml.explainability.global_summary import GlobalSummaryGenerator
from ml.cleaning.action_mapper import CleaningActionMapper
from ml.cleaning.plausibility_engine import PlausibilityEngine
from ml.cleaning.audit_ledger import AuditLedger

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    mode: str = "suggestion"           # suggestion | auto
    methods: str = "auto"              # auto | comma-separated names
    confidence_threshold: float = 0.5
    contamination_estimate: Optional[float] = None
    random_seed: int = 42
    require_user_approval: bool = False


@dataclass
class PipelineResult:
    run_id: str
    status: str
    selection_mode: str
    selected_methods: list[str]
    total_rows: int
    total_anomalies: int
    noise_percentage: float
    trust_score: float
    method_rankings: list
    anomaly_classifications: list
    cleaning_recommendations: list
    before_metrics: dict
    after_metrics: Optional[dict]
    global_summary: dict
    audit_entries: list
    duration_ms: int


class PipelineOrchestrator:
    """
    Orchestrates the 9-phase AutoClin Engine pipeline.
    
    Phase 1: Ingestion (schema inference)
    Phase 2: Profiling
    Phase 3: Clinical Mapping
    Phase 4: Preprocessing
    Phase 5: Multi-Engine Detection
    Phase 6: Method Benchmarking
    Phase 7: Ensemble & Selection
    Phase 8: Cleaning Recommendation
    Phase 9: Reporting & Export
    """

    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()
        self.progress_callback: Optional[Callable] = None

    def set_progress_callback(self, callback: Callable):
        """Set callback for phase progress updates: callback(phase, pct, message)"""
        self.progress_callback = callback

    def _update_progress(self, phase: int, pct: float, message: str):
        if self.progress_callback:
            self.progress_callback(phase, pct, message)
        logger.info(f"Phase {phase} ({pct:.0f}%): {message}")

    def run(self, df: pd.DataFrame, run_id: str = "run-001") -> PipelineResult:
        """Execute the full 9-phase pipeline."""
        pipeline_start = time.time()

        # ── Phase 1: Schema Inference ──
        self._update_progress(1, 0, "Inferring schema...")
        schema_engine = SchemaInferenceEngine()
        schema = schema_engine.infer(df)
        self._update_progress(1, 50, "Detecting duplicates...")
        dup_detector = DuplicateDetector()
        dup_report = dup_detector.detect(df)
        self._update_progress(1, 100, "Ingestion complete")

        # ── Phase 2: Profiling ──
        self._update_progress(2, 0, "Profiling dataset...")
        profiler = DataProfiler()
        profile = profiler.profile(df, schema=schema, duplicate_rate=dup_report.total_duplicate_rate)
        self._last_profile = profile  # Store for server access
        self._update_progress(2, 100, "Profiling complete")

        # ── Phase 3: Clinical Mapping ──
        self._update_progress(3, 0, "Detecting clinical fields...")
        mapper = ClinicalFieldMapper()
        clinical_mappings = mapper.map_fields(df)
        self._last_clinical_map = clinical_mappings  # Store for server
        self._update_progress(3, 100, "Clinical mapping complete")

        # ── Phase 4: Preprocessing ──
        self._update_progress(4, 0, "Preprocessing for ML...")
        X, feature_names = self._preprocess(df, schema, profile)
        self._update_progress(4, 50, "Computing dataset characterization...")
        dcv_calc = DCVCalculator()
        dcv = dcv_calc.compute(df, profile, clinical_mappings)
        contamination = self.config.contamination_estimate or profile.noise_estimate or 0.05
        contamination = max(0.01, min(contamination, 0.2))
        self._update_progress(4, 100, "Preprocessing complete")

        # ── Phase 5: Multi-Engine Detection ──
        self._update_progress(5, 0, "Running detection methods...")
        detectors = self._get_detectors(dcv.to_dict())
        method_results: dict[str, DetectionResult] = {}
        for i, det in enumerate(detectors):
            self._update_progress(5, int((i / len(detectors)) * 100),
                                  f"Running {det.name}...")
            try:
                result = det.fit_score(X, contamination=contamination)
                method_results[det.name] = result
            except Exception as e:
                logger.warning(f"Method {det.name} failed: {e}")
        self._update_progress(5, 100, f"{len(method_results)} methods completed")

        # ── Phase 6: Benchmarking ──
        self._update_progress(6, 0, "Scoring methods...")
        scorer = MethodScorer(bootstrap_samples=8)
        plausibility_engine = PlausibilityEngine()
        clinical_flags = plausibility_engine.compute_plausibility_flags(df, clinical_mappings)

        max_duration = max((r.duration_ms for r in method_results.values()), default=1) or 1
        method_scores = []
        for name, result in method_results.items():
            det = next((d for d in detectors if d.name == name), None)
            if det is None:
                continue
            det._last_duration_ms = result.duration_ms
            ms = scorer.score_method(
                name, result.anomaly_scores, X, det, contamination,
                clinical_flags=clinical_flags, max_duration=max_duration,
                dcv=dcv.to_dict(),
            )
            method_scores.append(ms)
        self._update_progress(6, 100, "Benchmarking complete")

        # ── Phase 7: Ensemble & Selection ──
        self._update_progress(7, 0, "Selecting best method...")
        selector = MethodSelector()
        selection = selector.select(method_scores)

        scores_dict = {name: r.anomaly_scores for name, r in method_results.items()}
        ensemble_engine = EnsembleEngine()
        ensemble_result = ensemble_engine.combine(
            scores_dict, selection.weights, contamination, selection.mode,
        )

        flagged_indices = np.where(ensemble_result.anomaly_flags)[0]
        self._update_progress(7, 100,
            f"{len(flagged_indices)} anomalies flagged ({selection.mode} mode)")

        # ── Explainability ──
        contrib_extractor = FeatureContributionExtractor()
        primary_method = selection.selected_methods[0]
        primary_result = method_results.get(primary_method)

        contributions_list = []
        if primary_result and len(flagged_indices) > 0:
            contributions_list = contrib_extractor.extract_batch(
                primary_method, X, flagged_indices, primary_result, feature_names,
            )

        # Counterfactual clean twins
        cf_generator = CounterfactualGenerator()
        clean_twins = cf_generator.generate_batch(
            flagged_indices, df, X, ensemble_result.anomaly_flags, feature_names,
        )

        # Taxonomy classification
        taxonomy = AnomalyTaxonomyClassifier(clinical_mappings)
        cluster_labels = None
        if primary_result and primary_result.cluster_labels is not None:
            cluster_labels = primary_result.cluster_labels

        classifications = taxonomy.classify_batch(
            df, flagged_indices, ensemble_result.final_scores,
            contributions_list, primary_method,
            cluster_labels=cluster_labels,
            agreement_scores=ensemble_result.method_agreement,
        )

        # Explanation cards
        card_gen = ExplanationCardGenerator()
        for i, cls in enumerate(classifications):
            row_data = df.iloc[cls.row_index].to_dict()
            contribs = contributions_list[i] if i < len(contributions_list) else {}
            clinical_ctx = next(
                (self._clinical_ctx(m) for m in clinical_mappings
                 if m.column_name in cls.flagged_columns and m.reference_range),
                None,
            )
            cls.explanation_card = card_gen.generate_card(
                cls.row_index, cls.anomaly_type, cls.severity, cls.confidence,
                cls.flagged_columns, contribs, row_data,
                clean_twin=clean_twins[i] if i < len(clean_twins) else None,
                clinical_context=clinical_ctx,
            )

        # Global summary
        summary_gen = GlobalSummaryGenerator()
        global_summary = summary_gen.generate(
            contributions_list, classifications, feature_names,
            len(df), clinical_mappings,
        )
        self._update_progress(7, 100, "Explainability complete")

        # ── Phase 8: Cleaning Recommendations ──
        self._update_progress(8, 0, "Mapping cleaning actions...")
        action_mapper = CleaningActionMapper()
        cleaning_recs = action_mapper.map_actions(
            classifications,
            row_data_fn=lambda idx: df.iloc[idx].to_dict() if idx < len(df) else {},
        )

        audit_ledger = AuditLedger(run_id=run_id, random_seed=self.config.random_seed)

        # Auto-clean if configured
        cleaned_df = None
        if self.config.mode == "auto":
            safe_actions = action_mapper.get_auto_safe_actions(cleaning_recs)
            cleaned_df = df.copy()
            for action in safe_actions:
                validation = plausibility_engine.validate_correction(
                    action.column_name,
                    action.proposed_value or 0,
                    action.anomaly_type,
                )
                if action.proposed_value is not None and validation["valid"]:
                    cleaned_df.at[action.row_index, action.column_name] = action.proposed_value
                    audit_ledger.log(
                        row_index=action.row_index, column_name=action.column_name,
                        original_value=action.original_value, new_value=action.proposed_value,
                        action=action.recommended_action, method_trigger=primary_method,
                        risk_level=action.auto_clean_risk, rationale=action.rationale,
                    )
        self._update_progress(8, 100, "Cleaning recommendations ready")

        # ── Phase 9: Compute Quality Metrics ──
        self._update_progress(9, 0, "Computing quality metrics...")
        before_metrics = self._compute_quality_metrics(
            df, ensemble_result.anomaly_flags, profile, dup_report, clinical_flags,
            classifications=classifications,
        )

        after_metrics = None
        if cleaned_df is not None:
            # Re-run detection on cleaned data (verification)
            X_clean, _ = self._preprocess(cleaned_df, schema, profile)
            if X_clean.shape == X.shape:
                try:
                    det_rerun = next(d for d in detectors if d.name == primary_method)
                    rerun_result = det_rerun.fit_score(X_clean, contamination=contamination)
                    threshold = np.quantile(rerun_result.anomaly_scores, 1.0 - contamination)
                    after_flags = rerun_result.anomaly_scores > threshold
                    dup_after = DuplicateDetector().detect(cleaned_df)
                    clinical_flags_after = plausibility_engine.compute_plausibility_flags(
                        cleaned_df, clinical_mappings,
                    )
                    after_profile = DataProfiler().profile(cleaned_df, duplicate_rate=dup_after.total_duplicate_rate)
                    after_metrics = self._compute_quality_metrics(
                        cleaned_df, after_flags, after_profile, dup_after, clinical_flags_after,
                    )
                except Exception as e:
                    logger.warning(f"Re-run verification failed: {e}")

        duration_ms = int((time.time() - pipeline_start) * 1000)
        noise_pct = before_metrics.get("noise_pct", 0)
        trust = before_metrics.get("trust_score", 0)
        self._update_progress(9, 100,
            f"Pipeline complete in {duration_ms/1000:.1f}s — "
            f"{len(flagged_indices)} anomalies, {noise_pct:.1f}% noise, trust={trust:.0f}")

        result = PipelineResult(
            run_id=run_id,
            status="completed",
            selection_mode=selection.mode,
            selected_methods=selection.selected_methods,
            total_rows=len(df),
            total_anomalies=len(flagged_indices),
            noise_percentage=noise_pct,
            trust_score=trust,
            method_rankings=[{
                "method": ms.method_name, "composite": ms.composite,
                "ad": ms.ad, "ss": ms.ss, "cp": ms.cp, "ex": ms.ex, "cc": ms.cc,
                "ndc": ms.ndc, "anomaly_count": ms.anomaly_count,
                "duration_ms": ms.duration_ms, "selected": ms.method_name in selection.selected_methods,
            } for ms in sorted(method_scores, key=lambda m: m.composite, reverse=True)],
            anomaly_classifications=classifications,
            cleaning_recommendations=cleaning_recs,
            before_metrics=before_metrics,
            after_metrics=after_metrics,
            global_summary=global_summary,
            audit_entries=audit_ledger.to_records(),
            duration_ms=duration_ms,
        )
        # Attach scores for transparency features (borderline detection)
        result._final_scores = ensemble_result.final_scores
        result._threshold = ensemble_result.threshold if hasattr(ensemble_result, 'threshold') else None
        return result

    def _get_detectors(self, dcv: dict) -> list:
        """Return applicable detectors based on config and DCV."""
        all_detectors = [
            IsolationForestDetector(random_state=self.config.random_seed),
            HDBSCANDetector(),
            LOFDetector(),
            AutoencoderDetector(random_state=self.config.random_seed),
            RobustPCADetector(random_state=self.config.random_seed),
        ]

        if self.config.methods != "auto":
            requested = set(self.config.methods.split(","))
            return [d for d in all_detectors if d.name in requested]

        return [d for d in all_detectors if d.is_applicable(dcv)]

    def _preprocess(self, df, schema, profile) -> tuple[np.ndarray, list[str]]:
        """Convert DataFrame to ML-ready numeric feature matrix.
        
        Handles:
        - Numeric columns → standardized
        - Low-cardinality categoricals → one-hot encoded
        - Free text / high-cardinality strings → text feature extraction
          (length, word count, char ratios, entropy, TF-IDF/SVD)
        - Identifiers → excluded (not informative for anomaly detection)
        """
        feature_cols = []
        encoded_parts = []

        for col_profile in profile.columns:
            col = col_profile.name
            if col not in df.columns:
                continue

            if col_profile.dtype == "numeric":
                numeric = pd.to_numeric(df[col], errors="coerce")
                median_val = numeric.median()
                numeric = numeric.fillna(median_val if not pd.isna(median_val) else 0)
                encoded_parts.append(numeric.values.reshape(-1, 1))
                feature_cols.append(col)

            elif col_profile.dtype == "categorical" and col_profile.unique_count <= 50:
                dummies = pd.get_dummies(df[col].fillna("__MISSING__"), prefix=col, drop_first=True)
                if dummies.shape[1] <= 20:
                    encoded_parts.append(dummies.values)
                    feature_cols.extend(dummies.columns.tolist())

        # Text feature extraction for free_text and high-cardinality columns
        try:
            from ml.preprocessing.text_features import extract_text_features_for_dataset
            X_text, text_names = extract_text_features_for_dataset(df, profile)
            if X_text is not None and X_text.shape[1] > 0:
                encoded_parts.append(X_text)
                feature_cols.extend(text_names)
                logger.info(f"Text features extracted: {len(text_names)} features from text columns")
        except Exception as e:
            logger.debug(f"Text feature extraction skipped: {e}")

        if not encoded_parts:
            return np.zeros((len(df), 1)), ["__empty__"]

        X = np.hstack(encoded_parts).astype(np.float64)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        # Standardize
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        return X, feature_cols

    def _compute_quality_metrics(self, df, anomaly_flags, profile, dup_report, clinical_flags,
                                 classifications=None):
        """Compute comprehensive quality metrics."""
        n = len(df)
        noise_pct = round(anomaly_flags.sum() / n * 100, 2) if n > 0 else 0
        miss_pct = round(profile.overall_missingness * 100, 2)
        dup_pct = round(dup_report.total_duplicate_rate * 100, 2)
        plausibility = round((1 - clinical_flags.mean()) * 100, 2) if len(clinical_flags) > 0 else 100.0

        # Trust score: weighted composite
        completeness = max(0, 100 - miss_pct)
        consistency = max(0, 100 - noise_pct)
        plaus = plausibility
        uniqueness = max(0, 100 - dup_pct)
        trust = round(
            0.25 * completeness + 0.30 * consistency + 0.25 * plaus + 0.20 * uniqueness, 1
        )

        # Column-wise error counts from actual classifications
        col_errors = {}
        if classifications:
            for cls in classifications:
                for col in cls.flagged_columns:
                    col_errors[col] = col_errors.get(col, 0) + 1
        else:
            for cp in profile.columns:
                col_errors[cp.name] = 0

        # Column error burden: sum of per-column anomaly counts / (rows * cols)
        total_col_errors = sum(col_errors.values())
        col_error_burden = round(total_col_errors / (n * len(df.columns)) * 100, 4) if n > 0 else 0

        return {
            "noise_pct": noise_pct,
            "trust_score": trust,
            "missingness_pct": miss_pct,
            "duplicate_pct": dup_pct,
            "plausibility_rate": plausibility,
            "column_errors": col_errors,
            "column_error_burden": col_error_burden,
            "total_rows": n,
            "total_anomalies": int(anomaly_flags.sum()),
        }

    def _clinical_ctx(self, mapping):
        if mapping.reference_range:
            return {"reference_range": mapping.reference_range}
        return None
