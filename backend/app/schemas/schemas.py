"""
AutoClin Engine API Schemas
Pydantic models for request/response validation.
"""
from pydantic import BaseModel, Field
from typing import Optional, Any
from datetime import datetime
from uuid import UUID
from enum import Enum


# ─── Enums ───────────────────────────────────────────────
class DatasetFormat(str, Enum):
    CSV = "csv"
    TSV = "tsv"
    XLSX = "xlsx"
    XLS = "xls"
    JSON = "json"
    XML = "xml"
    PARQUET = "parquet"
    SQL = "sql"
    REDCAP = "redcap"
    EDC = "edc"
    FHIR = "fhir"


class Severity(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class AutoCleanRisk(str, Enum):
    SAFE = "safe"
    CAUTION = "caution"
    MANUAL_ONLY = "manual_only"


class PipelineMode(str, Enum):
    SUGGESTION = "suggestion"
    AUTO = "auto"


class AdjudicationAction(str, Enum):
    ACCEPT = "accept"
    DISMISS = "dismiss"
    MODIFY = "modify"


# ─── Dataset ─────────────────────────────────────────────
class DatasetUploadResponse(BaseModel):
    dataset_id: UUID
    filename: str
    format: str
    size_bytes: int
    status: str


class ColumnProfile(BaseModel):
    name: str
    inferred_type: str
    type_confidence: float
    nullity_rate: float
    unique_count: int
    mixed_type: bool = False
    mixed_type_proportion: float = 0.0
    sample_values: list[Any] = []
    stats: Optional[dict] = None


class DatasetProfile(BaseModel):
    dataset_id: UUID
    row_count: int
    col_count: int
    overall_missingness: float
    duplicate_proportion: float
    columns: list[ColumnProfile]
    type_distribution: dict[str, float]  # {numeric: 0.4, categorical: 0.3, ...}
    noise_estimate: float


# ─── Clinical Mapping ────────────────────────────────────
class ClinicalFieldMapping(BaseModel):
    column_name: str
    clinical_type: Optional[str] = None  # patient_id, visit_date, lab_value, vital_sign, etc.
    confidence: float = 0.0
    reference_range: Optional[dict] = None  # {min, max, unit}
    user_confirmed: bool = False


class ClinicalMapUpdate(BaseModel):
    mappings: list[ClinicalFieldMapping]


# ─── Pipeline ────────────────────────────────────────────
class PipelineRunConfig(BaseModel):
    dataset_id: UUID
    mode: PipelineMode = PipelineMode.SUGGESTION
    methods: str = "auto"  # "auto" or comma-separated method names
    confidence_threshold: float = 0.5
    contamination_estimate: Optional[float] = None
    require_user_approval: bool = False


class PipelineStatus(BaseModel):
    run_id: UUID
    status: str
    current_phase: int
    phase_name: str
    progress_pct: float
    estimated_remaining_seconds: Optional[int] = None


# ─── Method Results ──────────────────────────────────────
class MethodScores(BaseModel):
    ad: float  # Anomaly Discrimination
    ss: float  # Stability Score
    cp: float  # Clinical Plausibility
    ex: float  # Explainability Score
    cc: float  # Computational Cost
    ndc: float  # Noise Detection Confidence
    composite: float


class MethodRanking(BaseModel):
    method: str
    scores: MethodScores
    anomaly_count: int
    duration_ms: int
    selected: bool


# ─── Anomalies ───────────────────────────────────────────
class FeatureContribution(BaseModel):
    feature: str
    contribution: float


class AnomalyExplanation(BaseModel):
    summary: str
    feature_contributions: dict[str, float]
    nearest_normal_id: Optional[int] = None
    clean_twin: Optional[dict] = None


class AnomalyResponse(BaseModel):
    anomaly_id: UUID
    row_index: int
    anomaly_type: str
    severity: Severity
    confidence: float
    primary_method: str
    ensemble_agreement: Optional[float] = None
    flagged_columns: list[str]
    explanation: AnomalyExplanation
    recommended_action: str
    alternative_actions: list[str]
    auto_clean_risk: AutoCleanRisk
    status: str = "pending"


class AnomalyListResponse(BaseModel):
    total: int
    page: int
    page_size: int
    anomalies: list[AnomalyResponse]


class AdjudicateRequest(BaseModel):
    action: AdjudicationAction
    modified_value: Optional[Any] = None
    notes: Optional[str] = None


# ─── Cleaning ────────────────────────────────────────────
class CleaningActionItem(BaseModel):
    anomaly_id: UUID
    row_index: int
    column_name: str
    action: str
    original_value: Any
    proposed_value: Optional[Any] = None
    risk_level: AutoCleanRisk
    rationale: str


class CleaningPlan(BaseModel):
    run_id: UUID
    total_actions: int
    auto_safe_count: int
    manual_review_count: int
    actions: list[CleaningActionItem]


# ─── Before vs After ─────────────────────────────────────
class QualitySnapshot(BaseModel):
    noise_pct: float
    trust_score: float
    missingness_pct: float
    duplicate_pct: float
    plausibility_rate: float
    column_errors: dict[str, int]


class BeforeAfterResponse(BaseModel):
    run_id: UUID
    before: QualitySnapshot
    after: QualitySnapshot
    noise_reduction_pct: float
    trust_improvement: float
    missingness_reduction_pct: float
    duplicate_elimination_pct: float
    cleaning_induced_artifacts: int


# ─── Reports ─────────────────────────────────────────────
class ReportGenerateRequest(BaseModel):
    run_id: UUID
    format: str = "pdf"  # pdf, xlsx, csv, docx
    sections: Optional[list[str]] = None  # None = all sections


class ReportResponse(BaseModel):
    report_id: UUID
    format: str
    file_path: str
    generated_at: datetime
    file_size_bytes: Optional[int] = None


# ─── Full Pipeline Result ────────────────────────────────
class PipelineResult(BaseModel):
    pipeline_run_id: UUID
    dataset_id: UUID
    status: str
    selection_mode: str  # single, ensemble, fallback
    selected_methods: list[str]
    total_rows: int
    total_anomalies: int
    noise_percentage: float
    trust_score: float
    method_rankings: list[MethodRanking]
    before_after: Optional[BeforeAfterResponse] = None
