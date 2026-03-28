"""
AutoClin Engine Database Models
All core tables with row-level security via org_id.
"""
import uuid
from datetime import datetime, timezone
from sqlalchemy import (
    Column, String, Integer, Float, Boolean, DateTime, ForeignKey, Text, Enum,
    Index, UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import relationship

from app.core.database import Base


def utcnow():
    return datetime.now(timezone.utc)


def new_uuid():
    return uuid.uuid4()


class Organization(Base):
    __tablename__ = "organizations"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=new_uuid)
    name = Column(String(255), nullable=False)
    plan = Column(String(50), default="free")  # free, pro, enterprise
    settings = Column(JSONB, default=dict)
    created_at = Column(DateTime(timezone=True), default=utcnow)
    
    users = relationship("User", back_populates="organization")
    projects = relationship("Project", back_populates="organization")


class User(Base):
    __tablename__ = "users"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=new_uuid)
    email = Column(String(255), unique=True, nullable=False, index=True)
    name = Column(String(255), nullable=False)
    hashed_password = Column(String(255), nullable=False)
    org_id = Column(UUID(as_uuid=True), ForeignKey("organizations.id"), nullable=False)
    role = Column(String(50), default="analyst")  # admin, manager, analyst, viewer
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), default=utcnow)
    updated_at = Column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)
    
    organization = relationship("Organization", back_populates="users")
    projects = relationship("Project", back_populates="owner")


class Project(Base):
    __tablename__ = "projects"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=new_uuid)
    name = Column(String(255), nullable=False)
    description = Column(Text, default="")
    org_id = Column(UUID(as_uuid=True), ForeignKey("organizations.id"), nullable=False)
    owner_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    status = Column(String(50), default="active")  # active, archived
    created_at = Column(DateTime(timezone=True), default=utcnow)
    
    organization = relationship("Organization", back_populates="projects")
    owner = relationship("User", back_populates="projects")
    datasets = relationship("Dataset", back_populates="project")
    
    __table_args__ = (
        Index("ix_projects_org", "org_id"),
    )


class Dataset(Base):
    __tablename__ = "datasets"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=new_uuid)
    project_id = Column(UUID(as_uuid=True), ForeignKey("projects.id"), nullable=False)
    org_id = Column(UUID(as_uuid=True), nullable=False, index=True)
    filename = Column(String(500), nullable=False)
    format = Column(String(50), nullable=False)  # csv, xlsx, json, parquet, etc.
    size_bytes = Column(Integer, default=0)
    row_count = Column(Integer)
    col_count = Column(Integer)
    upload_path = Column(String(1000), nullable=False)
    status = Column(String(50), default="uploaded")  # uploaded, profiling, profiled, analyzing, completed, error
    profile = Column(JSONB)  # Full profiling results
    clinical_map = Column(JSONB)  # Clinical field mappings
    created_at = Column(DateTime(timezone=True), default=utcnow)
    
    project = relationship("Project", back_populates="datasets")
    pipeline_runs = relationship("PipelineRun", back_populates="dataset")
    
    __table_args__ = (
        Index("ix_datasets_org", "org_id"),
        Index("ix_datasets_project", "project_id"),
    )


class PipelineRun(Base):
    __tablename__ = "pipeline_runs"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=new_uuid)
    dataset_id = Column(UUID(as_uuid=True), ForeignKey("datasets.id"), nullable=False)
    org_id = Column(UUID(as_uuid=True), nullable=False, index=True)
    status = Column(String(50), default="pending")  # pending, running, completed, failed
    current_phase = Column(Integer, default=0)  # 1-9
    config = Column(JSONB, default=dict)  # Pipeline configuration
    dcv = Column(JSONB)  # Dataset Characterization Vector
    started_at = Column(DateTime(timezone=True))
    completed_at = Column(DateTime(timezone=True))
    duration_ms = Column(Integer)
    error_message = Column(Text)
    
    dataset = relationship("Dataset", back_populates="pipeline_runs")
    method_results = relationship("MethodResult", back_populates="pipeline_run")
    anomaly_records = relationship("AnomalyRecord", back_populates="pipeline_run")
    cleaning_audits = relationship("CleaningAudit", back_populates="pipeline_run")
    quality_metrics = relationship("QualityMetric", back_populates="pipeline_run")
    reports = relationship("Report", back_populates="pipeline_run")


class MethodResult(Base):
    __tablename__ = "method_results"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=new_uuid)
    run_id = Column(UUID(as_uuid=True), ForeignKey("pipeline_runs.id"), nullable=False)
    method_name = Column(String(100), nullable=False)
    params = Column(JSONB, default=dict)
    scores = Column(JSONB)  # {ad, ss, cp, ex, cc, ndc, composite}
    anomaly_count = Column(Integer, default=0)
    duration_ms = Column(Integer)
    selected = Column(Boolean, default=False)
    
    pipeline_run = relationship("PipelineRun", back_populates="method_results")
    
    __table_args__ = (
        Index("ix_method_results_run", "run_id"),
    )


class AnomalyRecord(Base):
    __tablename__ = "anomaly_records"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=new_uuid)
    run_id = Column(UUID(as_uuid=True), ForeignKey("pipeline_runs.id"), nullable=False)
    org_id = Column(UUID(as_uuid=True), nullable=False, index=True)
    row_index = Column(Integer, nullable=False)
    anomaly_type = Column(String(100), nullable=False)
    severity = Column(String(20), nullable=False)  # critical, high, medium, low
    confidence = Column(Float, nullable=False)
    method = Column(String(100), nullable=False)
    ensemble_agreement = Column(Float)
    explanation = Column(JSONB)  # Full explanation object
    feature_contributions = Column(JSONB)  # {feature: contribution_score}
    clean_twin = Column(JSONB)  # Counterfactual nearest-normal values
    flagged_columns = Column(JSONB)  # List of column names
    action_recommended = Column(String(100))
    alternative_actions = Column(JSONB)
    auto_clean_risk = Column(String(20))  # safe, caution, manual_only
    action_taken = Column(String(100))
    reviewed_by = Column(UUID(as_uuid=True), ForeignKey("users.id"))
    reviewed_at = Column(DateTime(timezone=True))
    status = Column(String(50), default="pending")  # pending, accepted, dismissed, modified
    
    pipeline_run = relationship("PipelineRun", back_populates="anomaly_records")
    cleaning_audits = relationship("CleaningAudit", back_populates="anomaly_record")
    
    __table_args__ = (
        Index("ix_anomaly_records_run", "run_id"),
        Index("ix_anomaly_records_type", "anomaly_type"),
        Index("ix_anomaly_records_severity", "severity"),
    )


class CleaningAudit(Base):
    __tablename__ = "cleaning_audit"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=new_uuid)
    run_id = Column(UUID(as_uuid=True), ForeignKey("pipeline_runs.id"), nullable=False)
    anomaly_id = Column(UUID(as_uuid=True), ForeignKey("anomaly_records.id"))
    org_id = Column(UUID(as_uuid=True), nullable=False, index=True)
    transaction_id = Column(String(100), nullable=False, unique=True)
    row_index = Column(Integer, nullable=False)
    column_name = Column(String(255), nullable=False)
    original_value = Column(Text)
    new_value = Column(Text)
    action = Column(String(100), nullable=False)
    method_trigger = Column(String(100))
    risk_level = Column(String(20))  # safe, caution, manual_only
    approved_by = Column(UUID(as_uuid=True), ForeignKey("users.id"))
    rationale = Column(Text)
    reproducibility = Column(JSONB)  # {version, seed, params}
    timestamp = Column(DateTime(timezone=True), default=utcnow)
    
    pipeline_run = relationship("PipelineRun", back_populates="cleaning_audits")
    anomaly_record = relationship("AnomalyRecord", back_populates="cleaning_audits")
    
    __table_args__ = (
        Index("ix_cleaning_audit_run", "run_id"),
        Index("ix_cleaning_audit_txn", "transaction_id"),
    )


class QualityMetric(Base):
    __tablename__ = "quality_metrics"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=new_uuid)
    run_id = Column(UUID(as_uuid=True), ForeignKey("pipeline_runs.id"), nullable=False)
    phase = Column(String(10), nullable=False)  # before, after
    noise_pct = Column(Float)
    missingness_pct = Column(Float)
    duplicate_pct = Column(Float)
    trust_score = Column(Float)
    plausibility_rate = Column(Float)
    column_metrics = Column(JSONB)  # Per-column quality details
    site_metrics = Column(JSONB)  # Per-site quality details
    
    pipeline_run = relationship("PipelineRun", back_populates="quality_metrics")


class Report(Base):
    __tablename__ = "reports"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=new_uuid)
    run_id = Column(UUID(as_uuid=True), ForeignKey("pipeline_runs.id"), nullable=False)
    org_id = Column(UUID(as_uuid=True), nullable=False, index=True)
    format = Column(String(10), nullable=False)  # pdf, xlsx, csv, docx
    file_path = Column(String(1000), nullable=False)
    metadata = Column(JSONB)
    created_at = Column(DateTime(timezone=True), default=utcnow)
    
    pipeline_run = relationship("PipelineRun", back_populates="reports")
