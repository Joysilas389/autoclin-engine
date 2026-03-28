"""
AutoClin Engine Celery Worker
Async task definitions for pipeline execution.
"""
import os
import json
import logging
from celery import Celery

from app.core.config import settings

logger = logging.getLogger(__name__)

celery_app = Celery(
    "autoclin_engine",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND,
)
celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_acks_late=True,
    worker_prefetch_multiplier=1,
)


@celery_app.task(bind=True, name="clinengine.run_pipeline", max_retries=2)
def run_pipeline_task(self, dataset_id: str, run_id: str, config: dict):
    """
    Execute the full AutoClin Engine pipeline asynchronously.
    Updates task state for progress tracking via SSE.
    """
    import pandas as pd
    from ml.orchestrator import PipelineOrchestrator, PipelineConfig

    try:
        self.update_state(state="PROGRESS", meta={"phase": 1, "pct": 0, "msg": "Starting..."})

        # Load dataset from storage
        # TODO: Fetch from MinIO using dataset_id
        # For now, this is a placeholder
        upload_path = config.get("upload_path", "")
        if upload_path.endswith(".csv"):
            df = pd.read_csv(upload_path)
        elif upload_path.endswith((".xlsx", ".xls")):
            df = pd.read_excel(upload_path)
        elif upload_path.endswith(".json"):
            df = pd.read_json(upload_path)
        elif upload_path.endswith(".parquet"):
            df = pd.read_parquet(upload_path)
        elif upload_path.endswith(".tsv"):
            df = pd.read_csv(upload_path, sep="\t")
        else:
            raise ValueError(f"Unsupported format: {upload_path}")

        # Configure pipeline
        pipeline_config = PipelineConfig(
            mode=config.get("mode", "suggestion"),
            methods=config.get("methods", "auto"),
            confidence_threshold=config.get("confidence_threshold", 0.5),
            contamination_estimate=config.get("contamination_estimate"),
            random_seed=config.get("random_seed", 42),
        )

        # Run pipeline with progress callback
        orchestrator = PipelineOrchestrator(pipeline_config)
        orchestrator.set_progress_callback(
            lambda phase, pct, msg: self.update_state(
                state="PROGRESS", meta={"phase": phase, "pct": pct, "msg": msg}
            )
        )

        result = orchestrator.run(df, run_id=run_id)

        # TODO: Save results to database and MinIO
        # - Save anomaly records to anomaly_records table
        # - Save method results to method_results table
        # - Save quality metrics to quality_metrics table
        # - Save cleaned dataset to MinIO if auto-clean mode
        # - Update pipeline_run status to "completed"

        return {
            "status": "completed",
            "run_id": run_id,
            "total_anomalies": result.total_anomalies,
            "noise_pct": result.noise_percentage,
            "trust_score": result.trust_score,
            "duration_ms": result.duration_ms,
        }

    except Exception as exc:
        logger.exception(f"Pipeline failed for run {run_id}: {exc}")
        self.update_state(state="FAILURE", meta={"error": str(exc)})
        # TODO: Update pipeline_run status to "failed" with error_message
        raise


@celery_app.task(name="clinengine.generate_report")
def generate_report_task(run_id: str, format: str, output_path: str):
    """Generate report asynchronously."""
    # TODO: Load pipeline result from DB, generate report, save to MinIO
    logger.info(f"Generating {format} report for run {run_id}")
    return {"status": "completed", "path": output_path}
