"""
Pipeline endpoints — run analysis, check status, get methods, clean, before/after.
"""
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse
from uuid import UUID

from app.schemas.schemas import (
    PipelineRunConfig, PipelineStatus, MethodRanking,
    CleaningPlan, BeforeAfterResponse, PipelineResult,
)
from app.core.security import get_current_user

router = APIRouter()


@router.post("/run", response_model=PipelineStatus)
async def run_pipeline(
    config: PipelineRunConfig,
    user: dict = Depends(get_current_user),
):
    """
    Start a full pipeline run (Phases 1-9).
    Returns immediately with run_id; use /status for progress.
    """
    # TODO:
    # 1. Validate dataset exists and belongs to user's org
    # 2. Create PipelineRun record
    # 3. Dispatch Celery task for orchestrator.run_pipeline()
    # 4. Return run_id + initial status
    raise HTTPException(status_code=501, detail="Not yet implemented")


@router.get("/{run_id}/status")
async def get_pipeline_status(
    run_id: UUID,
    user: dict = Depends(get_current_user),
):
    """
    Get pipeline progress. Supports SSE streaming for real-time updates.
    """
    # TODO: Return current phase, progress %, ETA
    raise HTTPException(status_code=501, detail="Not yet implemented")


@router.get("/{run_id}/methods", response_model=list[MethodRanking])
async def get_method_comparison(
    run_id: UUID,
    user: dict = Depends(get_current_user),
):
    """Get method comparison results with scores."""
    raise HTTPException(status_code=501, detail="Not yet implemented")


@router.get("/{run_id}/result", response_model=PipelineResult)
async def get_pipeline_result(
    run_id: UUID,
    user: dict = Depends(get_current_user),
):
    """Get complete pipeline result summary."""
    raise HTTPException(status_code=501, detail="Not yet implemented")


@router.post("/{run_id}/clean", response_model=CleaningPlan)
async def apply_cleaning(
    run_id: UUID,
    action_ids: list[UUID] = Query(default=None),
    user: dict = Depends(get_current_user),
):
    """
    Apply cleaning actions. 
    If action_ids provided, apply only those. Otherwise apply all auto-safe actions.
    """
    raise HTTPException(status_code=501, detail="Not yet implemented")


@router.get("/{run_id}/before-after", response_model=BeforeAfterResponse)
async def get_before_after(
    run_id: UUID,
    user: dict = Depends(get_current_user),
):
    """Get before vs after quality metrics."""
    raise HTTPException(status_code=501, detail="Not yet implemented")


@router.get("/{run_id}/audit-trail")
async def get_audit_trail(
    run_id: UUID,
    page: int = 1,
    page_size: int = 50,
    user: dict = Depends(get_current_user),
):
    """Get full cleaning audit log."""
    raise HTTPException(status_code=501, detail="Not yet implemented")


@router.post("/simulate/clean")
async def simulate_cleaning(
    run_id: UUID,
    action_ids: list[UUID],
    user: dict = Depends(get_current_user),
):
    """Preview impact of proposed cleaning actions before applying."""
    raise HTTPException(status_code=501, detail="Not yet implemented")
