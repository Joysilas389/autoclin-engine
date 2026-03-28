"""
Anomaly endpoints — list, explain, adjudicate.
"""
from fastapi import APIRouter, Depends, HTTPException, Query
from uuid import UUID
from typing import Optional

from app.schemas.schemas import (
    AnomalyResponse, AnomalyListResponse, AnomalyExplanation, AdjudicateRequest,
)
from app.core.security import get_current_user

router = APIRouter()


@router.get("/by-run/{run_id}", response_model=AnomalyListResponse)
async def list_anomalies(
    run_id: UUID,
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=200),
    anomaly_type: Optional[str] = None,
    severity: Optional[str] = None,
    min_confidence: float = Query(0.0, ge=0.0, le=1.0),
    column: Optional[str] = None,
    site: Optional[str] = None,
    status: Optional[str] = None,
    user: dict = Depends(get_current_user),
):
    """
    List flagged anomalies with filtering and pagination.
    """
    raise HTTPException(status_code=501, detail="Not yet implemented")


@router.get("/{anomaly_id}/explain", response_model=AnomalyExplanation)
async def get_anomaly_explanation(
    anomaly_id: UUID,
    user: dict = Depends(get_current_user),
):
    """Get detailed explanation for a single anomaly."""
    raise HTTPException(status_code=501, detail="Not yet implemented")


@router.post("/{anomaly_id}/adjudicate")
async def adjudicate_anomaly(
    anomaly_id: UUID,
    request: AdjudicateRequest,
    user: dict = Depends(get_current_user),
):
    """Submit adjudication decision for an anomaly."""
    raise HTTPException(status_code=501, detail="Not yet implemented")
