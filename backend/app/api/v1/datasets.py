"""
Dataset endpoints — upload, profiling, clinical mapping.
"""
from fastapi import APIRouter, UploadFile, File, Depends, HTTPException, Query
from uuid import UUID

from app.schemas.schemas import (
    DatasetUploadResponse, DatasetProfile, ClinicalMapUpdate, ClinicalFieldMapping,
)
from app.core.security import get_current_user

router = APIRouter()


@router.post("/upload", response_model=DatasetUploadResponse)
async def upload_dataset(
    file: UploadFile = File(...),
    project_id: UUID = Query(...),
    user: dict = Depends(get_current_user),
):
    """
    Upload a dataset file. Triggers format detection and schema inference.
    Supports: CSV, TSV, Excel, JSON, XML, Parquet, REDCap, EDC exports.
    """
    # TODO:
    # 1. Validate file size against MAX_UPLOAD_SIZE_MB
    # 2. Detect format via extension + magic bytes
    # 3. Store raw file in MinIO under org prefix
    # 4. Create Dataset record in DB
    # 5. Trigger async schema inference task
    raise HTTPException(status_code=501, detail="Not yet implemented")


@router.get("/{dataset_id}/profile", response_model=DatasetProfile)
async def get_profile(
    dataset_id: UUID,
    user: dict = Depends(get_current_user),
):
    """Get data profiling results for a dataset."""
    raise HTTPException(status_code=501, detail="Not yet implemented")


@router.get("/{dataset_id}/clinical-map", response_model=list[ClinicalFieldMapping])
async def get_clinical_map(
    dataset_id: UUID,
    user: dict = Depends(get_current_user),
):
    """Get auto-detected clinical field mappings."""
    raise HTTPException(status_code=501, detail="Not yet implemented")


@router.put("/{dataset_id}/clinical-map")
async def update_clinical_map(
    dataset_id: UUID,
    update: ClinicalMapUpdate,
    user: dict = Depends(get_current_user),
):
    """Review and edit clinical field mappings."""
    raise HTTPException(status_code=501, detail="Not yet implemented")
