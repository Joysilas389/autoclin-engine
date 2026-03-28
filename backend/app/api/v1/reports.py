"""
Report endpoints — generate, download.
"""
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from uuid import UUID

from app.schemas.schemas import ReportGenerateRequest, ReportResponse
from app.core.security import get_current_user

router = APIRouter()


@router.post("/generate", response_model=ReportResponse)
async def generate_report(
    request: ReportGenerateRequest,
    user: dict = Depends(get_current_user),
):
    """Generate a report in the specified format (PDF, Excel, CSV, DOCX)."""
    raise HTTPException(status_code=501, detail="Not yet implemented")


@router.get("/{report_id}/download")
async def download_report(
    report_id: UUID,
    user: dict = Depends(get_current_user),
):
    """Download a generated report file."""
    raise HTTPException(status_code=501, detail="Not yet implemented")
