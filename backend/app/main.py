"""
AutoClin Engine API Gateway
Main application with lifespan, CORS, and route registration.
"""
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.core.database import init_db, close_db
from app.api.v1 import datasets, pipelines, anomalies, reports, auth


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown logic."""
    await init_db()
    yield
    await close_db()


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="Automated Unsupervised Clinical Data Quality Engine",
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register API routes
app.include_router(auth.router, prefix=f"{settings.API_V1_PREFIX}/auth", tags=["auth"])
app.include_router(datasets.router, prefix=f"{settings.API_V1_PREFIX}/datasets", tags=["datasets"])
app.include_router(pipelines.router, prefix=f"{settings.API_V1_PREFIX}/pipelines", tags=["pipelines"])
app.include_router(anomalies.router, prefix=f"{settings.API_V1_PREFIX}/anomalies", tags=["anomalies"])
app.include_router(reports.router, prefix=f"{settings.API_V1_PREFIX}/reports", tags=["reports"])


@app.get("/health")
async def health_check():
    return {"status": "healthy", "version": settings.APP_VERSION}
