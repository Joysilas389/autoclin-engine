"""
AutoClin Engine Core Configuration
Environment-based settings with sensible defaults.
"""
from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Optional
import os


class Settings(BaseSettings):
    # Application
    APP_NAME: str = "AutoClin Engine"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False
    API_V1_PREFIX: str = "/api/v1"
    
    # Server
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    WORKERS: int = 4
    
    # Database
    DATABASE_URL: str = "postgresql+asyncpg://clinengine:clinengine@localhost:5432/clinengine"
    DATABASE_POOL_SIZE: int = 20
    DATABASE_MAX_OVERFLOW: int = 10
    
    # Redis
    REDIS_URL: str = "redis://localhost:6379/0"
    CELERY_BROKER_URL: str = "redis://localhost:6379/1"
    CELERY_RESULT_BACKEND: str = "redis://localhost:6379/2"
    
    # Object Storage (MinIO / S3)
    S3_ENDPOINT: str = "localhost:9000"
    S3_ACCESS_KEY: str = "autoclin_engine"
    S3_SECRET_KEY: str = "clinengine_secret"
    S3_BUCKET_UPLOADS: str = "clinengine-uploads"
    S3_BUCKET_REPORTS: str = "clinengine-reports"
    S3_BUCKET_CLEANED: str = "clinengine-cleaned"
    S3_USE_SSL: bool = False
    
    # Auth
    SECRET_KEY: str = "CHANGE-ME-IN-PRODUCTION"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7
    ALGORITHM: str = "HS256"
    
    # Pipeline
    MAX_UPLOAD_SIZE_MB: int = 500
    CHUNK_SIZE_ROWS: int = 50000
    DEFAULT_CONTAMINATION_ESTIMATE: float = 0.05
    BOOTSTRAP_SAMPLES: int = 20
    MAX_METHODS_PARALLEL: int = 4
    
    # ML
    RANDOM_SEED: int = 42
    AUTOENCODER_EPOCHS: int = 100
    AUTOENCODER_BATCH_SIZE: int = 256
    AUTOENCODER_PATIENCE: int = 10
    
    # Reporting
    REPORT_TEMPLATE_DIR: str = "reporting/templates"
    MAX_REPORT_ANOMALIES: int = 500
    
    # Security
    ALLOWED_ORIGINS: list[str] = ["http://localhost:3000"]
    
    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()
