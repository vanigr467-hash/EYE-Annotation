import os
from typing import List

try:
    from pydantic_settings import BaseSettings
except ImportError:
    from pydantic import BaseModel
    class BaseSettings(BaseModel):
        pass

class Settings(BaseSettings):
    DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///./eye_annotation.db")
    STORAGE_ROOT: str = os.getenv("STORAGE_ROOT", "./storage")
    AUTO_APPROVE_THRESHOLD: float = float(os.getenv("AUTO_APPROVE_THRESHOLD", "85.0"))
    CORS_ORIGINS: List[str] = [
        "http://localhost:8501",
        "http://localhost:8502",
        "http://localhost:8000",
        "http://127.0.0.1:8501",
        "http://127.0.0.1:8000",
        "*"
    ]
    API_PREFIX: str = "/api/v1"

settings = Settings()
