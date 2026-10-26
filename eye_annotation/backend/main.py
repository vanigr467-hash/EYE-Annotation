from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.config import settings
from backend.database import init_db
from backend.routers import (
    projects, media, frames, annotations, inference, export, review
)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize the database tables on startup
    init_db()
    yield

app = FastAPI(
    title="Eye Annotation Platform API",
    version="1.0.0",
    description="Enterprise Automated Pupil & Iris Boundary Detection API",
    lifespan=lifespan
)

# Set up CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers - each router already declares its sub-prefix (e.g. /projects, /media)
app.include_router(projects.router, prefix=settings.API_PREFIX)
app.include_router(media.router, prefix=settings.API_PREFIX)
app.include_router(frames.router, prefix=settings.API_PREFIX)
app.include_router(annotations.router, prefix=settings.API_PREFIX)
app.include_router(inference.router, prefix=settings.API_PREFIX)
app.include_router(export.router, prefix=settings.API_PREFIX)
app.include_router(review.router, prefix=settings.API_PREFIX)

@app.get("/health", tags=["health"])
def health_check():
    return {"status": "healthy"}

@app.get(f"{settings.API_PREFIX}/health", tags=["health"])
def api_health_check():
    return {"status": "healthy"}

@app.get("/", tags=["root"])
def root():
    return {
        "name": "Eye Annotation Platform API",
        "version": "1.0.0",
        "docs_url": "/docs",
        "api_prefix": settings.API_PREFIX
    }
