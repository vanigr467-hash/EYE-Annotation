from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import PlainTextResponse
from sqlalchemy.orm import Session
from backend.database import get_db
from backend.services.export_service import (
    export_coco_json, export_csv, export_yolo, export_summary as generate_export_summary
)

router = APIRouter(prefix="/export", tags=["export"])

@router.get("/{project_id}/coco")
def export_coco(project_id: int, db: Session = Depends(get_db)):
    try:
        return export_coco_json(db, project_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{project_id}/csv", response_class=PlainTextResponse)
def export_project_csv(project_id: int, db: Session = Depends(get_db)):
    try:
        return export_csv(db, project_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{project_id}/yolo", response_class=PlainTextResponse)
def export_project_yolo(project_id: int, db: Session = Depends(get_db)):
    try:
        return export_yolo(db, project_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{project_id}/summary")
def export_project_summary(project_id: int, db: Session = Depends(get_db)):
    try:
        return generate_export_summary(db, project_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
