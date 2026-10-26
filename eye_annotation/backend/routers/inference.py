from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from backend.database import get_db
from backend.services.inference_service import run_detection_on_frame, run_batch_detection
from backend.services.storage import StorageService
from backend.schemas import AnnotationResponse

storage_service = StorageService()
router = APIRouter(prefix="/inference", tags=["inference"])

@router.post("/detect/{frame_id}")
def detect_frame(frame_id: int, db: Session = Depends(get_db)):
    try:
        anns = run_detection_on_frame(db, frame_id, storage_service)
        anns_data = [AnnotationResponse.model_validate(a).model_dump() for a in anns]
        return {"detected": len(anns_data), "annotations": anns_data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/detect-batch/{media_asset_id}")
def detect_batch(media_asset_id: int, db: Session = Depends(get_db)):
    try:
        return run_batch_detection(db, media_asset_id, storage_service)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
