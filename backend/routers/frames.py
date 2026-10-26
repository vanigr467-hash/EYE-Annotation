from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
from typing import List, Optional
from backend.database import get_db
from backend.models import MediaAsset, Frame
from backend.services.video_service import extract_frames

try:
    from backend.services.storage import StorageService
except ImportError:
    class StorageService:
        def save_file(self, filename, data, folder): return f"{folder}/{filename}"
        def get_local_path(self, url): return url

storage_service = StorageService()
router = APIRouter(prefix="/frames", tags=["frames"])

@router.post("/extract/{media_asset_id}")
def extract_video_frames(media_asset_id: int, fps: float = 5.0, db: Session = Depends(get_db)):
    asset = db.query(MediaAsset).filter(MediaAsset.id == media_asset_id).first()
    if not asset:
        raise HTTPException(status_code=404, detail="Media asset not found")
    if asset.media_type != "video":
        raise HTTPException(status_code=400, detail="Not a video")
    
    local_path = storage_service.get_local_path(asset.storage_url)
    try:
        frames_info = extract_frames(local_path, fps, storage_service)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
        
    for info in frames_info:
        frame = Frame(media_asset_id=asset.id, **info)
        db.add(frame)
        
    asset.frame_count = len(frames_info)
    asset.status = "ready"
    db.commit()
    
    return {"extracted": len(frames_info)}

@router.get("/{media_asset_id}")
def list_frames(media_asset_id: int, db: Session = Depends(get_db)):
    return db.query(Frame).filter(Frame.media_asset_id == media_asset_id).all()

@router.get("/{frame_id}/image")
def get_frame_image(frame_id: int, db: Session = Depends(get_db)):
    frame = db.query(Frame).filter(Frame.id == frame_id).first()
    if not frame:
        raise HTTPException(status_code=404, detail="Frame not found")
    return FileResponse(storage_service.get_local_path(frame.storage_url))
