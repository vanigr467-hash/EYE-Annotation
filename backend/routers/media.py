import os
from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, Form
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
from typing import List, Optional
from backend.database import get_db
from backend.models import MediaAsset, Frame
from backend.services.media_service import get_image_dimensions, get_video_metadata
from backend.services.storage import StorageService

storage_service = StorageService()
router = APIRouter(prefix="/media", tags=["media"])

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

@router.post("/upload", status_code=status.HTTP_201_CREATED)
async def upload_media(
    project_id: int = Form(...),
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    filename = file.filename or "uploaded_media.bin"
    ext = os.path.splitext(filename)[1].lower()
    content_type = file.content_type or ""
    
    if ext in VIDEO_EXTENSIONS or "video" in content_type:
        media_type = "video"
    else:
        media_type = "image"
        
    file_bytes = await file.read()
    storage_url = storage_service.save_file(file_bytes, category="uploads", filename=filename)
    local_path = storage_service.get_local_path(storage_url)
    
    asset = MediaAsset(
        project_id=project_id,
        filename=filename,
        original_filename=filename,
        media_type=media_type,
        storage_url=storage_url,
        status="uploaded"
    )
    db.add(asset)
    db.commit()
    db.refresh(asset)
    
    if media_type == "image":
        width, height = 0, 0
        try:
            width, height = get_image_dimensions(local_path)
        except Exception:
            pass
        asset.width = width
        asset.height = height
        
        frame = Frame(
            media_asset_id=asset.id,
            frame_index=0,
            timestamp_sec=0.0,
            storage_url=storage_url,
            width=width,
            height=height
        )
        db.add(frame)
        asset.status = "ready"
        asset.frame_count = 1
    else:
        try:
            meta = get_video_metadata(local_path)
            asset.fps = meta.get("fps")
            asset.width = meta.get("width", 0)
            asset.height = meta.get("height", 0)
            asset.duration_sec = meta.get("duration_sec", 0.0)
            asset.frame_count = meta.get("frame_count", 0)
        except Exception:
            pass
        
    db.commit()
    db.refresh(asset)
    return asset

@router.get("/")
def list_media(project_id: Optional[int] = None, media_type: Optional[str] = None, db: Session = Depends(get_db)):
    q = db.query(MediaAsset)
    if project_id:
        q = q.filter(MediaAsset.project_id == project_id)
    if media_type:
        q = q.filter(MediaAsset.media_type == media_type)
    return q.all()

@router.get("/{id}")
def get_media(id: int, db: Session = Depends(get_db)):
    asset = db.query(MediaAsset).filter(MediaAsset.id == id).first()
    if not asset:
        raise HTTPException(status_code=404, detail="Media asset not found")
    return asset

@router.delete("/{id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_media(id: int, db: Session = Depends(get_db)):
    asset = db.query(MediaAsset).filter(MediaAsset.id == id).first()
    if not asset:
        raise HTTPException(status_code=404, detail="Media asset not found")
    db.delete(asset)
    db.commit()

@router.get("/{id}/image")
def get_media_image(id: int, db: Session = Depends(get_db)):
    asset = db.query(MediaAsset).filter(MediaAsset.id == id).first()
    if not asset:
        raise HTTPException(status_code=404, detail="Media asset not found")
    file_path = storage_service.get_local_path(asset.storage_url)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Media file not found on disk")
    return FileResponse(file_path)
