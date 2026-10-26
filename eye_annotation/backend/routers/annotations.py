from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List, Optional
from backend.database import get_db
from backend.models import Annotation, AnnotationVersion
from backend.services.annotation_service import create_annotation_with_version, update_annotation_with_version
from backend.schemas import AnnotationCreate, AnnotationUpdate, AnnotationResponse, AnnotationVersionResponse

router = APIRouter(prefix="/annotations", tags=["annotations"])

@router.post("/", response_model=AnnotationResponse, status_code=status.HTTP_201_CREATED)
def create_annotation(ann: AnnotationCreate, db: Session = Depends(get_db)):
    ann_dict = ann.model_dump() if hasattr(ann, "model_dump") else ann.dict()
    return create_annotation_with_version(db, ann_dict)

@router.get("/", response_model=List[AnnotationResponse])
def list_annotations(
    frame_id: Optional[int] = None,
    media_asset_id: Optional[int] = None,
    label_type: Optional[str] = None,
    status: Optional[str] = None,
    source: Optional[str] = None,
    db: Session = Depends(get_db)
):
    q = db.query(Annotation)
    if frame_id is not None:
        q = q.filter(Annotation.frame_id == frame_id)
    if media_asset_id is not None:
        q = q.filter(Annotation.media_asset_id == media_asset_id)
    if label_type is not None:
        q = q.filter(Annotation.label_type == label_type)
    if status is not None:
        q = q.filter(Annotation.status == status)
    if source is not None:
        q = q.filter(Annotation.source == source)
    return q.all()

@router.get("/{id}", response_model=AnnotationResponse)
def get_annotation(id: int, db: Session = Depends(get_db)):
    ann = db.query(Annotation).filter(Annotation.id == id).first()
    if not ann:
        raise HTTPException(status_code=404, detail="Annotation not found")
    return ann

@router.put("/{id}", response_model=AnnotationResponse)
def update_annotation(id: int, update_data: AnnotationUpdate, db: Session = Depends(get_db)):
    data = update_data.model_dump(exclude_unset=True) if hasattr(update_data, "model_dump") else update_data.dict(exclude_unset=True)
    edited_by = data.pop("edited_by", "annotator")
    action = data.pop("action", "edit")
    return update_annotation_with_version(db, id, data, edited_by=edited_by, action=action)

@router.delete("/{id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_annotation(id: int, db: Session = Depends(get_db)):
    ann = db.query(Annotation).filter(Annotation.id == id).first()
    if not ann:
        raise HTTPException(status_code=404, detail="Annotation not found")
    db.delete(ann)
    db.commit()

@router.get("/{id}/versions", response_model=List[AnnotationVersionResponse])
def get_annotation_versions(id: int, db: Session = Depends(get_db)):
    return db.query(AnnotationVersion).filter(AnnotationVersion.annotation_id == id).order_by(AnnotationVersion.created_at.desc()).all()
