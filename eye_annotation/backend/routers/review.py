from fastapi import APIRouter, Depends, HTTPException, status, Body
from sqlalchemy.orm import Session
from typing import List, Optional
from backend.database import get_db
from backend.models import Annotation, AnnotationVersion, MediaAsset
from backend.services.annotation_service import update_annotation_with_version
from backend.schemas import AnnotationUpdate, AnnotationResponse, BatchApproveRequest

router = APIRouter(prefix="/review", tags=["review"])

@router.get("/queue", response_model=List[AnnotationResponse])
def get_review_queue(project_id: Optional[int] = None, db: Session = Depends(get_db)):
    q = db.query(Annotation).filter(Annotation.status == "pending_review")
    if project_id:
        q = q.join(MediaAsset, Annotation.media_asset_id == MediaAsset.id).filter(MediaAsset.project_id == project_id)
    return q.order_by(Annotation.confidence.asc()).all()

@router.put("/{annotation_id}/approve", response_model=AnnotationResponse)
def approve_annotation(
    annotation_id: int,
    reviewed_by: Optional[str] = "reviewer",
    db: Session = Depends(get_db)
):
    return update_annotation_with_version(
        db, annotation_id, {"status": "approved"},
        edited_by=reviewed_by or "reviewer", action="approve"
    )

@router.put("/{annotation_id}/reject", response_model=AnnotationResponse)
def reject_annotation(
    annotation_id: int,
    reviewed_by: Optional[str] = "reviewer",
    db: Session = Depends(get_db)
):
    return update_annotation_with_version(
        db, annotation_id, {"status": "rejected"},
        edited_by=reviewed_by or "reviewer", action="reject"
    )

@router.put("/{annotation_id}/edit", response_model=AnnotationResponse)
def edit_and_approve(
    annotation_id: int,
    update_data: AnnotationUpdate,
    reviewed_by: Optional[str] = "reviewer",
    db: Session = Depends(get_db)
):
    data = update_data.model_dump(exclude_unset=True) if hasattr(update_data, "model_dump") else update_data.dict(exclude_unset=True)
    data["status"] = "approved"
    data["source"] = "human"
    return update_annotation_with_version(
        db, annotation_id, data,
        edited_by=reviewed_by or "reviewer", action="edit"
    )

@router.post("/batch-approve")
def batch_approve(
    payload: BatchApproveRequest,
    db: Session = Depends(get_db)
):
    threshold = payload.threshold
    project_id = payload.project_id
    q = db.query(Annotation).filter(Annotation.status == "pending_review", Annotation.confidence >= threshold)
    if project_id:
        q = q.join(MediaAsset, Annotation.media_asset_id == MediaAsset.id).filter(MediaAsset.project_id == project_id)
        
    annotations = q.all()
    for ann in annotations:
        ann.status = "approved"
        version = AnnotationVersion(
            annotation_id=ann.id,
            cx=ann.cx,
            cy=ann.cy,
            rx=ann.rx,
            ry=ann.ry,
            angle=ann.angle or 0.0,
            boundary_points_json=ann.boundary_points_json,
            edited_by="reviewer",
            action="approve"
        )
        db.add(version)
    db.commit()
    return {"approved_count": len(annotations)}
