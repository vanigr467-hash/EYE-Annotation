from sqlalchemy.orm import Session
from typing import Dict, Any, Optional
from backend.models import Annotation, AnnotationVersion

def create_annotation_with_version(db: Session, annotation_data: Dict[str, Any]) -> Annotation:
    clean_data = dict(annotation_data)
    boundary_points_json = clean_data.get("boundary_points_json")
    created_by = clean_data.get("created_by") or "system"
    clean_data["created_by"] = created_by

    annotation = Annotation(**clean_data)
    db.add(annotation)
    db.commit()
    db.refresh(annotation)
    
    version = AnnotationVersion(
        annotation_id=annotation.id,
        cx=annotation.cx,
        cy=annotation.cy,
        rx=annotation.rx,
        ry=annotation.ry,
        angle=annotation.angle or 0.0,
        boundary_points_json=boundary_points_json,
        edited_by=created_by,
        action="create"
    )
    db.add(version)
    db.commit()
    
    return annotation

def update_annotation_with_version(
    db: Session,
    annotation_id: int,
    update_data: Dict[str, Any],
    edited_by: Optional[str] = None,
    action: str = "edit"
) -> Annotation:
    annotation = db.query(Annotation).filter(Annotation.id == annotation_id).first()
    if not annotation:
        raise ValueError("Annotation not found")
        
    for key, value in update_data.items():
        if hasattr(annotation, key):
            setattr(annotation, key, value)
        
    db.commit()
    db.refresh(annotation)
    
    editor_str = str(edited_by) if edited_by else "system"
    
    version = AnnotationVersion(
        annotation_id=annotation.id,
        cx=annotation.cx,
        cy=annotation.cy,
        rx=annotation.rx,
        ry=annotation.ry,
        angle=annotation.angle or 0.0,
        boundary_points_json=annotation.boundary_points_json,
        edited_by=editor_str,
        action=action
    )
    db.add(version)
    db.commit()
    
    return annotation
