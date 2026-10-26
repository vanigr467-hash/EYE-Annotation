import json
from sqlalchemy.orm import Session
from backend.models import Annotation, Project, Frame, MediaAsset

def export_coco_json(db: Session, project_id: int) -> dict:
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise ValueError("Project not found")
        
    coco = {
        "info": {"description": project.name},
        "images": [],
        "annotations": [],
        "categories": [
            {"id": 1, "name": "pupil"},
            {"id": 2, "name": "iris"}
        ]
    }
    
    frames = db.query(Frame).join(MediaAsset).filter(MediaAsset.project_id == project_id).all()
    for frame in frames:
        coco["images"].append({
            "id": frame.id,
            "file_name": frame.storage_url,
            "width": frame.width,
            "height": frame.height
        })
        
    annotations = db.query(Annotation).join(MediaAsset).filter(
        MediaAsset.project_id == project_id,
        Annotation.status == "approved"
    ).all()
    
    for ann in annotations:
        cat_id = 1 if ann.label_type == "pupil" else 2
        coco["annotations"].append({
            "id": ann.id,
            "image_id": ann.frame_id,
            "category_id": cat_id,
            "bbox": [ann.cx - ann.rx, ann.cy - ann.ry, ann.rx * 2, ann.ry * 2] if ann.cx and ann.rx else [],
            "segmentation": [json.loads(ann.boundary_points_json)] if ann.boundary_points_json else [],
            "area": (3.14159 * ann.rx * ann.ry) if ann.rx and ann.ry else 0
        })
        
    return coco

def export_csv(db: Session, project_id: int) -> str:
    annotations = db.query(Annotation).join(MediaAsset).filter(
        MediaAsset.project_id == project_id,
        Annotation.status == "approved"
    ).all()
    
    csv_lines = ["id,frame_id,label_type,cx,cy,rx,ry,angle"]
    for ann in annotations:
        csv_lines.append(f"{ann.id},{ann.frame_id},{ann.label_type},{ann.cx},{ann.cy},{ann.rx},{ann.ry},{ann.angle}")
        
    return "\n".join(csv_lines)

def export_yolo(db: Session, project_id: int) -> str:
    annotations = db.query(Annotation).join(MediaAsset).filter(
        MediaAsset.project_id == project_id,
        Annotation.status == "approved"
    ).all()
    
    yolo_lines = []
    for ann in annotations:
        cat_id = 0 if ann.label_type == "pupil" else 1
        frame = db.query(Frame).filter(Frame.id == ann.frame_id).first()
        if not frame or not frame.width or not frame.height:
            continue
            
        if ann.cx is not None and ann.rx is not None:
            nx = ann.cx / frame.width
            ny = ann.cy / frame.height
            nw = (ann.rx * 2) / frame.width
            nh = (ann.ry * 2) / frame.height
            yolo_lines.append(f"{cat_id} {nx} {ny} {nw} {nh}")
            
    return "\n".join(yolo_lines)

def export_summary(db: Session, project_id: int) -> dict:
    annotations = db.query(Annotation).join(MediaAsset).filter(
        MediaAsset.project_id == project_id
    ).all()
    
    summary = {
        "total_annotations": len(annotations),
        "approved": sum(1 for a in annotations if a.status == "approved"),
        "pending_review": sum(1 for a in annotations if a.status == "pending_review"),
        "rejected": sum(1 for a in annotations if a.status == "rejected"),
    }
    return summary
