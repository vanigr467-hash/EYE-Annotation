from sqlalchemy.orm import Session
from backend.models import Frame, MediaAsset
from backend.services.annotation_service import create_annotation_with_version
from backend.config import settings
import json
from typing import List, Dict, Any

try:
    from ml.inference.detector import detect_eye_boundaries
except ImportError:
    def detect_eye_boundaries(image_path: str):
        return None

def run_detection_on_frame(db: Session, frame_id: int, storage_service) -> List[Any]:
    frame = db.query(Frame).filter(Frame.id == frame_id).first()
    if not frame:
        raise ValueError(f"Frame with id {frame_id} not found")
        
    try:
        file_path = storage_service.get_local_path(frame.storage_url)
    except AttributeError:
        file_path = frame.storage_url
        
    results = detect_eye_boundaries(file_path)
    if not results or not isinstance(results, dict):
        return []
        
    auto_thresh = float(getattr(settings, "AUTO_APPROVE_THRESHOLD", 85.0))
    annotations = []

    # Process pupil
    pupil_data = results.get("pupil", {})
    if pupil_data.get("detected"):
        conf = float(pupil_data.get("confidence", 0.0))
        status = "approved" if conf >= auto_thresh else "pending_review"
        pts = pupil_data.get("boundary_points", [])
        pts_json = json.dumps(pts) if pts else None
        
        ann_data = {
            "frame_id": frame.id,
            "media_asset_id": frame.media_asset_id,
            "label_type": "pupil",
            "cx": pupil_data.get("cx", 0.0),
            "cy": pupil_data.get("cy", 0.0),
            "rx": pupil_data.get("rx", 0.0),
            "ry": pupil_data.get("ry", 0.0),
            "angle": pupil_data.get("angle", 0.0),
            "boundary_points_json": pts_json,
            "source": "ai",
            "confidence": conf,
            "status": status,
            "model_version": "v1.0"
        }
        ann = create_annotation_with_version(db, ann_data)
        annotations.append(ann)

    # Process iris
    iris_data = results.get("iris", {})
    if iris_data.get("detected"):
        conf = float(iris_data.get("confidence", 0.0))
        status = "approved" if conf >= auto_thresh else "pending_review"
        pts = iris_data.get("boundary_points", [])
        pts_json = json.dumps(pts) if pts else None
        
        ann_data = {
            "frame_id": frame.id,
            "media_asset_id": frame.media_asset_id,
            "label_type": "iris",
            "cx": iris_data.get("cx", 0.0),
            "cy": iris_data.get("cy", 0.0),
            "rx": iris_data.get("rx", 0.0),
            "ry": iris_data.get("ry", 0.0),
            "angle": iris_data.get("angle", 0.0),
            "boundary_points_json": pts_json,
            "source": "ai",
            "confidence": conf,
            "status": status,
            "model_version": "v1.0"
        }
        ann = create_annotation_with_version(db, ann_data)
        annotations.append(ann)
        
    return annotations

def run_batch_detection(db: Session, media_asset_id: int, storage_service, progress_callback=None) -> Dict[str, Any]:
    frames = db.query(Frame).filter(Frame.media_asset_id == media_asset_id).order_by(Frame.frame_index).all()
    if not frames:
        return {"total": 0, "processed": 0, "annotations_created": 0}
        
    total_annotations = 0
    for i, frame in enumerate(frames):
        try:
            anns = run_detection_on_frame(db, frame.id, storage_service)
            total_annotations += len(anns)
        except Exception as e:
            print(f"Error detecting on frame {frame.id}: {e}")
            
        if progress_callback:
            progress_callback(i + 1, len(frames))
            
    return {
        "total": len(frames),
        "processed": len(frames),
        "annotations_created": total_annotations
    }
