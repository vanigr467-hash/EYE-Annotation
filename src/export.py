import json
import pandas as pd
from typing import Dict, Any, List

def export_image_metrics_csv(metrics_dict: Dict[str, Any]) -> str:
    """Converts image detection metrics to CSV string."""
    df = pd.DataFrame([metrics_dict])
    return df.to_csv(index=False)

def export_video_metrics_csv(frame_logs: List[Dict[str, Any]]) -> str:
    """Converts frame-by-frame video detection logs to CSV string."""
    df = pd.DataFrame(frame_logs)
    return df.to_csv(index=False)

def export_video_metrics_json(
    video_summary: Dict[str, Any],
    frame_logs: List[Dict[str, Any]]
) -> str:
    """Exports structured video statistics and frame-by-frame metrics to JSON string."""
    data = {
        "video_summary": {
            "total_frames": video_summary.get("total_frames", 0),
            "processed_frames": video_summary.get("processed_frames", 0),
            "successful_detections": video_summary.get("successful_detections", 0),
            "predicted_frames": video_summary.get("predicted_frames", 0),
            "failed_frames": video_summary.get("failed_frames", 0),
            "pupil_detection_rate": video_summary.get("pupil_detection_rate", 0.0),
            "iris_detection_rate": video_summary.get("iris_detection_rate", 0.0),
            "avg_overall_confidence": video_summary.get("avg_overall_confidence", 0.0),
            "processing_fps": video_summary.get("processing_fps", 0.0)
        },
        "frame_logs": frame_logs
    }
    return json.dumps(data, indent=2)

def export_coco_keypoints_json(
    image_name: str,
    width: int,
    height: int,
    points_dict: Dict[str, Any],
    pupil_center: tuple,
    iris_center: tuple
) -> str:
    """
    Exports keypoint annotations in standard COCO Keypoints format for machine learning dataset training.
    """
    pupil_pts = points_dict.get("pupil_points", [])
    iris_pts = points_dict.get("iris_points", [])

    coco_keypoints = []
    # Add pupil center & points (visibility flag 2 = visible)
    coco_keypoints.extend([pupil_center[0], pupil_center[1], 2])
    for pt in pupil_pts:
        coco_keypoints.extend([pt[0], pt[1], 2])

    # Add iris center & points
    coco_keypoints.extend([iris_center[0], iris_center[1], 2])
    for pt in iris_pts:
        coco_keypoints.extend([pt[0], pt[1], 2])

    num_kpts = len(pupil_pts) + len(iris_pts) + 2

    coco_data = {
        "images": [
            {
                "id": 1,
                "file_name": image_name,
                "width": width,
                "height": height
            }
        ],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "num_keypoints": num_kpts,
                "keypoints": coco_keypoints
            }
        ],
        "categories": [
            {
                "id": 1,
                "name": "eye_boundary",
                "supercategory": "eye"
            }
        ]
    }
    return json.dumps(coco_data, indent=2)

def export_yolo_keypoints_txt(
    width: int,
    height: int,
    pupil_center: tuple,
    iris_center: tuple,
    pupil_radius: float,
    iris_radius: float,
    points_dict: Dict[str, Any]
) -> str:
    """
    Exports bounding box and keypoints normalized format (0-1) for YOLO-Pose model training.
    Format: <class_id> <bbox_center_x> <bbox_center_y> <bbox_w> <bbox_h> <px1> <py1> <p_vis1> ...
    """
    icx, icy = iris_center
    ir = iris_radius

    # Bounding box around iris
    bbox_cx = icx / float(width)
    bbox_cy = icy / float(height)
    bbox_w = (ir * 2.2) / float(width)
    bbox_h = (ir * 2.2) / float(height)

    kpt_tokens = [f"0 {bbox_cx:.6f} {bbox_cy:.6f} {bbox_w:.6f} {bbox_h:.6f}"]

    # Pupil center
    kpt_tokens.append(f"{pupil_center[0]/width:.6f} {pupil_center[1]/height:.6f} 2.0")
    # Iris center
    kpt_tokens.append(f"{iris_center[0]/width:.6f} {iris_center[1]/height:.6f} 2.0")

    for pt in points_dict.get("pupil_points", []):
        kpt_tokens.append(f"{pt[0]/width:.6f} {pt[1]/height:.6f} 2.0")
    for pt in points_dict.get("iris_points", []):
        kpt_tokens.append(f"{pt[0]/width:.6f} {pt[1]/height:.6f} 2.0")

    return " ".join(kpt_tokens)
