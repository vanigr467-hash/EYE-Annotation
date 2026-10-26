import os
import sys
import cv2
import numpy as np
import traceback
from pathlib import Path
from typing import Dict, Any, List, Union

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.image_quality import analyze_image_quality
from src.eye_roi import detect_eye_roi
from src.preprocessing import preprocess_eye_image
from src.eyelid_mask import detect_eyelid_mask
from src.pupil_detector import detect_pupil, PupilResult
from src.iris_detector import detect_iris, IrisResult
from src.validation import validate_detection_pair
from src.confidence import compute_overall_confidence
from src.annotation import draw_automatic_annotations

def _remap_points(points: List[List[float]], rx: int, ry: int) -> List[List[float]]:
    if not points:
        return points
    return [[float(p[0]) + rx, float(p[1]) + ry] for p in points]

def detect_eye_boundaries(
    image_input: Union[np.ndarray, str, Path],
    use_roi: bool = True,
    use_eyelid_mask: bool = True,
    sensitivity: float = 0.5,
    fit_mode: str = 'ellipse',
    num_points: int = 64
) -> Dict[str, Any]:
    """
    Clean, stateless detection function for the eye annotation pipeline.
    Accepts either an RGB numpy image array or a file path.
    Runs the full CV pipeline and returns structured data including boundary points.
    """
    # Resolve input to RGB numpy array
    if isinstance(image_input, (str, Path)):
        file_path = str(image_input)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Image not found at {file_path}")
        bgr = cv2.imread(file_path)
        if bgr is None:
            raise ValueError(f"Failed to read image at {file_path}")
        image_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    elif isinstance(image_input, np.ndarray):
        image_rgb = image_input
    else:
        raise ValueError("image_input must be an RGB numpy array or file path string/Path")

    result = {
        'pupil': {
            'detected': False, 'cx': 0.0, 'cy': 0.0, 'rx': 0.0, 'ry': 0.0, 'angle': 0.0,
            'confidence': 0.0, 'boundary_points': [], 'status': 'Not attempted'
        },
        'iris': {
            'detected': False, 'cx': 0.0, 'cy': 0.0, 'rx': 0.0, 'ry': 0.0, 'angle': 0.0,
            'confidence': 0.0, 'quality_score': 0.0, 'boundary_points': [], 'inlier_points': [], 'status': 'Not attempted'
        },
        'quality': {'score': 0.0, 'grade': 'UNKNOWN', 'warnings': []},
        'overall_confidence': 0.0,
        'detection_status': 'FAILED',
        'annotated_image': image_rgb.copy() if image_rgb is not None else None
    }
    
    try:
        if image_rgb is None or image_rgb.size == 0:
            return result
            
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        
        # 1. Quality
        quality_info = analyze_image_quality(gray)
        result['quality'] = quality_info
        
        # 2. ROI
        roi_crop = image_rgb
        rx, ry = 0, 0
        if use_roi:
            roi_obj = detect_eye_roi(image_rgb)
            if roi_obj and roi_obj.crop is not None:
                roi_crop = roi_obj.crop
                rx, ry, _, _ = roi_obj.bbox
        
        # 3. Preprocess
        prep_dict = preprocess_eye_image(roi_crop)
        
        # 4. Pupil
        pupil_res = detect_pupil(prep_dict, sensitivity=sensitivity, fit_mode=fit_mode)
        
        # 5. Eyelid mask
        eyelid_mask = None
        if use_eyelid_mask and pupil_res.detected:
            roi_gray = cv2.cvtColor(roi_crop, cv2.COLOR_RGB2GRAY)
            eyelid_obj = detect_eyelid_mask(roi_gray, pupil_res.center, pupil_res.radius)
            if eyelid_obj:
                eyelid_mask = eyelid_obj.valid_mask
                
        # 6. Iris
        iris_res = detect_iris(prep_dict, pupil_res, eyelid_mask=eyelid_mask, sensitivity=sensitivity, fit_mode=fit_mode)
        
        # Remap coordinates if ROI used
        h, w = image_rgb.shape[:2]
        
        if rx > 0 or ry > 0:
            if pupil_res.detected:
                pupil_res.center = (pupil_res.center[0] + rx, pupil_res.center[1] + ry)
                if pupil_res.ellipse:
                    pupil_res.ellipse = ((pupil_res.ellipse[0][0] + rx, pupil_res.ellipse[0][1] + ry), pupil_res.ellipse[1], pupil_res.ellipse[2])
                    
            if iris_res.detected:
                iris_res.center = (iris_res.center[0] + rx, iris_res.center[1] + ry)
                if iris_res.ellipse:
                    iris_res.ellipse = ((iris_res.ellipse[0][0] + rx, iris_res.ellipse[0][1] + ry), iris_res.ellipse[1], iris_res.ellipse[2])
                if hasattr(iris_res, 'inlier_points') and iris_res.inlier_points is not None:
                    iris_res.inlier_points = _remap_points(iris_res.inlier_points, rx, ry)
                
        # 7. Validate
        val_info = validate_detection_pair(pupil_res, iris_res, (h, w))
        
        # 8. Confidence
        conf_dict = compute_overall_confidence(pupil_res, iris_res, val_info, quality_info)
        result['overall_confidence'] = conf_dict.get('overall_confidence', 0.0)
        result['detection_status'] = conf_dict.get('status', 'FAILED')
        
        # 9. Annotate
        annotated_rgb, points_data = draw_automatic_annotations(image_rgb, pupil_res, iris_res, num_points=num_points)
        result['annotated_image'] = annotated_rgb
        
        # Populate result dict
        if pupil_res.detected:
            pupil_pts = points_data.get('pupil_points', []) if isinstance(points_data, dict) else []
            result['pupil'] = {
                'detected': True,
                'cx': float(pupil_res.center[0]), 'cy': float(pupil_res.center[1]),
                'rx': float(pupil_res.radius_x) if hasattr(pupil_res, 'radius_x') else float(pupil_res.radius),
                'ry': float(pupil_res.radius_y) if hasattr(pupil_res, 'radius_y') else float(pupil_res.radius),
                'angle': float(pupil_res.rotation) if hasattr(pupil_res, 'rotation') else 0.0,
                'confidence': float(pupil_res.confidence),
                'boundary_points': [[float(p[0]), float(p[1])] for p in pupil_pts],
                'status': getattr(pupil_res, 'status', 'Success')
            }
            
        if iris_res.detected:
            iris_pts = points_data.get('iris_points', []) if isinstance(points_data, dict) else []
            inliers = getattr(iris_res, 'inlier_points', []) or []
            result['iris'] = {
                'detected': True,
                'cx': float(iris_res.center[0]), 'cy': float(iris_res.center[1]),
                'rx': float(iris_res.radius_x) if hasattr(iris_res, 'radius_x') else float(iris_res.radius),
                'ry': float(iris_res.radius_y) if hasattr(iris_res, 'radius_y') else float(iris_res.radius),
                'angle': float(iris_res.rotation) if hasattr(iris_res, 'rotation') else 0.0,
                'confidence': float(iris_res.confidence),
                'quality_score': float(getattr(iris_res, 'quality_score', 0.0)),
                'boundary_points': [[float(p[0]), float(p[1])] for p in iris_pts],
                'inlier_points': [[float(p[0]), float(p[1])] for p in inliers],
                'status': getattr(iris_res, 'status', 'Success')
            }
            
    except Exception as e:
        print(f"Error in detect_eye_boundaries: {e}")
        traceback.print_exc()
        result['detection_status'] = f"ERROR: {str(e)}"
        
    return result
