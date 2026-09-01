import numpy as np
import cv2
from typing import Dict, Any, Tuple, List
from src.pupil_detection import PupilResult
from src.iris_detection import IrisResult

def compute_eye_analytics(
    prep_dict: Dict[str, Any],
    pupil_res: PupilResult,
    iris_res: IrisResult
) -> Dict[str, Any]:
    """
    Calculates detailed biometric measurements, center decentration, 
    eccentricity, area ratios, and intensity cross-section profile.
    """
    px, py = pupil_res.center
    pr = pupil_res.radius
    p_diam = pr * 2.0
    p_area = np.pi * (pr ** 2)

    ix, iy = iris_res.center
    ir = iris_res.radius
    i_diam = ir * 2.0
    i_area = np.pi * (ir ** 2)

    # Ratios
    ratio_diameter = i_diam / p_diam if p_diam > 0 else 0.0
    ratio_area = (p_area / i_area * 100.0) if i_area > 0 else 0.0

    # Center Decentration (Displacement between pupil and iris centers)
    dx = ix - px
    dy = iy - py
    decentration_dist = np.sqrt(dx**2 + dy**2)

    # Pupil Eccentricity / Ellipticity
    eccentricity = 0.0
    if pupil_res.ellipse is not None:
        ((_, _), (a, b), _) = pupil_res.ellipse
        major = max(a, b) / 2.0
        minor = min(a, b) / 2.0
        if major > 0:
            eccentricity = np.sqrt(max(0.0, 1.0 - (minor**2 / major**2)))

    # Radial Intensity Profile along horizontal axis passing through pupil center
    gray = prep_dict["gray_inpainted"]
    h, w = gray.shape[:2]
    
    line_y = int(py)
    profile_x = list(range(0, w))
    profile_intensity = []
    if 0 <= line_y < h:
        profile_intensity = [int(gray[line_y, x]) for x in profile_x]
    else:
        profile_intensity = [0] * w

    return {
        "pupil_center_x": px,
        "pupil_center_y": py,
        "pupil_radius_px": round(pr, 1),
        "pupil_diameter_px": round(p_diam, 1),
        "pupil_area_px2": round(p_area, 1),
        "pupil_eccentricity": round(eccentricity, 3),
        "iris_center_x": ix,
        "iris_center_y": iy,
        "iris_radius_px": round(ir, 1),
        "iris_diameter_px": round(i_diam, 1),
        "iris_area_px2": round(i_area, 1),
        "iris_pupil_diameter_ratio": round(ratio_diameter, 2),
        "pupil_iris_area_percentage": round(ratio_area, 2),
        "center_decentration_dx": dx,
        "center_decentration_dy": dy,
        "center_decentration_distance": round(decentration_dist, 2),
        "profile_x": profile_x,
        "profile_intensity": profile_intensity
    }
