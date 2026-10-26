"""
Eye biometric analytics — measurements, ratios, gaze vector, 
iris ellipse parameters, and PLR dilation velocity.
"""

import numpy as np
import cv2
from typing import Dict, Any, Tuple, List
from src.pupil_detection import PupilResult
from src.iris_detection import IrisResult
from src.gaze_tracking import calculate_gaze_vector


def compute_eye_analytics(
    prep_dict: Dict[str, Any],
    pupil_res: PupilResult,
    iris_res: IrisResult
) -> Dict[str, Any]:
    """
    Calculates detailed biometric measurements including iris ellipse
    parameters (rx, ry, rotation), 2D Gaze Vector, and intensity profile.
    """
    px, py = pupil_res.center
    pr = pupil_res.radius
    p_diam = pr * 2.0
    p_area = np.pi * (pr ** 2)

    ix, iy = iris_res.center
    ir = iris_res.radius
    irx = getattr(iris_res, 'radius_x', ir)
    iry = getattr(iris_res, 'radius_y', ir)
    i_rot = getattr(iris_res, 'rotation', 0.0)
    i_diam = ir * 2.0
    i_area = np.pi * irx * iry  # ellipse area

    # Ratios
    ratio_diameter = i_diam / p_diam if p_diam > 0 else 0.0
    ratio_area = (p_area / i_area * 100.0) if i_area > 0 else 0.0

    # Center Decentration
    dx = ix - px
    dy = iy - py
    decentration_dist = np.sqrt(dx**2 + dy**2)

    # 2D Eye Gaze Vector
    gaze_info = calculate_gaze_vector((px, py), (ix, iy), ir)

    # Pupil Eccentricity / Ellipticity
    eccentricity = 0.0
    if pupil_res.ellipse is not None:
        ((_, _), (a, b), _) = pupil_res.ellipse
        major = max(a, b) / 2.0
        minor = min(a, b) / 2.0
        if major > 0:
            eccentricity = np.sqrt(max(0.0, 1.0 - (minor**2 / major**2)))

    # Iris Eccentricity
    iris_eccentricity = 0.0
    if irx > 0 and iry > 0:
        i_major = max(irx, iry)
        i_minor = min(irx, iry)
        if i_major > 0:
            iris_eccentricity = np.sqrt(max(0.0, 1.0 - (i_minor**2 / i_major**2)))

    # Radial Intensity Profile along horizontal axis passing through pupil center
    gray = prep_dict["gray_inpainted"]
    h, w = gray.shape[:2]

    line_y = int(py)
    profile_x = list(range(0, w))
    if 0 <= line_y < h:
        profile_intensity = [int(gray[line_y, x]) for x in profile_x]
    else:
        profile_intensity = [0] * w

    # Quality diagnostics
    iris_quality = getattr(iris_res, 'quality_score', 0.0)
    iris_inlier_ratio = getattr(iris_res, 'inlier_ratio', 0.0)
    num_valid_rays = getattr(iris_res, 'num_valid_rays', 0)

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
        "iris_radius_x_px": round(irx, 1),
        "iris_radius_y_px": round(iry, 1),
        "iris_rotation_deg": round(i_rot, 1),
        "iris_eccentricity": round(iris_eccentricity, 3),
        "iris_diameter_px": round(i_diam, 1),
        "iris_area_px2": round(i_area, 1),
        "iris_pupil_diameter_ratio": round(ratio_diameter, 2),
        "pupil_iris_area_percentage": round(ratio_area, 2),
        "center_decentration_dx": dx,
        "center_decentration_dy": dy,
        "center_decentration_distance": round(decentration_dist, 2),
        "gaze_dx": gaze_info["dx"],
        "gaze_dy": gaze_info["dy"],
        "gaze_distance_px": gaze_info["distance_px"],
        "gaze_offset_pct": gaze_info["normalized_offset_pct"],
        "gaze_angle_deg": gaze_info["gaze_angle_deg"],
        "gaze_direction": gaze_info["direction_label"],
        "iris_boundary_quality": round(iris_quality, 1),
        "iris_inlier_ratio": round(iris_inlier_ratio, 3),
        "iris_valid_rays": num_valid_rays,
        "profile_x": profile_x,
        "profile_intensity": profile_intensity
    }


def compute_video_plr_dynamics(df_logs: Any) -> Any:
    """
    Computes Pupillary Light Reflex (PLR) Dilation / Constriction Velocity
    dr_p/dt across consecutive video frames.
    """
    if "pupil_radius" not in df_logs.columns or "timestamp_sec" not in df_logs.columns:
        df_logs["dilation_velocity_px_s"] = 0.0
        return df_logs

    radii = df_logs["pupil_radius"].values
    timestamps = df_logs["timestamp_sec"].values

    dt = np.diff(timestamps)
    dr = np.diff(radii)

    # Avoid division by zero
    dt[dt <= 0] = 0.04  # Default 25 FPS step

    velocity = np.zeros(len(radii), dtype=np.float32)
    velocity[1:] = np.round(dr / dt, 2)

    df_logs["dilation_velocity_px_s"] = velocity
    return df_logs
