"""
Geometric Plausibility Constraints Module for Eye Annotation Platform.
Enforces anatomical and physical boundaries on pupil and iris fits.
No detection that violates these constraints can bypass validation or receive high confidence.
"""

import math
import numpy as np
from typing import Dict, Any, List, Tuple, Optional, Union

from src.config_constants import (
    PUPIL_BORDER_MARGIN_RATIO,
    PUPIL_MIN_RADIUS_RATIO,
    PUPIL_MAX_RADIUS_RATIO,
    PUPIL_MIN_AXIS_RATIO,
    PUPIL_MIN_CONTAINMENT_RATIO,
    IRIS_MIN_RADIUS_RATIO,
    IRIS_MAX_RADIUS_RATIO,
    IRIS_MIN_AXIS_RATIO,
    IRIS_MIN_CONTAINMENT_RATIO,
    MAX_PUPIL_IRIS_CENTER_OFFSET_RATIO,
    MIN_PUPIL_IRIS_RADIUS_RATIO,
    MAX_PUPIL_IRIS_RADIUS_RATIO
)


def _extract_geometry(obj: Any) -> Optional[Dict[str, float]]:
    """Helper to extract (cx, cy, rx, ry, radius, angle, detected) from object or dict."""
    if obj is None:
        return None

    if isinstance(obj, dict):
        detected = obj.get("detected", True)
        if not detected:
            return None
        center = obj.get("center")
        if center is not None:
            cx, cy = float(center[0]), float(center[1])
        else:
            cx = float(obj.get("cx", 0.0))
            cy = float(obj.get("cy", 0.0))
        radius = float(obj.get("radius", 0.0))
        rx = float(obj.get("rx", radius))
        ry = float(obj.get("ry", radius))
        angle = float(obj.get("angle", obj.get("rotation", 0.0)))
        return {"cx": cx, "cy": cy, "rx": rx, "ry": ry, "radius": radius or max(rx, ry), "angle": angle}

    # Otherwise assume object like PupilResult / IrisResult
    if not getattr(obj, "detected", False):
        return None

    center = getattr(obj, "center", (0.0, 0.0))
    cx, cy = float(center[0]), float(center[1])
    radius = float(getattr(obj, "radius", 0.0))
    rx = float(getattr(obj, "radius_x", radius))
    ry = float(getattr(obj, "radius_y", radius))
    angle = float(getattr(obj, "rotation", 0.0))

    if hasattr(obj, "ellipse") and obj.ellipse is not None:
        try:
            ((ecx, ecy), (ew, eh), eang) = obj.ellipse
            cx, cy = float(ecx), float(ecy)
            rx, ry = float(ew / 2.0), float(eh / 2.0)
            angle = float(eang)
            radius = (rx + ry) / 2.0
        except Exception:
            pass

    return {"cx": cx, "cy": cy, "rx": rx, "ry": ry, "radius": radius or max(rx, ry), "angle": angle}


def _ellipse_containment_ratio(cx: float, cy: float, rx: float, ry: float, angle_deg: float, W: int, H: int, num_samples: int = 36) -> float:
    """Calculates fraction of ellipse perimeter points that lie within frame bounds [0, W-1] x [0, H-1]."""
    if rx <= 0 or ry <= 0:
        return 0.0

    # Fast path: if bounding box is strictly within frame, containment is 100%
    max_r = max(rx, ry)
    if (cx - max_r >= 0.0) and (cx + max_r < float(W)) and (cy - max_r >= 0.0) and (cy + max_r < float(H)):
        return 1.0

    rad = math.radians(angle_deg)
    cos_a = math.cos(rad)
    sin_a = math.sin(rad)

    inside = 0
    thetas = np.linspace(0, 2 * np.pi, num_samples, endpoint=False)
    for theta in thetas:
        dx = rx * math.cos(theta)
        dy = ry * math.sin(theta)
        x = cx + dx * cos_a - dy * sin_a
        y = cy + dx * sin_a + dy * cos_a
        if 0.0 <= x < float(W) and 0.0 <= y < float(H):
            inside += 1

    return inside / float(num_samples)


def validate_detection(
    pupil: Any,
    iris: Any,
    frame_shape: Tuple[int, int, ...]
) -> Tuple[bool, List[str], float]:
    """
    Validates pupil and iris geometry against physical eye plausibility constraints.

    Returns:
        is_valid: True only if all hard constraints pass.
        failures: List of descriptive constraint failure names.
        plausibility_score: Value in [0.0, 1.0] reflecting geometric quality.
    """
    H, W = frame_shape[:2]
    min_dim = float(min(H, W))

    failures: List[str] = []
    penalty = 0.0

    p_geom = _extract_geometry(pupil)
    i_geom = _extract_geometry(iris)

    if p_geom is None:
        return False, ["no_valid_pupil_seed"], 0.0

    pcx, pcy = p_geom["cx"], p_geom["cy"]
    prx, pry = p_geom["rx"], p_geom["ry"]
    pr = p_geom["radius"]
    p_ang = p_geom["angle"]

    # -------------------------------------------------------------
    # 1. Pupil Constraints
    # -------------------------------------------------------------
    margin = PUPIL_BORDER_MARGIN_RATIO * min_dim
    if pcx < margin or pcx > (W - margin) or pcy < margin or pcy > (H - margin):
        failures.append("pupil_center_near_border")
        penalty += 0.40

    min_pr = PUPIL_MIN_RADIUS_RATIO * min_dim
    max_pr = PUPIL_MAX_RADIUS_RATIO * min_dim
    if pr < min_pr:
        failures.append("pupil_radius_too_small")
        penalty += 0.35
    elif pr > max_pr:
        failures.append("pupil_radius_too_large")
        penalty += 0.40

    if max(prx, pry) > 0:
        p_axis_ratio = min(prx, pry) / max(prx, pry)
        if p_axis_ratio < PUPIL_MIN_AXIS_RATIO:
            failures.append("pupil_elongated_axis_ratio")
            penalty += 0.30
    else:
        failures.append("pupil_degenerate_radii")
        penalty += 0.50

    p_containment = _ellipse_containment_ratio(pcx, pcy, prx, pry, p_ang, W, H)
    if p_containment < PUPIL_MIN_CONTAINMENT_RATIO:
        failures.append("pupil_perimeter_clipped_by_frame")
        penalty += 0.40

    # -------------------------------------------------------------
    # 2. Iris Constraints (if iris present)
    # -------------------------------------------------------------
    if i_geom is not None:
        icx, icy = i_geom["cx"], i_geom["cy"]
        irx, iry = i_geom["rx"], i_geom["ry"]
        ir = i_geom["radius"]
        i_ang = i_geom["angle"]

        min_ir = min(IRIS_MIN_RADIUS_RATIO * min_dim, max(12.0, 0.85 * pr * 2.2))
        max_ir = IRIS_MAX_RADIUS_RATIO * min_dim
        if ir < min_ir:
            failures.append("iris_radius_too_small")
            penalty += 0.35
        elif ir > max_ir:
            failures.append("iris_radius_too_large")
            penalty += 0.40

        if max(irx, iry) > 0:
            i_axis_ratio = min(irx, iry) / max(irx, iry)
            if i_axis_ratio < IRIS_MIN_AXIS_RATIO:
                failures.append("iris_elongated_axis_ratio")
                penalty += 0.25
        else:
            failures.append("iris_degenerate_radii")
            penalty += 0.40

        i_containment = _ellipse_containment_ratio(icx, icy, irx, iry, i_ang, W, H)
        if i_containment < IRIS_MIN_CONTAINMENT_RATIO:
            failures.append("iris_perimeter_severely_clipped")
            penalty += 0.35

        # -------------------------------------------------------------
        # 3. Pupil-Iris Relationship Constraints
        # -------------------------------------------------------------
        # Pupil center must lie inside iris ellipse
        dx = pcx - icx
        dy = pcy - icy
        i_rad = math.radians(i_ang)
        cos_i = math.cos(i_rad)
        sin_i = math.sin(i_rad)
        x_rot = dx * cos_i + dy * sin_i
        y_rot = -dx * sin_i + dy * cos_i

        if irx > 0 and iry > 0:
            norm_dist_sq = (x_rot / irx) ** 2 + (y_rot / iry) ** 2
            if norm_dist_sq > 1.05:  # slight epsilon
                failures.append("pupil_center_outside_iris")
                penalty += 0.50

        # Center offset constraint
        center_dist = math.hypot(dx, dy)
        max_offset = MAX_PUPIL_IRIS_CENTER_OFFSET_RATIO * ir
        if center_dist > max_offset:
            failures.append("pupil_iris_centers_too_far")
            penalty += 0.40

        # Radius ratio constraint
        if ir > 0:
            r_ratio = pr / ir
            if r_ratio < MIN_PUPIL_IRIS_RADIUS_RATIO:
                failures.append("pupil_iris_radius_ratio_too_small")
                penalty += 0.35
            elif r_ratio > MAX_PUPIL_IRIS_RADIUS_RATIO:
                failures.append("pupil_iris_radius_ratio_too_large")
                penalty += 0.40

    is_valid = (len(failures) == 0)
    plausibility_score = max(0.0, min(1.0, 1.0 - penalty))

    # If invalid, score is capped strictly at 0.15
    if not is_valid:
        plausibility_score = min(plausibility_score, 0.15)

    return is_valid, failures, round(plausibility_score, 3)
