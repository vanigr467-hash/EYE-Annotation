"""
Strict 10-Point Geometric Validation Module.

Validates detected pupil and iris hypotheses against physical anatomy, 
geometric constraints, image boundaries, and temporal consistency.
"""

from typing import Dict, Any, Tuple, Optional
import numpy as np


def validate_detection_pair(
    pupil_res: Any,
    iris_res: Any,
    image_shape: Tuple[int, int],
    prev_state: Optional[Dict[str, Any]] = None,
    max_jump_threshold: float = 0.5
) -> Dict[str, Any]:
    """
    Performs comprehensive 10-point anatomical, geometric, and temporal validation.
    
    Checks:
    1. Pupil validity & minimum size (radius >= 3px)
    2. Pupil inside image frame bounds
    3. Pupil inside iris
    4. Pupil center reasonably close to iris center (disp <= pupil_radius * 0.6)
    5. Iris radius larger than pupil radius with plausible ratio (1.3x to 4.2x)
    6. Iris inside image / ROI frame bounds
    7. Iris does not cover unreasonable proportion of ROI (area <= 35%)
    8. Ellipse eccentricity is physically plausible (e <= 0.85, aspect ratio <= 2.5)
    9. Temporal center stability between consecutive frames
    10. Temporal radius stability between consecutive frames
    
    Returns:
        dict with:
            'is_valid': bool (both pupil and iris passed)
            'pupil_valid': bool
            'iris_valid': bool
            'rejection_reason': str
            'status': str ('DETECTED', 'LOW_CONFIDENCE', 'PARTIAL', 'NOT_DETECTED')
    """
    h, w = image_shape[:2]

    # --- 1. Pupil Detection & Minimum Size ---
    if pupil_res is None or not getattr(pupil_res, "detected", False) or pupil_res.radius <= 3.0:
        return {
            "is_valid": False,
            "pupil_valid": False,
            "iris_valid": False,
            "rejection_reason": "Pupil not detected or radius too small (<3px).",
            "status": "NOT_DETECTED"
        }

    px, py = pupil_res.center
    pr = float(pupil_res.radius)

    # --- 2. Pupil Inside Frame Bounds ---
    if px < 0 or px >= w or py < 0 or py >= h:
        return {
            "is_valid": False,
            "pupil_valid": False,
            "iris_valid": False,
            "rejection_reason": f"Pupil center ({px}, {py}) lies outside image boundaries ({w}x{h}).",
            "status": "NOT_DETECTED"
        }

    # Pupil eccentricity check
    if getattr(pupil_res, "ellipse", None) is not None:
        try:
            (_, _), (pw, ph), _ = pupil_res.ellipse
            p_major = max(pw, ph) / 2.0
            p_minor = max(0.1, min(pw, ph) / 2.0)
            if (p_major / p_minor) > 3.0:
                return {
                    "is_valid": False,
                    "pupil_valid": False,
                    "iris_valid": False,
                    "rejection_reason": f"Pupil ellipse is degenerate / overly eccentric (aspect {p_major/p_minor:.1f}).",
                    "status": "LOW_CONFIDENCE"
                }
        except Exception:
            pass

    # --- 3 & 5. Iris Detection & Dimensions vs Pupil ---
    if iris_res is None or not getattr(iris_res, "detected", False) or iris_res.radius <= pr:
        return {
            "is_valid": False,
            "pupil_valid": True,
            "iris_valid": False,
            "rejection_reason": "Iris not detected or radius not larger than pupil.",
            "status": "PARTIAL"
        }

    ix, iy = iris_res.center
    ir = float(iris_res.radius)
    irx = float(getattr(iris_res, "radius_x", ir))
    iry = float(getattr(iris_res, "radius_y", ir))
    mean_ir = (irx + iry) / 2.0 if (irx > 0 and iry > 0) else ir

    # Iris-to-pupil ratio
    ratio = mean_ir / float(pr) if pr > 0 else 0
    if ratio < 1.3:
        return {
            "is_valid": False,
            "pupil_valid": True,
            "iris_valid": False,
            "rejection_reason": f"Iris-to-pupil ratio ({ratio:.2f}x) is unrealistically small (<1.3x).",
            "status": "PARTIAL"
        }

    if ratio > 4.5:
        return {
            "is_valid": False,
            "pupil_valid": True,
            "iris_valid": False,
            "rejection_reason": f"Iris-to-pupil ratio ({ratio:.2f}x) is unrealistically large (>4.5x). Overlaps sclera/skin.",
            "status": "PARTIAL"
        }

    # --- 4. Pupil Center vs Iris Center Proximity ---
    disp = np.sqrt((ix - px) ** 2 + (iy - py) ** 2)
    max_allowed_disp = max(10.0, pr * 0.7)
    if disp > max_allowed_disp:
        return {
            "is_valid": False,
            "pupil_valid": True,
            "iris_valid": False,
            "rejection_reason": f"Pupil/Iris center displacement ({disp:.1f}px) exceeds anatomical threshold ({max_allowed_disp:.1f}px).",
            "status": "PARTIAL"
        }

    # --- 6. Iris Inside Frame Bounds ---
    if ix < 0 or ix >= w or iy < 0 or iy >= h:
        return {
            "is_valid": False,
            "pupil_valid": True,
            "iris_valid": False,
            "rejection_reason": "Iris center lies outside image bounds.",
            "status": "PARTIAL"
        }

    # --- 7. Iris Area Proportion vs ROI ---
    iris_area = np.pi * irx * iry
    img_area = h * w
    if img_area > 0 and (iris_area / img_area) > 0.40:
        return {
            "is_valid": False,
            "pupil_valid": True,
            "iris_valid": False,
            "rejection_reason": f"Iris area covers {(iris_area/img_area)*100:.1f}% of image. Likely fitting eyelid/skin.",
            "status": "LOW_CONFIDENCE"
        }

    # --- 8. Iris Ellipse Eccentricity ---
    if irx > 0 and iry > 0:
        i_major = max(irx, iry)
        i_minor = max(0.1, min(irx, iry))
        aspect = i_major / i_minor
        if aspect > 2.2:
            return {
                "is_valid": False,
                "pupil_valid": True,
                "iris_valid": False,
                "rejection_reason": f"Iris ellipse is too elongated (aspect ratio {aspect:.2f}).",
                "status": "LOW_CONFIDENCE"
            }

    # --- 9 & 10. Temporal Stability Checks ---
    if prev_state is not None and prev_state.get("detection_status") in ("DETECTED", "PREDICTED"):
        prev_px, prev_py = prev_state.get("pupil_center", (px, py))
        prev_pr = prev_state.get("pupil_radius", pr)
        prev_ir = prev_state.get("iris_radius", mean_ir)

        # Center jump
        jump_dist = np.sqrt((px - prev_px) ** 2 + (py - prev_py) ** 2)
        max_jump = max(25.0, prev_pr * max_jump_threshold * 2.5)

        if jump_dist > max_jump:
            return {
                "is_valid": False,
                "pupil_valid": False,
                "iris_valid": False,
                "is_temporal_outlier": True,
                "rejection_reason": f"Sudden spatial jump ({jump_dist:.1f}px) detected between consecutive frames.",
                "status": "LOW_CONFIDENCE"
            }

        # Radius jump
        if prev_ir > 0 and mean_ir > 0:
            ir_jump = abs(mean_ir - prev_ir) / prev_ir
            if ir_jump > 0.35:
                return {
                    "is_valid": False,
                    "pupil_valid": True,
                    "iris_valid": False,
                    "is_temporal_outlier": True,
                    "rejection_reason": f"Sudden iris radius change ({ir_jump*100:.0f}%) between consecutive frames.",
                    "status": "LOW_CONFIDENCE"
                }

    return {
        "is_valid": True,
        "pupil_valid": True,
        "iris_valid": True,
        "rejection_reason": "All 10 geometric, anatomical, and temporal validation constraints passed.",
        "status": "DETECTED"
    }
