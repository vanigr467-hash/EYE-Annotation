"""
Multi-Factor Confidence Fusion Module.

Combines quantitative evidence from image quality, pupil candidate scoring,
iris edge inlier support, geometric validity, ellipse fitting residuals, and temporal tracking:
- Pupil Detection Confidence (0-100%)
- Iris Detection Confidence (0-100%)
- Overall Detection Confidence (0-100%)
- Categorical Status: DETECTED / LOW CONFIDENCE / PARTIAL / NOT DETECTED
"""

from typing import Dict, Any, Optional, Tuple
import numpy as np

from src.config_constants import (
    CONF_WEIGHT_GRADIENT,
    CONF_WEIGHT_CONTRAST,
    CONF_WEIGHT_COMPLETENESS,
    CONF_WEIGHT_INLIER_RATIO,
    CONF_WEIGHT_PLAUSIBILITY
)
from src.plausibility import validate_detection


def compute_calibrated_confidence(
    gradient_support: float,
    darkness_contrast: float,
    contour_completeness: float,
    inlier_ratio: float,
    plausibility_score: float,
    is_valid: bool
) -> Tuple[float, Dict[str, Any]]:
    """
    Computes calibrated confidence score using the strict physical gating formula:
    confidence = (0.30 * gradient + 0.20 * contrast + 0.15 * completeness + 
                  0.15 * inliers + 0.20 * plausibility) * hard_constraint_gate
    
    If any hard constraint fails, hard_constraint_gate = 0.0, ensuring confidence < 20% (0%).
    """
    g = float(np.clip(gradient_support, 0.0, 1.0))
    d = float(np.clip(darkness_contrast, 0.0, 1.0))
    c = float(np.clip(contour_completeness, 0.0, 1.0))
    r = float(np.clip(inlier_ratio, 0.0, 1.0))
    p = float(np.clip(plausibility_score, 0.0, 1.0))

    raw_score = (
        CONF_WEIGHT_GRADIENT * g +
        CONF_WEIGHT_CONTRAST * d +
        CONF_WEIGHT_COMPLETENESS * c +
        CONF_WEIGHT_INLIER_RATIO * r +
        CONF_WEIGHT_PLAUSIBILITY * p
    )

    hard_constraint_gate = 1.0 if is_valid else 0.0
    final_confidence = round(float(raw_score * hard_constraint_gate * 100.0), 1)

    components = {
        "gradient_support": round(g, 3),
        "darkness_contrast": round(d, 3),
        "contour_completeness": round(c, 3),
        "inlier_ratio": round(r, 3),
        "plausibility_score": round(p, 3),
        "hard_constraint_gate": hard_constraint_gate,
        "raw_score": round(raw_score, 3),
        "confidence": final_confidence
    }

    return final_confidence, components


def compute_overall_confidence(
    pupil_res: Any,
    iris_res: Any,
    validation_info: Optional[Dict[str, Any]] = None,
    quality_info: Optional[Dict[str, Any]] = None,
    weights: Optional[Dict[str, float]] = None,
    frame_shape: Optional[Tuple[int, int, ...]] = None
) -> Dict[str, Any]:
    """
    Computes unified composite detection confidence from multi-factor evidence,
    strictly gated by geometric plausibility constraints.
    """
    default_weights = {
        "pupil": 0.40,
        "iris": 0.35,
        "geometry": 0.15,
        "quality": 0.10
    }
    w = weights if weights is not None else default_weights

    # Run plausibility validation if frame_shape provided
    is_valid = True
    failures = []
    plaus_score = 1.0
    if frame_shape is not None and pupil_res is not None:
        is_valid, failures, plaus_score = validate_detection(pupil_res, iris_res, frame_shape)
    elif validation_info is not None:
        is_valid = validation_info.get("is_valid", True)
        failures = validation_info.get("failures", [])
        plaus_score = validation_info.get("plausibility_score", 1.0 if is_valid else 0.15)

    hard_constraint_gate = 1.0 if is_valid else 0.0

    # 1. Pupil Confidence
    if pupil_res is None or not getattr(pupil_res, "detected", False):
        p_conf = 0.0
        p_level = "Low"
    else:
        p_raw = float(getattr(pupil_res, "confidence", 50.0))
        p_conf = p_raw * hard_constraint_gate
        p_level = "High" if p_conf >= 75.0 else ("Medium" if p_conf >= 50.0 else "Low")

    # 2. Iris Confidence
    if iris_res is None or not getattr(iris_res, "detected", False):
        i_conf = 0.0
        i_level = "Low"
        i_quality = 0.0
        i_inlier = 0.0
    else:
        i_raw = float(getattr(iris_res, "confidence", 50.0))
        i_conf = i_raw * hard_constraint_gate
        i_quality = float(getattr(iris_res, "quality_score", i_raw)) * hard_constraint_gate
        i_inlier = float(getattr(iris_res, "inlier_ratio", 0.0))
        i_level = "High" if i_conf >= 70.0 else ("Medium" if i_conf >= 45.0 else "Low")

    # 3. Geometric Validity Score
    geo_score = (plaus_score * 100.0) if is_valid else 0.0

    # 4. Image Quality Modifier
    q_score = 80.0
    if quality_info is not None:
        q_score = float(quality_info.get("quality_score", 75.0))

    # Composite Score
    if not is_valid or (p_conf == 0.0 and i_conf == 0.0):
        overall_conf = 0.0
    elif p_conf > 0 and i_conf == 0.0:
        overall_conf = (p_conf * 0.55 + geo_score * 0.30 + q_score * 0.15) * hard_constraint_gate
    else:
        overall_conf = (
            w.get("pupil", 0.40) * p_conf +
            w.get("iris", 0.35) * i_conf +
            w.get("geometry", 0.15) * geo_score +
            w.get("quality", 0.10) * q_score
        ) * hard_constraint_gate

    overall_conf = float(np_clip(overall_conf, 0.0, 98.0))

    # Categorical Status
    if is_valid and overall_conf >= 70.0:
        overall_level = "High"
        status = "DETECTED"
    elif is_valid and overall_conf >= 40.0:
        overall_level = "Medium"
        status = "LOW CONFIDENCE" if (p_conf > 0 and i_conf > 0) else "PARTIAL"
    elif is_valid and p_conf > 0 and i_conf == 0:
        overall_level = "Low"
        status = "PARTIAL"
    else:
        overall_level = "Low"
        status = "NOT DETECTED"

    # Warnings
    warnings = []
    if not is_valid:
        warnings.append(f"Hard constraint violation: {', '.join(failures)}.")
    if p_level == "Low" and is_valid:
        warnings.append("Pupil detection confidence is low.")
    if i_level == "Low" and p_level != "Low" and is_valid:
        warnings.append("Iris boundary is partially occluded, blurry, or low contrast.")

    warning_msg = " ".join(warnings) if warnings else "High detection confidence. Boundaries validated."

    return {
        "pupil_confidence": round(p_conf, 1),
        "pupil_level": p_level,
        "iris_confidence": round(i_conf, 1),
        "iris_level": i_level,
        "iris_quality_score": round(i_quality, 1),
        "iris_inlier_ratio": round(i_inlier, 3),
        "overall_confidence": round(overall_conf, 1),
        "overall_level": overall_level,
        "hard_constraint_gate": hard_constraint_gate,
        "plausibility_score": round(plaus_score, 3),
        "failures": failures,
        "status": status,
        "warning": warning_msg
    }


def np_clip(val: float, low: float, high: float) -> float:
    return max(low, min(high, val))

