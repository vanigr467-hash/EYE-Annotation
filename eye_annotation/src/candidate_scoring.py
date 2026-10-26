"""
Candidate Scoring Module for Pupil and Iris Boundary Hypotheses.

Provides multi-factor evaluation models that score candidate detections based on
relative local darkness, circularity, edge gradient support, contrast against surroundings,
centrality, temporal stability, and boundary penalties.
"""

from typing import Dict, Any, Optional, Tuple, Union, List
import math
import cv2
import numpy as np

from src.config_constants import (
    WEIGHT_RADIAL_GRADIENT,
    WEIGHT_DARKNESS_CONTRAST,
    WEIGHT_CIRCULARITY,
    WEIGHT_CONTOUR_COMPLETENESS,
    WEIGHT_PLAUSIBILITY_PRIOR
)
from src.plausibility import validate_detection


def score_pupil_candidate(
    candidate: Dict[str, Any],
    gray_img: np.ndarray,
    roi_shape: Tuple[int, int],
    eyelid_mask: Optional[np.ndarray] = None,
    glint_mask: Optional[np.ndarray] = None,
    prev_center: Optional[Tuple[float, float]] = None,
    weights: Optional[Dict[str, float]] = None,
    return_breakdown: bool = False
) -> Union[float, Tuple[float, Dict[str, float]]]:
    """
    Computes a composite multi-factor score (0.0 to 1.0) for a pupil candidate.
    
    Factors:
    1. Radial gradient support (0.35): outward-increasing gradient along perimeter normals,
       penalized by gradient variance across angles.
    2. Darkness contrast (0.20): inside mean intensity vs immediate exterior annulus.
       Rejects eyelash clusters which lack a bright exterior iris ring.
    3. Circularity (0.15): axis ratio min(rx, ry) / max(rx, ry).
    4. Contour completeness (0.15): fraction of perimeter exhibiting supporting edge step.
    5. Plausibility prior (0.15): strict anatomical bounds. Hard violations return 0.0.
    """
    h, w_img = roi_shape[:2]
    
    # Extract candidate center and radii
    center = candidate.get("center", (0, 0))
    cx, cy = float(center[0]), float(center[1])
    r = float(candidate.get("radius", 0.0))
    rx = float(candidate.get("rx", candidate.get("radius_x", r)))
    ry = float(candidate.get("ry", candidate.get("radius_y", r)))
    ang = float(candidate.get("angle", candidate.get("rotation", 0.0)))

    if r <= 3 or rx <= 0 or ry <= 0 or cx < 0 or cx >= w_img or cy < 0 or cy >= h:
        empty_breakdown = {
            "gradient": 0.0, "contrast": 0.0, "circularity": 0.0,
            "completeness": 0.0, "plausibility": 0.0
        }
        return (0.0, empty_breakdown) if return_breakdown else 0.0

    # -------------------------------------------------------------
    # 5. Plausibility Prior (Strict Gate)
    # -------------------------------------------------------------
    p_dict = {
        "cx": cx, "cy": cy, "rx": rx, "ry": ry, "radius": r, "angle": ang, "detected": True
    }
    is_valid, failures, plausibility_score = validate_detection(p_dict, iris=None, frame_shape=(h, w_img))
    if not is_valid:
        # Immediate hard disqualification for border-clipping, giant, or degenerate candidates
        empty_breakdown = {
            "gradient": 0.0, "contrast": 0.0, "circularity": 0.0,
            "completeness": 0.0, "plausibility": 0.0, "failures": failures
        }
        return (0.0, empty_breakdown) if return_breakdown else 0.0

    # -------------------------------------------------------------
    # 1. Radial Gradient Support & 4. Contour Completeness
    # -------------------------------------------------------------
    num_samples = 32
    thetas = np.linspace(0, 2 * np.pi, num_samples, endpoint=False)
    rad_ang = math.radians(ang)
    cos_a, sin_a = math.cos(rad_ang), math.sin(rad_ang)

    delta = 2.5
    edge_steps = []
    inner_vals = []
    outer_vals = []

    for th in thetas:
        cos_th = math.cos(th)
        sin_th = math.sin(th)

        # Ellipse points at r, r - delta, r + delta
        # Unrotated offsets
        dx_mid = rx * cos_th
        dy_mid = ry * sin_th
        dx_in = max(1.0, rx - delta) * cos_th
        dy_in = max(1.0, ry - delta) * sin_th
        dx_out = (rx + delta) * cos_th
        dy_out = (ry + delta) * sin_th

        # Rotated to image frame
        px_in = int(round(cx + dx_in * cos_a - dy_in * sin_a))
        py_in = int(round(cy + dx_in * sin_a + dy_in * cos_a))
        px_out = int(round(cx + dx_out * cos_a - dy_out * sin_a))
        py_out = int(round(cy + dx_out * sin_a + dy_out * cos_a))

        # Check bounds
        if 0 <= px_in < w_img and 0 <= py_in < h and 0 <= px_out < w_img and 0 <= py_out < h:
            val_in = float(gray_img[py_in, px_in])
            val_out = float(gray_img[py_out, px_out])
            grad_step = val_out - val_in  # Positive when interior is darker than exterior
            edge_steps.append(grad_step)
            inner_vals.append(val_in)
            outer_vals.append(val_out)
        else:
            edge_steps.append(0.0)

    if edge_steps:
        pos_steps = [max(0.0, s) for s in edge_steps]
        mean_step = float(np.mean(pos_steps))
        std_step = float(np.std(pos_steps))

        # Normalized gradient strength
        grad_strength = min(1.0, mean_step / 18.0)
        # Variance penalty: penalize erratic, spiky eyelash gradients
        var_penalty = 1.0 - 0.45 * min(1.0, std_step / (mean_step + 1e-4))
        f_gradient = float(np.clip(grad_strength * var_penalty, 0.0, 1.0))

        # Completeness: fraction of angles with clear outward positive edge step
        edge_thresh = 4.0
        supporting_rays = sum(1 for s in edge_steps if s >= edge_thresh)
        f_completeness = float(supporting_rays / float(num_samples))
    else:
        f_gradient = 0.0
        f_completeness = 0.0

    # -------------------------------------------------------------
    # 2. Darkness Contrast vs Local Exterior Annulus
    # -------------------------------------------------------------
    # Sample interior core (0.65 * r) and exterior ring (1.45 * r)
    core_samples = []
    annulus_samples = []
    for th in thetas:
        cos_th = math.cos(th)
        sin_th = math.sin(th)

        px_core = int(round(cx + (0.65 * rx * cos_th) * cos_a - (0.65 * ry * sin_th) * sin_a))
        py_core = int(round(cy + (0.65 * rx * cos_th) * sin_a + (0.65 * ry * sin_th) * cos_a))

        px_ann = int(round(cx + (1.45 * rx * cos_th) * cos_a - (1.45 * ry * sin_th) * sin_a))
        py_ann = int(round(cy + (1.45 * rx * cos_th) * sin_a + (1.45 * ry * sin_th) * cos_a))

        if 0 <= px_core < w_img and 0 <= py_core < h:
            core_samples.append(float(gray_img[py_core, px_core]))
        if 0 <= px_ann < w_img and 0 <= py_ann < h:
            annulus_samples.append(float(gray_img[py_ann, px_ann]))

    if core_samples and annulus_samples:
        mean_core = float(np.mean(core_samples))
        mean_ann = float(np.mean(annulus_samples))
        contrast_diff = mean_ann - mean_core  # Exterior iris is brighter than interior pupil
        f_contrast = float(np.clip(contrast_diff / 28.0, 0.0, 1.0))
    else:
        f_contrast = 0.0

    # -------------------------------------------------------------
    # 3. Circularity (Axis Ratio)
    # -------------------------------------------------------------
    axis_ratio = min(rx, ry) / max(rx, ry)
    f_circularity = float(np.clip(axis_ratio, 0.0, 1.0))

    # -------------------------------------------------------------
    # Composite Multi-Factor Score
    # -------------------------------------------------------------
    w_grad = weights.get("gradient", WEIGHT_RADIAL_GRADIENT) if weights else WEIGHT_RADIAL_GRADIENT
    w_cont = weights.get("contrast", WEIGHT_DARKNESS_CONTRAST) if weights else WEIGHT_DARKNESS_CONTRAST
    w_circ = weights.get("circularity", WEIGHT_CIRCULARITY) if weights else WEIGHT_CIRCULARITY
    w_comp = weights.get("completeness", WEIGHT_CONTOUR_COMPLETENESS) if weights else WEIGHT_CONTOUR_COMPLETENESS
    w_plaus = weights.get("plausibility", WEIGHT_PLAUSIBILITY_PRIOR) if weights else WEIGHT_PLAUSIBILITY_PRIOR

    composite_score = (
        w_grad * f_gradient +
        w_cont * f_contrast +
        w_circ * f_circularity +
        w_comp * f_completeness +
        w_plaus * plausibility_score
    )

    # Optional eyelid occlusion penalty
    if eyelid_mask is not None:
        pupil_circle_mask = np.zeros((h, w_img), dtype=np.uint8)
        cv2.circle(pupil_circle_mask, (int(round(cx)), int(round(cy))), int(round(r)), 255, -1)
        occluded = np.sum((pupil_circle_mask == 255) & (eyelid_mask == 0))
        total_p = np.sum(pupil_circle_mask == 255)
        if total_p > 0 and (occluded / float(total_p)) > 0.55:
            composite_score *= 0.65

    # Optional temporal prior penalty if large jump
    if prev_center is not None and prev_center[0] > 0:
        jump = math.hypot(cx - prev_center[0], cy - prev_center[1])
        if jump > 3.0 * r:
            composite_score *= 0.5

    final_score = float(np.clip(composite_score, 0.0, 1.0))

    breakdown = {
        "gradient": round(f_gradient, 3),
        "contrast": round(f_contrast, 3),
        "circularity": round(f_circularity, 3),
        "completeness": round(f_completeness, 3),
        "plausibility": round(plausibility_score, 3),
        "final_score": round(final_score, 3)
    }

    if return_breakdown:
        return final_score, breakdown
    return final_score


def score_iris_candidate(
    iris_radius: float,
    iris_center: Tuple[float, float],
    pupil_radius: float,
    pupil_center: Tuple[float, float],
    edge_inlier_ratio: float,
    texture_contrast_score: float,
    eyelid_occlusion_pct: float
) -> float:
    """
    Computes a composite score (0.0 to 1.0) for an iris boundary candidate.
    """
    if pupil_radius <= 0 or iris_radius <= pupil_radius:
        return 0.0

    ratio = iris_radius / pupil_radius
    if 1.5 <= ratio <= 3.4:
        ratio_score = 1.0
    elif 1.25 <= ratio <= 4.0:
        ratio_score = 0.70
    else:
        ratio_score = 0.20

    dx = iris_center[0] - pupil_center[0]
    dy = iris_center[1] - pupil_center[1]
    disp = np.sqrt(dx**2 + dy**2)
    disp_score = float(np.clip(1.0 - (disp / max(8.0, pupil_radius * 0.7)), 0.0, 1.0))

    occ_score = float(np.clip(1.0 - (eyelid_occlusion_pct / 80.0), 0.0, 1.0))

    score = (
        ratio_score * 0.30 +
        edge_inlier_ratio * 0.30 +
        texture_contrast_score * 0.25 +
        disp_score * 0.10 +
        occ_score * 0.05
    )

    return float(np.clip(score, 0.0, 1.0))
