"""
Detection Diagnostics and Candidate Visual Debugger Module.

Generates a comprehensive visual debug grid illustrating every intermediate stage of 
the hybrid computer vision pipeline:
1. Original Eye ROI
2. Preprocessed Grayscale & CLAHE
3. Specular Reflection Mask & Inpainted
4. Binary Dark-Region Threshold & Canny Edges
5. Eyelid Occlusion Mask & Parabolic Curves
6. Pupil Candidates (Accepted vs Rejected)
7. Iris Candidates & Annular Search Region
8. Final Refined Multi-Layer Annotation
"""

from typing import Dict, Any, Optional, List
import cv2
import numpy as np


def generate_diagnostics_image_grid(
    original_rgb: np.ndarray,
    prep_dict: Dict[str, Any],
    pupil_res: Any,
    iris_res: Any,
    eyelid_obj: Optional[Any] = None,
    glint_mask: Optional[np.ndarray] = None
) -> Dict[str, np.ndarray]:
    """
    Renders diagnostic visual representations of all pipeline stages:
    1. Raw vs Inpainted Grayscale
    2. Multi-Threshold Sweep Mask
    3. Candidate Overlay (Rejected & Winning with Scores)
    4. Radial Gradient & Eyelid Sector Profile
    5. Iris Annular Search & Limbal Inliers
    6. Edge Gradient Magnitude Map
    """
    h, w = original_rgb.shape[:2]
    gray = prep_dict.get("gray", cv2.cvtColor(original_rgb, cv2.COLOR_RGB2GRAY) if len(original_rgb.shape) == 3 else original_rgb)
    gray_inpainted = prep_dict.get("gray_inpainted", gray)
    clahe = prep_dict.get("clahe", gray)
    gray_rgb = cv2.cvtColor(gray_inpainted, cv2.COLOR_GRAY2RGB)

    # 1. Raw vs Glint Inpainted
    inpainted_vis = gray_rgb.copy()
    if glint_mask is not None and np.sum(glint_mask) > 0:
        # Highlight inpainted areas with soft cyan
        inpainted_vis[glint_mask > 0] = [0, 240, 255]
    cv2.putText(inpainted_vis, "Cyan = Inpainted Glint", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 240, 255), 2, cv2.LINE_AA)

    # 2. Multi-Threshold Sweep Mask
    p10 = int(np.percentile(gray_inpainted, 10))
    p25 = int(np.percentile(gray_inpainted, 25))
    _, bin1 = cv2.threshold(gray_inpainted, p10, 255, cv2.THRESH_BINARY_INV)
    _, bin2 = cv2.threshold(gray_inpainted, p25, 255, cv2.THRESH_BINARY_INV)
    thresh_composite = np.zeros((h, w, 3), dtype=np.uint8)
    thresh_composite[bin2 > 0] = [60, 60, 180]    # P25 = Darker blue
    thresh_composite[bin1 > 0] = [100, 220, 100]  # P10 = Green
    cv2.putText(thresh_composite, "P10 (Green) / P25 (Blue)", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

    # 3. Candidate Overlay with Scores
    cand_vis = gray_rgb.copy()
    all_cands = getattr(pupil_res, "all_candidates", [])
    for idx, c in enumerate(all_cands):
        cx, cy = c.get("center", (0, 0))
        cr = int(round(c.get("radius", 0.0)))
        score = c.get("score", 0.0)
        # Winning candidate is green, others graded yellow to red
        if idx == 0 and getattr(pupil_res, "detected", False):
            color = (0, 255, 0)
            thick = 2
        elif score > 0.4:
            color = (0, 200, 255)
            thick = 1
        else:
            color = (0, 0, 255)
            thick = 1
        cv2.circle(cand_vis, (int(cx), int(cy)), max(2, cr), color, thick, cv2.LINE_AA)
        cv2.putText(cand_vis, f"{score:.2f}", (int(cx) - 15, int(cy) - cr - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)

    if pupil_res and getattr(pupil_res, "detected", False):
        px, py = pupil_res.center
        cv2.drawMarker(cand_vis, (int(px), int(py)), (0, 255, 0), cv2.MARKER_CROSS, 10, 2, cv2.LINE_AA)

    # 4. Radial Gradient & Eyelid Profile
    eyelid_vis = gray_rgb.copy()
    if eyelid_obj is not None and getattr(eyelid_obj, "valid_mask", None) is not None:
        v_mask = eyelid_obj.valid_mask
        eyelid_vis[v_mask == 0] = (eyelid_vis[v_mask == 0] * 0.4 + np.array([200, 50, 0]) * 0.6).astype(np.uint8)

    if pupil_res and getattr(pupil_res, "detected", False):
        px, py = int(pupil_res.center[0]), int(pupil_res.center[1])
        pr = float(pupil_res.radius)
        # Draw ±30° lateral sector cones around horizontal
        for ang in [-30, 30, 150, 210]:
            rad = np.radians(ang)
            ex = int(px + pr * 3.0 * np.cos(rad))
            ey = int(py + pr * 3.0 * np.sin(rad))
            cv2.line(eyelid_vis, (px, py), (ex, ey), (0, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(eyelid_vis, "±30° Lateral Valid Cones", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2, cv2.LINE_AA)

    # 5. Iris Annular Candidates & Inliers
    iris_vis = gray_rgb.copy()
    if pupil_res and getattr(pupil_res, "detected", False):
        px, py = int(pupil_res.center[0]), int(pupil_res.center[1])
        pr = float(pupil_res.radius)
        # Annular boundaries
        cv2.circle(iris_vis, (px, py), int(round(pr * 1.4)), (180, 180, 180), 1, cv2.LINE_AA)
        cv2.circle(iris_vis, (px, py), int(round(min(pr * 4.0, min(w, h) * 0.45))), (180, 180, 180), 1, cv2.LINE_AA)

    if iris_res:
        for pt in getattr(iris_res, "outlier_points", []):
            cv2.circle(iris_vis, (int(round(pt[0])), int(round(pt[1]))), 2, (0, 0, 255), -1, cv2.LINE_AA)
        for pt in getattr(iris_res, "inlier_points", []):
            cv2.circle(iris_vis, (int(round(pt[0])), int(round(pt[1]))), 2, (0, 255, 0), -1, cv2.LINE_AA)
        if getattr(iris_res, "detected", False) and iris_res.radius > 0:
            if getattr(iris_res, "ellipse", None) is not None:
                cv2.ellipse(iris_vis, iris_res.ellipse, (255, 140, 0), 2, cv2.LINE_AA)
            else:
                cv2.circle(iris_vis, (int(iris_res.center[0]), int(iris_res.center[1])), int(round(iris_res.radius)), (255, 140, 0), 2, cv2.LINE_AA)

    # 6. Edge Map
    blurred = cv2.GaussianBlur(clahe, (5, 5), 1.5)
    edges = cv2.Canny(blurred, 30, 90)
    edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)

    return {
        "1. Raw vs Inpainted Grayscale": inpainted_vis,
        "2. Multi-Threshold Sweep": thresh_composite,
        "3. Candidate Overlays & Scores": cand_vis,
        "4. Radial Gradient & Eyelid Profile": eyelid_vis,
        "5. Iris Annular Region & Inliers": iris_vis,
        "6. Canny Edge Map": edges_rgb
    }


def get_diagnostics_report(
    pupil_res: Any,
    iris_res: Any,
    conf_dict: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Builds tabular confidence breakdown and plain-language failure descriptions.
    """
    p_bd = getattr(pupil_res, "confidence_breakdown", {})
    failures = conf_dict.get("failures", p_bd.get("failures", []))

    # Tabular confidence components
    components = [
        {"Component": "Radial Gradient Support", "Weight": "30%", "Value": f"{p_bd.get('gradient_support', 0.0):.2f}", "Contribution": f"{p_bd.get('gradient_support', 0.0) * 30:.1f}%"},
        {"Component": "Darkness Contrast vs Annulus", "Weight": "20%", "Value": f"{p_bd.get('darkness_contrast', 0.0):.2f}", "Contribution": f"{p_bd.get('darkness_contrast', 0.0) * 20:.1f}%"},
        {"Component": "Contour Completeness", "Weight": "15%", "Value": f"{p_bd.get('contour_completeness', 0.0):.2f}", "Contribution": f"{p_bd.get('contour_completeness', 0.0) * 15:.1f}%"},
        {"Component": "Boundary Inlier Ratio", "Weight": "15%", "Value": f"{p_bd.get('inlier_ratio', 0.0):.2f}", "Contribution": f"{p_bd.get('inlier_ratio', 0.0) * 15:.1f}%"},
        {"Component": "Geometric Plausibility Prior", "Weight": "20%", "Value": f"{p_bd.get('plausibility_score', 1.0):.2f}", "Contribution": f"{p_bd.get('plausibility_score', 1.0) * 20:.1f}%"},
        {"Component": "Hard Constraint Gate", "Weight": "Multiplier", "Value": f"{p_bd.get('hard_constraint_gate', 1.0):.1f}", "Contribution": f"x{p_bd.get('hard_constraint_gate', 1.0):.1f}"}
    ]

    # Plain language failure explanations
    failure_explanations = {
        "pupil_center_near_border": "Pupil center is too close to frame margin (locks onto outer corner / makeup / eyelashes).",
        "pupil_radius_too_small": "Pupil radius is smaller than physiological minimum (noise speckle).",
        "pupil_radius_too_large": "Pupil radius is too large (> 18% of frame dimension, likely eyelash cluster).",
        "pupil_elongated_axis_ratio": "Pupil axis ratio is too flat/elongated (min/max < 0.55).",
        "pupil_perimeter_clipped_by_frame": "Pupil ellipse is clipped by the image border (> 8% outside frame).",
        "iris_radius_too_small": "Iris radius is smaller than physiological minimum.",
        "iris_radius_too_large": "Iris radius spans too large a fraction of the frame (> 45%).",
        "iris_perimeter_severely_clipped": "Iris ellipse is clipped heavily by image border.",
        "pupil_center_outside_iris": "Pupil center lies outside the iris ellipse.",
        "pupil_iris_centers_too_far": "Pupil and iris centers are displaced by > 35% of iris radius.",
        "pupil_iris_radius_ratio_too_small": "Pupil is unnaturally small relative to iris (< 15%).",
        "pupil_iris_radius_ratio_too_large": "Pupil is unnaturally large relative to iris (> 75%).",
        "no_valid_pupil_seed": "Iris detection skipped because no valid pupil seed was found."
    }

    plain_failures = [failure_explanations.get(f, f) for f in failures]

    return {
        "components_table": components,
        "is_valid": conf_dict.get("hard_constraint_gate", 1.0) > 0.0 and len(failures) == 0,
        "failures": failures,
        "plain_failures": plain_failures,
        "overall_confidence": conf_dict.get("overall_confidence", 0.0),
        "status": conf_dict.get("status", "NOT_DETECTED")
    }

